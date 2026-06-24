using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Regression guard for the lean output-only SDPA in the fused MHA float forward
/// (<see cref="CpuEngine.ScaledDotProductAttentionFloatInto"/>). MHA discards the attention
/// weights, so the forward routes through the output-only kernel instead of the general
/// ScaledDotProductAttention, which would allocate a weightsData array, an attentionWeights
/// tensor and an output tensor and then copy out. On [128,32,64] h=4 that cut per-call
/// allocation from ~7.2 MB to ~1.0 MB (~6.9×) and sustained mean latency ~2.7×. This test
/// fails if the forward regresses back to the weight-materializing path.
/// </summary>
// Disable xUnit parallel execution for this class — the gate samples
// GC.GetTotalAllocatedBytes, a PROCESS-WIDE counter (deliberately, so it also
// catches the SDPA path's parallel per-head worker-thread allocations that a
// thread-local counter would miss). If any other test class allocates
// concurrently during the measured window, those bytes are attributed to our
// per-call figure and the threshold trips on the noise — observed as 37986 KB/call
// under a full parallel suite run while the real path is ~1024 KB (passes in
// isolation). Same rationale and pattern as OptimizerKernelsAllocSerial.
[CollectionDefinition("MhaLeanSdpaAllocSerial", DisableParallelization = true)]
public class MhaLeanSdpaAllocSerialCollection { }

[Collection("MhaLeanSdpaAllocSerial")]
public class MhaLeanSdpaAllocTests
{
    private static Tensor<float> Rand(Random r, params int[] s)
    {
        var t = new Tensor<float>(s);
        var sp = t.AsWritableSpan();
        for (int i = 0; i < sp.Length; i++) sp[i] = (float)(r.NextDouble() * 2 - 1);
        return t;
    }

#if NET5_0_OR_GREATER
    // Category=Performance so the coverage-instrumented CI correctness run (which filters
    // out Category!=Performance) excludes this allocation-measurement gate. Coverlet's
    // per-line hit counters allocate on every instrumented call across all parallel SDPA
    // worker threads, and GC.GetTotalAllocatedBytes counts ALL threads — so under coverage
    // this measured 9017 KB/call (higher than even the old 7200 KB path) while the real
    // path is ~1024 KB locally (where this passes). The threshold is unchanged; the gate
    // is just moved out of the instrumented lane where the measurement is meaningless,
    // exactly like PrePackSpeedupTest's wall-clock gates.
    [Trait("Category", "Performance")]
    [Fact]
    public void MhaForward_AllocatesFarLessThanWeightMaterializingPath()
    {
        var eng = new CpuEngine();
        const int batch = 128, seq = 32, dModel = 64, numHeads = 4;
        var r = new Random(2026);
        var input = Rand(r, batch, seq, dModel);
        var qW = Rand(r, dModel, dModel); var kW = Rand(r, dModel, dModel);
        var vW = Rand(r, dModel, dModel); var oW = Rand(r, dModel, dModel);

        for (int w = 0; w < 5; w++) _ = eng.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);

        const int n = 20;
        // Measure allocations on the CALLING thread only. GC.GetTotalAllocatedBytes is
        // process-wide, so when this runs in the full parallel suite the allocations of
        // every other concurrently-executing test land in the window and inflate the
        // figure (it measured ~9 MB/call under that contention vs ~1 MB real) — a load-
        // dependent false positive. The regression this guards against (reverting to the
        // weight-materializing SDPA, which allocates weightsData + attentionWeights +
        // output on the CALLING thread) shows up fully in the per-thread counter, so the
        // 3 MB threshold is unchanged and the guard is now immune to neighbouring tests.
        long a0 = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < n; i++) _ = eng.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        long a1 = GC.GetAllocatedBytesForCurrentThread();
        double kbPerCall = (a1 - a0) / 1024.0 / n;

        // Lean path measures ~1.0 MB/call; the old weight-materializing path was ~7.2 MB/call.
        // 3 MB threshold catches a regression with wide margin against measurement noise.
        Assert.True(kbPerCall < 3072,
            $"MHA forward allocated {kbPerCall:F0} KB/call; expected < 3072 KB (lean ~1024, " +
            "old weight-materializing path ~7200). Regressed to the non-lean SDPA?");
    }

    private static Tensor<double> RandD(Random r, params int[] s)
    {
        var t = new Tensor<double>(s);
        var sp = t.AsWritableSpan();
        for (int i = 0; i < sp.Length; i++) sp[i] = r.NextDouble() * 2 - 1;
        return t;
    }

    // #478 / #1675 regression guard: <double> MHA forward must use the pooled per-head fast path
    // (MultiHeadAttentionForwardDouble), NOT fall to MultiHeadAttentionForwardGeneric — which
    // materializes the full [B*H, seq, seq] attention scores + weights + four transpose tensors per
    // call (unpooled, > 2^20 elems), measured at 54x the float path's allocation per MHA layer and
    // the direct cause of a <double> transformer forward being ~65x slower than <float> on a
    // memory-bounded host (GC/heap churn, not math). Measured here at a seq where the generic path's
    // full-scores allocation is large enough to be unambiguous.
    [Trait("Category", "Performance")]
    [Fact]
    public void MhaForward_Double_UsesPooledFastPath_NotGenericFullScores()
    {
        var eng = new CpuEngine();
        const int batch = 1, seq = 256, dModel = 1280, numHeads = 20;
        var r = new Random(2026);
        var input = RandD(r, batch, seq, dModel);
        var qW = RandD(r, dModel, dModel); var kW = RandD(r, dModel, dModel);
        var vW = RandD(r, dModel, dModel); var oW = RandD(r, dModel, dModel);

        for (int w = 0; w < 5; w++) _ = eng.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);

        const int n = 10;
        long a0 = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < n; i++) _ = eng.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        long a1 = GC.GetAllocatedBytesForCurrentThread();
        double kbPerCall = (a1 - a0) / 1024.0 / n;

        // Fast path: pooled Q/K/V/concat scratch + one output tensor — a few MB/call. The generic
        // full-scores path it replaced allocated ~tens of MB/call at this shape (the [1,20,256,256]
        // scores + weights alone are ~10 MB each in double). 16 MB threshold catches a revert with
        // a wide margin against measurement noise.
        Assert.True(kbPerCall < 16384,
            $"Double MHA forward allocated {kbPerCall:F0} KB/call; expected < 16384 KB (pooled fast " +
            "path). Regressed to MultiHeadAttentionForwardGeneric's full-scores path (#478/#1675)?");
    }

    // #478 regression guard for the generic (double) LSTM forward: it must reuse ping-pong state
    // buffers + a pooled hidden-GEMM buffer across timesteps, NOT allocate a fresh hCurr/cCurr +
    // hidden GEMM output EVERY step (which measured ~25-28x the float fast path's allocation on a
    // 64-step LSTM and is the same GC-churn class as the MHA #478 fix).
    [Trait("Category", "Performance")]
    [Fact]
    public void LstmForward_Double_ReusesBuffers_NotPerStepAlloc()
    {
        var eng = new CpuEngine();
        const int batch = 4, seq = 128, inF = 256, hidden = 256;
        var r = new Random(2026);
        var input = RandD(r, batch, seq, inF);
        var wIh = RandD(r, 4 * hidden, inF);
        var wHh = RandD(r, 4 * hidden, hidden);
        var bIh = RandD(r, 4 * hidden);
        var bHh = RandD(r, 4 * hidden);

        for (int w = 0; w < 5; w++) _ = eng.LstmSequenceForward(input, null, null, wIh, wHh, bIh, bHh, returnSequences: true);

        const int n = 10;
        long a0 = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < n; i++) _ = eng.LstmSequenceForward(input, null, null, wIh, wHh, bIh, bHh, returnSequences: true);
        long a1 = GC.GetAllocatedBytesForCurrentThread();
        double kbPerCall = (a1 - a0) / 1024.0 / n;

        // After the ping-pong + pooled-GEMM fix this is a small constant per call (state buffers +
        // one-time input projection + output). The pre-fix per-step path allocated 2*seqLen state
        // tensors + seqLen GEMM outputs => many MB/call that scales with seq. 8 MB threshold catches
        // a revert to per-step allocation with a wide margin.
        Assert.True(kbPerCall < 8192,
            $"Double LSTM forward allocated {kbPerCall:F0} KB/call; expected < 8192 KB (buffer-reusing " +
            "path). Regressed to the per-timestep CreateZeros + per-step hidden-GEMM allocation (#478)?");
    }

    // #478 regression guard for the systemic zero-copy result hand-off: BatchNorm (and the conv /
    // SDPA / norm backward paths) must wrap their freshly-allocated result arrays with
    // Vector<T>.FromMemory, NOT `new Vector<T>(arr)`, which routes through the IEnumerable<T> ctor and
    // COPIES the whole array — doubling the result allocation (BatchNorm forward measured 4x the float
    // path; FromMemory brings it to ~2x = just the bytes).
    [Trait("Category", "Performance")]
    [Fact]
    public void BatchNormForward_Double_ZeroCopyHandoff_NotVectorCopy()
    {
        var eng = new CpuEngine();
        const int n = 8, c = 256, h = 16, w = 16;
        var r = new Random(2026);
        var input = RandD(r, n, c, h, w);
        var gamma = RandD(r, c); var beta = RandD(r, c);

        for (int wi = 0; wi < 5; wi++) _ = eng.BatchNorm(input, gamma, beta, 1e-5, out _, out _);

        const int reps = 10;
        long a0 = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < reps; i++) _ = eng.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
        long a1 = GC.GetAllocatedBytesForCurrentThread();
        double kbPerCall = (a1 - a0) / 1024.0 / reps;

        // Output [8,256,16,16] double ≈ 4 MB; the fixed path hands it off zero-copy. The reverted
        // `new Vector<double>(outDArr)` path COPIED it (~8 MB/call). 6 MB threshold catches a revert.
        Assert.True(kbPerCall < 6144,
            $"Double BatchNorm allocated {kbPerCall:F0} KB/call; expected < 6144 KB (zero-copy " +
            "FromMemory hand-off). Regressed to new Vector<double>(arr) which copies the result (#478)?");
    }
#endif
}
