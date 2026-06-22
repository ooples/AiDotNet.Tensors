using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// #1662 lever #3 wiring: the tape-path engine backward
/// <see cref="CpuEngine.FlashAttentionBackward{T}"/> can route through the tiled
/// <see cref="FusedAttention{T}.Backward"/> (flag <c>CpuEngine.UseTiledFusedAttentionBackward</c>).
///
/// <para>This gate proves the swap is BEHAVIOURALLY IDENTICAL to the legacy in-place blocked
/// kernel (<c>FlashAttentionBackwardFloat/Double</c>) the tape used before — across plain, causal,
/// rank-4 <c>[B,H,Sq,Sk]</c> bias AND rank-3 <c>[H,Sq,Sk]</c> (batch-broadcast) bias, which the
/// engine accepts but the standalone FusedAttention parity test did not previously cover. Tiled
/// (Sk &gt; 128) and full-path (Sk ≤ 128) shapes are both exercised. dQ/dK/dV from the two paths
/// must agree to 1e-3 (both are FlashAttention math; the difference is recompute-vs-stored-stats
/// and GEMM-vs-scalar accumulation order).</para>
/// </summary>
public class FusedAttentionTapeWiringTests
{
    private readonly ITestOutputHelper _out;
    public FusedAttentionTapeWiringTests(ITestOutputHelper o) => _out = o;

    public enum BiasMode { None, Causal, Bias4D, Bias3D }

    [Theory]
    [InlineData(2, 4, 64, 16, BiasMode.None)]    // full-matrix path
    [InlineData(1, 8, 256, 32, BiasMode.None)]   // tiled (2 key tiles)
    [InlineData(2, 4, 200, 16, BiasMode.None)]   // tiled, uneven last tile
    [InlineData(1, 4, 256, 32, BiasMode.Causal)] // tiled causal
    [InlineData(2, 4, 200, 16, BiasMode.Causal)] // tiled causal, uneven
    [InlineData(1, 4, 256, 32, BiasMode.Bias4D)] // tiled rank-4 bias
    [InlineData(2, 4, 200, 16, BiasMode.Bias3D)] // tiled rank-3 (batch-broadcast) bias
    [InlineData(1, 2, 64, 16, BiasMode.Bias3D)]  // full-path rank-3 bias
    public void TapeBackward_Tiled_MatchesLegacyBlocked(int B, int H, int S, int Dh, BiasMode mode)
    {
        var engine = new CpuEngine();
        var q  = RandomTensor(new[] { B, H, S, Dh }, 101);
        var k  = RandomTensor(new[] { B, H, S, Dh }, 102);
        var v  = RandomTensor(new[] { B, H, S, Dh }, 103);
        var dO = RandomTensor(new[] { B, H, S, Dh }, 104);

        bool isCausal = mode == BiasMode.Causal;
        Tensor<float>? bias = mode switch
        {
            BiasMode.Bias4D => RandomTensor(new[] { B, H, S, S }, 105),
            BiasMode.Bias3D => RandomTensor(new[] { H, S, S }, 106),
            _ => null,
        };

        double scale = 1.0 / Math.Sqrt(Dh);
        var output = engine.FlashAttention<float>(q, k, v, scale, isCausal, out var stats, bias);

        var prev = CpuEngine.UseTiledFusedAttentionBackward;
        try
        {
            CpuEngine.UseTiledFusedAttentionBackward = false; // legacy blocked kernel
            engine.FlashAttentionBackward<float>(dO, q, k, v, output, stats, scale, isCausal,
                out var lq, out var lk, out var lv, bias);

            CpuEngine.UseTiledFusedAttentionBackward = true;  // tiled FusedAttention.Backward
            engine.FlashAttentionBackward<float>(dO, q, k, v, output, stats, scale, isCausal,
                out var tq, out var tk, out var tv, bias);

            AssertClose(lq.AsSpan().ToArray(), tq.AsSpan().ToArray(), 1e-3f, "dQ");
            AssertClose(lk.AsSpan().ToArray(), tk.AsSpan().ToArray(), 1e-3f, "dK");
            AssertClose(lv.AsSpan().ToArray(), tv.AsSpan().ToArray(), 1e-3f, "dV");
        }
        finally
        {
            CpuEngine.UseTiledFusedAttentionBackward = prev;
        }
    }

    // Informational benchmark (not a hard perf gate beyond a loose sanity bound): measures the
    // per-call time + main-thread allocation of the tape backward with the tiled path ON vs OFF
    // for a realistic long-sequence shape. Establishes the "not a regression" evidence behind
    // making the tiled path the default. Dumps numbers to a file for inspection.
    [Fact]
    public void TapeBackward_Tiled_NotARegression_Bench()
    {
        var engine = new CpuEngine();
        const int B = 2, H = 8, S = 512, Dh = 64;
        var q  = RandomTensor(new[] { B, H, S, Dh }, 201);
        var k  = RandomTensor(new[] { B, H, S, Dh }, 202);
        var v  = RandomTensor(new[] { B, H, S, Dh }, 203);
        var dO = RandomTensor(new[] { B, H, S, Dh }, 204);
        double scale = 1.0 / Math.Sqrt(Dh);
        var output = engine.FlashAttention<float>(q, k, v, scale, false, out var stats, null);

        long Measure(bool tiled, int warmup, int reps, out double msPerCall)
        {
            CpuEngine.UseTiledFusedAttentionBackward = tiled;
            for (int i = 0; i < warmup; i++)
                engine.FlashAttentionBackward<float>(dO, q, k, v, output, stats, scale, false,
                    out _, out _, out _, null);
#if NET5_0_OR_GREATER
            long a0 = GC.GetAllocatedBytesForCurrentThread();
#endif
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < reps; i++)
                engine.FlashAttentionBackward<float>(dO, q, k, v, output, stats, scale, false,
                    out _, out _, out _, null);
            sw.Stop();
            msPerCall = sw.Elapsed.TotalMilliseconds / reps;
#if NET5_0_OR_GREATER
            return (GC.GetAllocatedBytesForCurrentThread() - a0) / reps;
#else
            return -1;
#endif
        }

        var prev = CpuEngine.UseTiledFusedAttentionBackward;
        try
        {
            long offAlloc = Measure(false, 5, 20, out double offMs);
            long onAlloc  = Measure(true,  5, 20, out double onMs);
            var msg = $"FlashAttentionBackward [B={B},H={H},S={S},Dh={Dh}] " +
                      $"OFF(blocked): {offMs:F3} ms/call, {offAlloc} B/call | " +
                      $"ON(tiled): {onMs:F3} ms/call, {onAlloc} B/call | " +
                      $"ratio time {onMs / Math.Max(offMs, 1e-9):F2}x";
            _out.WriteLine(msg);
            System.IO.File.WriteAllText(@"C:\Users\cheat\flashbwd_bench.txt", msg);

            // INFORMATIONAL ONLY — no wall-clock assertion. A timing threshold under `dotnet test`
            // (Debug build, tests running in parallel on a shared machine) is inherently flaky:
            // observed OFF/ON ratios swung 2.9×–9.3× run-to-run. The durable conclusion the bench
            // established — the tiled recompute path is consistently SLOWER than the blocked kernel,
            // with no memory win (both O(block)) — is what makes the legacy blocked kernel the
            // DEFAULT and the tiled path opt-in (AIDOTNET_TILED_ATTN_BWD=1). Correctness of the
            // tiled tape path is gated separately by TapeBackward_Tiled_MatchesLegacyBlocked.
            // Sanity-only: both paths produced finite, non-trivial timings.
            Assert.True(offMs > 0 && onMs > 0, "benchmark produced no timing");
        }
        finally
        {
            CpuEngine.UseTiledFusedAttentionBackward = prev;
        }
    }

    private static void AssertClose(float[] expected, float[] actual, float tol, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        int worst = -1; float worstAbs = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = Math.Abs(expected[i] - actual[i]);
            float bound = tol * (1f + Math.Abs(expected[i]));
            if (d > bound && d > worstAbs) { worstAbs = d; worst = i; }
        }
        Assert.True(worst < 0,
            $"{name} mismatch at {worst}: legacy {(worst >= 0 ? expected[worst] : 0)} vs tiled " +
            $"{(worst >= 0 ? actual[worst] : 0)} (absdiff {worstAbs}, tol {tol})");
    }

    private static Tensor<float> RandomTensor(int[] shape, int seed)
    {
        var r = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(r.NextDouble() - 0.5);
        return t;
    }
}
