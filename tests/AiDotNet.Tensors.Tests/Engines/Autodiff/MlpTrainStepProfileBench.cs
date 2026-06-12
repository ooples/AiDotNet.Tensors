using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Localizes the third-party-benchmark MLP gradient/memory gap vs PyTorch.
/// Splits a single MLP train step into inference-forward (no tape), training-forward
/// (tape recording), and backward (ComputeGradients), and reports allocated bytes per
/// step. ~468K-param MLP (256→512→512→128→10), batch 64 — the param-matched model in
/// the report. Run manually:
///   dotnet test --filter "FullyQualifiedName~MlpTrainStepProfileBench" -- (uses ITestOutputHelper)
/// </summary>
public class MlpTrainStepProfileBench
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public MlpTrainStepProfileBench(ITestOutputHelper output) => _output = output;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }

    [Fact(Skip = "Benchmark — run manually")]
    [Trait("Category", "Benchmark")]
    public void Profile_MlpTrainStep()
    {
        const int B = 64;
        int[] dims = { 256, 512, 512, 128, 10 };
        var x = Rand(new[] { B, dims[0] }, 1);
        var W = new Tensor<float>[4];
        var bb = new Tensor<float>[4];
        for (int l = 0; l < 4; l++)
        {
            W[l] = Rand(new[] { dims[l], dims[l + 1] }, 10 + l);
            bb[l] = Rand(new[] { dims[l + 1] }, 20 + l);
        }

        // Inference forward (no tape) — the optimized path.
        Tensor<float> InferenceForward()
        {
            var h = x;
            for (int l = 0; l < 4; l++)
            {
                var lin = _engine.TensorMatMul(h, W[l]);
                var bia = _engine.TensorBroadcastAdd(lin, bb[l]);
                h = l < 3 ? _engine.ReLU(bia) : bia;
            }
            return h;
        }

        // Full train step (forward records the tape, then backward).
        void TrainStep()
        {
            using var tape = new GradientTape<float>();
            var h = x;
            for (int l = 0; l < 4; l++)
            {
                var lin = _engine.TensorMatMul(h, W[l]);
                var bia = _engine.TensorBroadcastAdd(lin, bb[l]);
                h = l < 3 ? _engine.ReLU(bia) : bia;
            }
            var loss = _engine.ReduceSum(h, new[] { 0, 1 }, keepDims: false);
            tape.ComputeGradients(loss, new[] { W[0], bb[0], W[1], bb[1], W[2], bb[2], W[3], bb[3] });
        }

        // Same step, but via ComputeGradientsScope — gradient tensors are returned to the pool
        // on scope dispose and reused by the next step's backward (the existing #327 mechanism).
        var sources = new[] { W[0], bb[0], W[1], bb[1], W[2], bb[2], W[3], bb[3] };
        float sink = 0f;
        void TrainStepScoped()
        {
            using var tape = new GradientTape<float>();
            var h = x;
            for (int l = 0; l < 4; l++)
            {
                var lin = _engine.TensorMatMul(h, W[l]);
                var bia = _engine.TensorBroadcastAdd(lin, bb[l]);
                h = l < 3 ? _engine.ReLU(bia) : bia;
            }
            var loss = _engine.ReduceSum(h, new[] { 0, 1 }, keepDims: false);
            using var scope = tape.ComputeGradientsScope(loss, sources);
            // Touch each gradient like an optimizer would (so the buffers are genuinely produced).
            foreach (var g in scope.Grads.Values) sink += g.GetFlat(0);
        }

        // Warmup (JIT + autotune).
        for (int i = 0; i < 30; i++) { InferenceForward(); TrainStep(); TrainStepScoped(); }

        double MinMs(Action a, int iters)
        {
            double best = double.MaxValue;
            for (int i = 0; i < iters; i++)
            {
                var sw = Stopwatch.StartNew();
                a();
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            return best;
        }

        double infMs = MinMs(() => InferenceForward(), 300);
        double trainMs = MinMs(TrainStep, 300);

        // Allocations per train step (steady state).
        // Use GC.GetTotalAllocatedBytes(precise: true) — the per-thread variant
        // misses allocations made on ThreadPool worker threads (Parallel.For,
        // Task continuations, BlasManaged parallel-GEMM dispatch), which under-
        // counts the steady-state cost of every kernel that fans out.
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long before = GC.GetTotalAllocatedBytes(precise: true);
        const int allocIters = 50;
        for (int i = 0; i < allocIters; i++) TrainStep();
        long after = GC.GetTotalAllocatedBytes(precise: true);
        double allocKbPerStep = (after - before) / 1024.0 / allocIters;

        // Allocations for inference forward (for contrast).
        long ib = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < allocIters; i++) InferenceForward();
        long ia = GC.GetTotalAllocatedBytes(precise: true);
        double infAllocKb = (ia - ib) / 1024.0 / allocIters;

        // Forward WITH tape recording but NO backward — isolates tape-recording allocation
        // from backward (gradient output + intermediate) allocation.
        void ForwardWithTape()
        {
            using var tape = new GradientTape<float>();
            var h = x;
            for (int l = 0; l < 4; l++)
            {
                var lin = _engine.TensorMatMul(h, W[l]);
                var bia = _engine.TensorBroadcastAdd(lin, bb[l]);
                h = l < 3 ? _engine.ReLU(bia) : bia;
            }
            _engine.ReduceSum(h, new[] { 0, 1 }, keepDims: false);
        }
        long fb = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < allocIters; i++) ForwardWithTape();
        long fa = GC.GetTotalAllocatedBytes(precise: true);
        double fwdTapeKb = (fa - fb) / 1024.0 / allocIters;

        // Allocation with ComputeGradientsScope (pooled gradient buffers across steps).
        for (int i = 0; i < 10; i++) TrainStepScoped();
        long sb = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < allocIters; i++) TrainStepScoped();
        long sa = GC.GetTotalAllocatedBytes(precise: true);
        double scopedKb = (sa - sb) / 1024.0 / allocIters;

        // Raw single GEMM [64,256]x[256,512] to separate dispatch overhead from GEMM speed.
        var w0 = W[0];
        double rawGemmMs = MinMs(() => _engine.TensorMatMul(x, w0), 500);
        long rb = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < allocIters; i++) _engine.TensorMatMul(x, w0);
        long ra = GC.GetTotalAllocatedBytes(precise: true);
        double rawGemmKb = (ra - rb) / 1024.0 / allocIters;
        double gemmGflops = (2.0 * B * dims[0] * dims[1]) / (rawGemmMs * 1e6);

        _output.WriteLine($"ENGINE = {_engine.GetType().Name}");
        _output.WriteLine($"  raw GEMM [64,256]x[256,512]: {rawGemmMs:F4} ms ({gemmGflops:F1} GFLOP/s), alloc {rawGemmKb:F1} KB");
        // Direct BlasManaged.Gemm (no engine dispatch, pre-allocated C) at M=256 — isolates the
        // kernel from the eager TensorMatMul path (tape check, alloc, contiguity, dispatch).
        {
            int dm = 256, dk = 256, dn = 512;
            var daArr = Rand(new[] { dm, dk }, 7).GetDataArray();
            var dbArr = w0.GetDataArray();
            var dc = new float[dm * dn];
            void DirectGemm() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
                daArr.AsSpan(), dk, false, dbArr.AsSpan(), dn, false, dc.AsSpan(), dn, dm, dn, dk,
                new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float>());
            for (int i = 0; i < 30; i++) DirectGemm();
            double dms = MinMs(DirectGemm, 300);
            double dgf = (2.0 * dm * dk * dn) / (dms * 1e6);
            _output.WriteLine($"  DIRECT BlasManaged.Gemm M=256 (autotune ON):  {dms:F4} ms, {dgf:F1} GFLOP/s");
            // Same, but DisableAutotune — exactly what the engine's SimdGemm.Sgemm shim passes.
            void DirectGemmNoAuto() => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
                daArr.AsSpan(), dk, false, dbArr.AsSpan(), dn, false, dc.AsSpan(), dn, dm, dn, dk,
                new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float>
                { PackingMode = AiDotNet.Tensors.Engines.BlasManaged.PackingMode.DisableAutotune });
            for (int i = 0; i < 30; i++) DirectGemmNoAuto();
            double dms2 = MinMs(DirectGemmNoAuto, 300);
            double dgf2 = (2.0 * dm * dk * dn) / (dms2 * 1e6);
            _output.WriteLine($"  DIRECT BlasManaged.Gemm M=256 (DisableAutotune=engine path): {dms2:F4} ms, {dgf2:F1} GFLOP/s");
        }
        // M-scaling: same K=256,N=512, growing M — isolates per-call packing/dispatch overhead.
        foreach (int m in new[] { 32, 64, 128, 256, 512, 1024 })
        {
            var xm = Rand(new[] { m, 256 }, 99);
            for (int i = 0; i < 20; i++) _engine.TensorMatMul(xm, w0);
            double ms = MinMs(() => _engine.TensorMatMul(xm, w0), 200);
            double gf = (2.0 * m * 256 * 512) / (ms * 1e6);
            _output.WriteLine($"    M={m,5}: {ms:F4} ms, {gf,6:F1} GFLOP/s");
        }
        _output.WriteLine($"MLP {string.Join("→", dims)} batch={B}");
        _output.WriteLine($"  inference forward (no tape): {infMs:F3} ms,  alloc {infAllocKb:F1} KB/step");
        _output.WriteLine($"  forward WITH tape (no bwd):  alloc {fwdTapeKb:F1} KB/step  (tape-record cost {fwdTapeKb - infAllocKb:F1} KB)");
        _output.WriteLine($"  full train step (fwd+bwd):   {trainMs:F3} ms,  alloc {allocKbPerStep:F1} KB/step  (backward cost {allocKbPerStep - fwdTapeKb:F1} KB)");
        _output.WriteLine($"  train step via ComputeGradientsScope (pooled): alloc {scopedKb:F1} KB/step  ({allocKbPerStep - scopedKb:F1} KB saved, {(1 - scopedKb / Math.Max(allocKbPerStep, 1e-9)) * 100:F0}%)  sink={sink:E1}");
        _output.WriteLine($"  backward overhead (train-inf): {trainMs - infMs:F3} ms  ({(trainMs / Math.Max(infMs, 1e-9)):F1}× inference)");
        _output.WriteLine($"  alloc amplification (train/inf): {allocKbPerStep / Math.Max(infAllocKb, 1e-6):F1}×");

        // Memory metric comparability: PyTorch reports process RSS (psutil); the AiDotNet harness
        // reports managed heap (GC.GetTotalMemory). Show both — and the per-window peak RSS — to
        // confirm the reported gap is real (process RSS >= managed heap, so true RSS is even higher).
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        double managedMb = MemoryMetrics.ManagedHeapMb;
        double rssMb = MemoryMetrics.CurrentProcessRssMb;
        double peakWindowMb;
        using (var sampler = new MemoryMetrics.PeakRssSampler(sampleIntervalMs: 2))
        {
            for (int i = 0; i < 200; i++) TrainStep();
            peakWindowMb = sampler.PeakMb;
        }
        _output.WriteLine($"  MEMORY: managed-heap {managedMb:F1} MB | process-RSS {rssMb:F1} MB | peak-RSS over 200 train steps {peakWindowMb:F1} MB");
        _output.WriteLine($"  (PyTorch comparable metric = peak-RSS; the harness's managedRssMbPeak measures managed-heap and understates footprint)");
    }
}
