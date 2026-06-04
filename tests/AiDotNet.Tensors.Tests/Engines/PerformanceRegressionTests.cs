using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Performance regression tests that catch scalar loops and ensure critical ops
/// run within acceptable bounds. These prevent the class of bugs from issues
/// #100, #102, #104 where operations silently used scalar loops instead of BLAS/SIMD.
///
/// Each test measures a critical operation and asserts it completes within a
/// time budget. The budgets are generous (10-50x headroom) so they pass on
/// any reasonable hardware, but catch 60x regressions from scalar loops.
///
/// These are runnable perf gates (Category=Perf + AIDOTNET_RUN_PERF_GATES=1 opt-in,
/// matching FusedConv2DResnetSpeedupTests): a no-op on default/shared CI runners,
/// executable on dedicated hardware. Run with:
///   AIDOTNET_RUN_PERF_GATES=1 dotnet test --filter "Category=Perf"
/// (A bare [Fact(Skip=...)] never runs even with --filter, so it cannot gate.)
/// </summary>
public class PerformanceRegressionTests
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // Budgets: set to catch scalar loop regressions (60x+ slower than BLAS) while
    // tolerating CI noise from parallel test execution and cold JIT.
    //
    // PyTorch CPU baselines (measured on reference hardware):
    //   MatMul 32x256@256x256: 0.023ms   | MatMul 16x768@768x768: 0.074ms
    //   FusedLinear 32x256+ReLU: 0.033ms  | FusedLinear double: 0.068ms
    //   Add 100K: 0.015ms                 | Multiply 100K: 0.013ms
    //   BatchMatMul [4,32,64]: 0.022ms    | ReLU backward 1M: 1.500ms
    //
    // Our clean-run performance:
    //   MatMul: 0.033ms (1.4x PT) | FusedLinear: 0.129ms (3.9x PT)
    //   Add: 0.084ms (5.6x PT)    | BatchMatMul: 0.241ms (11x PT)
    //
    // Budgets are generous enough for CI but catch any regression to scalar loops.
    // Scalar loops produce 2-20ms for these ops, so 1-2ms budgets catch them.
    private const double MatMul256BudgetMs = 1.5;       // Scalar would be ~3ms
    private const double MatMul768BudgetMs = 5.0;       // Scalar would be ~20ms
    private const double FusedLinear256BudgetMs = 2.0;   // Scalar would be ~3ms
    private const double FusedLinearDoubleBudgetMs = 2.0; // Scalar would be ~3ms
    private const double Elementwise100KBudgetMs = 1.0;  // Scalar would be ~2ms
    private const double BatchMatMulBudgetMs = 2.0;      // Scalar would be ~5ms
    private const double ReLUBackwardBudgetMs = 5.0;     // Scalar would be ~10ms

    // AIsEval CNN inference perf gate — see ooples/AiDotNet.Tensors#436.
    // The AIsEval (PyTorch-vs-AiDotNet) benchmark surfaced that AiDotNet's CNN
    // bs=128 inference is currently ~6.7x faster than PyTorch's nn.Conv2d path
    // on the same shapes (6.21 ms vs 41.42 ms on the reference rig). These two
    // gates lock in the two Conv2D forward shapes that dominate the CNN
    // benchmark (the layer-1 and layer-2 conv at SmallCNN's input/intermediate
    // resolutions) so a future refactor of the conv path can't silently regress
    // the lead. Budgets are 4-5x our measured numbers — generous enough for CI
    // noise, tight enough to catch any 5x+ regression.
    private const double CnnAiseval_L1_BudgetMs = 30.0;  // Measured ~6 ms on RTX-class CPU
    private const double CnnAiseval_L2_BudgetMs = 35.0;  // Measured ~8 ms on RTX-class CPU

    // AIsEval LSTM + MHA inference perf gates (issue #436, post-Stage 5-7).
    // Budgets are sized to catch a regression to the previous-stage path,
    // not just generic CI noise:
    //
    //   LSTM:
    //     Stage 3 (generic-T fused, pre float fast path): 44 ms/iter
    //     Stage 5 (float fast path, current):              8 ms/iter
    //     30 ms budget catches Stage 5 -> Stage 3 regression with ~3x
    //     headroom over the Stage 5 measurement.
    //
    //   MHA:
    //     Stage 4 (wrapper path, no float fast path):     22 ms/iter
    //     Stage 6 (float fast path):                      10 ms/iter
    //     Stage 7 (+ parallel transposes, current):        8.5 ms/iter
    //     15 ms budget catches Stage 7/6 -> Stage 4 regression while still
    //     tolerating ~1.7x CI variance over the Stage 7 measurement. The
    //     previous 30 ms budget (CodeRabbit review on PR #437) would have
    //     passed a wrapper-path fallback unchanged — it caught only outright
    //     unfused dispatch, not the regression this test is named for.
    // (These gates run on net5+ only; see PerfGatesEnabled — net471 perf timing is
    // dominated by the OS scheduler quantum, not the kernel.)
    private const double LstmAiseval_BudgetMs = 30.0;
    private const double MhaAiseval_BudgetMs = 15.0;

    // AIsEval MLP inference perf gate (issue #436 P1). The MlpForward fused
    // primitive shipped in #437; the AIsEval rerun measured the framework's
    // unfused per-layer Predict path at 8.94 ms on
    // Dense(784->512)->Dense(512->128)->Dense(128->10) at bs=128.
    // The #436 fresh head-to-head (min-of-7-rounds) then measured the fused
    // primitive at ~1.65 ms median / 3.24 ms p95, and the native-BLAS
    // thread-cap (one thread per 16 rows; see CpuEngine.Mlp.cs) brought it to
    // ~1.37 ms median / 1.75 ms p95 on a 16-core box — the thread cap removed
    // the small-GEMM oversubscription jitter. Budget tightened to 4 ms (net10.0):
    // it still catches a regression to the 8.94 ms unfused path or the un-capped
    // p95 (~3.2 ms) while leaving ~2.9x headroom over the capped median for slower
    // CI hardware. (Beating PyTorch's ~0.6 ms median outright is blocked by
    // OpenBLAS-vs-MKL small-GEMM kernel quality — tracked as a follow-up.)
    // (net5+ only; see PerfGatesEnabled — net471 perf timing is dominated by the
    // OS scheduler quantum, not the kernel.)
    private const double MlpAiseval_BudgetMs = 4.0;

    public PerformanceRegressionTests(ITestOutputHelper output) => _output = output;

    // Perf gates execute only on dedicated hardware (AIDOTNET_RUN_PERF_GATES=1).
    // On shared/CI runners they no-op so sub-millisecond latency noise can't fail
    // them; the Category=Perf trait also keeps them out of the default CI filter.
    // This replaces the old [Fact(Skip=...)] pattern, which never runs even with
    // --filter and therefore could not actually gate anything.
    private bool PerfGatesEnabled()
    {
#if NET471
        // Perf gates target the performance-critical modern runtime (net5+), where
        // System.Runtime.Intrinsics is available. On net471 the fast paths fall back
        // to Vector<T>/scalar kernels, and — more decisively — the parallel dispatch
        // paths sleep-and-wake on the ~15.6 ms Windows scheduler quantum rather than
        // spin-waiting, so sub-millisecond ops report ~15.78 ms identically across
        // unrelated kernels (TensorMatMul-768, TensorAdd-100K, Tensor.Add all land at
        // the same ~15.8 ms). That floor is the OS timer, not a regression signal, so
        // gating perf on net471 is meaningless. net471 correctness is covered by the
        // functional suites; perf is gated on net5+ only. No-op here.
        _output.WriteLine("Skip: perf gates are net5+-only (net471 sub-ms timing hits the OS scheduler quantum).");
        return false;
#else
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_GATES") == "1") return true;
        _output.WriteLine("Skip: AIDOTNET_RUN_PERF_GATES != 1 (shared-runner perf gates are flaky).");
        return false;
#endif
    }

    // Per-iter MEDIAN (robust to a single GC pause, unlike a mean) over `iters`
    // timed runs, after `warmup` untimed runs to settle JIT + caches.
    private static double MedianMs(int warmup, int iters, Action body)
    {
        for (int w = 0; w < warmup; w++) body();
        var samples = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            body();
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        return samples[iters / 2];
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void TensorMatMul_256x256_NotScalarLoop()
    {
        if (!PerfGatesEnabled()) return;

        var a = Tensor<float>.CreateRandom([32, 256]);
        var b = Tensor<float>.CreateRandom([256, 256]);

        double ms = MedianMs(5, 20, () => _engine.TensorMatMul(a, b));

        _output.WriteLine($"TensorMatMul 32x256 @ 256x256: {ms:F3}ms median (budget: {MatMul256BudgetMs}ms)");
        Assert.True(ms < MatMul256BudgetMs,
            $"TensorMatMul took {ms:F3}ms — exceeds {MatMul256BudgetMs}ms budget. " +
            "Likely using scalar loops instead of BLAS. Check CpuEngine.TensorMatMul2D.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void TensorMatMul_768x768_NotScalarLoop()
    {
        if (!PerfGatesEnabled()) return;

        var a = Tensor<float>.CreateRandom([16, 768]);
        var b = Tensor<float>.CreateRandom([768, 768]);

        double ms = MedianMs(3, 10, () => _engine.TensorMatMul(a, b));

        _output.WriteLine($"TensorMatMul 16x768 @ 768x768: {ms:F3}ms median (budget: {MatMul768BudgetMs}ms)");
        Assert.True(ms < MatMul768BudgetMs,
            $"TensorMatMul took {ms:F3}ms — exceeds {MatMul768BudgetMs}ms budget. " +
            "Likely using scalar loops instead of BLAS.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void FusedLinear_256x256_NotScalarLoop()
    {
        if (!PerfGatesEnabled()) return;

        var input = Tensor<float>.CreateRandom([32, 256]);
        var weights = Tensor<float>.CreateRandom([256, 256]);
        var bias = Tensor<float>.CreateRandom([1, 256]);

        double ms = MedianMs(5, 20, () => _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU));

        _output.WriteLine($"FusedLinear 32x256 + ReLU: {ms:F3}ms median (budget: {FusedLinear256BudgetMs}ms)");
        Assert.True(ms < FusedLinear256BudgetMs,
            $"FusedLinear took {ms:F3}ms — exceeds {FusedLinear256BudgetMs}ms budget. " +
            "Check CpuFusedOperations.FusedGemmBiasActivation for scalar loops.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void FusedLinear_Double_NotScalarLoop()
    {
        if (!PerfGatesEnabled()) return;

        var input = new Tensor<double>(Enumerable.Range(0, 32 * 256).Select(i => (double)i / 1000).ToArray(), [32, 256]);
        var weights = new Tensor<double>(Enumerable.Range(0, 256 * 256).Select(i => (double)i / 100000).ToArray(), [256, 256]);
        var bias = new Tensor<double>(Enumerable.Range(0, 256).Select(i => 0.01 * i).ToArray(), [1, 256]);

        double ms = MedianMs(5, 20, () => _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU));

        _output.WriteLine($"FusedLinear double 32x256 + ReLU: {ms:F3}ms median (budget: {FusedLinearDoubleBudgetMs}ms)");
        Assert.True(ms < FusedLinearDoubleBudgetMs,
            $"FusedLinear double took {ms:F3}ms — exceeds {FusedLinearDoubleBudgetMs}ms budget.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void TensorAdd_100K_NotScalarLoop()
    {
        if (!PerfGatesEnabled()) return;

        var a = Tensor<float>.CreateRandom([100000]);
        var b = Tensor<float>.CreateRandom([100000]);

        double ms = MedianMs(10, 50, () => _engine.TensorAdd(a, b));

        _output.WriteLine($"TensorAdd 100K: {ms:F3}ms median (budget: {Elementwise100KBudgetMs}ms)");
        Assert.True(ms < Elementwise100KBudgetMs,
            $"TensorAdd took {ms:F3}ms — exceeds {Elementwise100KBudgetMs}ms budget.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void TensorMultiply_100K_NotScalarLoop()
    {
        if (!PerfGatesEnabled()) return;

        var a = Tensor<float>.CreateRandom([100000]);
        var b = Tensor<float>.CreateRandom([100000]);

        double ms = MedianMs(10, 50, () => _engine.TensorMultiply(a, b));

        _output.WriteLine($"TensorMultiply 100K: {ms:F3}ms median (budget: {Elementwise100KBudgetMs}ms)");
        Assert.True(ms < Elementwise100KBudgetMs,
            $"TensorMultiply took {ms:F3}ms — exceeds {Elementwise100KBudgetMs}ms budget.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void TensorInstanceMethod_MatrixMultiply_UsesBLAS()
    {
        if (!PerfGatesEnabled()) return;

        // Ensures Tensor.MatrixMultiply routes through engine (not scalar loops)
        var a = Tensor<float>.CreateRandom([32, 256]);
        var b = Tensor<float>.CreateRandom([256, 256]);

        double ms = MedianMs(5, 20, () => a.MatrixMultiply(b));

        _output.WriteLine($"Tensor.MatrixMultiply 32x256 @ 256x256: {ms:F3}ms median (budget: {MatMul256BudgetMs}ms)");
        Assert.True(ms < MatMul256BudgetMs,
            $"Tensor.MatrixMultiply took {ms:F3}ms — likely not routing through engine BLAS.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void TensorInstanceMethod_Add_UsesSIMD()
    {
        if (!PerfGatesEnabled()) return;

        var a = Tensor<float>.CreateRandom([100000]);
        var b = Tensor<float>.CreateRandom([100000]);

        double ms = MedianMs(10, 50, () => a.Add(b));

        _output.WriteLine($"Tensor.Add 100K: {ms:F3}ms median (budget: {Elementwise100KBudgetMs}ms)");
        Assert.True(ms < Elementwise100KBudgetMs,
            $"Tensor.Add took {ms:F3}ms — likely not routing through engine SIMD.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void TensorInstanceMethod_PointwiseMultiply_UsesSIMD()
    {
        if (!PerfGatesEnabled()) return;

        var a = Tensor<float>.CreateRandom([100000]);
        var b = Tensor<float>.CreateRandom([100000]);

        double ms = MedianMs(10, 50, () => a.PointwiseMultiply(b));

        _output.WriteLine($"Tensor.PointwiseMultiply 100K: {ms:F3}ms median (budget: {Elementwise100KBudgetMs}ms)");
        Assert.True(ms < Elementwise100KBudgetMs,
            $"Tensor.PointwiseMultiply took {ms:F3}ms — likely not routing through engine SIMD.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void BatchMatMul_3D_UsesBLAS()
    {
        if (!PerfGatesEnabled()) return;

        var a = Tensor<float>.CreateRandom([4, 32, 64]);
        var b = Tensor<float>.CreateRandom([4, 64, 32]);

        double ms = MedianMs(5, 20, () => _engine.BatchMatMul(a, b));

        _output.WriteLine($"BatchMatMul [4,32,64]@[4,64,32]: {ms:F3}ms median (budget: {BatchMatMulBudgetMs}ms)");
        Assert.True(ms < BatchMatMulBudgetMs,
            $"BatchMatMul took {ms:F3}ms — check if BLAS path is active for batch slices.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void ReLUBackward_1M_UsesSIMD()
    {
        if (!PerfGatesEnabled()) return;

        var gradOutput = Tensor<float>.CreateRandom([1000000]);
        var input = Tensor<float>.CreateRandom([1000000]);

        // SIMD ReLU backward on 1M: ~1-2ms. Scalar: ~10ms+
        double ms = MedianMs(5, 20, () => _engine.ReluBackward(gradOutput, input));

        _output.WriteLine($"ReLU backward 1M: {ms:F3}ms median (budget: {ReLUBackwardBudgetMs}ms)");
        Assert.True(ms < ReLUBackwardBudgetMs,
            $"ReLU backward took {ms:F3}ms — check SimdKernels.ReluBackwardUnsafe.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void Conv2D_AiseValCnnL1_Forward_PreservesPyTorchLead()
    {
        if (!PerfGatesEnabled()) return;

        // AIsEval CNN layer-1 shape: nn.Conv2d(1, 16, kernel=3, padding=1) at
        // input [128, 1, 28, 28] (MNIST-shape, bs=128). PyTorch baseline for
        // this exact shape was 41.42 ms (bs=128 steady-state latency including
        // ReLU + MaxPool, but Conv2D dominates). Our measured pure-Conv2D was
        // ~5-6 ms. Locking the gate at 30 ms catches a 5x+ regression while
        // tolerating CI noise and cold JIT.
        var input = Tensor<float>.CreateRandom([128, 1, 28, 28]);
        var kernel = Tensor<float>.CreateRandom([16, 1, 3, 3]);

        double ms = MedianMs(3, 10, () => _engine.Conv2D(input, kernel, stride: 1, padding: 1));

        _output.WriteLine($"Conv2D AIsEval-L1 [128,1,28,28]@[16,1,3,3] pad=1: {ms:F3}ms median (budget: {CnnAiseval_L1_BudgetMs}ms)");
        Assert.True(ms < CnnAiseval_L1_BudgetMs,
            $"Conv2D AIsEval-L1 took {ms:F3}ms — exceeds {CnnAiseval_L1_BudgetMs}ms budget. " +
            "The AIsEval benchmark relies on this shape being well below PyTorch's 41.42ms baseline; " +
            "a regression here would erase the CNN inference lead documented in issue #436.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void LstmSequenceForward_Aiseval_FloatFastPath_BeatsPyTorch()
    {
        if (!PerfGatesEnabled()) return;

        // Locks in the AIsEval LSTM win. Stage 3 took LSTM from "doesn't
        // finish in 3+ min" to 44 ms (generic-T path). Stage 5 added a
        // float fast path using SimdGemm + vectorized sigmoid/tanh +
        // ArrayPool scratch, bringing the AIsEval shape to 8.19 ms on
        // net10.0 — faster than PyTorch's 11.76 ms nn.LSTM. This gate
        // catches any regression that would lose that lead.
        const int batch = 128, seq = 32, inFeatures = 32, hidden = 64;
        var input = Tensor<float>.CreateRandom(batch, seq, inFeatures);
        var wIh   = Tensor<float>.CreateRandom(4 * hidden, inFeatures);
        var wHh   = Tensor<float>.CreateRandom(4 * hidden, hidden);

        // LstmSequenceForward is on IEngine (the float fast path lives in
        // CpuEngine and is picked up by DirectGpuTensorEngine via inheritance),
        // so no downcast needed.
        double ms = MedianMs(3, 10, () => _engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null));

        _output.WriteLine($"LstmSequenceForward AIsEval [128,32,32->64]: {ms:F2} ms median (budget: {LstmAiseval_BudgetMs} ms; PyTorch baseline 11.76 ms)");
        Assert.True(ms < LstmAiseval_BudgetMs,
            $"LstmSequenceForward took {ms:F2} ms — exceeds {LstmAiseval_BudgetMs} ms budget. " +
            "This indicates a regression away from the Stage 5 float fast path that put AiDotNet ahead of PyTorch on this shape.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void MultiHeadAttentionForward_Aiseval_FloatFastPath()
    {
        if (!PerfGatesEnabled()) return;

        // Locks in the Stage 6 MHA float fast path. Stage 4 wrapper measured
        // 22.31 ms on net10.0; Stage 6 brought it to 10.13 ms via direct
        // SimdGemm projections + ArrayPool scratch + explicit-loop transpose.
        // Budget at 15 ms catches regression back to the wrapper-only path.
        const int batch = 128, seq = 32, dModel = 64, numHeads = 4;
        var input = Tensor<float>.CreateRandom(batch, seq, dModel);
        var qW = Tensor<float>.CreateRandom(dModel, dModel);
        var kW = Tensor<float>.CreateRandom(dModel, dModel);
        var vW = Tensor<float>.CreateRandom(dModel, dModel);
        var oW = Tensor<float>.CreateRandom(dModel, dModel);

        // MultiHeadAttentionForward is on IEngine (the float fast path lives in
        // CpuEngine and is picked up by DirectGpuTensorEngine via inheritance),
        // so no downcast needed.
        double ms = MedianMs(3, 10, () => _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads));

        _output.WriteLine($"MultiHeadAttentionForward AIsEval [128,32,64] h=4: {ms:F2} ms median (budget: {MhaAiseval_BudgetMs} ms)");
        Assert.True(ms < MhaAiseval_BudgetMs,
            $"MultiHeadAttentionForward took {ms:F2} ms — exceeds {MhaAiseval_BudgetMs} ms budget. " +
            "Stage 6 float fast path measured 10.13 ms on net10.0; a regression to the Stage 4 wrapper path would push this back to ~22 ms.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void MlpForward_Aiseval_FusedDispatch_BeatsUnfusedPath()
    {
        if (!PerfGatesEnabled()) return;

        // AIsEval MLP inference shape: Dense(784->512)->Dense(512->128)->Dense(128->10)
        // at bs=128 — three GEMMs of (128,784,512), (128,512,128), (128,128,10).
        // PyTorch did this in 1.91 ms; the framework's per-layer
        // MatMul+Add+Activation Predict path measured 8.94 ms (issue #436 P1).
        // MlpForward collapses the 3-layer stack to 3 fused-linear dispatches; the
        // #436 head-to-head measured it at ~1.65 ms median pre-cap and ~1.37 ms
        // median / 1.75 ms p95 once the native-BLAS thread cap (one thread / 16
        // rows; see CpuEngine.Mlp.cs) removed the small-GEMM oversubscription
        // jitter — i.e. it already meets the issue's <= 3 ms acceptance, so the
        // remaining work is framework-side wiring to consume it (ooples/AiDotNet#1447).
        // The 4 ms budget catches a regression back to the 8.94 ms unfused dispatch
        // while leaving ~7x headroom over the measured fused median for CI noise.
        var input = Tensor<float>.CreateRandom([128, 784]);
        var weights = new List<Tensor<float>>
        {
            Tensor<float>.CreateRandom([784, 512]),
            Tensor<float>.CreateRandom([512, 128]),
            Tensor<float>.CreateRandom([128, 10]),
        };
        var biases = new List<Tensor<float>?>
        {
            Tensor<float>.CreateRandom([512]),
            Tensor<float>.CreateRandom([128]),
            Tensor<float>.CreateRandom([10]),
        };

        // MlpForward is on IEngine (the fused path lives in CpuEngine and is
        // picked up by DirectGpuTensorEngine via inheritance), so no downcast.
        double ms = MedianMs(5, 50, () => _engine.MlpForward(input, weights, biases, FusedActivationType.ReLU, FusedActivationType.None));

        _output.WriteLine($"MlpForward AIsEval [128,784->512->128->10]: {ms:F3} ms/iter median " +
            $"(budget: {MlpAiseval_BudgetMs} ms; PyTorch baseline 1.91 ms; framework unfused path 8.94 ms)");
        Assert.True(ms < MlpAiseval_BudgetMs,
            $"MlpForward took {ms:F3} ms median — exceeds {MlpAiseval_BudgetMs} ms budget. " +
            "The fused 3-dispatch path should stay below the 8.94 ms unfused-dispatch number " +
            "from issue #436; a regression here means the primitive is no longer collapsing the dense stack.");
    }

    [Fact]
    [Trait("Category", "Perf")]
    public void Conv2D_AiseValCnnL2_Forward_PreservesPyTorchLead()
    {
        if (!PerfGatesEnabled()) return;

        // AIsEval CNN layer-2 shape: nn.Conv2d(16, 32, kernel=3, padding=1) at
        // input [128, 16, 14, 14] (post-first-MaxPool). This is the heavier of
        // the two convs by FLOPs (~ 36M MACs vs ~6M for L1). Budget at 35 ms
        // covers the same 5x-regression-catching headroom.
        var input = Tensor<float>.CreateRandom([128, 16, 14, 14]);
        var kernel = Tensor<float>.CreateRandom([32, 16, 3, 3]);

        double ms = MedianMs(3, 10, () => _engine.Conv2D(input, kernel, stride: 1, padding: 1));

        _output.WriteLine($"Conv2D AIsEval-L2 [128,16,14,14]@[32,16,3,3] pad=1: {ms:F3}ms median (budget: {CnnAiseval_L2_BudgetMs}ms)");
        Assert.True(ms < CnnAiseval_L2_BudgetMs,
            $"Conv2D AIsEval-L2 took {ms:F3}ms — exceeds {CnnAiseval_L2_BudgetMs}ms budget. " +
            "Layer-2 conv dominates the CNN benchmark's FLOPs; a regression here is what would " +
            "show up first in the AIsEval bs=128 numbers.");
    }
}
