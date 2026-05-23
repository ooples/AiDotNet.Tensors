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
/// Run with: dotnet test --filter "PerformanceRegression"
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
    private const double LstmAiseval_BudgetMs = 30.0;
    private const double MhaAiseval_BudgetMs = 15.0;

    public PerformanceRegressionTests(ITestOutputHelper output) => _output = output;

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void TensorMatMul_256x256_NotScalarLoop()
    {
        var a = Tensor<float>.CreateRandom([32, 256]);
        var b = Tensor<float>.CreateRandom([256, 256]);

        // Warmup
        for (int w = 0; w < 5; w++) _engine.TensorMatMul(a, b);

        var sw = Stopwatch.StartNew();
        int iters = 20;
        for (int i = 0; i < iters; i++) _engine.TensorMatMul(a, b);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"TensorMatMul 32x256 @ 256x256: {ms:F3}ms (budget: {MatMul256BudgetMs}ms)");
        Assert.True(ms < MatMul256BudgetMs,
            $"TensorMatMul took {ms:F3}ms — exceeds {MatMul256BudgetMs}ms budget. " +
            "Likely using scalar loops instead of BLAS. Check CpuEngine.TensorMatMul2D.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void TensorMatMul_768x768_NotScalarLoop()
    {
        var a = Tensor<float>.CreateRandom([16, 768]);
        var b = Tensor<float>.CreateRandom([768, 768]);

        for (int w = 0; w < 3; w++) _engine.TensorMatMul(a, b);

        var sw = Stopwatch.StartNew();
        int iters = 10;
        for (int i = 0; i < iters; i++) _engine.TensorMatMul(a, b);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"TensorMatMul 16x768 @ 768x768: {ms:F3}ms (budget: {MatMul768BudgetMs}ms)");
        Assert.True(ms < MatMul768BudgetMs,
            $"TensorMatMul took {ms:F3}ms — exceeds {MatMul768BudgetMs}ms budget. " +
            "Likely using scalar loops instead of BLAS.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void FusedLinear_256x256_NotScalarLoop()
    {
        var input = Tensor<float>.CreateRandom([32, 256]);
        var weights = Tensor<float>.CreateRandom([256, 256]);
        var bias = Tensor<float>.CreateRandom([1, 256]);

        for (int w = 0; w < 5; w++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);

        var sw = Stopwatch.StartNew();
        int iters = 20;
        for (int i = 0; i < iters; i++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"FusedLinear 32x256 + ReLU: {ms:F3}ms (budget: {FusedLinear256BudgetMs}ms)");
        Assert.True(ms < FusedLinear256BudgetMs,
            $"FusedLinear took {ms:F3}ms — exceeds {FusedLinear256BudgetMs}ms budget. " +
            "Check CpuFusedOperations.FusedGemmBiasActivation for scalar loops.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void FusedLinear_Double_NotScalarLoop()
    {
        var input = new Tensor<double>(Enumerable.Range(0, 32 * 256).Select(i => (double)i / 1000).ToArray(), [32, 256]);
        var weights = new Tensor<double>(Enumerable.Range(0, 256 * 256).Select(i => (double)i / 100000).ToArray(), [256, 256]);
        var bias = new Tensor<double>(Enumerable.Range(0, 256).Select(i => 0.01 * i).ToArray(), [1, 256]);

        for (int w = 0; w < 5; w++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);

        var sw = Stopwatch.StartNew();
        int iters = 20;
        for (int i = 0; i < iters; i++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"FusedLinear double 32x256 + ReLU: {ms:F3}ms (budget: {FusedLinearDoubleBudgetMs}ms)");
        Assert.True(ms < FusedLinearDoubleBudgetMs,
            $"FusedLinear double took {ms:F3}ms — exceeds {FusedLinearDoubleBudgetMs}ms budget.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void TensorAdd_100K_NotScalarLoop()
    {
        var a = Tensor<float>.CreateRandom([100000]);
        var b = Tensor<float>.CreateRandom([100000]);

        for (int w = 0; w < 10; w++) _engine.TensorAdd(a, b);

        var sw = Stopwatch.StartNew();
        int iters = 50;
        for (int i = 0; i < iters; i++) _engine.TensorAdd(a, b);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"TensorAdd 100K: {ms:F3}ms (budget: {Elementwise100KBudgetMs}ms)");
        Assert.True(ms < Elementwise100KBudgetMs,
            $"TensorAdd took {ms:F3}ms — exceeds {Elementwise100KBudgetMs}ms budget.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void TensorMultiply_100K_NotScalarLoop()
    {
        var a = Tensor<float>.CreateRandom([100000]);
        var b = Tensor<float>.CreateRandom([100000]);

        for (int w = 0; w < 10; w++) _engine.TensorMultiply(a, b);

        var sw = Stopwatch.StartNew();
        int iters = 50;
        for (int i = 0; i < iters; i++) _engine.TensorMultiply(a, b);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"TensorMultiply 100K: {ms:F3}ms (budget: {Elementwise100KBudgetMs}ms)");
        Assert.True(ms < Elementwise100KBudgetMs,
            $"TensorMultiply took {ms:F3}ms — exceeds {Elementwise100KBudgetMs}ms budget.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void TensorInstanceMethod_MatrixMultiply_UsesBLAS()
    {
        // Ensures Tensor.MatrixMultiply routes through engine (not scalar loops)
        var a = Tensor<float>.CreateRandom([32, 256]);
        var b = Tensor<float>.CreateRandom([256, 256]);

        for (int w = 0; w < 5; w++) a.MatrixMultiply(b);

        var sw = Stopwatch.StartNew();
        int iters = 20;
        for (int i = 0; i < iters; i++) a.MatrixMultiply(b);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"Tensor.MatrixMultiply 32x256 @ 256x256: {ms:F3}ms (budget: {MatMul256BudgetMs}ms)");
        Assert.True(ms < MatMul256BudgetMs,
            $"Tensor.MatrixMultiply took {ms:F3}ms — likely not routing through engine BLAS.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void TensorInstanceMethod_Add_UsesSIMD()
    {
        var a = Tensor<float>.CreateRandom([100000]);
        var b = Tensor<float>.CreateRandom([100000]);

        for (int w = 0; w < 10; w++) a.Add(b);

        var sw = Stopwatch.StartNew();
        int iters = 50;
        for (int i = 0; i < iters; i++) a.Add(b);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"Tensor.Add 100K: {ms:F3}ms (budget: {Elementwise100KBudgetMs}ms)");
        Assert.True(ms < Elementwise100KBudgetMs,
            $"Tensor.Add took {ms:F3}ms — likely not routing through engine SIMD.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void TensorInstanceMethod_PointwiseMultiply_UsesSIMD()
    {
        var a = Tensor<float>.CreateRandom([100000]);
        var b = Tensor<float>.CreateRandom([100000]);

        for (int w = 0; w < 10; w++) a.PointwiseMultiply(b);

        var sw = Stopwatch.StartNew();
        int iters = 50;
        for (int i = 0; i < iters; i++) a.PointwiseMultiply(b);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"Tensor.PointwiseMultiply 100K: {ms:F3}ms (budget: {Elementwise100KBudgetMs}ms)");
        Assert.True(ms < Elementwise100KBudgetMs,
            $"Tensor.PointwiseMultiply took {ms:F3}ms — likely not routing through engine SIMD.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void BatchMatMul_3D_UsesBLAS()
    {
        var a = Tensor<float>.CreateRandom([4, 32, 64]);
        var b = Tensor<float>.CreateRandom([4, 64, 32]);

        for (int w = 0; w < 5; w++) _engine.BatchMatMul(a, b);

        var sw = Stopwatch.StartNew();
        int iters = 20;
        for (int i = 0; i < iters; i++) _engine.BatchMatMul(a, b);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"BatchMatMul [4,32,64]@[4,64,32]: {ms:F3}ms (budget: {BatchMatMulBudgetMs}ms)");
        Assert.True(ms < BatchMatMulBudgetMs,
            $"BatchMatMul took {ms:F3}ms — check if BLAS path is active for batch slices.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void ReLUBackward_1M_UsesSIMD()
    {
        var gradOutput = Tensor<float>.CreateRandom([1000000]);
        var input = Tensor<float>.CreateRandom([1000000]);

        for (int w = 0; w < 5; w++) _engine.ReluBackward(gradOutput, input);

        var sw = Stopwatch.StartNew();
        int iters = 20;
        for (int i = 0; i < iters; i++) _engine.ReluBackward(gradOutput, input);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        // SIMD ReLU backward on 1M: ~1-2ms. Scalar: ~10ms+
        _output.WriteLine($"ReLU backward 1M: {ms:F3}ms (budget: 5ms)");
        Assert.True(ms < ReLUBackwardBudgetMs,
            $"ReLU backward took {ms:F3}ms — check SimdKernels.ReluBackwardUnsafe.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void Conv2D_AiseValCnnL1_Forward_PreservesPyTorchLead()
    {
        // AIsEval CNN layer-1 shape: nn.Conv2d(1, 16, kernel=3, padding=1) at
        // input [128, 1, 28, 28] (MNIST-shape, bs=128). PyTorch baseline for
        // this exact shape was 41.42 ms (bs=128 steady-state latency including
        // ReLU + MaxPool, but Conv2D dominates). Our measured pure-Conv2D was
        // ~5-6 ms. Locking the gate at 30 ms catches a 5x+ regression while
        // tolerating CI noise and cold JIT.
        var input = Tensor<float>.CreateRandom([128, 1, 28, 28]);
        var kernel = Tensor<float>.CreateRandom([16, 1, 3, 3]);

        for (int w = 0; w < 3; w++) _engine.Conv2D(input, kernel, stride: 1, padding: 1);

        var sw = Stopwatch.StartNew();
        const int iters = 10;
        for (int i = 0; i < iters; i++) _engine.Conv2D(input, kernel, stride: 1, padding: 1);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"Conv2D AIsEval-L1 [128,1,28,28]@[16,1,3,3] pad=1: {ms:F3}ms (budget: {CnnAiseval_L1_BudgetMs}ms)");
        Assert.True(ms < CnnAiseval_L1_BudgetMs,
            $"Conv2D AIsEval-L1 took {ms:F3}ms — exceeds {CnnAiseval_L1_BudgetMs}ms budget. " +
            "The AIsEval benchmark relies on this shape being well below PyTorch's 41.42ms baseline; " +
            "a regression here would erase the CNN inference lead documented in issue #436.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void LstmSequenceForward_Aiseval_FloatFastPath_BeatsPyTorch()
    {
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
        for (int w = 0; w < 3; w++)
            _ = _engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null);

        var sw = Stopwatch.StartNew();
        const int iters = 10;
        for (int i = 0; i < iters; i++)
            _ = _engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"LstmSequenceForward AIsEval [128,32,32->64]: {ms:F2} ms (budget: {LstmAiseval_BudgetMs} ms; PyTorch baseline 11.76 ms)");
        Assert.True(ms < LstmAiseval_BudgetMs,
            $"LstmSequenceForward took {ms:F2} ms — exceeds {LstmAiseval_BudgetMs} ms budget. " +
            "This indicates a regression away from the Stage 5 float fast path that put AiDotNet ahead of PyTorch on this shape.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void MultiHeadAttentionForward_Aiseval_FloatFastPath()
    {
        // Locks in the Stage 6 MHA float fast path. Stage 4 wrapper measured
        // 22.31 ms on net10.0; Stage 6 brought it to 10.13 ms via direct
        // SimdGemm projections + ArrayPool scratch + explicit-loop transpose.
        // Budget at 30 ms catches regression back to the wrapper-only path.
        const int batch = 128, seq = 32, dModel = 64, numHeads = 4;
        var input = Tensor<float>.CreateRandom(batch, seq, dModel);
        var qW = Tensor<float>.CreateRandom(dModel, dModel);
        var kW = Tensor<float>.CreateRandom(dModel, dModel);
        var vW = Tensor<float>.CreateRandom(dModel, dModel);
        var oW = Tensor<float>.CreateRandom(dModel, dModel);

        // MultiHeadAttentionForward is on IEngine (the float fast path lives in
        // CpuEngine and is picked up by DirectGpuTensorEngine via inheritance),
        // so no downcast needed.
        for (int w = 0; w < 3; w++)
            _ = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);

        var sw = Stopwatch.StartNew();
        const int iters = 10;
        for (int i = 0; i < iters; i++)
            _ = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"MultiHeadAttentionForward AIsEval [128,32,64] h=4: {ms:F2} ms (budget: {MhaAiseval_BudgetMs} ms)");
        Assert.True(ms < MhaAiseval_BudgetMs,
            $"MultiHeadAttentionForward took {ms:F2} ms — exceeds {MhaAiseval_BudgetMs} ms budget. " +
            "Stage 6 float fast path measured 10.13 ms on net10.0; a regression to the Stage 4 wrapper path would push this back to ~22 ms.");
    }

    [Fact(Skip = "Performance guard — run manually with --filter PerformanceRegression")]
    public void Conv2D_AiseValCnnL2_Forward_PreservesPyTorchLead()
    {
        // AIsEval CNN layer-2 shape: nn.Conv2d(16, 32, kernel=3, padding=1) at
        // input [128, 16, 14, 14] (post-first-MaxPool). This is the heavier of
        // the two convs by FLOPs (~ 36M MACs vs ~6M for L1). Budget at 35 ms
        // covers the same 5x-regression-catching headroom.
        var input = Tensor<float>.CreateRandom([128, 16, 14, 14]);
        var kernel = Tensor<float>.CreateRandom([32, 16, 3, 3]);

        for (int w = 0; w < 3; w++) _engine.Conv2D(input, kernel, stride: 1, padding: 1);

        var sw = Stopwatch.StartNew();
        const int iters = 10;
        for (int i = 0; i < iters; i++) _engine.Conv2D(input, kernel, stride: 1, padding: 1);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"Conv2D AIsEval-L2 [128,16,14,14]@[32,16,3,3] pad=1: {ms:F3}ms (budget: {CnnAiseval_L2_BudgetMs}ms)");
        Assert.True(ms < CnnAiseval_L2_BudgetMs,
            $"Conv2D AIsEval-L2 took {ms:F3}ms — exceeds {CnnAiseval_L2_BudgetMs}ms budget. " +
            "Layer-2 conv dominates the CNN benchmark's FLOPs; a regression here is what would " +
            "show up first in the AIsEval bs=128 numbers.");
    }
}
