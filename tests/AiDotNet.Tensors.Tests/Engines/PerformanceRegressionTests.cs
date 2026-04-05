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

    public PerformanceRegressionTests(ITestOutputHelper output) => _output = output;

    [Fact]
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

    [Fact]
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

    [Fact]
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

    [Fact]
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
            $"FusedLinear double took {ms:F3}ms — exceeds {FusedLinear256BudgetMs}ms budget.");
    }

    [Fact]
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

    [Fact]
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

    [Fact]
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

    [Fact]
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

    [Fact]
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

    [Fact]
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

    [Fact]
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
}
