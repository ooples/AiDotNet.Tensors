using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Before/after benchmarks for issue #100: scalar GEMM loops vs BLAS.
/// Run with Skip removed to capture timing data.
/// </summary>
public class ScalarLoopPerfTests
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public ScalarLoopPerfTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Benchmark_FusedLinear_Float_256x256()
    {
        var input = Tensor<float>.CreateRandom([32, 256]);
        var weights = Tensor<float>.CreateRandom([256, 256]);
        var bias = Tensor<float>.CreateRandom([1, 256]);
        int iters = 50;

        // Warmup
        for (int w = 0; w < 5; w++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;
        _output.WriteLine($"FusedLinear float 32x256 @ 256x256 + ReLU: {ms:F3}ms");
    }

    [Fact]
    public void Benchmark_FusedLinear_Double_256x256()
    {
        var input = new Tensor<double>(Enumerable.Range(0, 32 * 256).Select(i => (double)i / 1000).ToArray(), [32, 256]);
        var weights = new Tensor<double>(Enumerable.Range(0, 256 * 256).Select(i => (double)i / 100000).ToArray(), [256, 256]);
        var bias = new Tensor<double>(Enumerable.Range(0, 256).Select(i => 0.01 * i).ToArray(), [1, 256]);
        int iters = 50;

        for (int w = 0; w < 5; w++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;
        _output.WriteLine($"FusedLinear double 32x256 @ 256x256 + ReLU: {ms:F3}ms");
    }

    [Fact]
    public void Benchmark_FusedLinear_Float_768x768()
    {
        // Transformer-scale: typical attention projection
        var input = Tensor<float>.CreateRandom([16, 768]);
        var weights = Tensor<float>.CreateRandom([768, 768]);
        var bias = Tensor<float>.CreateRandom([1, 768]);
        int iters = 20;

        for (int w = 0; w < 3; w++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.GELU);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.GELU);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;
        _output.WriteLine($"FusedLinear float 16x768 @ 768x768 + GELU: {ms:F3}ms");
    }

    [Fact]
    public void Benchmark_TensorMatMul_VsFusedLinear_Float()
    {
        // Compare: TensorMatMul (uses BLAS) vs FusedLinear (scalar loops)
        var input = Tensor<float>.CreateRandom([32, 256]);
        var weights = Tensor<float>.CreateRandom([256, 256]);
        var bias = Tensor<float>.CreateRandom([1, 256]);
        int iters = 50;

        // Warmup both
        for (int w = 0; w < 5; w++)
        {
            _engine.TensorMatMul(input, weights);
            _engine.FusedLinear(input, weights, bias, FusedActivationType.None);
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            _engine.TensorMatMul(input, weights);
        sw.Stop();
        double matmulMs = sw.Elapsed.TotalMilliseconds / iters;

        sw.Restart();
        for (int i = 0; i < iters; i++)
            _engine.FusedLinear(input, weights, bias, FusedActivationType.None);
        sw.Stop();
        double fusedMs = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"TensorMatMul (BLAS): {matmulMs:F3}ms");
        _output.WriteLine($"FusedLinear (scalar): {fusedMs:F3}ms");
        _output.WriteLine($"Ratio: FusedLinear is {fusedMs / matmulMs:F1}x slower");
    }
}
