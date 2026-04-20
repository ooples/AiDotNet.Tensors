#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using TorchSharp;
using static TorchSharp.torch;

namespace AiDotNet.Tensors.Benchmarks.Parity210;

/// <summary>
/// Special-math benchmarks for issue #210.  Exercises element-wise special
/// functions (erfc, lgamma, digamma, I0, I1, erfinv) on a representative
/// [512, 1024] float matrix.
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(20)]
[MaxIterationCount(80)]
public class Parity210SpecialMathBenchmarks
{
    private readonly CpuEngine _engine = new();
    private Tensor<float> _input = null!;
    private Tensor<float> _inputPositive = null!;
    private Tensor<float> _inputUnit = null!;
    private torch.Tensor _tinput = null!;
    private torch.Tensor _tinputPositive = null!;
    private torch.Tensor _tinputUnit = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int total = 512 * 1024;
        var arr = new float[total];
        var pos = new float[total];
        var unit = new float[total];
        for (int i = 0; i < total; i++)
        {
            arr[i] = (float)(rng.NextDouble() * 4 - 2);     // -2..2 for erfc / I0 / I1
            pos[i] = (float)(rng.NextDouble() * 10 + 0.1);  // 0.1..10 for lgamma / digamma
            unit[i] = (float)(rng.NextDouble() * 1.8 - 0.9); // -0.9..0.9 for erfinv
        }
        _input = new Tensor<float>(arr, new[] { 512, 1024 });
        _inputPositive = new Tensor<float>(pos, new[] { 512, 1024 });
        _inputUnit = new Tensor<float>(unit, new[] { 512, 1024 });
        _tinput = torch.randn(new long[] { 512, 1024 });
        _tinputPositive = torch.rand(new long[] { 512, 1024 }) * 9.9f + 0.1f;
        _tinputUnit = torch.rand(new long[] { 512, 1024 }) * 1.8f - 0.9f;
    }

    [Benchmark(Baseline = true, Description = "Ours: Erfc [512,1024]")]
    public Tensor<float> Ours_Erfc() => _engine.TensorErfc(_input);

    [Benchmark(Description = "PyTorch: erfc [512,1024]")]
    public torch.Tensor PyTorch_Erfc() => torch.special.erfc(_tinput);

    [Benchmark(Description = "Ours: Lgamma [512,1024]")]
    public Tensor<float> Ours_Lgamma() => _engine.TensorLgamma(_inputPositive);

    [Benchmark(Description = "PyTorch: lgamma [512,1024]")]
    public torch.Tensor PyTorch_Lgamma() => torch.special.gammaln(_tinputPositive);

    [Benchmark(Description = "Ours: Digamma [512,1024]")]
    public Tensor<float> Ours_Digamma() => _engine.TensorDigamma(_inputPositive);

    [Benchmark(Description = "PyTorch: digamma [512,1024]")]
    public torch.Tensor PyTorch_Digamma() => torch.special.digamma(_tinputPositive);

    [Benchmark(Description = "Ours: I0 [512,1024]")]
    public Tensor<float> Ours_I0() => _engine.TensorI0(_input);

    [Benchmark(Description = "PyTorch: i0 [512,1024]")]
    public torch.Tensor PyTorch_I0() => torch.special.i0(_tinput);

    [Benchmark(Description = "Ours: Erfinv [512,1024]")]
    public Tensor<float> Ours_Erfinv() => _engine.TensorErfinv(_inputUnit);

    [Benchmark(Description = "PyTorch: erfinv [512,1024]")]
    public torch.Tensor PyTorch_Erfinv() => torch.special.erfinv(_tinputUnit);
}
#endif
