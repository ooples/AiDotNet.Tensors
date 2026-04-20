#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using TorchSharp;
using static TorchSharp.torch;

namespace AiDotNet.Tensors.Benchmarks.Parity210;

/// <summary>
/// Cumulative-op benchmarks for issue #210's "beat PyTorch" gate.
/// Exercises CumSum / CumProd / CumMax / LogCumSumExp against PyTorch
/// eager at a mid-size shape representative of log-likelihood / scan
/// workloads (batch × seq × features = 8 × 512 × 64).
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(20)]
[MaxIterationCount(100)]
public class Parity210CumulativeBenchmarks
{
    private readonly CpuEngine _engine = new();
    private Tensor<float> _input = null!;
    private torch.Tensor _tinput = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int total = 8 * 512 * 64;
        var arr = new float[total];
        for (int i = 0; i < total; i++) arr[i] = (float)(rng.NextDouble() * 2 - 1);
        _input = new Tensor<float>(arr, new[] { 8, 512, 64 });
        _tinput = torch.randn(new long[] { 8, 512, 64 });
    }

    // CumSum along seq axis (axis=1) — scan pattern.
    [Benchmark(Baseline = true, Description = "Ours: CumSum axis=1 [8,512,64]")]
    public Tensor<float> Ours_CumSum() => _engine.TensorCumSum(_input, axis: 1);

    [Benchmark(Description = "PyTorch: cumsum dim=1 [8,512,64]")]
    public torch.Tensor PyTorch_CumSum() => torch.cumsum(_tinput, dim: 1);

    [Benchmark(Description = "Ours: CumProd axis=1 [8,512,64]")]
    public Tensor<float> Ours_CumProd() => _engine.TensorCumProd(_input, axis: 1);

    [Benchmark(Description = "PyTorch: cumprod dim=1 [8,512,64]")]
    public torch.Tensor PyTorch_CumProd() => torch.cumprod(_tinput, dim: 1);

    [Benchmark(Description = "Ours: CumMax axis=1 [8,512,64]")]
    public Tensor<float> Ours_CumMax() => _engine.TensorCumMax(_input, axis: 1);

    [Benchmark(Description = "PyTorch: cummax dim=1 [8,512,64]")]
    public torch.Tensor PyTorch_CumMax() => torch.cummax(_tinput, dim: 1).values;

    [Benchmark(Description = "Ours: LogCumSumExp axis=1 [8,512,64]")]
    public Tensor<float> Ours_LogCumSumExp() => _engine.TensorLogCumSumExp(_input, axis: 1);

    [Benchmark(Description = "PyTorch: logcumsumexp dim=1 [8,512,64]")]
    public torch.Tensor PyTorch_LogCumSumExp() => torch.logcumsumexp(_tinput, dim: 1);
}
#endif
