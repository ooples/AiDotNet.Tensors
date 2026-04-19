#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using TorchSharp;
using static TorchSharp.torch;

namespace AiDotNet.Tensors.Benchmarks.Parity210;

/// <summary>
/// Movement benchmarks for issue #210.  Exercises Roll / Flip / Triu /
/// Tril on a [64, 512, 512] transformer-attention-shaped workload —
/// the same sizes we hit in masked attention.
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(20)]
[MaxIterationCount(80)]
public class Parity210MovementBenchmarks
{
    private readonly CpuEngine _engine = new();
    private Tensor<float> _mat = null!;
    private torch.Tensor _tmat = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int total = 64 * 512 * 512;
        var arr = new float[total];
        for (int i = 0; i < total; i++) arr[i] = (float)(rng.NextDouble() * 2 - 1);
        _mat = new Tensor<float>(arr, new[] { 64, 512, 512 });
        _tmat = torch.randn(new long[] { 64, 512, 512 });
    }

    [Benchmark(Baseline = true, Description = "Ours: Roll axis=1 by 7 [64,512,512]")]
    public Tensor<float> Ours_Roll() => _engine.TensorRoll(_mat, new[] { 7 }, new[] { 1 });

    [Benchmark(Description = "PyTorch: roll dim=1 shifts=7 [64,512,512]")]
    public torch.Tensor PyTorch_Roll() => torch.roll(_tmat, 7, 1);

    [Benchmark(Description = "Ours: Flip axis=[1,2] [64,512,512]")]
    public Tensor<float> Ours_Flip() => _engine.TensorFlip(_mat, new[] { 1, 2 });

    [Benchmark(Description = "PyTorch: flip dims=[1,2] [64,512,512]")]
    public torch.Tensor PyTorch_Flip() => torch.flip(_tmat, new long[] { 1, 2 });

    [Benchmark(Description = "Ours: Triu [64,512,512] diag=0")]
    public Tensor<float> Ours_Triu() => _engine.TensorTriu(_mat, diagonal: 0);

    [Benchmark(Description = "PyTorch: triu diagonal=0 [64,512,512]")]
    public torch.Tensor PyTorch_Triu() => torch.triu(_tmat, diagonal: 0);

    [Benchmark(Description = "Ours: Tril [64,512,512] diag=0")]
    public Tensor<float> Ours_Tril() => _engine.TensorTril(_mat, diagonal: 0);

    [Benchmark(Description = "PyTorch: tril diagonal=0 [64,512,512]")]
    public torch.Tensor PyTorch_Tril() => torch.tril(_tmat, diagonal: 0);
}
#endif
