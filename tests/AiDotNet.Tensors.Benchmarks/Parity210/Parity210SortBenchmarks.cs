#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using TorchSharp;
using static TorchSharp.torch;

namespace AiDotNet.Tensors.Benchmarks.Parity210;

/// <summary>
/// Sort + TopK benchmarks for issue #210.  Workload matches a typical
/// ranking / beam-search pattern: sort across the final axis of a
/// [32, 4096] score matrix (batch × candidates).
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(20)]
[MaxIterationCount(80)]
public class Parity210SortBenchmarks
{
    private readonly CpuEngine _engine = new();
    private Tensor<float> _scores = null!;
    private torch.Tensor _tscores = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int total = 32 * 4096;
        var arr = new float[total];
        for (int i = 0; i < total; i++) arr[i] = (float)(rng.NextDouble() * 10 - 5);
        _scores = new Tensor<float>(arr, new[] { 32, 4096 });
        _tscores = torch.randn(new long[] { 32, 4096 });
    }

    [Benchmark(Baseline = true, Description = "Ours: Sort descending last-axis [32,4096]")]
    public Tensor<float> Ours_Sort()
    {
        var (values, _) = _engine.TensorSort(_scores, axis: -1, descending: true);
        return values;
    }

    [Benchmark(Description = "PyTorch: sort descending dim=-1 [32,4096]")]
    public torch.Tensor PyTorch_Sort() => torch.sort(_tscores, dim: -1, descending: true).values;

    [Benchmark(Description = "Ours: TopK k=128 last-axis [32,4096]")]
    public Tensor<float> Ours_TopK()
    {
        return _engine.TensorTopK(_scores, k: 128, axis: -1, out var _);
    }

    [Benchmark(Description = "PyTorch: topk k=128 dim=-1 [32,4096]")]
    public torch.Tensor PyTorch_TopK() => torch.topk(_tscores, k: 128, dim: -1).values;
}
#endif
