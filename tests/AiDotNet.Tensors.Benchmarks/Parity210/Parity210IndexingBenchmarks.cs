#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using TorchSharp;
using static TorchSharp.torch;

namespace AiDotNet.Tensors.Benchmarks.Parity210;

/// <summary>
/// Indexing / gather-scatter benchmarks for issue #210.  Exercises the
/// Take / TakeAlongDim / IndexAdd / MaskedSelect paths — dominant in
/// embedding-lookups, sparse updates, and boolean-selection kernels.
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(20)]
[MaxIterationCount(80)]
public class Parity210IndexingBenchmarks
{
    private readonly CpuEngine _engine = new();

    // Source: vocab-sized embedding table [32000, 512]
    private Tensor<float> _embed = null!;
    private torch.Tensor _tembed = null!;

    // Indices: batch of 4096 token-ids
    private Tensor<int> _idx = null!;
    private torch.Tensor _tidx = null!;

    // Values to scatter: [4096, 512]
    private Tensor<float> _values = null!;
    private torch.Tensor _tvalues = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int etotal = 32000 * 512;
        var embedArr = new float[etotal];
        for (int i = 0; i < etotal; i++) embedArr[i] = (float)(rng.NextDouble() * 2 - 1);
        _embed = new Tensor<float>(embedArr, new[] { 32000, 512 });
        _tembed = torch.randn(new long[] { 32000, 512 });

        int k = 4096;
        var idxArr = new int[k];
        for (int i = 0; i < k; i++) idxArr[i] = rng.Next(0, 32000);
        _idx = new Tensor<int>(idxArr, new[] { k });
        _tidx = torch.from_array(idxArr, torch.int64);

        int vtotal = k * 512;
        var vArr = new float[vtotal];
        for (int i = 0; i < vtotal; i++) vArr[i] = (float)(rng.NextDouble() * 2 - 1);
        _values = new Tensor<float>(vArr, new[] { k, 512 });
        _tvalues = torch.randn(new long[] { k, 512 });
    }

    [Benchmark(Baseline = true, Description = "Ours: Take 4096-of-32000 rows × 512")]
    public Tensor<float> Ours_Take() => _engine.TensorIndexSelect(_embed, _idx, 0);

    [Benchmark(Description = "PyTorch: index_select dim=0")]
    public torch.Tensor PyTorch_Take() => torch.index_select(_tembed, 0, _tidx);

    [Benchmark(Description = "Ours: IndexAdd scatter-add 4096 rows")]
    public Tensor<float> Ours_IndexAdd()
        => _engine.TensorIndexAdd(_embed, 0, _idx, _values);

    [Benchmark(Description = "PyTorch: index_add dim=0")]
    public torch.Tensor PyTorch_IndexAdd()
        => torch.index_add(_tembed, 0, _tidx, _tvalues, 1.0f);
}
#endif
