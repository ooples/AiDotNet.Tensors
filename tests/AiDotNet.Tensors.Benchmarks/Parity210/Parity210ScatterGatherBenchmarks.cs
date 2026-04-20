#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using TorchSharp;
using static TorchSharp.torch;

namespace AiDotNet.Tensors.Benchmarks.Parity210;

/// <summary>
/// Scatter / gather / masked-select / scatter-add / scatter-reduce benchmarks
/// for issue #210. Exercises the full indexing family — the acceptance
/// criteria explicitly call for a <c>ScatterGatherBenchmarks</c> fixture,
/// which the existing <c>Parity210IndexingBenchmarks</c> doesn't cover
/// (it only hits IndexSelect + IndexAdd).
///
/// Workload: embedding-table scatter/gather with a 32000-row table, 512-dim
/// embeddings, and 4096-index batches — the dominant shape in transformer
/// embedding lookups, sparse gradient accumulation, and MoE routing.
/// </summary>
[MemoryDiagnoser]
[MinIterationCount(20)]
[MaxIterationCount(80)]
public class Parity210ScatterGatherBenchmarks
{
    private readonly CpuEngine _engine = new();
    private Tensor<float> _embed = null!;
    private torch.Tensor _tembed = null!;
    private Tensor<int> _idx = null!;
    private torch.Tensor _tidx = null!;
    private Tensor<float> _values = null!;
    private torch.Tensor _tvalues = null!;
    private Tensor<Bit> _mask = null!;
    private torch.Tensor _tmask = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        // Embedding table: [32000, 512]
        int etotal = 32000 * 512;
        var embedArr = new float[etotal];
        for (int i = 0; i < etotal; i++) embedArr[i] = (float)(rng.NextDouble() * 2 - 1);
        _embed = new Tensor<float>(embedArr, new[] { 32000, 512 });
        _tembed = torch.randn(new long[] { 32000, 512 });

        // Batch of 4096 token-ids
        int k = 4096;
        var idxArr = new int[k];
        for (int i = 0; i < k; i++) idxArr[i] = rng.Next(0, 32000);
        _idx = new Tensor<int>(idxArr, new[] { k });
        _tidx = torch.from_array(idxArr, torch.int64);

        // Values to scatter: [4096, 512]
        int vtotal = k * 512;
        var vArr = new float[vtotal];
        for (int i = 0; i < vtotal; i++) vArr[i] = (float)(rng.NextDouble() * 2 - 1);
        _values = new Tensor<float>(vArr, new[] { k, 512 });
        _tvalues = torch.randn(new long[] { k, 512 });

        // Boolean mask over [32000, 512] with ~25% true density.
        int mtotal = 32000 * 512;
        var mArr = new Bit[mtotal];
        for (int i = 0; i < mtotal; i++) mArr[i] = rng.NextDouble() < 0.25 ? Bit.True : Bit.False;
        _mask = new Tensor<Bit>(mArr, new[] { 32000, 512 });
        // TorchSharp bool tensor with matching density.
        _tmask = torch.rand(new long[] { 32000, 512 }) < 0.25f;
    }

    // ---- Gather family ----

    [Benchmark(Baseline = true, Description = "Ours: Gather axis=0 [32k,512] via 4k idx")]
    public Tensor<float> Ours_Gather() => _engine.TensorGather(_embed, _idx, axis: 0);

    [Benchmark(Description = "PyTorch: gather dim=0")]
    public torch.Tensor PyTorch_Gather()
    {
        // torch.gather needs index of same rank; use index_select for rows.
        return torch.index_select(_tembed, 0, _tidx);
    }

    [Benchmark(Description = "Ours: IndexSelect axis=0 [32k,512] via 4k idx")]
    public Tensor<float> Ours_IndexSelect() => _engine.TensorIndexSelect(_embed, _idx, 0);

    [Benchmark(Description = "PyTorch: index_select dim=0")]
    public torch.Tensor PyTorch_IndexSelect() => torch.index_select(_tembed, 0, _tidx);

    // ---- Scatter family ----

    [Benchmark(Description = "Ours: ScatterAdd axis=0 [32k,512] from 4k rows")]
    public Tensor<float> Ours_ScatterAdd()
        => _engine.TensorScatterAdd(_embed, _idx, _values, axis: 0);

    [Benchmark(Description = "PyTorch: scatter_add_ dim=0")]
    public torch.Tensor PyTorch_ScatterAdd()
    {
        // Replicate our [32k,512] shape with index expanded to the row-shape
        // that scatter_add_ expects.
        var clone = _tembed.clone();
        // PyTorch scatter_add_ requires the index to match the src shape —
        // broadcast the [4096] row-index to [4096,512].
        var idxExpanded = _tidx.unsqueeze(1).expand(new long[] { 4096, 512 });
        return clone.scatter_add_(0, idxExpanded, _tvalues);
    }

    [Benchmark(Description = "Ours: IndexAdd axis=0 [32k,512]")]
    public Tensor<float> Ours_IndexAdd() => _engine.TensorIndexAdd(_embed, 0, _idx, _values);

    [Benchmark(Description = "PyTorch: index_add dim=0")]
    public torch.Tensor PyTorch_IndexAdd()
        => torch.index_add(_tembed, 0, _tidx, _tvalues, 1.0f);

    // ---- Masked select ----

    [Benchmark(Description = "Ours: MaskedSelect at 25% density [32k,512]")]
    public Tensor<float> Ours_MaskedSelect()
        => _engine.TensorMaskedSelect(_embed, _mask);

    [Benchmark(Description = "PyTorch: masked_select at 25% density")]
    public torch.Tensor PyTorch_MaskedSelect()
        => torch.masked_select(_tembed, _tmask);
}
#endif
