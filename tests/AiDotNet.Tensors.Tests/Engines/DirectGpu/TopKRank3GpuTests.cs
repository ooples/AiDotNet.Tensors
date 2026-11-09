using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// TensorTopK on the last axis of a rank-3 tensor (LM logits [B, S, V]) runs the GPU row kernel and keeps the
/// leading shape. Before, it fell through to the CPU, which rejected every rank other than 2.
/// </summary>
public class TopKRank3GpuTests
{
    [SkippableFact]
    public void Rank3_last_axis_topk_matches_the_reference()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        int[] shape = { 2, 3, 500 };
        // Distinct values: the GPU sort's tie order is not part of the contract.
        var data = Enumerable.Range(0, 2 * 3 * 500).Select(i => (float)((i * 7919) % 3000)).ToArray();
        var values = ((IEngine)gpu).TensorTopK(new Tensor<float>(data, shape), 8, -1, out var indices);
        var (ev, ei) = TopKAnyAxisTests.Reference(data, shape, 2, 8, largest: true);
        Assert.Equal(new[] { 2, 3, 8 }, values.Shape.ToArray());
        Assert.Equal(new[] { 2, 3, 8 }, indices.Shape.ToArray());
        Assert.Equal(ev, values.ToArray());
        Assert.Equal(ei, indices.ToArray());
    }

    // k past the axis length clamps to the axis: the leading shape is restored with the selected count, not k.
    [SkippableFact]
    public void Rank3_last_axis_topk_clamps_k_to_the_axis_length()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        int[] shape = { 2, 3, 5 };
        var data = Enumerable.Range(0, 2 * 3 * 5).Select(i => (float)((i * 7) % 30)).ToArray();
        var values = ((IEngine)gpu).TensorTopK(new Tensor<float>(data, shape), 8, -1, out var indices);
        var (ev, ei) = TopKAnyAxisTests.Reference(data, shape, 2, 5, largest: true);
        Assert.Equal(new[] { 2, 3, 5 }, values.Shape.ToArray());
        Assert.Equal(new[] { 2, 3, 5 }, indices.Shape.ToArray());
        Assert.Equal(ev, values.ToArray());
        Assert.Equal(ei, indices.ToArray());
    }

    // An empty last axis divided by zero before any backend was consulted; it now returns empty outputs.
    [Fact]
    public void An_empty_last_axis_returns_empty_outputs()
    {
        using var gpu = new DirectGpuTensorEngine();
        var values = ((IEngine)gpu).TensorTopK(new Tensor<float>(new float[0], new[] { 2, 3, 0 }), 2, -1, out var indices);
        Assert.Equal(new[] { 2, 3, 0 }, values.Shape.ToArray());
        Assert.Equal(new[] { 2, 3, 0 }, indices.Shape.ToArray());
    }

    // The GPU row sort carries indices as floats, exact only up to 2^24. Two tied maxima at 2^24 and 2^24 + 1 came
    // back as the first index twice; a row that long now takes the CPU selection, which keeps every index exact.
    [Fact]
    public void A_row_past_float_index_precision_keeps_exact_indices()
    {
        using var gpu = new DirectGpuTensorEngine();
        int columns = DirectGpuTensorEngine.MaxFloatExactSortIndexColumns + 1;
        var data = new float[columns];
        data[columns - 2] = 1f;
        data[columns - 1] = 1f;
        ((IEngine)gpu).TensorTopK(new Tensor<float>(data, new[] { 1, 1, columns }), 2, -1, out var indices);
        Assert.Equal(new[] { columns - 2, columns - 1 }, indices.ToArray());
    }

    /// <summary>
    /// Small k over long rows takes the chunk tournament (TopKChunk = 128): an LM-vocabulary row (three rounds) and a
    /// row whose length is not a multiple of the chunk (padded chunk). Distinct values, so the order is fully defined.
    /// </summary>
    [SkippableTheory]
    [InlineData(64, 49152, 32)]
    [InlineData(3, 1000, 20)]
    [InlineData(5, 4096, 64)]
    public void Chunk_tournament_topk_matches_the_reference(int rows, int columns, int k)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        var rng = new System.Random(rows * 31 + columns);
        var data = new float[rows * columns];
        for (int r = 0; r < rows; r++)
        {
            // A random permutation of 0..columns-1 per row, scaled: distinct values, worst cases spread over chunks.
            var perm = Enumerable.Range(0, columns).OrderBy(_ => rng.Next()).ToArray();
            for (int c = 0; c < columns; c++) data[r * columns + c] = perm[c] * 0.5f - 7f;
        }
        int[] shape = { rows, columns };
        var values = ((IEngine)gpu).TensorTopK(new Tensor<float>(data, shape), k, 1, out var indices);
        var (ev, ei) = TopKAnyAxisTests.Reference(data, shape, 1, k, largest: true);
        Assert.Equal(new[] { rows, k }, values.Shape.ToArray());
        Assert.Equal(ev, values.ToArray());
        Assert.Equal(ei, indices.ToArray());
    }
}
