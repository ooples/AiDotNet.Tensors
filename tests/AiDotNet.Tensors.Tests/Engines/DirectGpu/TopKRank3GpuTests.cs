using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
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
        var rng = RandomHelper.CreateSeededRandom(rows * 31 + columns);
        var data = new float[rows * columns];
        for (int r = 0; r < rows; r++)
        {
            // A random permutation of 0..columns-1 per row, scaled: distinct values, worst cases spread over chunks.
            var perm = Enumerable.Range(0, columns).OrderBy(_ => rng.Next()).ToArray();
            for (int c = 0; c < columns; c++) data[r * columns + c] = perm[c] * 0.5f - 7f;
        }
        int[] shape = { rows, columns };
        int tournamentsBefore = gpu.TopKTournamentsCompleted;
        var values = ((IEngine)gpu).TensorTopK(new Tensor<float>(data, shape), k, 1, out var indices);
        // The tournament itself ran on the device: a full row sort or a CPU fallback returns the same answer.
        Assert.Equal(tournamentsBefore + 1, gpu.TopKTournamentsCompleted);
        var (ev, ei) = TopKAnyAxisTests.Reference(data, shape, 1, k, largest: true);
        Assert.Equal(new[] { rows, k }, values.Shape.ToArray());
        Assert.Equal(ev, values.ToArray());
        Assert.Equal(ei, indices.ToArray());
    }

    /// <summary>
    /// The tournament pads each row to a whole chunk with -inf at index -1. A row whose real entries are -inf ties with
    /// those pads, and a pad must never be returned for it: every selected index has to be a real, distinct column.
    /// The sort's comparator (bitonic_sorts_before) orders a pad after every real entry whatever the values; this pins it.
    /// Row 0 is all -inf; row 1 has fewer finite entries than k, so the rest of its top k are real -inf columns.
    /// </summary>
    [SkippableFact]
    public void Chunk_tournament_never_returns_a_pad_for_a_negative_infinity_row()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        const int rows = 2, columns = 1000, k = 20;
        var data = Enumerable.Repeat(float.NegativeInfinity, rows * columns).ToArray();
        var finite = new[] { (Column: 7, Value: 3f), (Column: 500, Value: 2f), (Column: 999, Value: 1f) };
        foreach (var (column, value) in finite) data[columns + column] = value;

        int tournamentsBefore = gpu.TopKTournamentsCompleted;
        var values = ((IEngine)gpu).TensorTopK(new Tensor<float>(data, new[] { rows, columns }), k, 1, out var indices);
        Assert.Equal(tournamentsBefore + 1, gpu.TopKTournamentsCompleted);

        var v = values.ToArray();
        var ix = indices.ToArray();
        for (int r = 0; r < rows; r++)
        {
            var rowIndices = ix.Skip(r * k).Take(k).ToArray();
            Assert.All(rowIndices, c => Assert.InRange(c, 0, columns - 1));
            Assert.Equal(k, rowIndices.Distinct().Count());
            for (int i = 0; i < k; i++) Assert.Equal(data[r * columns + rowIndices[i]], v[r * k + i]);
        }
        Assert.All(v.Take(k), x => Assert.Equal(float.NegativeInfinity, x));
        for (int i = 0; i < finite.Length; i++)
        {
            Assert.Equal(finite[i].Column, ix[k + i]);
            Assert.Equal(finite[i].Value, v[k + i]);
        }
        Assert.All(v.Skip(k + finite.Length), x => Assert.Equal(float.NegativeInfinity, x));
    }

}
