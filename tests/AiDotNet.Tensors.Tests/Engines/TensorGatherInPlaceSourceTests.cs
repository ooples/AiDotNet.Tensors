using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// CpuEngine.TensorGather reads a contiguous source in place (no full-table copy). Checked against direct indexing on
/// a contiguous view with a non-zero storage offset, a non-contiguous (transposed) source, out-of-range indices
/// (row left zero), and the general-axis path.
/// </summary>
public class TensorGatherInPlaceSourceTests
{
    private static readonly CpuEngine Engine = new();

    private static Tensor<float> Iota(params int[] shape)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = i * 0.5f + 1f;
        return t;
    }

    [Fact]
    public void Axis0_OnOffsetView_ReadsTheViewsRows()
    {
        var table = Iota(10, 6).Slice(0, 3, 9);          // rows 3..8 of a 10x6 table: storage offset 18
        Assert.True(table.IsContiguous);
        var idx = new Tensor<int>(new[] { 5, 0, 2, 2, 7, -1 }, new[] { 6 });
        var g = Engine.TensorGather(table, idx, 0);
        Assert.Equal(new[] { 6, 6 }, g.Shape.ToArray());
        int[] expectRow = { 5, 0, 2, 2, -1, -1 };          // 7 and -1 are out of range for a 6-row view
        for (int r = 0; r < 6; r++)
            for (int c = 0; c < 6; c++)
                Assert.Equal(expectRow[r] < 0 ? 0f : table[expectRow[r], c], g[r, c]);
    }

    [Fact]
    public void Axis0_NonContiguousSource_MatchesIndexing()
    {
        var table = Iota(6, 8).Transpose(new[] { 1, 0 });   // [8, 6] strided view
        Assert.False(table.IsContiguous);
        var idx = new Tensor<int>(new[] { 7, 1, 1, 4 }, new[] { 4 });
        var g = Engine.TensorGather(table, idx, 0);
        int[] rows = { 7, 1, 1, 4 };
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 6; c++)
                Assert.Equal(table[rows[r], c], g[r, c]);
    }

    [Fact]
    public void Axis1_GeneralPath_OnContiguousSource()
    {
        var src = Iota(3, 5, 4);
        var idx = new Tensor<int>(new[] { 4, 0, 3 }, new[] { 3 });
        var g = Engine.TensorGather(src, idx, 1);
        int[] cols = { 4, 0, 3 };
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                for (int c = 0; c < 4; c++)
                    Assert.Equal(src[a, cols[b], c], g[a, b, c]);
    }
}
