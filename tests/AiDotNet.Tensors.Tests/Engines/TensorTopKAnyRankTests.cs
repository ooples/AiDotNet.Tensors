using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// TensorTopK used to accept only [batch, features] along axis 1. It now takes any rank and axis, and reads a
/// non-contiguous view through its strides (the old 2-D path read through the indexer).
/// </summary>
public sealed class TensorTopKAnyRankTests
{
    [Fact]
    public void Topk_on_a_transposed_view_matches_topk_on_its_contiguous_copy()
    {
        var eng = new CpuEngine();
        var x = new Tensor<float>(new[] { 3, 5 }, new Vector<float>(new float[]
        {
            4, 9, 1, 7, 3,
            8, 2, 6, 0, 5,
            1, 1, 9, 4, 2,
        }));
        var view = eng.TensorTranspose(x); // [5, 3], non-contiguous
        var copy = new Tensor<float>(new[] { 5, 3 }, new Vector<float>(view.ToArray()));
        for (int axis = 0; axis < 2; axis++)
        {
            var a = eng.TensorTopK(view, 2, axis, out var ia);
            var b = eng.TensorTopK(copy, 2, axis, out var ib);
            Assert.Equal(b.Shape.ToArray(), a.Shape.ToArray());
            Assert.Equal(b.ToArray(), a.ToArray());
            Assert.Equal(ib.ToArray(), ia.ToArray());
        }
    }

    [Fact]
    public void Topk_along_a_middle_axis_returns_descending_values_and_their_positions()
    {
        var eng = new CpuEngine();
        // [2, 4, 3]: along axis 1 each (outer, inner) fibre is a column of four values.
        var data = Enumerable.Range(0, 24).Select(i => (float)((i * 7) % 11)).ToArray();
        var x = new Tensor<float>(new[] { 2, 4, 3 }, new Vector<float>(data));
        var v = eng.TensorTopK(x, 2, 1, out var idx);
        Assert.Equal(new[] { 2, 2, 3 }, v.Shape.ToArray());
        var vals = v.ToArray();
        var pos = idx.ToArray();
        for (int o = 0; o < 2; o++)
            for (int k = 0; k < 3; k++)
            {
                var fibre = Enumerable.Range(0, 4).Select(j => data[(o * 4 + j) * 3 + k]).OrderByDescending(f => f).ToArray();
                for (int j = 0; j < 2; j++)
                {
                    int p = (o * 2 + j) * 3 + k;
                    Assert.Equal(fibre[j], vals[p]);
                    Assert.Equal(vals[p], data[(o * 4 + pos[p]) * 3 + k]);
                }
            }
    }
}
