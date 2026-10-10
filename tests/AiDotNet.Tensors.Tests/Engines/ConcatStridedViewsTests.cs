using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// CpuEngine.Concat copies whole runs: straight from the storage of a view that is packed from the concat axis inward
/// (a narrow or offset slice), and from a materialized copy otherwise. Every combination of dense inputs, narrowed
/// views (packed and not packed from the axis) and a permuted view must give the element-wise concatenation.
/// </summary>
public class ConcatStridedViewsTests
{
    private static Tensor<double> Seq(params int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = i * 0.5 + 1;
        return t;
    }

    public static TheoryData<string, int> Cases()
    {
        var d = new TheoryData<string, int>();
        foreach (var kind in new[] { "dense", "narrow0", "narrow1", "narrow2", "permuted" })
            for (int axis = 0; axis < 3; axis++) d.Add(kind, axis);
        return d;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Concat_OfViews_MatchesElementwise(string kind, int axis)
    {
        var engine = new CpuEngine();
        // Target view shape [4, 5, 6] for every kind, so the other operand can be shaped to match.
        Tensor<double> view = kind switch
        {
            "dense" => Seq(4, 5, 6),
            "narrow0" => engine.TensorNarrow(Seq(7, 5, 6), 0, 2, 4),
            "narrow1" => engine.TensorNarrow(Seq(4, 9, 6), 1, 3, 5),
            "narrow2" => engine.TensorNarrow(Seq(4, 5, 10), 2, 1, 6),
            _ => engine.TensorPermute(Seq(6, 4, 5), new[] { 1, 2, 0 }),
        };
        var otherShape = new[] { 4, 5, 6 }; otherShape[axis] = 3;
        var other = Seq(otherShape);
        for (int i = 0; i < other.Length; i++) other[i] = -1000 - i;

        foreach (var (first, second) in new[] { (view, other), (other, view) })
        {
            var result = engine.Concat(new[] { first, second }, axis);
            var expectShape = new[] { 4, 5, 6 }; expectShape[axis] = first._shape[axis] + second._shape[axis];
            Assert.Equal(expectShape, result._shape);
            for (int a = 0; a < expectShape[0]; a++)
                for (int b = 0; b < expectShape[1]; b++)
                    for (int c = 0; c < expectShape[2]; c++)
                    {
                        var idx = new[] { a, b, c };
                        var src = first;
                        if (idx[axis] >= first._shape[axis]) { idx[axis] -= first._shape[axis]; src = second; }
                        Assert.Equal(src[idx[0], idx[1], idx[2]], result[a, b, c]);
                    }
        }
    }
}
