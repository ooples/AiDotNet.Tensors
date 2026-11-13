using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// <c>Cast</c> converts a tensor's own elements in logical order. It used to read the backing storage from index 0,
/// which is the tensor's content only for a dense, zero-offset, row-major layout: offset and strided views converted
/// other elements of their base while every shape still matched.
/// </summary>
public class CastViewTests
{
    private static Tensor<float> Iota(params int[] shape)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = i;
        return t;
    }

    private static void AssertSameElements(Tensor<float> source, Tensor<double> cast)
    {
        Assert.Equal(source.Shape.ToArray(), cast.Shape.ToArray());
        var expected = source.ToArray();
        var actual = cast.ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void OffsetView_CastsItsOwnElements()
    {
        // Rows 1..2 of a [3, 4] iota: contiguous, but starting at storage offset 4. The old cast returned 0..7.
        var view = AiDotNetEngine.Current.TensorNarrow(Iota(3, 4), 0, 1, 2);

        var cast = view.Cast<double>();

        Assert.Equal(new[] { 4.0, 5, 6, 7, 8, 9, 10, 11 }, cast.ToArray());
    }

    [Fact]
    public void StridedNarrowView_CastsItsOwnElements()
    {
        // Column 1 of a [3, 2] iota is [1, 3, 5]; a flat read of the storage gave [0, 1, 2].
        var view = AiDotNetEngine.Current.TensorNarrow(Iota(3, 2), 1, 1, 1);

        var cast = view.Cast<double>();

        Assert.Equal(new[] { 1.0, 3, 5 }, cast.ToArray());
    }

    [Fact]
    public void PermutedView_CastsInLogicalOrder()
    {
        var view = Iota(2, 3, 4).Transpose(new[] { 2, 0, 1 });

        AssertSameElements(view, view.Cast<double>());
    }
}
