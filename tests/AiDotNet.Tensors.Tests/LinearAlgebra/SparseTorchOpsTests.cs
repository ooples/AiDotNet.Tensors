using System;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// The sparse COO operations against torch 2.12. S is torch.sparse_coo_tensor([[0, 1, 2, 2], [1, 0, 2, 0]],
/// [1, 2, 3, 4], (3, 3)); the expected values were printed by torch for the same inputs.
/// </summary>
public class SparseTorchOpsTests
{
    private static SparseTensor<double> S() =>
        new SparseTensor<double>(3, 3, new[] { 0, 1, 2, 2 }, new[] { 1, 0, 2, 0 }, new double[] { 1, 2, 3, 4 });

    private static Tensor<double> Arange(int rows, int cols) =>
        new Tensor<double>(Enumerable.Range(0, rows * cols).Select(i => (double)i).ToArray(), new[] { rows, cols });

    [Fact]
    public void Indices_AreTheCoalescedCooIndices()
    {
        var idx = S().Indices();
        Assert.Equal(new[] { 2, 4 }, idx.Shape.ToArray());
        Assert.Equal(new[] { 0, 1, 2, 2, 1, 0, 0, 2 }, idx.ToArray());
    }

    [Fact]
    public void SparseMask_TakesTheSourceValuesAtTheMaskPattern()
    {
        var m = SparseTensor<double>.SparseMask(Arange(3, 3), S());
        Assert.Equal(new[] { 0, 1, 2, 2 }, m.RowIndices);
        Assert.Equal(new[] { 1, 0, 0, 2 }, m.ColumnIndices);
        Assert.Equal(new double[] { 1, 3, 6, 8 }, m.Values);
    }

    [Fact]
    public void Smm_Hsmm_Hspmm_MatchTorch()
    {
        var expected = new double[] { 2, 3, 0, 2, 12, 19 };
        foreach (var r in new[] { SparseTensor<double>.Smm(S(), Arange(3, 2)), SparseTensor<double>.Hsmm(S(), Arange(3, 2)), SparseTensor<double>.Hspmm(S(), Arange(3, 2)) })
        {
            Assert.Equal(new[] { 3, 2 }, r.Shape.ToArray());
            Assert.Equal(6, r.NonZeroCount);   // every entry of each row that stores something, as torch
            Assert.Equal(expected, r.ToDense().ToArray());
        }
    }

    [Fact]
    public void Saddmm_And_Sspaddmm_MatchTorch()
    {
        var expected = new double[] { 1.5, 4, 2.5, 4, 1, 2, 17, 12.5, 22 };
        var a = SparseTensor<double>.Saddmm(S(), S(), Arange(3, 3), 2.0, 0.5);
        var b = SparseTensor<double>.Sspaddmm(S(), S(), Arange(3, 3), 2.0, 0.5);
        Assert.Equal(expected, a.ToDense().ToArray());
        Assert.Equal(expected, b.ToDense().ToArray());
        Assert.Equal(9, a.NonZeroCount);   // torch: 13 uncoalesced, 9 coalesced
    }

    [Fact]
    public void SparseResize_GrowsKeepingTheElements()
    {
        var s = S();
        s.SparseResize(4, 5);
        Assert.Equal(new[] { 4, 5 }, s.Shape.ToArray());
        Assert.Equal(4, s.NonZeroCount);
        Assert.Equal(1.0, s.ToDense()[0, 1]);
        Assert.Equal(0.0, s.ToDense()[3, 4]);
    }

    [Fact]
    public void SparseResize_RefusesToShrinkANonEmptyTensor_AsTorchDoes()
        => Assert.Throws<InvalidOperationException>(() => S().SparseResize(2, 2));

    [Fact]
    public void SparseResizeAndClear_DropsEveryElement()
    {
        var s = S();
        s.SparseResizeAndClear(2, 2);
        Assert.Equal(new[] { 2, 2 }, s.Shape.ToArray());
        Assert.Equal(0, s.NonZeroCount);
        Assert.Equal(new double[4], s.ToDense().ToArray());
    }

    [Fact]
    public void ResizeAsSparse_TakesTheOtherSize()
    {
        var s = S();
        s.ResizeAsSparse(new SparseTensor<double>(5, 6, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<double>()));
        Assert.Equal(new[] { 5, 6 }, s.Shape.ToArray());
        Assert.Equal(4, s.NonZeroCount);
    }

    [Fact]
    public void SparseResize_OfACsrTensor_ExtendsTheRowPointers()
    {
        var csr = S().ToCsr();
        csr.SparseResize(5, 3);
        Assert.Equal(6, csr.RowPointers.Length);
        Assert.Equal(S().ToDense().ToArray().Concat(new double[6]).ToArray(), csr.ToDense().ToArray());
    }
}
