using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// The rank-2/3/4 fixed-arity indexers (bound instead of the params indexer for t[i, j], t[i, j, k], t[i, j, k, l])
/// must address exactly what the params indexer addresses -- including strided and offset views -- keep its
/// validation and version semantics, defer to SparseTensor, and not allocate.
/// </summary>
public class FixedArityIndexerTests
{
    private static Tensor<float> Iota(params int[] shape)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = i;
        return t;
    }

    [Fact]
    public void Rank4_PermutedView_MatchesParamsIndexer()
    {
        var t = Iota(2, 3, 4, 5).Transpose(new[] { 3, 1, 0, 2 });   // strided, non-contiguous view
        Assert.False(t.IsContiguous, "the permuted tensor must be a strided view for this test to mean anything");
        for (int a = 0; a < 5; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 2; c++) for (int d = 0; d < 4; d++)
            Assert.Equal(t[new[] { a, b, c, d }], t[a, b, c, d]);
    }

    [Fact]
    public void Rank3_And_Rank2_OffsetViews_MatchParamsIndexer()
    {
        var t3 = Iota(4, 3, 5).Slice(0, 1, 3);                         // storage offset 15
        for (int a = 0; a < 2; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 5; c++)
            Assert.Equal(t3[new[] { a, b, c }], t3[a, b, c]);
        var t2 = Iota(6, 7).Transpose(new[] { 1, 0 });
        for (int a = 0; a < 7; a++) for (int b = 0; b < 6; b++)
            Assert.Equal(t2[new[] { a, b }], t2[a, b]);
    }

    [Fact]
    public void Setter_WritesTheViewedElement_AndBumpsVersion()
    {
        var baseT = Iota(3, 4, 5);
        var view = baseT.Slice(0, 1, 3);
        int v0 = view.Version;
        view[1, 2, 3] = -7f;
        Assert.True(view.Version > v0);
        Assert.Equal(-7f, view[new[] { 1, 2, 3 }]);
        Assert.Equal(-7f, baseT[2, 2, 3]);
    }

    [Fact]
    public void WrongRank_And_OutOfRange_Throw_LikeParamsIndexer()
    {
        var t = Iota(2, 3);
        Assert.Throws<ArgumentException>(() => t[0, 0, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => t[2, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => t[0, -1]);
    }

    [Fact]
    public void SparseTensor_GoesThroughItsOverride()
    {
        var dense = new Tensor<float>(new[] { 3, 4 });
        dense[new[] { 1, 2 }] = 5f;
        var sparse = SparseTensor<float>.FromDense(dense);
        Assert.Equal(5f, sparse[1, 2]);
        Assert.Equal(0f, sparse[0, 0]);
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void Rank4_Read_DoesNotAllocate()
    {
        var t = Iota(2, 3, 4, 5);
        float s = t[1, 1, 1, 1];
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++) s += t[1, 2, 3, i % 5];
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated == 0, $"1000 reads allocated {allocated} bytes (sum {s})");
    }
#endif
}
