using System;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210AdvancedIndexingTests
{
    [Fact]
    public void BooleanMaskIndexer_SelectsMaskedElements()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var mask = new Tensor<Bit>(new[] { Bit.True, Bit.False, Bit.True, Bit.False }, new[] { 4 });
        var r = x[mask];
        Assert.Equal(new[] { 1f, 3f }, r.GetDataArray());
    }

    [Fact]
    public void IndexTensorIndexer_SelectsRowsAlongAxisZero()
    {
        // [3, 2] picked by indices [2, 0] → two rows in new order.
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 3, 2 });
        var idx = new Tensor<int>(new[] { 2, 0 }, new[] { 2 });
        var r = x[idx];
        Assert.Equal(new[] { 2, 2 }, r.Shape.ToArray());
        Assert.Equal(new[] { 5f, 6f, 1f, 2f }, r.GetDataArray());
    }

    [Fact]
    public void Unsqueeze_InsertsAxisOfSizeOne()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var r = x.Unsqueeze(0);
        Assert.Equal(new[] { 1, 3 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f }, r.GetDataArray());
    }

    [Fact]
    public void Unsqueeze_NegativeAxis_InsertsFromEnd()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var r = x.Unsqueeze(-1);
        Assert.Equal(new[] { 3, 1 }, r.Shape.ToArray());
    }

    [Fact]
    public void Unsqueeze_AxisAtRank_AppendsTrailingAxis()
    {
        var x = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var r = x.Unsqueeze(1);
        Assert.Equal(new[] { 2, 1 }, r.Shape.ToArray());
    }

    [Fact]
    public void SelectAlong_PicksSingleElementAndReducesRank()
    {
        // [3, 2] — select row 1 along axis 0 → shape [2].
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 3, 2 });
        var r = x.SelectAlong(0, 1);
        Assert.Equal(new[] { 2 }, r.Shape.ToArray());
        Assert.Equal(new[] { 3f, 4f }, r.GetDataArray());
    }

    [Fact]
    public void SelectAlong_NegativeAxis_Works()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var r = x.SelectAlong(-1, 0);
        Assert.Equal(new[] { 2 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 3f }, r.GetDataArray());
    }

    [Fact]
    public void SelectAlong_NegativeIndex_CountsFromEnd()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var r = x.SelectAlong(0, -1);
        Assert.Equal(new[] { 3f, 4f }, r.GetDataArray());
    }

    [Fact]
    public void NormalizeAxis_WrapsNegativesAndRejectsOutOfRange()
    {
        Assert.Equal(2, Tensor<float>.NormalizeAxis(-1, 3));
        Assert.Equal(0, Tensor<float>.NormalizeAxis(0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor<float>.NormalizeAxis(5, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor<float>.NormalizeAxis(-4, 3));
    }
}
