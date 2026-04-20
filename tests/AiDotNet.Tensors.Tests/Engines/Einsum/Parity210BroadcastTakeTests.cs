using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210BroadcastTakeTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static Tensor<int> I(int[] data, params int[] shape) => new Tensor<int>(data, shape);

    // --- BroadcastTo --------------------------------------------------

    [Fact]
    public void BroadcastTo_ExpandsLeadingDim()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var r = E.TensorBroadcastTo(x, new[] { 2, 3 });
        Assert.Equal(new[] { 2, 3 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 1f, 2f, 3f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void BroadcastTo_ExpandsSize1Dim()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3, 1);
        var r = E.TensorBroadcastTo(x, new[] { 3, 2 });
        Assert.Equal(new[] { 3, 2 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 1f, 2f, 2f, 3f, 3f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void BroadcastTo_Incompatible_Throws()
    {
        var x = T(new[] { 1f, 2f }, 2);
        Assert.Throws<System.ArgumentException>(
            () => E.TensorBroadcastTo(x, new[] { 3, 3 }));
    }

    // --- Take ---------------------------------------------------------

    [Fact]
    public void Take_PicksByFlatIndex()
    {
        var x = T(new[] { 10f, 20f, 30f, 40f, 50f, 60f }, 2, 3);
        var idx = I(new[] { 0, 2, 4 }, 3);
        var r = E.TensorTake(x, idx);
        Assert.Equal(new[] { 10f, 30f, 50f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Take_PreservesIndicesShape()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 6);
        var idx = I(new[] { 0, 5, 2, 3 }, 2, 2);
        var r = E.TensorTake(x, idx);
        Assert.Equal(new[] { 2, 2 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 6f, 3f, 4f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Take_OutOfRange_Throws()
    {
        var x = T(new[] { 1f, 2f }, 2);
        var idx = I(new[] { 5 }, 1);
        Assert.Throws<System.IndexOutOfRangeException>(() => E.TensorTake(x, idx));
    }

    // --- TakeAlongDim -------------------------------------------------

    [Fact]
    public void TakeAlongDim_Gathers()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        // Pick column 2 for row 0, column 0 for row 1.
        var idx = I(new[] { 2, 0 }, 2, 1);
        var r = E.TensorTakeAlongDim(x, idx, dim: 1);
        Assert.Equal(new[] { 2, 1 }, r.Shape.ToArray());
        Assert.Equal(new float[] { 3f, 4f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void TakeAlongDim_RankMismatch_Throws()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var idx = I(new[] { 0, 1 }, 2);  // Rank 1, tensor rank 2.
        Assert.Throws<System.ArgumentException>(
            () => E.TensorTakeAlongDim(x, idx, dim: 0));
    }
}
