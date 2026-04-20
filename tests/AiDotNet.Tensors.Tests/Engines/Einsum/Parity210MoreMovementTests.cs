using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210MoreMovementTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void Fliplr_ReversesLastAxis()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var r = E.TensorFliplr(x);
        Assert.Equal(new[] { 3f, 2f, 1f, 6f, 5f, 4f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Fliplr_RejectsRank1()
    {
        var x = T(new[] { 1f, 2f }, 2);
        Assert.Throws<System.ArgumentException>(() => E.TensorFliplr(x));
    }

    [Fact]
    public void Flipud_ReversesFirstAxis()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var r = E.TensorFlipud(x);
        // Row 0 and row 1 swapped.
        Assert.Equal(new[] { 4f, 5f, 6f, 1f, 2f, 3f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Rot90_k1_Transposes_ThenFlips()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var r = E.TensorRot90(x);
        // 90° ccw: [[1,2],[3,4]] -> [[2,4],[1,3]]
        Assert.Equal(new[] { 2f, 4f, 1f, 3f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Rot90_k2_DoubleFlip()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var r = E.TensorRot90(x, k: 2);
        // 180°: reverses everything.
        Assert.Equal(new[] { 4f, 3f, 2f, 1f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Rot90_k4_IsIdentity()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var r = E.TensorRot90(x, k: 4);
        Assert.Equal(x.AsSpan().ToArray(), r.AsSpan().ToArray());
    }

    [Fact]
    public void SwapAxes_2D_Transposes()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var r = E.TensorSwapAxes(x, 0, 1);
        // Transpose: shape becomes [3, 2]; Transpose returns a non-contiguous
        // view (O(1) stride rewrite), so materialise before taking the span.
        Assert.Equal(new[] { 3, 2 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 4f, 2f, 5f, 3f, 6f }, r.Contiguous().AsSpan().ToArray());
    }

    [Fact]
    public void MoveDim_MovesDimensionToFront()
    {
        // Shape [2, 3, 4] → move axis 2 to axis 0 → shape [4, 2, 3].
        var x = T(new float[24], 2, 3, 4);
        var r = E.TensorMoveDim(x, source: 2, destination: 0);
        Assert.Equal(new[] { 4, 2, 3 }, r.Shape.ToArray());
    }

    [Fact]
    public void MoveDim_IdentityWhenSourceEqualsDestination()
    {
        var x = T(new float[6], 2, 3);
        var r = E.TensorMoveDim(x, 1, 1);
        Assert.Equal(new[] { 2, 3 }, r.Shape.ToArray());
    }

    [Fact]
    public void AtLeast1D_ScalarBecomes1D()
    {
        // Rank-0 scalar tensor (empty-shape construction).
        var x = T(new[] { 42f }, new int[0]);
        var r = E.TensorAtLeast1D(x);
        Assert.Equal(new[] { 1 }, r.Shape.ToArray());
    }

    [Fact]
    public void AtLeast2D_PromotesRank1()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var r = E.TensorAtLeast2D(x);
        Assert.Equal(new[] { 1, 3 }, r.Shape.ToArray());
    }

    [Fact]
    public void AtLeast3D_PromotesRank2()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var r = E.TensorAtLeast3D(x);
        Assert.Equal(new[] { 1, 2, 2 }, r.Shape.ToArray());
    }
}
