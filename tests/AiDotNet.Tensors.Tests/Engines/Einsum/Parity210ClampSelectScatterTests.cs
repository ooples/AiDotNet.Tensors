using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210ClampSelectScatterTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void ClampTensor_WithBothBounds_ClampsPerPosition()
    {
        var x = T(new[] { 1f, 5f, -3f, 10f }, 4);
        var min = T(new[] { 0f, 0f, 0f, 0f }, 4);
        var max = T(new[] { 4f, 4f, 4f, 4f }, 4);
        var r = E.TensorClampTensor(x, min, max);
        Assert.Equal(new[] { 1f, 4f, 0f, 4f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ClampTensor_MinOnly()
    {
        var x = T(new[] { -1f, 2f, -5f }, 3);
        var min = T(new[] { 0f, 0f, 0f }, 3);
        var r = E.TensorClampTensor(x, min, null);
        Assert.Equal(new[] { 0f, 2f, 0f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ClampTensor_MaxOnly()
    {
        var x = T(new[] { 10f, 5f, -1f }, 3);
        var max = T(new[] { 5f, 5f, 5f }, 3);
        var r = E.TensorClampTensor(x, null, max);
        Assert.Equal(new[] { 5f, 5f, -1f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ClampTensor_PerPositionVaryingBounds()
    {
        var x = T(new[] { 1f, 5f, 10f }, 3);
        var min = T(new[] { 0f, 6f, 0f }, 3);
        var max = T(new[] { 2f, 10f, 8f }, 3);
        var r = E.TensorClampTensor(x, min, max);
        // position 1: 5 clamped to min=6 → 6; position 2: 10 clamped to max=8 → 8.
        Assert.Equal(new[] { 1f, 6f, 8f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ClampTensor_NoBounds_Throws()
    {
        var x = T(new[] { 1f }, 1);
        Assert.Throws<System.ArgumentException>(() => E.TensorClampTensor(x, null, null));
    }

    [Fact]
    public void SelectScatter_WritesRow()
    {
        var x = T(new[] { 0f, 0f, 0f, 0f, 0f, 0f }, 2, 3);
        var row = T(new[] { 1f, 2f, 3f }, 3);
        var r = E.TensorSelectScatter(x, row, dim: 0, index: 0);
        // Row 0 filled with source; row 1 still zero.
        Assert.Equal(new[] { 1f, 2f, 3f, 0f, 0f, 0f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void SelectScatter_WritesColumn()
    {
        var x = T(new[] { 0f, 0f, 0f, 0f, 0f, 0f }, 2, 3);
        var col = T(new[] { 1f, 2f }, 2);
        var r = E.TensorSelectScatter(x, col, dim: 1, index: 1);
        Assert.Equal(new[] { 0f, 1f, 0f, 0f, 2f, 0f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void SelectScatter_RankMismatch_Throws()
    {
        var x = T(new[] { 0f, 0f, 0f, 0f }, 2, 2);
        var bad = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);  // same rank — not allowed
        Assert.Throws<System.ArgumentException>(
            () => E.TensorSelectScatter(x, bad, 0, 0));
    }
}
