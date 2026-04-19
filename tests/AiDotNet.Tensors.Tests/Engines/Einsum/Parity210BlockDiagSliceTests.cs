using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210BlockDiagSliceTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void BlockDiag_TwoMatrices_BuildsBlockDiagonal()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 5f, 6f, 7f, 8f, 9f, 10f }, 2, 3);
        var r = E.TensorBlockDiag(new[] { a, b });
        Assert.Equal(new[] { 4, 5 }, r.Shape.ToArray());
        // Top-left 2x2 = a, bottom-right 2x3 = b, rest zero.
        var expected = new float[] {
            1, 2, 0, 0, 0,
            3, 4, 0, 0, 0,
            0, 0, 5, 6, 7,
            0, 0, 8, 9, 10
        };
        Assert.Equal(expected, r.AsSpan().ToArray());
    }

    [Fact]
    public void BlockDiag_SingleMatrix_IsIdentityCopy()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var r = E.TensorBlockDiag(new[] { a });
        Assert.Equal(a.AsSpan().ToArray(), r.AsSpan().ToArray());
    }

    [Fact]
    public void BlockDiag_NonMatrix_Throws()
    {
        var v = T(new[] { 1f, 2f, 3f }, 3);
        Assert.Throws<System.ArgumentException>(() => E.TensorBlockDiag(new[] { v }));
    }

    [Fact]
    public void SliceScatter_OverwritesAxisSlice()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 6);
        var src = T(new[] { 10f, 20f, 30f }, 3);
        var r = E.TensorSliceScatter(x, src, dim: 0, start: 1, length: 3);
        Assert.Equal(new[] { 1f, 10f, 20f, 30f, 5f, 6f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void SliceScatter_2D_Axis1()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var src = T(new[] { 99f, 88f }, 2, 1);
        var r = E.TensorSliceScatter(x, src, dim: 1, start: 1, length: 1);
        Assert.Equal(new[] { 1f, 99f, 3f, 4f, 88f, 6f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void SliceScatter_ShapeMismatch_Throws()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 4);
        var src = T(new[] { 10f }, 1);  // length should be 2
        Assert.Throws<System.ArgumentException>(
            () => E.TensorSliceScatter(x, src, dim: 0, start: 1, length: 2));
    }
}
