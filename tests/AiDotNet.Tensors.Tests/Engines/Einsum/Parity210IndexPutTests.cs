using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210IndexPutTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static Tensor<int> I(int[] data, params int[] shape) => new Tensor<int>(data, shape);

    [Fact]
    public void IndexPut_2D_WritesAtListedPositions()
    {
        var x = T(new[] { 0f, 0f, 0f, 0f, 0f, 0f }, 2, 3);
        var i0 = I(new[] { 0, 1 }, 2);
        var i1 = I(new[] { 2, 0 }, 2);
        var src = T(new[] { 9f, 5f }, 2);
        var r = E.TensorIndexPut(x, new[] { i0, i1 }, src);
        // (0, 2) = 9; (1, 0) = 5.
        Assert.Equal(new[] { 0f, 0f, 9f, 5f, 0f, 0f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void IndexPut_Accumulate_SumsDuplicates()
    {
        var x = T(new[] { 0f, 0f }, 2);
        var i0 = I(new[] { 0, 0, 1 }, 3);
        var src = T(new[] { 10f, 20f, 5f }, 3);
        var r = E.TensorIndexPut(x, new[] { i0 }, src, accumulate: true);
        Assert.Equal(new[] { 30f, 5f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void IndexPut_NoAccumulate_OverwritesDuplicates()
    {
        var x = T(new[] { 0f, 0f }, 2);
        var i0 = I(new[] { 0, 0 }, 2);
        var src = T(new[] { 10f, 20f }, 2);
        var r = E.TensorIndexPut(x, new[] { i0 }, src, accumulate: false);
        Assert.Equal(20f, r[0]);  // last write wins
    }

    [Fact]
    public void IndexPut_3D_WritesAtMultiAxisPositions()
    {
        var x = T(new float[2 * 2 * 2], 2, 2, 2);
        var i0 = I(new[] { 0, 1 }, 2);
        var i1 = I(new[] { 1, 0 }, 2);
        var i2 = I(new[] { 1, 0 }, 2);
        var src = T(new[] { 7f, 9f }, 2);
        var r = E.TensorIndexPut(x, new[] { i0, i1, i2 }, src);
        // (0,1,1) = 7; (1,0,0) = 9.
        Assert.Equal(7f, r[0, 1, 1]);
        Assert.Equal(9f, r[1, 0, 0]);
    }

    [Fact]
    public void IndexPut_IndexCountMismatch_Throws()
    {
        var x = T(new[] { 0f, 0f, 0f, 0f }, 2, 2);
        var i0 = I(new[] { 0 }, 1);
        var src = T(new[] { 1f }, 1);
        Assert.Throws<System.ArgumentException>(
            () => E.TensorIndexPut(x, new[] { i0 }, src));  // only 1 index, need 2
    }

    [Fact]
    public void IndexPut_OutOfRange_Throws()
    {
        var x = T(new[] { 0f, 0f }, 2);
        var i0 = I(new[] { 5 }, 1);
        var src = T(new[] { 1f }, 1);
        Assert.Throws<System.IndexOutOfRangeException>(
            () => E.TensorIndexPut(x, new[] { i0 }, src));
    }
}
