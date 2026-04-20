using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210CartesianProdTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void CartesianProd_TwoVectors()
    {
        var a = T(new[] { 1f, 2f }, 2);
        var b = T(new[] { 3f, 4f, 5f }, 3);
        var r = E.TensorCartesianProd(new[] { a, b });
        Assert.Equal(new[] { 6, 2 }, r.Shape.ToArray());
        // Row-major: (a[0],b[0]), (a[0],b[1]), (a[0],b[2]), (a[1],b[0]), ...
        var expected = new[] { 1f, 3f, 1f, 4f, 1f, 5f, 2f, 3f, 2f, 4f, 2f, 5f };
        Assert.Equal(expected, r.AsSpan().ToArray());
    }

    [Fact]
    public void CartesianProd_Single_ReturnsColumn()
    {
        var a = T(new[] { 1f, 2f, 3f }, 3);
        var r = E.TensorCartesianProd(new[] { a });
        Assert.Equal(new[] { 3, 1 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void CartesianProd_Three_2x2x2()
    {
        var a = T(new[] { 0f, 1f }, 2);
        var b = T(new[] { 0f, 1f }, 2);
        var c = T(new[] { 0f, 1f }, 2);
        var r = E.TensorCartesianProd(new[] { a, b, c });
        Assert.Equal(new[] { 8, 3 }, r.Shape.ToArray());
        // Binary counting 0..7.
        Assert.Equal(0f, r[0, 0]); Assert.Equal(0f, r[0, 1]); Assert.Equal(0f, r[0, 2]);
        Assert.Equal(1f, r[7, 0]); Assert.Equal(1f, r[7, 1]); Assert.Equal(1f, r[7, 2]);
    }

    [Fact]
    public void CartesianProd_NonRank1_Throws()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        Assert.Throws<System.ArgumentException>(() => E.TensorCartesianProd(new[] { a }));
    }
}
