using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210PDistCDistTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b, float tol = 1e-4f) => MathF.Abs(a - b) <= tol * (1 + MathF.Abs(a) + MathF.Abs(b));

    [Fact]
    public void PDist_Euclidean_Basic()
    {
        // 3 points in 2-D: (0,0), (3,0), (0,4). Pair distances: 3, 4, 5.
        var x = T(new[] { 0f, 0f, 3f, 0f, 0f, 4f }, 3, 2);
        var r = E.TensorPDist(x);
        Assert.Equal(new[] { 3 }, r.Shape.ToArray());
        Assert.True(Close(3f, r[0]));
        Assert.True(Close(4f, r[1]));
        Assert.True(Close(5f, r[2]));
    }

    [Fact]
    public void PDist_L1_Norm()
    {
        // Points (0,0), (1,1): L1 = |1|+|1| = 2.
        var x = T(new[] { 0f, 0f, 1f, 1f }, 2, 2);
        var r = E.TensorPDist(x, p: 1.0);
        Assert.True(Close(2f, r[0]));
    }

    [Fact]
    public void PDist_Empty_Returns_Empty()
    {
        var x = T(Array.Empty<float>(), 0, 2);
        var r = E.TensorPDist(x);
        Assert.Equal(new[] { 0 }, r.Shape.ToArray());
    }

    [Fact]
    public void CDist_Euclidean_Basic()
    {
        var x1 = T(new[] { 0f, 0f, 3f, 0f }, 2, 2);
        var x2 = T(new[] { 0f, 4f, 6f, 8f }, 2, 2);
        var r = E.TensorCDist(x1, x2);
        // r[0,0] = sqrt(0+16) = 4; r[0,1] = sqrt(36+64) = sqrt(100) = 10;
        // r[1,0] = sqrt(9+16) = 5; r[1,1] = sqrt(9+64) = sqrt(73) ≈ 8.544
        Assert.Equal(new[] { 2, 2 }, r.Shape.ToArray());
        Assert.True(Close(4f, r[0, 0]));
        Assert.True(Close(10f, r[0, 1]));
        Assert.True(Close(5f, r[1, 0]));
        Assert.True(Close(MathF.Sqrt(73f), r[1, 1], 1e-3f));
    }

    [Fact]
    public void CDist_FeatureDimMismatch_Throws()
    {
        var x1 = T(new[] { 1f, 2f, 3f }, 1, 3);
        var x2 = T(new[] { 1f, 2f }, 1, 2);
        Assert.Throws<ArgumentException>(() => E.TensorCDist(x1, x2));
    }
}
