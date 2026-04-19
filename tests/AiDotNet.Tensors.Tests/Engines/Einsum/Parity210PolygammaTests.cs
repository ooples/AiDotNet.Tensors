using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210PolygammaTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b, float tol = 1e-3f) => MathF.Abs(a - b) <= tol * (1 + MathF.Abs(a) + MathF.Abs(b));

    [Fact]
    public void Polygamma_N0_IsDigamma()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var pg0 = E.TensorPolygamma(0, x);
        var dg = E.TensorDigamma(x);
        for (int i = 0; i < 3; i++)
            Assert.True(Close(dg[i], pg0[i]));
    }

    [Fact]
    public void Trigamma_AtOne_IsPiSquaredOverSix()
    {
        // ψ'(1) = π²/6 ≈ 1.6449340668
        var x = T(new[] { 1f }, 1);
        var r = E.TensorPolygamma(1, x);
        Assert.True(Close(1.6449340668f, r[0], 1e-3f), $"got {r[0]}");
    }

    [Fact]
    public void Trigamma_AtTwo_IsPiSquaredOverSixMinusOne()
    {
        // ψ'(2) = ψ'(1) - 1 = π²/6 - 1 ≈ 0.6449340668
        var x = T(new[] { 2f }, 1);
        var r = E.TensorPolygamma(1, x);
        Assert.True(Close(0.6449340668f, r[0], 1e-3f), $"got {r[0]}");
    }

    [Fact]
    public void Polygamma_UnsupportedN_Throws()
    {
        var x = T(new[] { 1f }, 1);
        Assert.Throws<NotImplementedException>(() => E.TensorPolygamma(2, x));
    }
}
