using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210ExpBesselFrexpTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b, float tol = 1e-3f) => MathF.Abs(a - b) <= tol * (1 + MathF.Abs(a) + MathF.Abs(b));

    [Fact]
    public void I0e_AtZero_IsOne()
    {
        var x = T(new[] { 0f }, 1);
        var r = E.TensorI0e(x);
        Assert.True(Close(1f, r[0]));
    }

    [Fact]
    public void I0e_AtOne_MatchesKnown()
    {
        // I0e(1) = e^-1 · I0(1) ≈ 0.36788 · 1.26607 ≈ 0.4658
        var x = T(new[] { 1f }, 1);
        var r = E.TensorI0e(x);
        Assert.True(Close(0.4657596f, r[0], 1e-4f));
    }

    [Fact]
    public void I1e_AtZero_IsZero()
    {
        var x = T(new[] { 0f }, 1);
        var r = E.TensorI1e(x);
        Assert.True(MathF.Abs(r[0]) < 1e-6f);
    }

    [Fact]
    public void I1e_AtOne_MatchesKnown()
    {
        // I1e(1) = e^-1 · I1(1) ≈ 0.36788 · 0.56516 ≈ 0.2079
        var x = T(new[] { 1f }, 1);
        var r = E.TensorI1e(x);
        Assert.True(Close(0.2079104f, r[0], 1e-4f));
    }

    [Fact]
    public void Frexp_Of4_Returns_0_5_and_3()
    {
        // 4 = 0.5 · 2^3
        var x = T(new[] { 4f }, 1);
        var (m, e) = E.TensorFrexp(x);
        Assert.True(Close(0.5f, m[0]));
        Assert.Equal(3, e[0]);
    }

    [Fact]
    public void Frexp_OfZero_ReturnsZeroAndZero()
    {
        var x = T(new[] { 0f }, 1);
        var (m, e) = E.TensorFrexp(x);
        Assert.Equal(0f, m[0]);
        Assert.Equal(0, e[0]);
    }

    [Fact]
    public void Frexp_Of1_Point5_Returns_0_75_and_1()
    {
        // 1.5 = 0.75 · 2^1
        var x = T(new[] { 1.5f }, 1);
        var (m, e) = E.TensorFrexp(x);
        Assert.True(Close(0.75f, m[0]));
        Assert.Equal(1, e[0]);
    }

    [Fact]
    public void Frexp_Negative_PreservesSign()
    {
        var x = T(new[] { -8f }, 1);
        var (m, e) = E.TensorFrexp(x);
        // -8 = -0.5 · 2^4
        Assert.True(Close(-0.5f, m[0]));
        Assert.Equal(4, e[0]);
    }
}
