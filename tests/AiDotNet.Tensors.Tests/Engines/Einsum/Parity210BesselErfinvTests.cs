using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210BesselErfinvTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b, float tol = 1e-3f) => MathF.Abs(a - b) <= tol * (1 + MathF.Abs(a) + MathF.Abs(b));

    [Fact]
    public void Erfinv_AtZero_IsZero()
    {
        var x = T(new[] { 0f }, 1);
        var r = E.TensorErfinv(x);
        Assert.True(MathF.Abs(r[0]) < 1e-3f, $"got {r[0]}");
    }

    [Fact]
    public void Erfinv_AtHalfMatchesKnownValue()
    {
        // erfinv(0.5) ≈ 0.4769362762...
        var x = T(new[] { 0.5f }, 1);
        var r = E.TensorErfinv(x);
        Assert.True(Close(0.4769362762f, r[0], 1e-3f), $"got {r[0]}");
    }

    [Fact]
    public void Erfinv_AtNegativeHalf()
    {
        var x = T(new[] { -0.5f }, 1);
        var r = E.TensorErfinv(x);
        Assert.True(Close(-0.4769362762f, r[0], 1e-3f), $"got {r[0]}");
    }

    [Fact]
    public void Erfinv_AtOne_IsInfinity()
    {
        var x = T(new[] { 1f }, 1);
        var r = E.TensorErfinv(x);
        Assert.True(float.IsInfinity(r[0]));
    }

    [Fact]
    public void I0_AtZero_IsOne()
    {
        var x = T(new[] { 0f }, 1);
        var r = E.TensorI0(x);
        Assert.True(Close(1f, r[0]));
    }

    [Fact]
    public void I0_AtOne_MatchesKnown()
    {
        // I0(1) ≈ 1.2660658732
        var x = T(new[] { 1f }, 1);
        var r = E.TensorI0(x);
        Assert.True(Close(1.2660658732f, r[0], 1e-4f));
    }

    [Fact]
    public void I1_AtZero_IsZero()
    {
        var x = T(new[] { 0f }, 1);
        var r = E.TensorI1(x);
        Assert.True(MathF.Abs(r[0]) < 1e-6f, $"got {r[0]}");
    }

    [Fact]
    public void I1_AtOne_MatchesKnown()
    {
        // I1(1) ≈ 0.5651591040
        var x = T(new[] { 1f }, 1);
        var r = E.TensorI1(x);
        Assert.True(Close(0.5651591040f, r[0], 1e-4f));
    }
}
