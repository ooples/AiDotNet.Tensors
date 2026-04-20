using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210LogAddExpTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b, float tol = 1e-4f) => MathF.Abs(a - b) <= tol * (1 + MathF.Abs(a) + MathF.Abs(b));

    [Fact]
    public void LogAddExp_BasicEqualInputs()
    {
        // log(exp(a) + exp(a)) = a + log(2).
        var a = T(new[] { 5f }, 1);
        var b = T(new[] { 5f }, 1);
        var r = E.TensorLogAddExp(a, b);
        Assert.True(Close(5f + MathF.Log(2f), r[0]));
    }

    [Fact]
    public void LogAddExp_LargeInputs_NoOverflow()
    {
        // exp(1000) overflows — naive implementation fails. Stable version returns 1000 + log(2).
        var a = T(new[] { 1000f }, 1);
        var b = T(new[] { 1000f }, 1);
        var r = E.TensorLogAddExp(a, b);
        Assert.False(float.IsInfinity(r[0]));
        Assert.True(Close(1000f + MathF.Log(2f), r[0], 1e-3f));
    }

    [Fact]
    public void LogAddExp_VeryDifferent_EqualsLarger()
    {
        // When b is much smaller than a, log(exp(a) + exp(b)) ≈ a.
        var a = T(new[] { 100f }, 1);
        var b = T(new[] { -100f }, 1);
        var r = E.TensorLogAddExp(a, b);
        Assert.True(Close(100f, r[0], 1e-3f));
    }

    [Fact]
    public void LogAddExp2_MatchesBaseTwo()
    {
        // log2(2^3 + 2^3) = log2(16) = 4.
        var a = T(new[] { 3f }, 1);
        var b = T(new[] { 3f }, 1);
        var r = E.TensorLogAddExp2(a, b);
        Assert.True(Close(4f, r[0]));
    }

    [Fact]
    public void LogAddExp2_AsymmetricInputs()
    {
        // log2(2^5 + 2^1) = log2(34) ≈ 5.087
        var a = T(new[] { 5f }, 1);
        var b = T(new[] { 1f }, 1);
        var r = E.TensorLogAddExp2(a, b);
        Assert.True(Close(5.087463f, r[0], 1e-3f));
    }
}
