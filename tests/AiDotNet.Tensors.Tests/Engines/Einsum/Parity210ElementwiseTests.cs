using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210ElementwiseTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b, float eps = 1e-5f) => MathF.Abs(a - b) <= eps * (1 + MathF.Abs(a) + MathF.Abs(b));

    // --- Hypot --------------------------------------------------------

    [Fact]
    public void Hypot_3_4_Is_5()
    {
        var a = T(new[] { 3f, 6f }, 2);
        var b = T(new[] { 4f, 8f }, 2);
        var r = E.TensorHypot(a, b);
        Assert.True(Close(5f, r[0]));
        Assert.True(Close(10f, r[1]));
    }

    [Fact]
    public void Hypot_WithNegative_TakesMagnitude()
    {
        var a = T(new[] { -3f }, 1);
        var b = T(new[] { -4f }, 1);
        var r = E.TensorHypot(a, b);
        Assert.True(Close(5f, r[0]));
    }

    // --- Copysign -----------------------------------------------------

    [Fact]
    public void Copysign_PositiveB_LeavesPositive()
    {
        var a = T(new[] { -2f, 3f }, 2);
        var b = T(new[] { 1f, 1f }, 2);
        var r = E.TensorCopysign(a, b);
        Assert.Equal(2f, r[0]);
        Assert.Equal(3f, r[1]);
    }

    [Fact]
    public void Copysign_NegativeB_MakesNegative()
    {
        var a = T(new[] { 5f, -5f }, 2);
        var b = T(new[] { -1f, -1f }, 2);
        var r = E.TensorCopysign(a, b);
        Assert.Equal(-5f, r[0]);
        Assert.Equal(-5f, r[1]);
    }

    // --- Fmod ---------------------------------------------------------

    [Fact]
    public void Fmod_KeepsSignOfDividend()
    {
        var a = T(new[] { 7f, -7f, 7f, -7f }, 4);
        var b = T(new[] { 3f, 3f, -3f, -3f }, 4);
        var r = E.TensorFmod(a, b);
        // fmod(7, 3) = 1; fmod(-7, 3) = -1; fmod(7, -3) = 1; fmod(-7, -3) = -1.
        Assert.Equal(1f, r[0]);
        Assert.Equal(-1f, r[1]);
        Assert.Equal(1f, r[2]);
        Assert.Equal(-1f, r[3]);
    }

    // --- Remainder ----------------------------------------------------

    [Fact]
    public void Remainder_KeepsSignOfDivisor()
    {
        var a = T(new[] { 7f, -7f, 7f, -7f }, 4);
        var b = T(new[] { 3f, 3f, -3f, -3f }, 4);
        var r = E.TensorRemainder(a, b);
        // remainder = a - floor(a/b)*b.
        // (7, 3) = 1; (-7, 3) = 2; (7, -3) = -2; (-7, -3) = -1.
        Assert.Equal(1f, r[0]);
        Assert.Equal(2f, r[1]);
        Assert.Equal(-2f, r[2]);
        Assert.Equal(-1f, r[3]);
    }

    // --- FloatPower ---------------------------------------------------

    [Fact]
    public void FloatPower_CubesValues()
    {
        var a = T(new[] { 2f, 3f }, 2);
        var b = T(new[] { 3f, 3f }, 2);
        var r = E.TensorFloatPower(a, b);
        Assert.Equal(8f, r[0]);
        Assert.Equal(27f, r[1]);
    }

    // --- Erfc ---------------------------------------------------------

    [Fact]
    public void Erfc_AtZero_IsOne()
    {
        var x = T(new[] { 0f }, 1);
        var r = E.TensorErfc(x);
        Assert.True(Close(1f, r[0], 1e-4f));
    }

    [Fact]
    public void Erfc_AtInfinityProxy_IsSmall()
    {
        var x = T(new[] { 3f }, 1);
        var r = E.TensorErfc(x);
        // erfc(3) ≈ 2.2e-5.
        Assert.True(r[0] < 0.001f, $"erfc(3) ≈ {r[0]}, expected near zero");
        Assert.True(r[0] >= 0f);
    }

    // --- Xlogy / Xlog1py ---------------------------------------------

    [Fact]
    public void Xlogy_StandardCase()
    {
        var x = T(new[] { 2f, 3f }, 2);
        var y = T(new[] { MathF.E, MathF.E * MathF.E }, 2);
        var r = E.TensorXlogy(x, y);
        // 2 * log(e) = 2; 3 * log(e²) = 6.
        Assert.True(Close(2f, r[0]));
        Assert.True(Close(6f, r[1]));
    }

    [Fact]
    public void Xlogy_ZeroX_ReturnsZero_EvenWhenYZero()
    {
        // Whole point of xlogy: treat 0·log(0) as 0 rather than NaN.
        var x = T(new[] { 0f }, 1);
        var y = T(new[] { 0f }, 1);
        var r = E.TensorXlogy(x, y);
        Assert.Equal(0f, r[0]);
    }

    [Fact]
    public void Xlog1py_StandardCase()
    {
        var x = T(new[] { 2f }, 1);
        var y = T(new[] { MathF.E - 1f }, 1);
        var r = E.TensorXlog1py(x, y);
        // 2 * log(1 + (e-1)) = 2 * log(e) = 2.
        Assert.True(Close(2f, r[0], 1e-4f));
    }

    [Fact]
    public void Xlog1py_ZeroX_ReturnsZero()
    {
        var x = T(new[] { 0f }, 1);
        var y = T(new[] { -1f }, 1); // would be log(0) otherwise
        var r = E.TensorXlog1py(x, y);
        Assert.Equal(0f, r[0]);
    }
}
