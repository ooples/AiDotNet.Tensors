using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210PolygammaTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool CloseF(float a, float b, float tol = 1e-3f) => MathF.Abs(a - b) <= tol * (1 + MathF.Abs(a) + MathF.Abs(b));
    private static bool CloseD(double a, double b, double tol = 1e-4) => Math.Abs(a - b) <= tol * (1 + Math.Abs(a) + Math.Abs(b));

    [Fact]
    public void Polygamma_N0_IsDigamma()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var pg0 = E.TensorPolygamma(0, x);
        var dg = E.TensorDigamma(x);
        for (int i = 0; i < 3; i++)
            Assert.True(CloseF(dg[i], pg0[i]));
    }

    [Fact]
    public void Trigamma_AtOne_IsPiSquaredOverSix()
    {
        // ψ'(1) = π²/6 ≈ 1.6449340668
        var x = T(new[] { 1f }, 1);
        var r = E.TensorPolygamma(1, x);
        Assert.True(CloseF(1.6449340668f, r[0], 1e-3f), $"got {r[0]}");
    }

    [Fact]
    public void Trigamma_AtTwo_IsPiSquaredOverSixMinusOne()
    {
        // ψ'(2) = ψ'(1) - 1 = π²/6 - 1 ≈ 0.6449340668
        var x = T(new[] { 2f }, 1);
        var r = E.TensorPolygamma(1, x);
        Assert.True(CloseF(0.6449340668f, r[0], 1e-3f), $"got {r[0]}");
    }

    [Fact]
    public void Polygamma_N2_AtOne_IsMinusTwoZeta3()
    {
        // ψ''(1) = -2·ζ(3) ≈ -2.4041138063
        var x = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var r = E.TensorPolygamma(2, x);
        Assert.True(CloseD(-2.4041138063, r[0], 1e-4), $"ψ²(1) got {r[0]}");
    }

    [Fact]
    public void Polygamma_N2_AtTwo_UsesRecurrence()
    {
        // ψ''(2) = ψ''(1) - d/dx(-1/x²) = -2·ζ(3) + 2 ≈ -0.4041138063
        var x = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var r = E.TensorPolygamma(2, x);
        Assert.True(CloseD(-0.4041138063, r[0], 1e-4), $"ψ²(2) got {r[0]}");
    }

    [Fact]
    public void Polygamma_N3_AtOne_IsPi4Over15()
    {
        // ψ'''(1) = π⁴/15 ≈ 6.4939394023
        var x = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var r = E.TensorPolygamma(3, x);
        Assert.True(CloseD(6.4939394023, r[0], 1e-4), $"ψ³(1) got {r[0]}");
    }

    [Fact]
    public void Polygamma_N4_AtOne_IsMinus24Zeta5()
    {
        // ψ⁴(1) = -24·ζ(5) ≈ -24.8862661234
        var x = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var r = E.TensorPolygamma(4, x);
        Assert.True(CloseD(-24.8862661234, r[0], 5e-3), $"ψ⁴(1) got {r[0]}");
    }

    [Fact]
    public void Polygamma_NegativeN_Throws()
    {
        var x = T(new[] { 1f }, 1);
        Assert.Throws<ArgumentOutOfRangeException>(() => E.TensorPolygamma(-1, x));
    }

    [Fact]
    public void Polygamma_AtPole_ReturnsInfinity()
    {
        // ψ^(n)(0) diverges for all n ≥ 0.
        var x = new Tensor<double>(new[] { 0.0, -1.0, -2.0 }, new[] { 3 });
        var r = E.TensorPolygamma(2, x);
        for (int i = 0; i < 3; i++)
            Assert.True(double.IsPositiveInfinity(r[i]) || double.IsInfinity(r[i]),
                $"expected Inf at pole, got r[{i}]={r[i]}");
    }
}
