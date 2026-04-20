using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210ZetaTests
{
    private static CpuEngine E => new CpuEngine();
    private static bool Close(double a, double b, double tol = 1e-5)
        => Math.Abs(a - b) <= tol * (1 + Math.Abs(a) + Math.Abs(b));

    [Fact]
    public void Zeta_RiemannAt2_IsPiSquaredOverSix()
    {
        // ζ(2, 1) = π²/6 ≈ 1.6449340668
        var x = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var q = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var r = E.TensorZeta(x, q);
        Assert.True(Close(1.6449340668, r[0]), $"got {r[0]}");
    }

    [Fact]
    public void Zeta_RiemannAt4_IsPi4Over90()
    {
        // ζ(4, 1) = π⁴/90 ≈ 1.0823232337
        var x = new Tensor<double>(new[] { 4.0 }, new[] { 1 });
        var q = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var r = E.TensorZeta(x, q);
        Assert.True(Close(1.0823232337, r[0]), $"got {r[0]}");
    }

    [Fact]
    public void Zeta_AtQTwo_MatchesShiftedRiemann()
    {
        // ζ(s, 2) = ζ(s) - 1  (drop the k=0 term from the Riemann sum)
        // ζ(2, 2) = π²/6 - 1 ≈ 0.6449340668
        var x = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var q = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var r = E.TensorZeta(x, q);
        Assert.True(Close(0.6449340668, r[0], 1e-4), $"got {r[0]}");
    }

    [Fact]
    public void Zeta_AtSEqualsOne_IsInfinity()
    {
        // Simple pole at s = 1.
        var x = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var q = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var r = E.TensorZeta(x, q);
        Assert.True(double.IsPositiveInfinity(r[0]) || double.IsInfinity(r[0]), $"got {r[0]}");
    }

    [Fact]
    public void Zeta_QLessThanZero_NonIntegerConverges()
    {
        // ζ(2, 0.5) = 4·ζ(2) - 1 - 4 by the series/reflection; alternative:
        // ζ(s, 1/2) = (2^s - 1) · ζ(s). For s=2: 3·ζ(2) ≈ 4.9348022005
        var x = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var q = new Tensor<double>(new[] { 0.5 }, new[] { 1 });
        var r = E.TensorZeta(x, q);
        Assert.True(Close(4.9348022005, r[0], 5e-3), $"got {r[0]}");
    }

    [Fact]
    public void Zeta_ShapeMismatch_Throws()
    {
        var x = new Tensor<double>(new[] { 2.0, 3.0 }, new[] { 2 });
        var q = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        Assert.Throws<ArgumentException>(() => E.TensorZeta(x, q));
    }
}
