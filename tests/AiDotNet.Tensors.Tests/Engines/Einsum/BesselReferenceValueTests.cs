using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Einsum;

/// <summary>
/// The modified Bessel functions I0, I1, I0e and I1e against scipy.special (i0, i1, i0e, i1e) beyond the series'
/// comfortable range, the Half element type, and the I1e gradient against torch's i1e_backward. The CPU ops summed
/// 25 series terms (I0(50) is about 2.9e20, far past that), read elements with Convert.ToDouble (which throws for
/// Half), and MathHelper.BesselI0 divided each term by an extra i! (I0(5) read 12.71).
/// </summary>
public class BesselReferenceValueTests
{
    private static readonly double[] Arguments = { 5.0, 25.0, 30.0, 50.0, -7.5 };

    // scipy.special 1.x: i0, i0e, i1, i1e at Arguments.
    private static readonly double[] I0 = { 27.239871823604442, 5774560606.4663105, 781672297823.9775, 2.9325537838493355e+20, 268.1613115151893 };
    private static readonly double[] I0e = { 0.18354081260932834, 0.08019677354743669, 0.0731459464822373, 0.056561626647454184, 0.1483158300773955 };
    private static readonly double[] I1 = { 24.335642142450524, 5657865129.878702, 768532038938.9569, 2.9030785901035566e+20, -249.58436542268802 };
    private static readonly double[] I1e = { 0.16397226694454234, 0.07857611331929278, 0.07191633059864755, 0.055993123892895395, -0.13804121154855414 };

    private const double RelativeTolerance = 1e-12;

    private static void AssertRelative(double expected, double actual, string label)
        => Assert.True(Math.Abs(actual - expected) <= RelativeTolerance * Math.Abs(expected),
            $"{label}: expected {expected:R}, got {actual:R}");

    [Fact]
    public void MathHelperBesselI0_MatchesScipy()
    {
        for (int i = 0; i < Arguments.Length; i++)
            AssertRelative(I0[i], MathHelper.BesselI0(Arguments[i]), $"BesselI0({Arguments[i]})");
    }

    [Fact]
    public void TensorBesselFamily_MatchesScipy_PastTheSeriesRange()
    {
        var engine = new CpuEngine();
        var x = new Tensor<double>((double[])Arguments.Clone(), new[] { Arguments.Length });
        var i0 = engine.TensorI0(x);
        var i0e = engine.TensorI0e(x);
        var i1 = engine.TensorI1(x);
        var i1e = engine.TensorI1e(x);
        for (int i = 0; i < Arguments.Length; i++)
        {
            AssertRelative(I0[i], i0[i], $"TensorI0({Arguments[i]})");
            AssertRelative(I0e[i], i0e[i], $"TensorI0e({Arguments[i]})");
            AssertRelative(I1[i], i1[i], $"TensorI1({Arguments[i]})");
            AssertRelative(I1e[i], i1e[i], $"TensorI1e({Arguments[i]})");
        }
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void TensorBesselFamily_AcceptsHalf()
    {
        var engine = new CpuEngine();
        var x = new Tensor<Half>(new[] { (Half)5.0, (Half)(-7.5) }, new[] { 2 });
        var i0 = engine.TensorI0(x);
        var i0e = engine.TensorI0e(x);
        var i1 = engine.TensorI1(x);
        var i1e = engine.TensorI1e(x);
        // Half carries 11 significant bits: each result is the reference rounded to Half.
        Assert.Equal((Half)I0[0], i0[0]);
        Assert.Equal((Half)I0[4], i0[1]);
        Assert.Equal((Half)I0e[0], i0e[0]);
        Assert.Equal((Half)I1[4], i1[1]);
        Assert.Equal((Half)I1e[0], i1e[0]);
    }
#endif

    [Fact]
    public void I1eGradient_MatchesTorchI1eBackward()
    {
        // torch's i1e_backward: 0.5 at x = 0, else i0e(x) − i1e(x)·(sgn(x) + 1/x). At −1 that is
        // 0.4657596075936404 − (−0.2079104153497085)·(−2) = 0.049938776894223436.
        var engine = new CpuEngine();
        var x = new Tensor<double>(new[] { -1.0, 0.0, 5.0 }, new[] { 3 });
        double[] expected =
        {
            0.049938776894223436,
            0.5,
            I0e[0] - I1e[0] * (1.0 + 1.0 / 5.0),
        };
        using var tape = new GradientTape<double>();
        var loss = engine.ReduceSum(engine.TensorI1e(x), null);
        var grad = tape.ComputeGradients(loss, new[] { x })[x];
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(grad[i] - expected[i]) <= 1e-12, $"d I1e/dx at {x[i]}: expected {expected[i]:R}, got {grad[i]:R}");
    }
}
