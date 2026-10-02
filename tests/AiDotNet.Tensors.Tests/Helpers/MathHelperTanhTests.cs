using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// MathHelper.Tanh returned NaN for every float input above ~44 (and double above ~355): it computed
/// (e^2x - 1) / (e^2x + 1), which is Inf / Inf once e^2x overflows. Downstream, a policy whose raw
/// mean exceeded that range emitted NaN actions (ooples/AiDotNet#2216).
/// </summary>
public class MathHelperTanhTests
{
    public static readonly double[] Inputs =
    {
        0.0, 1e-12, -1e-12, 1e-8, 0.5, -0.5, 1.0, -3.0, 10.0, 20.0, 44.0, 45.0, 50.0, -50.0,
        100.0, -100.0, 355.0, 356.0, 1000.0, -1000.0, 1e30, -1e30,
        double.PositiveInfinity, double.NegativeInfinity,
    };

    [Fact]
    public void Float_MatchesMathTanh_AcrossTheOverflowRange()
    {
        foreach (double x in Inputs)
        {
            float expected = (float)Math.Tanh((float)x);
            float actual = MathHelper.Tanh((float)x);
            Assert.False(float.IsNaN(actual), $"tanh({x}) was NaN");
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void Double_MatchesMathTanh_AcrossTheOverflowRange()
    {
        foreach (double x in Inputs)
        {
            double actual = MathHelper.Tanh(x);
            Assert.False(double.IsNaN(actual), $"tanh({x}) was NaN");
            Assert.Equal(Math.Tanh(x), actual);
        }
    }

    [Fact]
    public void Decimal_MatchesMathTanh_ThroughTheGenericConversion()
    {
        // decimal's range ends near 7.9e28, so the sweep stops well inside it.
        foreach (double x in new[] { 0.0, 1e-8, -0.5, 1.0, -3.0, 20.0, 50.0, -50.0, 1000.0, -1000.0 })
        {
            decimal actual = MathHelper.Tanh((decimal)x);
            Assert.Equal((decimal)Math.Tanh(x), actual);
        }
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void Half_MatchesMathTanh_ThroughTheGenericConversion()
    {
        // Half overflows above 65504; e^2x left its range from x ~ 5.5 under the old formula.
        foreach (double x in new[] { 0.0, 0.001, -0.5, 1.0, -3.0, 6.0, 10.0, 50.0, -50.0, 60000.0, -60000.0 })
        {
            Half actual = MathHelper.Tanh((Half)x);
            Assert.False(Half.IsNaN(actual), $"tanh({x}) was NaN");
            Assert.Equal((Half)Math.Tanh((double)(Half)x), actual);
        }
    }
#endif

    [Fact]
    public void NaN_StaysNaN()
    {
        Assert.True(double.IsNaN(MathHelper.Tanh(double.NaN)));
        Assert.True(float.IsNaN(MathHelper.Tanh(float.NaN)));
    }
}
