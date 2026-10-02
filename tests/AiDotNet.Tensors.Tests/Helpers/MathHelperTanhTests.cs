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
    public void NaN_StaysNaN()
    {
        Assert.True(double.IsNaN(MathHelper.Tanh(double.NaN)));
        Assert.True(float.IsNaN(MathHelper.Tanh(float.NaN)));
    }
}
