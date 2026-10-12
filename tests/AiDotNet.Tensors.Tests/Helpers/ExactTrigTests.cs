using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// .NET Framework's Math.Sin/Cos do not reduce large arguments exactly (sin(1e300) came back as 1e300 there), which broke
// the large-argument Bessel and Airy forms on net471. ExactTrig has to agree with an exactly reducing libm everywhere.
public class ExactTrigTests
{
    // Reference values from the Windows UCRT (exact reduction), which CPython's math.sin/cos call.
    [Theory]
    [InlineData(1e14, -0.2094083074964523, -0.9778282879685325)]
    [InlineData(-1e14, 0.2094083074964523, -0.9778282879685325)]
    [InlineData(1e300, -0.8178819121159085, -0.5753861119575491)]
    [InlineData(-1e300, 0.8178819121159085, -0.5753861119575491)]
    [InlineData(6.7e20, 0.8167797893414578, 0.5769495434813375)]
    [InlineData(1048576.0, 0.3304931400217347, 0.943808393901312)]
    [InlineData(1e22, -0.8522008497671888, 0.523214785395139)]
    public void LargeArguments_MatchAnExactlyReducingLibm(double x, double sin, double cos)
    {
        Assert.True(Math.Abs(ExactTrig.Sin(x) - sin) <= 1e-15 * Math.Abs(sin), $"sin({x:R}) = {ExactTrig.Sin(x):R}, expected {sin:R}");
        Assert.True(Math.Abs(ExactTrig.Cos(x) - cos) <= 1e-15 * Math.Abs(cos), $"cos({x:R}) = {ExactTrig.Cos(x):R}, expected {cos:R}");
    }

    [Fact]
    public void SinSquaredPlusCosSquared_IsOne_AcrossTheRange()
    {
        var rng = new Random(7);
        for (int i = 0; i < 2000; i++)
        {
            double x = Math.Pow(10, 6 + rng.NextDouble() * 302) * (rng.Next(2) == 0 ? 1 : -1);
            double s = ExactTrig.Sin(x), c = ExactTrig.Cos(x);
            Assert.True(Math.Abs(s * s + c * c - 1) < 1e-15, $"x = {x:R}: sin² + cos² = {s * s + c * c:R}");
        }
    }

    [Fact]
    public void SmallAndNonFiniteArguments_FollowMath()
    {
        foreach (var x in new[] { 0.0, 1.0, -3.5, 1000.0, 1048575.0 })
        {
            Assert.Equal(Math.Sin(x), ExactTrig.Sin(x));
            Assert.Equal(Math.Cos(x), ExactTrig.Cos(x));
        }
        Assert.True(double.IsNaN(ExactTrig.Sin(double.NaN)));
        Assert.True(double.IsNaN(ExactTrig.Cos(double.PositiveInfinity)));
    }
}
