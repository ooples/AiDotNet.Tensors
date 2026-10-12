using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// ln|Γ(x)| against mpmath at 40 digits. It was computed as ln(Γ(x)), which overflowed to +∞ for x &gt; 171.6 (a
/// negative-binomial likelihood takes lgamma of 1/α, beyond 171 whenever α &lt; 0.006).
/// </summary>
public class GammaFunctionTests
{
    private static readonly (double X, double LogGamma)[] Reference =
    {
        (0.1, 2.252712651734206),
        (0.5, 0.5723649429247001),
        (1.0, 0.0),
        (1.5, -0.12078223763524522),
        (2.5, 0.2846828704729192),
        (7.25, 7.0521854507385395),
        (10.0, 12.801827480081469),
        (50.5, 146.51925549072064),
        (171.0, 706.5730622457874),
        (171.7, 710.171612940375),
        (172.0, 711.71472580229),
        (500.0, 2605.115850361734),
        (100000.0, 1051287.7089736569),
        (-0.5, 1.2655121234846454),
        (-2.3, 0.36956666345500805),
        (-10.7, -15.416789096416636),
    };

    [Fact]
    public void LogGamma_MatchesMpmath_AndStaysFiniteForLargeArguments()
    {
        var input = new Tensor<double>(new[] { Reference.Length });
        for (int i = 0; i < Reference.Length; i++) input[i] = Reference[i].X;
        var engine = new CpuEngine().TensorLgamma(input);
        for (int i = 0; i < Reference.Length; i++)
        {
            var (x, expected) = Reference[i];
            double tolerance = 1e-13 + 1e-14 * Math.Abs(expected);
            foreach (var (name, actual) in new[] { ("GammaFunction", GammaFunction.LogGamma(x)), ("MathHelper", MathHelper.LogGamma(x)), ("TensorLgamma", engine[i]) })
                Assert.True(Math.Abs(expected - actual) <= tolerance, $"{name}({x:R}): expected {expected:R}, got {actual:R}");
        }
    }
}
