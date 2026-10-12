using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// erf and erfc against mpmath at 40 digits, rounded to double (centre, all three of Cody's regions and their
/// boundaries, the tail, and negative arguments). The previous Abramowitz-Stegun 7.1.26 approximation was off by up to
/// 1.5e-7 absolute, and its erfc = 1 - erf lost all relative precision in the tail.
/// </summary>
public class ErrorFunctionTests
{
    private static readonly (double X, double Erf, double Erfc)[] Reference =
    {
        (0.0, 0.0, 1.0),
        (1e-20, 1.1283791670955125e-20, 1.0),
        (1e-10, 1.1283791670955126e-10, 0.999999999887162),
        (1e-05, 1.1283791670579e-05, 0.9999887162083294),
        (0.01, 0.011283415555849618, 0.9887165844441503),
        (0.1, 0.1124629160182849, 0.887537083981715),
        (0.25, 0.27632639016823696, 0.7236736098317631),
        (0.4, 0.42839235504666845, 0.5716076449533315),
        (0.46875, 0.49261347321793797, 0.507386526782062),
        (0.46876, 0.4926225311068465, 0.5073774688931535),
        (0.5, 0.5204998778130465, 0.4795001221869535),
        (0.75, 0.7111556336535151, 0.28884436634648486),
        (1.0, 0.8427007929497149, 0.15729920705028513),
        (1.5, 0.9661051464753108, 0.033894853524689274),
        (2.0, 0.9953222650189527, 0.004677734981047266),
        (2.5, 0.999593047982555, 0.0004069520174449589),
        (3.0, 0.9999779095030014, 2.209049699858544e-05),
        (3.5, 0.9999992569016276, 7.430983723414128e-07),
        (3.9999, 0.9999999845700388, 1.5429961215558176e-08),
        (4.0, 0.9999999845827421, 1.541725790028002e-08),
        (4.0001, 0.9999999845954353, 1.5404564743590145e-08),
        (4.5, 0.9999999998033839, 1.9661604415428876e-10),
        (5.0, 0.9999999999984626, 1.537459794428035e-12),
        (6.0, 1.0, 2.1519736712498913e-17),
        (8.0, 1.0, 1.1224297172982926e-29),
        (10.0, 1.0, 2.088487583762545e-45),
        (15.0, 1.0, 7.212994172451207e-100),
        (20.0, 1.0, 5.395865611607901e-176),
        (26.0, 1.0, 5.663192408856143e-296),
        (-1e-20, -1.1283791670955125e-20, 1.0),
        (-1e-10, -1.1283791670955126e-10, 1.000000000112838),
        (-1e-05, -1.1283791670579e-05, 1.0000112837916706),
        (-0.01, -0.011283415555849618, 1.0112834155558497),
        (-0.1, -0.1124629160182849, 1.1124629160182848),
        (-0.25, -0.27632639016823696, 1.276326390168237),
        (-0.4, -0.42839235504666845, 1.4283923550466684),
        (-0.46875, -0.49261347321793797, 1.492613473217938),
        (-0.46876, -0.4926225311068465, 1.4926225311068466),
        (-0.5, -0.5204998778130465, 1.5204998778130465),
        (-0.75, -0.7111556336535151, 1.7111556336535152),
        (-1.0, -0.8427007929497149, 1.8427007929497148),
        (-1.5, -0.9661051464753108, 1.9661051464753108),
        (-2.0, -0.9953222650189527, 1.9953222650189528),
        (-2.5, -0.999593047982555, 1.999593047982555),
        (-3.0, -0.9999779095030014, 1.9999779095030015),
        (-3.5, -0.9999992569016276, 1.9999992569016276),
        (-3.9999, -0.9999999845700388, 1.9999999845700387),
        (-4.0, -0.9999999845827421, 1.999999984582742),
        (-4.0001, -0.9999999845954353, 1.9999999845954353),
        (-4.5, -0.9999999998033839, 1.999999999803384),
        (-5.0, -0.9999999999984626, 1.9999999999984626),
        (-6.0, -1.0, 2.0),
        (-8.0, -1.0, 2.0),
        (-10.0, -1.0, 2.0),
        (-15.0, -1.0, 2.0),
        (-20.0, -1.0, 2.0),
        (-26.0, -1.0, 2.0),
    };

    private static void AssertRelative(double expected, double actual, double tolerance, string what)
    {
        double scale = Math.Max(Math.Abs(expected), double.Epsilon);
        Assert.True(Math.Abs(expected - actual) <= tolerance * scale, $"{what}: expected {expected:R}, got {actual:R}");
    }

    [Fact]
    public void Erf_MatchesMpmath_ToDoublePrecision()
    {
        foreach (var (x, erf, _) in Reference) AssertRelative(erf, ErrorFunction.Erf(x), 4e-16, $"erf({x:R})");
    }

    [Fact]
    public void Erfc_MatchesMpmath_InRelativeTerms_IncludingTheTail()
    {
        foreach (var (x, _, erfc) in Reference) AssertRelative(erfc, ErrorFunction.Erfc(x), 2e-15, $"erfc({x:R})");
    }

    [Fact]
    public void MathHelper_AndTheEngineOp_UseTheAccurateFunctions()
    {
        var input = new Tensor<double>(new[] { Reference.Length });
        for (int i = 0; i < Reference.Length; i++) input[i] = Reference[i].X;
        var erfc = new CpuEngine().TensorErfc(input);
        for (int i = 0; i < Reference.Length; i++)
        {
            var (x, erf, expected) = Reference[i];
            AssertRelative(erf, MathHelper.Erf(x), 4e-16, $"MathHelper.Erf({x:R})");
            AssertRelative(expected, MathHelper.Erfc(x), 2e-15, $"MathHelper.Erfc({x:R})");
            AssertRelative(expected, erfc[i], 2e-15, $"TensorErfc({x:R})");
        }
    }
}
