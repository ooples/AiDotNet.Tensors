using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>The double-precision special functions behind the torch.special parity ops, against reference values.</summary>
public class SpecialFunctionsTests
{
    private static void Close(double expected, double actual, double relative = 1e-10)
        => Assert.True(Math.Abs(expected - actual) <= relative * Math.Max(Math.Abs(expected), 1e-300),
            $"expected {expected:R}, got {actual:R}");

    [Theory]
    [InlineData(0.5, 0.52049987781304654)]
    [InlineData(1.0, 0.84270079294971487)]
    [InlineData(-2.0, -0.99532226501895273)]
    [InlineData(3.0, 0.99997790950300141)]
    public void Erf(double x, double expected) => Close(expected, SpecialFunctions.Erf(x));

    [Theory]
    [InlineData(1.0, 0.42758357615580705)]
    [InlineData(5.0, 0.11070463773306863)]
    [InlineData(-1.0, 5.008980080762283)]
    public void Erfcx(double x, double expected) => Close(expected, SpecialFunctions.Erfcx(x));

    [Theory]
    [InlineData(0.0, 0.5)]
    [InlineData(1.0, 0.84134474606854293)]
    [InlineData(-3.0, 0.0013498980316300946)]
    public void Ndtr(double x, double expected) => Close(expected, SpecialFunctions.Ndtr(x));

    [Theory]
    [InlineData(-10.0, -53.231285150512466)]
    [InlineData(-1.0, -1.8410216450092636)]
    [InlineData(2.0, -0.023012909328963493)]
    public void LogNdtr(double x, double expected) => Close(expected, SpecialFunctions.LogNdtr(x));

    [Theory]
    [InlineData(0.975, 1.959963984540054)]
    [InlineData(0.5, 0.0)]
    [InlineData(1e-10, -6.3613409024040557)]
    public void Ndtri(double p, double expected)
        => Assert.True(Math.Abs(expected - SpecialFunctions.Ndtri(p)) <= 1e-12 * Math.Max(1, Math.Abs(expected)));

    [Theory]
    [InlineData(0.5, 0.57236494292470008)]
    [InlineData(10.0, 12.801827480081469)]
    [InlineData(-0.5, 1.2655121234846454)]
    public void LogGamma(double x, double expected) => Close(expected, SpecialFunctions.LogGamma(x), 1e-12);

    [Theory]
    [InlineData(1.0, -0.57721566490153287)]
    [InlineData(0.5, -1.9635100260214235)]
    [InlineData(10.0, 2.2517525890667211)]
    public void Digamma(double x, double expected) => Close(expected, SpecialFunctions.Digamma(x), 1e-12);

    [Theory]
    [InlineData(2.0, 1.0, 0.26424111765711533)]
    [InlineData(0.5, 2.0, 0.95449973610364158)]
    [InlineData(10.0, 3.0, 0.0011024881301155)]
    public void Igamma(double a, double x, double expected)
    {
        Close(expected, SpecialFunctions.Igamma(a, x), 1e-9);
        Close(1 - expected, SpecialFunctions.Igammac(a, x), 1e-9);
    }

    [Theory]
    [InlineData(0, 1.0, 0.76519768655796655)]
    [InlineData(1, 1.0, 0.44005058574493352)]
    [InlineData(0, 10.0, -0.24593576445134834)]
    [InlineData(1, 10.0, 0.043472746168861438)]
    [InlineData(0, 40.0, 0.0073668905842372)]
    public void BesselJ(int n, double x, double expected) => Close(expected, SpecialFunctions.BesselJ(n, x), 1e-9);

    [Theory]
    [InlineData(0, 1.0, 0.088256964215676956)]
    [InlineData(1, 1.0, -0.78121282130028868)]
    [InlineData(0, 10.0, 0.055671167283599395)]
    [InlineData(1, 10.0, 0.24901542420695388)]
    [InlineData(0, 0.01, -3.005455637083646)]
    public void BesselY(int n, double x, double expected) => Close(expected, SpecialFunctions.BesselY(n, x), 1e-9);

    [Theory]
    [InlineData(0, 1.0, 1.2660658777520084)]
    [InlineData(1, 1.0, 0.56515910399248503)]
    [InlineData(0, 10.0, 2815.7166284662544)]
    [InlineData(1, 10.0, 2670.9883037012547)]
    public void BesselI(int n, double x, double expected) => Close(expected, SpecialFunctions.BesselI(n, x), 1e-11);

    [Theory]
    [InlineData(0, 1.0, 0.42102443824070834)]
    [InlineData(1, 1.0, 0.60190723019723457)]
    [InlineData(0, 10.0, 1.7780062316167652e-05)]
    [InlineData(1, 10.0, 1.8648773453825585e-05)]
    [InlineData(0, 0.001, 7.0236888005623825)]
    public void BesselK(int n, double x, double expected) => Close(expected, SpecialFunctions.BesselK(n, x), 1e-10);

    [Theory]
    [InlineData(0.0, 0.35502805388781724)]
    [InlineData(1.0, 0.13529241631288141)]
    [InlineData(-1.0, 0.53556088329235211)]
    [InlineData(3.0, 0.0065911393574607191)]
    [InlineData(-10.0, 0.040241238486441955)]
    [InlineData(-20.0, -0.17640612707798468)]
    public void AiryAi(double x, double expected) => Close(expected, SpecialFunctions.AiryAi(x), 1e-8);

    [Fact]
    public void Polynomials_MatchClosedForms()
    {
        double x = 0.3;
        Close(Math.Cos(5 * Math.Acos(x)), SpecialFunctions.Polynomial(SpecialFunctions.PolynomialKind.ChebyshevT, x, 5), 1e-13);
        Close(Math.Sin(6 * Math.Acos(x)) / Math.Sin(Math.Acos(x)), SpecialFunctions.Polynomial(SpecialFunctions.PolynomialKind.ChebyshevU, x, 5), 1e-13);
        Close(8 * x * x * x - 12 * x, SpecialFunctions.Polynomial(SpecialFunctions.PolynomialKind.HermiteH, x, 3), 1e-13);
        Close(x * x * x - 3 * x, SpecialFunctions.Polynomial(SpecialFunctions.PolynomialKind.HermiteHe, x, 3), 1e-13);
        Close((x * x - 4 * x + 2) / 2, SpecialFunctions.Polynomial(SpecialFunctions.PolynomialKind.LaguerreL, x, 2), 1e-13);
        Close((3 * x * x - 1) / 2, SpecialFunctions.Polynomial(SpecialFunctions.PolynomialKind.LegendreP, x, 2), 1e-13);
        Assert.Equal(0.0, SpecialFunctions.Polynomial(SpecialFunctions.PolynomialKind.LegendreP, x, -1));
    }
}
