using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The special functions at their edges: ±∞, NaN, ±0, subnormal-adjacent and huge arguments. Expected values are
/// mpmath (330 digits, enough to reduce 1e300 exactly) wherever the input and PyTorch's float64 result are finite,
/// and PyTorch float64 otherwise, except at ±∞ for Ai, J₀, J₁, Y₀, Y₁, I₀, I₁ and sinc: PyTorch returns NaN there,
/// but each has a well-defined limit (0, or ±∞ for I), which is what the engine returns.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class TorchSpecialEdgeCaseTests
{
    private static readonly double[] Inputs = { double.NegativeInfinity, double.PositiveInfinity, double.NaN, 0.0, -0.0, 1e-300, -1e-300, 0.75, -3.5, 60.0, 5000.0, 10000.0, 1000000.0, 100000000000000.0, -100000000000000.0, 1e+300, -1e+300 };

    private static readonly Dictionary<string, double[]> Expected = new Dictionary<string, double[]>
    {
        ["TensorErf"] = new[] { -1.0, 1.0, double.NaN, 0.0, 0.0, 1.1283791670955126e-300, -1.1283791670955126e-300, 0.7111556336535151, -0.9999992569016276, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0 },
        ["TensorErfcx"] = new[] { double.PositiveInfinity, 0.0, double.NaN, 1.0, 1.0, 1.0, 1.0, 0.5069376502931449, 417962.4224457703, 0.009401854275176388, 0.00011283791445279306, 5.641895807268084e-05, 5.641895835474742e-07, 5.6418958354775626e-15, double.PositiveInfinity, 5.641895835477562e-301, double.PositiveInfinity },
        ["TensorNdtr"] = new[] { 0.0, 1.0, double.NaN, 0.5, 0.5, 0.5, 0.5, 0.7733726476231318, 0.00023262907903552504, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0 },
        ["TensorLogNdtr"] = new[] { double.NegativeInfinity, -0.0, double.NaN, -0.6931471805599453, -0.6931471805599453, -0.6931471805599453, -0.6931471805599453, -0.25699426683836524, -8.366065308344092, 0.0, 0.0, 0.0, 0.0, 0.0, -5e+27, 0.0, double.NegativeInfinity },
        ["TensorEntr"] = new[] { double.NegativeInfinity, double.NegativeInfinity, double.NaN, 0.0, 0.0, 6.9077552789821376e-298, double.NegativeInfinity, 0.2157615543388357, double.NegativeInfinity, -245.66067373332604, -42585.965957081185, -92103.40371976183, -13815510.557964275, -3223619130191664.0, double.NegativeInfinity, -6.907755278982137e+302, double.NegativeInfinity },
        ["TensorSinc"] = new[] { 0.0, 0.0, double.NaN, 1.0, 1.0, 1.0, 1.0, 0.30010543871903533, -0.09094568176679733, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        ["TensorBesselJ0"] = new[] { 0.0, 0.0, double.NaN, 1.0, 1.0, 1.0, 1.0, 0.8642422751666486, -0.3801277399872634, -0.09147180408906187, -0.0066489842514483475, -0.0070961603533888015, 0.00033104301373987376, -6.698265203680453e-08, -6.698265203680453e-08, -7.860673062724093e-151, -7.860673062724093e-151 },
        ["TensorBesselJ1"] = new[] { 0.0, 0.0, double.NaN, 0.0, 0.0, 5e-301, -5e-301, 0.34924360217486217, -0.1373775273623272, 0.046598383758166315, -0.00911740571364616, 0.0036474507555295803, -0.000725968356813763, 4.335345487723154e-08, -4.335345487723154e-08, -1.3681360450342481e-151, 1.3681360450342481e-151 },
        ["TensorBesselY0"] = new[] { double.NaN, 0.0, double.NaN, double.NegativeInfinity, double.NegativeInfinity, -439.8351636227653, double.NaN, -0.1371727693857724, double.NaN, 0.0473589522094494, -0.009116740769643963, 0.0036478055589866058, -0.0007259685223351791, 4.335345487723188e-08, double.NaN, -1.3681360450342481e-151, double.NaN },
        ["TensorBesselY1"] = new[] { double.NaN, 0.0, double.NaN, double.NegativeInfinity, double.NegativeInfinity, -6.366197723675813e+299, double.NaN, -1.0375945507692854, double.NaN, 0.09186960936986689, 0.00664807261062542, 0.007096342752536495, -0.00033104337672417626, 6.698265203680474e-08, double.NaN, 7.860673062724093e-151, double.NaN },
        ["TensorModifiedBesselI0"] = new[] { double.PositiveInfinity, double.PositiveInfinity, double.NaN, 1.0, 1.0, 1.0, 1.0, 1.1456467780440014, 7.3782034322254795, 5.894077055609801e+24, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity },
        ["TensorModifiedBesselI1"] = new[] { double.NegativeInfinity, double.PositiveInfinity, double.NaN, 0.0, 0.0, 5e-301, -5e-301, 0.4019924615809222, -6.205834922258365, 5.844751588390468e+24, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity },
        ["TensorModifiedBesselK0"] = new[] { double.NaN, 0.0, double.NaN, double.PositiveInfinity, double.PositiveInfinity, 690.8914594138721, double.NaN, 0.6105824221164641, double.NaN, 1.4138978405591078e-27, 0.0, 0.0, 0.0, 0.0, double.NaN, 0.0, double.NaN },
        ["TensorModifiedBesselK1"] = new[] { double.NaN, 0.0, double.NaN, double.PositiveInfinity, double.PositiveInfinity, 9.999999999999999e+299, double.NaN, 0.9495804669621403, double.NaN, 1.4256320265171043e-27, 0.0, 0.0, 0.0, 0.0, double.NaN, 0.0, double.NaN },
        ["TensorScaledModifiedBesselK0"] = new[] { double.NaN, 0.0, double.NaN, double.PositiveInfinity, double.PositiveInfinity, 690.8914594138721, double.NaN, 1.2926029977639617, double.NaN, 0.16146817823629392, 0.017724095445432315, 0.012532984717699286, 0.0012533139806513213, 1.2533141373154987e-07, double.NaN, 1.2533141373155002e-150, double.NaN },
        ["TensorScaledModifiedBesselK1"] = new[] { double.NaN, 0.0, double.NaN, double.PositiveInfinity, double.PositiveInfinity, 9.999999999999999e+299, double.NaN, 2.010261864333922, double.NaN, 0.16280823094404426, 0.017725867766374102, 0.012533611351270506, 0.0012533146073081549, 1.253314137315505e-07, double.NaN, 1.2533141373155002e-150, double.NaN },
        ["TensorSphericalBesselJ0"] = new[] { 0.0, 0.0, double.NaN, 1.0, 1.0, 1.0, 1.0, 0.9088516800311123, -0.10022377933989139, -0.005080177018370278, -0.00019759328775335538, -3.056143888882521e-05, -3.4999350217129296e-07, -2.094083074964523e-15, -2.094083074964523e-15, -8.178819121159085e-301, -8.178819121159085e-301 },
        ["TensorAiryAi"] = new[] { 0.0, 0.0, double.NaN, 0.3550280538878172, 0.3550280538878172, 0.3550280538878172, 0.3550280538878172, 0.17933630547864524, -0.37553382314043193, 2.7831487094969354e-136, 0.0, 0.0, 0.0, 0.0, -0.00017391172622874227, 0.0, double.NaN }
    };

    [Theory]
    [InlineData("TensorErf")]
    [InlineData("TensorErfcx")]
    [InlineData("TensorNdtr")]
    [InlineData("TensorLogNdtr")]
    [InlineData("TensorEntr")]
    [InlineData("TensorSinc")]
    [InlineData("TensorBesselJ0")]
    [InlineData("TensorBesselJ1")]
    [InlineData("TensorBesselY0")]
    [InlineData("TensorBesselY1")]
    [InlineData("TensorModifiedBesselI0")]
    [InlineData("TensorModifiedBesselI1")]
    [InlineData("TensorModifiedBesselK0")]
    [InlineData("TensorModifiedBesselK1")]
    [InlineData("TensorScaledModifiedBesselK0")]
    [InlineData("TensorScaledModifiedBesselK1")]
    [InlineData("TensorSphericalBesselJ0")]
    [InlineData("TensorAiryAi")]
    public void MatchesReferenceAtTheEdges(string op)
    {
        var engine = new CpuEngine();
        var method = typeof(CpuEngine).GetMethods().Single(m => m.Name == op && m.GetParameters().Length == 1).MakeGenericMethod(typeof(double));
        var actual = (Tensor<double>)(method.Invoke(engine, new object[] { new Tensor<double>(Inputs, new[] { Inputs.Length }) })
            ?? throw new InvalidOperationException(op + " returned null"));
        var expected = Expected[op];
        var failures = new List<string>();
        for (int i = 0; i < Inputs.Length; i++)
        {
            double e = expected[i], a = actual.GetFlat(i);
            const double tolerance = 1e-12;
            bool ok = double.IsNaN(e) ? double.IsNaN(a)
                : double.IsInfinity(e) ? a == e
                // An absolute allowance only for an exact-zero reference: for a tiny nonzero one (J1(1e-300) = 5e-301)
                // it would accept a wrong 0.
                : e == 0 ? Math.Abs(a) <= 1e-300
                : Math.Abs(a - e) <= tolerance * Math.Abs(e);
            if (!ok) failures.Add($"x={Inputs[i]:R}: expected {e:R}, got {a:R}");
        }
        Assert.True(failures.Count == 0, op + ": " + string.Join("; ", failures));
    }
}
