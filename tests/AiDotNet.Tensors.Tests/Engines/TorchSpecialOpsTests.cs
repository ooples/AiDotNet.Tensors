using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The torch.* / torch.special element-wise parity ops against arbitrary-precision references (mpmath; PyTorch's own float64 Bessel/Airy
/// are off by up to 1e-7), the polynomials against PyTorch float64, their gradients inside each op's domain against finite differences, and the integer, comparison and NaN
/// reduction ops.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class TorchSpecialOpsTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    private static readonly double[] X = { -2.5, -1.0, -0.3, 0.0, 0.4, 1.0, 2.0, 6.5 };
    private static readonly double[] P = { 0.01, 0.2, 0.5, 0.7, 0.99 };
    private static readonly double[] Pos = { 0.05, 0.5, 1.0, 3.0, 12.0, 30.0 };

    private static Tensor<double> T(double[] values) => new Tensor<double>((double[])values.Clone(), new[] { values.Length });

    // Calls the engine op named op with double tensors (the ops under test share one shape: tensors in, tensor out).
    private Tensor<double> Invoke(string op, params Tensor<double>[] args)
    {
        var method = typeof(CpuEngine).GetMethods()
            .Single(m => m.Name == op && m.GetParameters().Count(p => !p.IsOptional) == args.Length)
            .MakeGenericMethod(typeof(double));
        var parameters = method.GetParameters();
        var call = parameters.Select((p, i) => i < args.Length ? args[i] : p.DefaultValue).ToArray();
        return method.Invoke(_engine, call) as Tensor<double>
            ?? throw new InvalidOperationException($"{op} returned no tensor.");
    }

    private static void Close(double[] expected, Tensor<double> actual, string what, double tolerance = 1e-8)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i], a = actual.GetFlat(i);
            if (double.IsNaN(e)) { Assert.True(double.IsNaN(a), $"{what}[{i}]: expected NaN, got {a}"); continue; }
            if (double.IsInfinity(e)) { Assert.True(e == a, $"{what}[{i}]: expected {e}, got {a}"); continue; }
            Assert.True(Math.Abs(e - a) <= tolerance * Math.Max(1, Math.Abs(e)), $"{what}[{i}]: torch {e:R}, ours {a:R}");
        }
    }

    [Theory]
    [InlineData("TensorErf", new[] { -0.999593047982555, -0.8427007929497149, -0.3286267594591274, 0.0, 0.42839235504666845, 0.8427007929497149, 0.9953222650189527, 1.0 })]
    [InlineData("TensorSinc", new[] { 0.12732395447351627, 0.0, 0.8583936913341398, 1.0, 0.7568267286406569, 0.0, 0.0, 0.04897075172058318 })]
    [InlineData("TensorErfcx", new[] { 1035.814842972623, 5.008980080762283, 1.4537492328427655, 1.0, 0.6707877852947615, 0.427583576155807, 0.25539567631050575, 0.08580567010489461 })]
    [InlineData("TensorNdtr", new[] { 0.006209665325776135, 0.15865525393145705, 0.3820885778110474, 0.5, 0.6554217416103242, 0.8413447460685429, 0.9772498680518208, 0.99999999995984 })]
    [InlineData("TensorLogNdtr", new[] { -5.08164827727869, -1.8410216450092636, -0.9621028181688507, -0.6931471805599453, -0.42247637022777607, -0.17275377902344988, -0.02301290932896349, -4.016000583939759e-11 })]
    [InlineData("TensorBesselJ0", new[] { -0.048383776468198, 0.7651976865579666, 0.9776262465382961, 1.0, 0.9603982266595634, 0.7651976865579666, 0.22389077914123567, 0.2600946055816064 })]
    [InlineData("TensorBesselJ1", new[] { -0.49709410246427405, -0.4400505857449335, -0.148318816273104, 0.0, 0.19602657795531875, 0.4400505857449335, 0.5767248077568734, -0.15384130140997185 })]
    [InlineData("TensorModifiedBesselI0", new[] { 3.289839144050123, 1.2660658777520084, 1.022626879351597, 1.0, 1.0404017822293412, 1.2660658777520084, 2.2795853023360673, 106.2928582439956 })]
    [InlineData("TensorModifiedBesselI1", new[] { -2.5167162452886984, -0.565159103992485, -0.15169384000359276, 0.0, 0.2040267557335706, 0.565159103992485, 1.590636854637329, 97.73501077403152 })]
    [InlineData("TensorAiryAi", new[] { -0.11232506769296609, 0.5355608832923521, 0.43090309528558085, 0.3550280538878172, 0.2547423542956763, 0.13529241631288141, 0.03492413042327438, 2.7958823432049136e-06 })]
    [InlineData("TensorSphericalBesselJ0", new[] { 0.2393888576415826, 0.8414709848078965, 0.9850673555377986, 1.0, 0.9735458557716262, 0.8414709848078965, 0.45464871341284085, 0.03309538278274085 })]
    public void UnaryOnX_MatchesTorch(string op, double[] expected)
    {
        Close(expected, Invoke(op, T(X)), op);
    }

    [Theory]
    [InlineData("TensorBesselY0", new[] { -1.9793110008172097, -0.44451873350670656, 0.08825696421567696, 0.3768500100127904, -0.22523731263436145, -0.11729573168666403 })]
    [InlineData("TensorBesselY1", new[] { -12.78985517117497, -1.471472392670243, -0.7812128213002887, 0.3246744247918, -0.05709921826089652, 0.08442557066174723 })]
    [InlineData("TensorModifiedBesselK0", new[] { 3.11423402947199, 0.9244190712276659, 0.42102443824070834, 0.03473950438627925, 2.2008253973114916e-06, 2.1324774964630563e-14 })]
    [InlineData("TensorModifiedBesselK1", new[] { 19.909674325882506, 1.656441120003301, 0.6019072301972346, 0.040156431128194184, 2.290757464767188e-06, 2.1677320018915495e-14 })]
    [InlineData("TensorScaledModifiedBesselK0", new[] { 3.273904222534542, 1.5241093857739094, 1.144463079806895, 0.6977615980438517, 0.3581948784890782, 0.22788666561625373 })]
    [InlineData("TensorScaledModifiedBesselK1", new[] { 20.93046515706008, 2.731009708211786, 1.6361534862632583, 0.8065634801287869, 0.37283175336970986, 0.2316541293777118 })]
    [InlineData("TensorEntr", new[] { 0.14978661367769955, 0.34657359027997264, 0.0, -3.295836866004329, -29.818879797456002, -102.03592144986466 })]
    public void UnaryOnPos_MatchesTorch(string op, double[] expected)
    {
        Close(expected, Invoke(op, T(Pos)), op);
    }

    [Fact]
    public void Logit_Ndtri_Mvlgamma_Igamma_MatchTorch()
    {
        Close(new[] { -4.59511985013459, -1.3862943611198906, 0.0, 0.8472978603872034, 4.595119850134589 }, _engine.TensorLogit(T(P)), "logit");
        Close(new[] { -2.326347874040841, -0.8416212335729142, 0.0, 0.5244005127080407, 2.3263478740408408 }, _engine.TensorNdtri(T(P)), "ndtri");
        Close(new[] { 2.052466467738472, 1.596312591138855, 1.8809954616117741, 7.1635644711916715, 61.69873299261713, 225.70000022955645 }, _engine.TensorMvlgamma(T(Pos.Select(v => v + 1.5).ToArray()), 3), "mvlgamma");
        var a = T(new[] { 0.5, 1.0, 2.0, 4.0, 10.0, 3.0 });
        Close(new[] { 0.24817036595415073, 0.3934693402873666, 0.26424111765711533, 0.35276811121776874, 0.7576078383294876, 0.9999999999549898 }, _engine.TensorIgamma(a, T(Pos)), "igamma");
        Close(new[] { 0.7518296340458492, 0.6065306597126334, 0.7357588823428847, 0.6472318887822313, 0.24239216167051233, 4.501016648012124e-11 }, _engine.TensorIgammac(a, T(Pos)), "igammac");
    }

    [Theory]
    [InlineData("TensorChebyshevPolynomialT", new[] { 1.0, -0.4, -1.0, -0.792, -0.07583999999999991, -0.6068780500000006, 44.696799999999996, 2589.8481280000015 })]
    [InlineData("TensorChebyshevPolynomialU", new[] { 1.0, -0.8, -1.0, -0.984, -0.8236799999999999, 1.8112338999999964, 99.95359999999998, 5497.425856000002 })]
    [InlineData("TensorChebyshevPolynomialV", new[] { 1.0, -1.8, -1.0, -0.344, 0.4227200000000001, -0.7341471000000006, 67.4496, 6819.052096000005 })]
    [InlineData("TensorChebyshevPolynomialW", new[] { 1.0, 0.19999999999999996, -1.0, -1.624, -2.07008, 4.35661489999999, 132.45759999999999, 4175.799616000003 })]
    [InlineData("TensorShiftedChebyshevPolynomialT", new[] { 1.0, -1.8, 1.0, 0.944, 0.8451199999999999, -0.9998784, 220.34079999999997, 753146.3726720003 })]
    [InlineData("TensorHermitePolynomialH", new[] { 1.0, -0.8, -2.0, -3.384, 39.92832, 334.20559389999994, 6.913599999999995, -623.1741439999996 })]
    [InlineData("TensorHermitePolynomialHe", new[] { 1.0, -0.4, -1.0, -0.873, 6.91776, -25.276687391406256, -5.987900000000001, -35.20409600000002 })]
    [InlineData("TensorLaguerrePolynomialL", new[] { 1.0, 1.4, 1.0, 0.2305, -0.5336480000000001, -0.11568604841874398, -0.05732916666666654, 103.3686214222222 })]
    [InlineData("TensorLegendrePolynomialP", new[] { 1.0, -0.4, -0.5, -0.3825, -0.15263999999999994, 0.011227208544920955, 26.0779375, 1207.1808640000002 })]
    public void Polynomials_MatchTorch(string op, double[] expected)
    {
        var x = T(new[] { -0.9, -0.4, 0.0, 0.3, 0.6, 0.95, 1.7, -2.2 });
        var n = T(new double[] { 0, 1, 2, 3, 5, 7, 4, 6 });
        Close(expected, Invoke(op, x, n), op, 1e-11);
    }

    [Fact]
    public void IntegerOps_MatchTorch()
    {
        var a = new Tensor<int>(new[] { 12, -18, 7, 0, 255 }, new[] { 5 });
        var b = new Tensor<int>(new[] { 8, 12, -3, 5, 15 }, new[] { 5 });
        Assert.Equal(new[] { 4, 6, 1, 5, 15 }, _engine.TensorGcd(a, b).ToArray());
        Assert.Equal(new[] { 24, 36, 21, 0, 255 }, _engine.TensorLcm(a, b).ToArray());
        Assert.Equal(new[] { 12 & 8, -18 & 12, 7 & -3, 0 & 5, 255 & 15 }, _engine.TensorBitwiseAnd(a, b).ToArray());
        Assert.Equal(new[] { 12 | 8, -18 | 12, 7 | -3, 0 | 5, 255 | 15 }, _engine.TensorBitwiseOr(a, b).ToArray());
        Assert.Equal(new[] { 12 ^ 8, -18 ^ 12, 7 ^ -3, 0 ^ 5, 255 ^ 15 }, _engine.TensorBitwiseXor(a, b).ToArray());
        Assert.Equal(new[] { ~12, ~-18, ~7, ~0, ~255 }, _engine.TensorBitwiseNot(a).ToArray());
        var shift = new Tensor<int>(new[] { 1, 2, 3, 0, 4 }, new[] { 5 });
        Assert.Equal(new[] { 24, -72, 56, 0, 4080 }, _engine.TensorBitwiseLeftShift(a, shift).ToArray());
        Assert.Equal(new[] { 6, -5, 0, 0, 15 }, _engine.TensorBitwiseRightShift(a, shift).ToArray());
        // floor division rounds toward -inf, as torch.floor_divide does
        Assert.Equal(new[] { 1, -2, -3, 0, 17 }, _engine.TensorFloorDivide(a, b).ToArray());
        Assert.Equal(new byte[] { 0xF0 }, _engine.TensorBitwiseNot(new Tensor<byte>(new byte[] { 0x0F }, new[] { 1 })).ToArray());
        Assert.Throws<NotSupportedException>(() => _engine.TensorGcd(T(new[] { 1.0 }), T(new[] { 2.0 })));
    }

    [Fact]
    public void ComparisonsAndIndicators_MatchTorch()
    {
        var a = T(new[] { 1.0, 2.0, 3.0, double.NaN, -0.0, double.PositiveInfinity, double.NegativeInfinity });
        var b = T(new[] { 2.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0 });
        Assert.Equal(new[] { 0.0, 1, 1, 0, 1, 1, 0 }, _engine.TensorGreaterEqual(a, b).ToArray());
        Assert.Equal(new[] { 1.0, 1, 0, 0, 1, 0, 1 }, _engine.TensorLessEqual(a, b).ToArray());
        // .NET's double.NaN carries the sign bit (0xFFF8...), unlike Python's float('nan'): signbit reads the bit.
        Assert.Equal(new[] { 0.0, 0, 0, 1, 1, 0, 1 }, _engine.TensorSignbit(a).ToArray());
        Assert.Equal(new[] { 0.0, 0, 0, 0, 0, 1, 0 }, _engine.TensorIsPosInf(a).ToArray());
        Assert.Equal(new[] { 0.0, 0, 0, 0, 0, 0, 1 }, _engine.TensorIsNegInf(a).ToArray());
        var fmax = _engine.TensorFmax(a, b).ToArray();
        Assert.Equal(new[] { 2.0, 2, 3, 1, 0, double.PositiveInfinity, 1 }, fmax);
        var step = _engine.TensorHeaviside(T(new[] { -1.5, 0.0, 2.0 }), T(new[] { 0.5, 0.5, 0.5 })).ToArray();
        Assert.Equal(new[] { 0.0, 0.5, 1.0 }, step);
        Assert.Equal(new[] { 0.0, 0.0, 0.0 }, _engine.TensorDeg2Rad(T(new[] { 0.0, 0.0, 0.0 })).ToArray());
        Assert.Equal(Math.PI, _engine.TensorDeg2Rad(T(new[] { 180.0 })).GetFlat(0), 12);
    }

    [Fact]
    public void NanSumAndNanMean_SkipNaN_AndRouteGradientsAroundIt()
    {
        var x = new Tensor<double>(new[] { 1.0, double.NaN, 3.0, 4.0, double.NaN, 6.0 }, new[] { 2, 3 });
        Assert.Equal(new[] { 4.0, 10.0 }, _engine.TensorNanSum(x, new[] { 1 }).ToArray());
        Assert.Equal(new[] { 2.0, 5.0 }, _engine.TensorNanMean(x, new[] { 1 }).ToArray());
        Assert.Equal(14.0, _engine.TensorNanSum(x).GetFlat(0));

        using var tape = new GradientTape<double>();
        var loss = _engine.ReduceSum(_engine.TensorNanMean(x, new[] { 1 }), null, false);
        var g = tape.ComputeGradients(loss, new[] { x })[x].ToArray();
        Assert.Equal(new[] { 0.5, 0.0, 0.5, 0.5, 0.0, 0.5 }, g.Select(v => Math.Round(v, 12)).ToArray());
    }

    [Theory]
    [InlineData("TensorLogit", 0.2, 0.8)]
    [InlineData("TensorNdtri", 0.05, 0.95)]
    [InlineData("TensorBesselY1", 0.5, 8.0)]
    [InlineData("TensorModifiedBesselK1", 0.3, 6.0)]
    [InlineData("TensorScaledModifiedBesselK1", 0.3, 6.0)]
    [InlineData("TensorAiryAi", -6.0, 4.0)]
    [InlineData("TensorLogNdtr", -12.0, 3.0)]
    [InlineData("TensorErfcx", -2.0, 9.0)]
    public void Gradients_MatchFiniteDifferences_InsideTheDomain(string op, double lo, double hi)
    {
        var values = Enumerable.Range(0, 7).Select(i => lo + (hi - lo) * (i + 0.5) / 7).ToArray();
        var x = T(values);
        Tensor<double> grad;
        using (var tape = new GradientTape<double>())
        {
            var y = Invoke(op, x);
            grad = tape.ComputeGradients(_engine.ReduceSum(y, null, false), new[] { x })[x];
        }
        for (int i = 0; i < values.Length; i++)
        {
            double h = 1e-6 * Math.Max(1, Math.Abs(values[i]));
            double fd = (Invoke(op, T(new[] { values[i] + h })).GetFlat(0) - Invoke(op, T(new[] { values[i] - h })).GetFlat(0)) / (2 * h);
            Assert.True(Math.Abs(fd - grad.GetFlat(i)) <= 1e-5 * Math.Max(1, Math.Abs(fd)), $"{op}'({values[i]}): fd {fd}, analytic {grad.GetFlat(i)}");
        }
    }

    [Fact]
    public void IgammaGradient_FlowsToX_AsInTorch()
    {
        var a = T(new[] { 0.5, 2.0, 5.0 });
        var x = T(new[] { 0.7, 1.5, 6.0 });
        Tensor<double> gx;
        using (var tape = new GradientTape<double>())
        {
            var y = _engine.TensorIgamma(a, x);
            gx = tape.ComputeGradients(_engine.ReduceSum(y, null, false), new[] { x })[x];
        }
        for (int i = 0; i < 3; i++)
        {
            double h = 1e-6;
            double fd = (_engine.TensorIgamma(T(new[] { a.GetFlat(i) }), T(new[] { x.GetFlat(i) + h })).GetFlat(0)
                - _engine.TensorIgamma(T(new[] { a.GetFlat(i) }), T(new[] { x.GetFlat(i) - h })).GetFlat(0)) / (2 * h);
            Assert.Equal(fd, gx.GetFlat(i), 6);
        }
    }

    [Fact]
    public void PolynomialDegree_TruncatesTowardZero_LikeTorch()
    {
        // torch casts n with static_cast<int64_t>: 2.7 -> 2, -0.7 -> 0, 3.9 -> 3, -2.5 -> -2 (negative degree: 0).
        Close(new[] { -0.5, 1.0 }, _engine.TensorChebyshevPolynomialT(T(new[] { 0.5, 0.3 }), T(new[] { 2.7, -0.7 })), "chebyshev_t", 1e-15);
        Close(new[] { -0.4375, 0.0 }, _engine.TensorLegendrePolynomialP(T(new[] { 0.5, 0.3 }), T(new[] { 3.9, -2.5 })), "legendre_p", 1e-15);
    }
}
