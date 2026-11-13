using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The second batch of sweep data ops on the CPU engine. The caller-Random draws (Box-Muller in each convention, clamped
/// normal, gamma, Bernoulli, Rademacher) against the scalar loops they replace, value for value and draw for draw;
/// TensorToBytes against BitConverter and NumPy 2.5.4 (<c>np.float64(x).astype(np.float16)</c>,
/// <c>astype(np.float32).tobytes()</c>); TensorCast against Math.Round / Truncate / Floor / Ceiling by hand; the row
/// gather / scatter by hand; and TensorLinearFilter against <c>scipy.signal.lfilter</c> (SciPy 1.18.1, float64), its
/// gradients against central differences of the same SciPy call (h = 1e-6).
/// </summary>
public sealed class SweepDataOpsTests
{
    private static readonly CpuEngine Cpu = new();

    private static Tensor<double> D(double[] values, params int[] shape) => new(values, shape);

    private static void AssertSame(double[] expected, double[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.DoubleToInt64Bits(expected[i]) == BitConverter.DoubleToInt64Bits(actual[i]),
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }

    private static void AssertClose(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }

    // ------------------------------------------------------------------ Box-Muller conventions

    // The scalar loops of the AiDotNet call sites, verbatim.
    private static double MetaLearnerSampleNormal(Random r, double mean, double stddev)
    {
        double u1 = 1.0 - r.NextDouble();
        double u2 = r.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + stddev * z;
    }

    private static double RandomExtensionsNextGaussian(Random r)
    {
        double u1 = r.NextDouble();
        double u2 = r.NextDouble();
        u1 = Math.Max(u1, double.Epsilon);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private static double ThompsonStandardNormal(Random r)
    {
        double u1, u2;
        do
        {
            u1 = r.NextDouble();
            u2 = r.NextDouble();
        } while (u1 <= double.Epsilon);
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    private static double HorseshoeSampleNormal(Random r)
    {
        double u1 = 1.0 - r.NextDouble();
        double u2 = 1.0 - r.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private static readonly BoxMullerVariant MetaLearner =
        new(BoxMullerFirstUniform.OneMinus, 0.0, BoxMullerSecondUniform.Raw, BoxMullerBranch.Cosine);
    private static readonly BoxMullerVariant NextGaussian =
        new(BoxMullerFirstUniform.Raw, double.Epsilon, BoxMullerSecondUniform.Raw, BoxMullerBranch.Sine);
    private static readonly BoxMullerVariant Thompson =
        new(BoxMullerFirstUniform.RedrawPairWhileZero, 0.0, BoxMullerSecondUniform.Raw, BoxMullerBranch.Cosine);
    private static readonly BoxMullerVariant Horseshoe =
        new(BoxMullerFirstUniform.OneMinus, 0.0, BoxMullerSecondUniform.OneMinus, BoxMullerBranch.Sine);

    public static IEnumerable<object[]> Conventions()
    {
        yield return new object[] { "metalearner" };
        yield return new object[] { "nextgaussian" };
        yield return new object[] { "thompson" };
        yield return new object[] { "horseshoe" };
    }

    private static (BoxMullerVariant Variant, Func<Random, double> Scalar) Convention(string name) => name switch
    {
        "metalearner" => (MetaLearner, r => MetaLearnerSampleNormal(r, 0.0, 1.0)),
        "nextgaussian" => (NextGaussian, RandomExtensionsNextGaussian),
        "thompson" => (Thompson, ThompsonStandardNormal),
        _ => (Horseshoe, HorseshoeSampleNormal),
    };

    [Theory]
    [MemberData(nameof(Conventions))]
    public void BoxMuller_ReproducesTheScalarLoop_ValuesAndStream(string name)
    {
        var (variant, scalar) = Convention(name);
        var a = RandomHelper.CreateSeededRandom(1234);
        var b = RandomHelper.CreateSeededRandom(1234);
        var drawn = Cpu.TensorNormalBoxMullerFromRng<double>(a, new[] { 7, 9 }, variant, 0.25, 1.5).ToArray();
        var expected = new double[63];
        for (int i = 0; i < expected.Length; i++) expected[i] = 0.25 + 1.5 * scalar(b);
        AssertSame(expected, drawn, name);
        Assert.Equal(b.NextDouble(), a.NextDouble());
    }

    [Fact]
    public void BoxMuller_Float_IsTheDoubleValueRounded()
    {
        var a = RandomHelper.CreateSeededRandom(5);
        var b = RandomHelper.CreateSeededRandom(5);
        var f = Cpu.TensorNormalBoxMullerFromRng<float>(a, new[] { 100 }, MetaLearner, -1.0, 0.5).ToArray();
        for (int i = 0; i < f.Length; i++) Assert.Equal((float)MetaLearnerSampleNormal(b, -1.0, 0.5), f[i]);
    }

    [Fact]
    public void BoxMuller_RedrawWhileZero_SkipsZeroFirstUniforms()
    {
        var a = new SequenceRandom(0.0, 0.0, 0.5, 0.25, 0.75, 0.125);
        var z = Cpu.TensorNormalBoxMullerFromRng<double>(a,
            new[] { 2 }, new BoxMullerVariant(BoxMullerFirstUniform.RedrawWhileZero, 0.0, BoxMullerSecondUniform.Raw,
                BoxMullerBranch.Cosine)).ToArray();
        AssertSame(new[]
        {
            Math.Sqrt(-2.0 * Math.Log(0.5)) * Math.Cos(2.0 * Math.PI * 0.25),
            Math.Sqrt(-2.0 * Math.Log(0.75)) * Math.Cos(2.0 * Math.PI * 0.125),
        }, z, "redraw");
        Assert.Equal(6, a.Calls);
    }

    [Fact]
    public void TruncatedNormal_ClampsTheNextGaussianDraw()
    {
        var a = RandomHelper.CreateSeededRandom(77);
        var b = RandomHelper.CreateSeededRandom(77);
        var t = Cpu.TensorTruncatedNormalFromRng<double>(a, new[] { 500 }, NextGaussian, 0.0, 0.02, -0.02, 0.02).ToArray();
        for (int i = 0; i < t.Length; i++)
        {
            double v = 0.0 + 0.02 * RandomExtensionsNextGaussian(b);
            Assert.Equal(Math.Max(-0.02, Math.Min(0.02, v)), t[i]);
        }
        Assert.Contains(t, v => v == 0.02);
        Assert.Contains(t, v => v == -0.02);
    }

    [Fact]
    public void NormalFromRng_CallsTheSamplerOncePerElementInOrder()
    {
        var a = RandomHelper.CreateSeededRandom(9);
        var b = RandomHelper.CreateSeededRandom(9);
        var t = Cpu.TensorNormalFromRng<float>(a, new[] { 3, 4 }, r => r.NextDouble() * 3 - 1).ToArray();
        for (int i = 0; i < 12; i++) Assert.Equal((float)(b.NextDouble() * 3 - 1), t[i]);
        Assert.Equal(b.NextDouble(), a.NextDouble());
    }

    // ------------------------------------------------------------------ gamma

    // HorseshoeSelector.SampleGamma, verbatim.
    private static double HorseshoeGamma(double shape, double scale, Random rand)
    {
        if (shape < 1)
        {
            double u = rand.NextDouble();
            return HorseshoeGamma(shape + 1, scale, rand) * Math.Pow(u, 1.0 / shape);
        }
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double x, v;
            do
            {
                x = HorseshoeSampleNormal(rand);
                v = 1.0 + c * x;
            } while (v <= 0);
            v = v * v * v;
            double u = rand.NextDouble();
            if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return d * v * scale;
            if (Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v))) return d * v * scale;
        }
    }

    [Theory]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(3.7)]
    public void Gamma_SharedShape_ReproducesTheScalarSampler(double shape)
    {
        var a = RandomHelper.CreateSeededRandom(21);
        var b = RandomHelper.CreateSeededRandom(21);
        var scales = Enumerable.Range(0, 40).Select(i => 0.5 + 0.1 * i).ToArray();
        var drawn = Cpu.TensorGammaFromRng(a, shape, D(scales, 40), Horseshoe).ToArray();
        var expected = scales.Select(s => HorseshoeGamma(shape, s, b)).ToArray();
        AssertSame(expected, drawn, $"gamma({shape})");
        Assert.Equal(b.NextDouble(), a.NextDouble());
    }

    [Fact]
    public void Gamma_PerElementShape_ReproducesTheScalarSampler_AndHasTheGammaMean()
    {
        var a = RandomHelper.CreateSeededRandom(22);
        var b = RandomHelper.CreateSeededRandom(22);
        int n = 20000;
        var shapes = Enumerable.Range(0, n).Select(i => i % 2 == 0 ? 0.3 : 4.0).ToArray();
        var drawn = Cpu.TensorGammaFromRng(a, D(shapes, n), D(Enumerable.Repeat(2.0, n).ToArray(), n), Horseshoe).ToArray();
        for (int i = 0; i < 50; i++) Assert.Equal(HorseshoeGamma(shapes[i], 2.0, b), drawn[i]);
        // E[Gamma(k, theta)] = k theta: 0.6 and 8.0; 10000 draws each put the sample mean within a few standard errors.
        Assert.InRange(drawn.Where((_, i) => i % 2 == 0).Average(), 0.55, 0.65);
        Assert.InRange(drawn.Where((_, i) => i % 2 == 1).Average(), 7.85, 8.15);
    }

    [Fact]
    public void Gamma_RejectsNonPositiveShape()
    {
        var r = RandomHelper.CreateSeededRandom(1);
        Assert.Throws<ArgumentOutOfRangeException>(() => Cpu.TensorGammaFromRng(r, 0.0, D(new[] { 1.0 }, 1), Horseshoe));
    }

    // ------------------------------------------------------------------ Bernoulli / Rademacher

    [Fact]
    public void Bernoulli_PerElement_ComparesEachDrawWithItsProbability()
    {
        var a = RandomHelper.CreateSeededRandom(3);
        var b = RandomHelper.CreateSeededRandom(3);
        var p = Enumerable.Range(0, 64).Select(i => i / 63.0).ToArray();
        var drawn = Cpu.TensorBernoulliFromRng(a, D(p, 8, 8), UniformDrawComparison.DrawBelowThreshold).ToArray();
        for (int i = 0; i < 64; i++) Assert.Equal(b.NextDouble() < p[i] ? 1.0 : 0.0, drawn[i]);
    }

    [Fact]
    public void Bernoulli_Above_AndRademacher_ResolveTheExactHalfAsTheirLoops()
    {
        // Draws 0.5 exactly, below and above: "> 0.5" is false at 0.5, "< 0.5" is false at 0.5.
        var above = Cpu.TensorBernoulliFromRng<double>(new SequenceRandom(0.5, 0.25, 0.75), new[] { 3 }, 0.5,
            UniformDrawComparison.DrawAboveThreshold).ToArray();
        Assert.Equal(new[] { 0.0, 0.0, 1.0 }, above);
        var negBelow = Cpu.TensorRademacherFromRng<double>(new SequenceRandom(0.5, 0.25, 0.75), new[] { 3 },
            RademacherRule.NegativeWhenDrawBelowHalf).ToArray();
        Assert.Equal(new[] { 1.0, -1.0, 1.0 }, negBelow);
        var posAbove = Cpu.TensorRademacherFromRng<double>(new SequenceRandom(0.5, 0.25, 0.75), new[] { 3 },
            RademacherRule.PositiveWhenDrawAboveHalf).ToArray();
        Assert.Equal(new[] { -1.0, -1.0, 1.0 }, posAbove);
    }

    [Fact]
    public void Rademacher_ReproducesTheSpsaLoop()
    {
        var a = RandomHelper.CreateSeededRandom(8);
        var b = RandomHelper.CreateSeededRandom(8);
        var drawn = Cpu.TensorRademacherFromRng<float>(a, new[] { 300 }, RademacherRule.NegativeWhenDrawBelowHalf).ToArray();
        for (int i = 0; i < 300; i++) Assert.Equal(b.NextDouble() < 0.5 ? -1f : 1f, drawn[i]);
    }

    // ------------------------------------------------------------------ TensorToBytes

    [Fact]
    public void ToBytes_Float64_IsBitConverterInEitherOrder()
    {
        var values = new[] { 0.1, -2.5, double.MaxValue, double.Epsilon, double.NegativeInfinity, -0.0 };
        var le = Cpu.TensorToBytes(D(values, 6), NumericStorageType.Float64);
        var be = Cpu.TensorToBytes(D(values, 6), NumericStorageType.Float64, littleEndian: false);
        for (int i = 0; i < values.Length; i++)
        {
            var bytes = BitConverter.GetBytes(values[i]);
            if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
            Assert.Equal(bytes, le.Skip(8 * i).Take(8).ToArray());
            Assert.Equal(bytes.Reverse().ToArray(), be.Skip(8 * i).Take(8).ToArray());
        }
    }

    [Fact]
    public void ToBytes_Float32_MatchesNumPy()
    {
        // np.array([1.5, -1.5, 2.5, -7.9, 300, -300, 1e20, -1e20]).astype(np.float32).tobytes().hex()
        var bytes = Cpu.TensorToBytes(D(new[] { 1.5, -1.5, 2.5, -7.9, 300.0, -300.0, 1e20, -1e20 }, 8), NumericStorageType.Float32);
        Assert.Equal("0000c03f0000c0bf00002040cdccfcc000009643000096c3ec78ad60ec78ade0",
            string.Concat(bytes.Select(b => b.ToString("x2", CultureInfo.InvariantCulture))));
    }

    [Fact]
    public void ToBytes_Float16_RoundsFromTheDoubleAsNumPy()
    {
        var values = new[]
        {
            0.0, -0.0, 1.0, -2.5, 65504.0, 65519.99, 65520.0, 1e-8, 5.960464477539063e-08, 2.9802322387695312e-08,
            2.98023223876953125e-08 * 1.0000001, 6.103515625e-05, 1.0009765625, 1.00048828125, 1.00146484375,
            double.PositiveInfinity, double.NegativeInfinity, 3.14159,
        };
        // np.array(values).astype(np.float16).view(np.uint16)
        var expected = new ushort[]
        {
            0x0, 0x8000, 0x3c00, 0xc100, 0x7bff, 0x7bff, 0x7c00, 0x0, 0x1, 0x0, 0x1, 0x400, 0x3c01, 0x3c00, 0x3c02, 0x7c00,
            0xfc00, 0x4248,
        };
        var bytes = Cpu.TensorToBytes(D(values, values.Length), NumericStorageType.Float16);
        for (int i = 0; i < values.Length; i++)
            Assert.True(expected[i] == (ushort)(bytes[2 * i] | (bytes[2 * i + 1] << 8)), $"[{i}] {values[i]:R}");
    }

    [Fact]
    public void ToBytes_Integers_TruncateAndSaturate_NaNAsZero()
    {
        var values = new[] { 2.9, -2.9, 1e10, -1e10, double.NaN, 127.5, -128.5, 255.9 };
        var i8 = Cpu.TensorToBytes(D(values, 8), NumericStorageType.Int8).Select(b => (sbyte)b).ToArray();
        Assert.Equal(new sbyte[] { 2, -2, 127, -128, 0, 127, -128, 127 }, i8);
        var u8 = Cpu.TensorToBytes(D(values, 8), NumericStorageType.UInt8);
        Assert.Equal(new byte[] { 2, 0, 255, 0, 0, 127, 0, 255 }, u8);
        var i32 = Cpu.TensorToBytes(D(values, 8), NumericStorageType.Int32);
        Assert.Equal(new[] { 2, -2, int.MaxValue, int.MinValue, 0, 127, -128, 255 },
            Enumerable.Range(0, 8).Select(i => BitConverter.ToInt32(i32, 4 * i)).ToArray());
        var i64 = Cpu.TensorToBytes(new Tensor<long>(new[] { long.MaxValue, long.MinValue, 9007199254740993L }, new[] { 3 }),
            NumericStorageType.Int64);
        Assert.Equal(new[] { long.MaxValue, long.MinValue, 9007199254740993L },
            Enumerable.Range(0, 3).Select(i => BitConverter.ToInt64(i64, 8 * i)).ToArray());
    }

    [Fact]
    public void ToBytes_ThenFromBytes_RoundTripsEveryFloatType()
    {
        var rng = RandomHelper.CreateSeededRandom(4);
        var values = Enumerable.Range(0, 257).Select(_ => (float)((rng.NextDouble() - 0.5) * 1e4)).ToArray();
        var t = new Tensor<float>(values, new[] { 257 });
        foreach (bool le in new[] { true, false })
            Assert.Equal(values, Cpu.TensorFromBytes<float>(Cpu.TensorToBytes(t, NumericStorageType.Float32, le),
                NumericStorageType.Float32, new[] { 257 }, le).ToArray());
        Assert.Equal(values, Cpu.TensorFromBytes<float>(Cpu.TensorToBytes(t, NumericStorageType.Float64),
            NumericStorageType.Float64, new[] { 257 }).ToArray());
    }

    // ------------------------------------------------------------------ TensorCast

    [Fact]
    public void Cast_RoundsUnderEachMode()
    {
        var x = D(new[] { 2.5, -2.5, 3.5, 0.49999999999999994, -0.7, 1.2 }, 6);
        Assert.Equal(new[] { 2, -2, 3, 0, 0, 1 }, Cpu.TensorCast<double, int>(x, RoundingMode.TowardZero).ToArray());
        Assert.Equal(new[] { 2, -2, 4, 0, -1, 1 }, Cpu.TensorCast<double, int>(x, RoundingMode.ToNearestEven).ToArray());
        Assert.Equal(new[] { 3, -3, 4, 0, -1, 1 }, Cpu.TensorCast<double, int>(x, RoundingMode.ToNearestAwayFromZero).ToArray());
        Assert.Equal(new[] { 2, -3, 3, 0, -1, 1 }, Cpu.TensorCast<double, int>(x, RoundingMode.TowardNegativeInfinity).ToArray());
        Assert.Equal(new[] { 3, -2, 4, 1, 0, 2 }, Cpu.TensorCast<double, int>(x, RoundingMode.TowardPositiveInfinity).ToArray());
    }

    [Fact]
    public void Cast_SaturatesOrThrows()
    {
        var x = D(new[] { 1e30, -1e30, double.NaN, 255.5, -0.5 }, 5);
        Assert.Equal(new[] { int.MaxValue, int.MinValue, 0, 255, 0 }, Cpu.TensorCast<double, int>(x).ToArray());
        Assert.Equal(new byte[] { 255, 0, 0, 255, 0 }, Cpu.TensorCast<double, byte>(x).ToArray());
        Assert.Equal(new[] { long.MaxValue, long.MinValue, 0L, 255L, 0L }, Cpu.TensorCast<double, long>(x).ToArray());
        Assert.Throws<OverflowException>(() => Cpu.TensorCast<double, int>(x, RoundingMode.TowardZero, CastOverflowBehavior.Throw));
        var big = new Tensor<long>(new[] { long.MaxValue, 300L, -1L }, new[] { 3 });
        Assert.Equal(new[] { int.MaxValue, 300, -1 }, Cpu.TensorCast<long, int>(big).ToArray());
        Assert.Equal(new byte[] { 255, 255, 0 }, Cpu.TensorCast<long, byte>(big).ToArray());
    }

    [Fact]
    public void Cast_ToFloatingTypes_ConvertsAsTheElementTypes()
    {
        var x = D(new[] { 0.1, 1e300, -2.5 }, 3);
        Assert.Equal(new[] { 0.1f, float.PositiveInfinity, -2.5f }, Cpu.TensorCast<double, float>(x).ToArray());
        var ints = new Tensor<int>(new[] { 3, -7 }, new[] { 2 });
        Assert.Equal(new[] { 3.0, -7.0 }, Cpu.TensorCast<int, double>(ints).ToArray());
    }

    // ------------------------------------------------------------------ TensorFormat

    [Fact]
    public void Format_IsEachDoubleValueToString()
    {
        var x = new Tensor<float>(new[] { 0.1f, -2.5f, 1e-7f }, new[] { 3 });
        var expected = new[] { 0.1f, -2.5f, 1e-7f }.Select(v => ((double)v).ToString(CultureInfo.InvariantCulture)).ToArray();
        Assert.Equal(expected, Cpu.TensorFormat(x, CultureInfo.InvariantCulture));
        Assert.Equal("0.10|-2.50|0.00", Cpu.TensorFormatJoined(x, "F2", CultureInfo.InvariantCulture, "|"));
        Assert.Equal(new[] { "0,5" }, Cpu.TensorFormat(D(new[] { 0.5 }, 1), new CultureInfo("de-DE")));
    }

    // ------------------------------------------------------------------ rows

    private static Tensor<double> Table() => D(Enumerable.Range(0, 12).Select(i => (double)i).ToArray(), 4, 3);

    [Fact]
    public void GatherRows_TakesRowsAndZeroesOutOfRangeIds()
    {
        var rows = new Tensor<long>(new[] { 2L, -1L, 0L, 4L, 2L, 3L }, new[] { 2, 3 });
        var g = Cpu.TensorGatherRows(Table(), rows);
        Assert.Equal(new[] { 2, 3, 3 }, g.Shape.ToArray());
        Assert.Equal(new double[] { 6, 7, 8, 0, 0, 0, 0, 1, 2, 0, 0, 0, 6, 7, 8, 9, 10, 11 }, g.ToArray());
    }

    [Fact]
    public void GatherRows_Gradient_IndexAddsRepeatedRows()
    {
        var table = Table();
        var rows = new Tensor<long>(new[] { 2L, 7L, 2L, 0L }, new[] { 4 });
        var w = D(Enumerable.Range(0, 12).Select(i => 0.5 * i - 1).ToArray(), 4, 3);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var g = Cpu.TensorGatherRows(table, rows);
            grads = tape.ComputeGradients(Cpu.ReduceSum(Cpu.TensorMultiply(g, w), null), new[] { table });
        }
        // Row 2 gets w rows 0 and 2, row 0 gets w row 3, rows 1 and 3 nothing (id 7 is out of range).
        Assert.Equal(new double[] { 3.5, 4, 4.5, 0, 0, 0, 1, 2, 3, 0, 0, 0 }, grads[table].ToArray());
    }

    [Fact]
    public void ScatterRows_WritesInPlace_SkipsOutOfRange_LastRepeatWins()
    {
        var table = Table();
        var rows = new Tensor<long>(new[] { 1L, 9L, 1L, 3L }, new[] { 4 });
        var source = D(new double[] { 100, 101, 102, 200, 201, 202, 300, 301, 302, 400, 401, 402 }, 4, 3);
        Cpu.TensorScatterRowsInPlace(table, rows, source);
        Assert.Equal(new double[] { 0, 1, 2, 300, 301, 302, 6, 7, 8, 400, 401, 402 }, table.ToArray());
    }

    // ------------------------------------------------------------------ linear filter (scipy.signal.lfilter)

    private static readonly double[] Signals =
    {
        0.5, -1.25, 2.0, 0.75, -0.5, 1.5, 0.0, -2.25,
        1.0, 0.25, -0.75, 1.75, -1.5, 0.5, 2.5, -0.25,
    };

    private static readonly double[] Numerator = { 0.5, -0.2, 0.1 };
    private static readonly double[] Denominator = { 2.0, -0.6, 0.3, 0.05 };
    private static readonly double[] InitialState = { 0.1, -0.2, 0.3, 0.0, 0.4, -0.1 };

    [Fact]
    public void LinearFilter_MatchesScipy_WithAndWithoutInitialState()
    {
        // scipy.signal.lfilter([0.5, -0.2, 0.1], [2.0, -0.6, 0.3, 0.05], x, axis=-1)
        var y = Cpu.TensorLinearFilter(D(Numerator, 3), D(Denominator, 4), D(Signals, 2, 8)).ToArray();
        AssertClose(new[]
        {
            0.125, -0.325, 0.53375, 0.13074999999999998, -0.1327125, 0.38973, -0.04144287500000002, -0.5550745499999999,
            0.25, 0.03749999999999999, -0.18875, 0.4565, -0.4231750000000001, 0.17179125, 0.603601125, -0.12160897500000001,
        }, y, 1e-15, "y");
        // ... and zi = [[0.1, -0.2, 0.3], [0.0, 0.4, -0.1]]
        var y2 = Cpu.TensorLinearFilter(D(Numerator, 3), D(Denominator, 4), D(Signals, 2, 8), D(InitialState, 2, 3)).ToArray();
        AssertClose(new[]
        {
            0.225, -0.495, 0.76775, 0.22394999999999998, -0.13560250000000001, 0.369033, -0.04954847500000001, -0.55432943,
            0.25, 0.43750000000000006, -0.16874999999999998, 0.4025, -0.452375, 0.17063125, 0.608983125, -0.119090375,
        }, y2, 1e-15, "y zi");
    }

    [Fact]
    public void LinearFilter_MaResidualAndFir_MatchScipy()
    {
        // lfilter([1], [1, 0.4, -0.25], x[0]): the MA residual recursion e[t] = x[t] - 0.4 e[t-1] + 0.25 e[t-2]
        var e = Cpu.TensorLinearFilter(D(new[] { 1.0 }, 1), D(new[] { 1.0, 0.4, -0.25 }, 3), D(Signals.Take(8).ToArray(), 8)).ToArray();
        AssertClose(new[] { 0.5, -1.45, 2.705, -0.6945000000000001, 0.45405000000000006, 1.144755, -0.3443895, -1.8260554500000001 },
            e, 1e-15, "ma");
        // lfilter([0.25, 0.5, 0.25], [1], x[1])
        var fir = Cpu.TensorLinearFilter(D(new[] { 0.25, 0.5, 0.25 }, 3), D(new[] { 1.0 }, 1), D(Signals.Skip(8).ToArray(), 8)).ToArray();
        AssertClose(new[] { 0.25, 0.5625, 0.1875, 0.125, 0.3125, -0.1875, 0.5, 1.3125 }, fir, 1e-15, "fir");
    }

    [Fact]
    public void LinearFilterGuarded_ClampsAndZeroesNaN_BeforeFeedback()
    {
        // y[n] = x[n] + 2 y[n-1] runs away; the guard clamps each output to 3 before it feeds the next step.
        var y = Cpu.TensorLinearFilterGuarded(D(new[] { 1.0 }, 1), D(new[] { 1.0, -2.0 }, 2), D(new[] { 1.0, 1.0, 1.0, double.NaN, 1.0 }, 5),
            null, 3.0).ToArray();
        Assert.Equal(new[] { 1.0, 3.0, 3.0, 0.0, 1.0 }, y);
    }

    [Fact]
    public void LinearFilter_RejectsZeroLeadingDenominator() =>
        Assert.Throws<ArgumentException>(() => Cpu.TensorLinearFilter(D(new[] { 1.0 }, 1), D(new[] { 0.0, 1.0 }, 2), D(Signals, 16)));

    [Fact]
    public void LinearFilter_Gradients_MatchScipyFiniteDifferences()
    {
        var b = D(Numerator, 3); var a = D(Denominator, 4); var x = D(Signals, 2, 8); var zi = D(InitialState, 2, 3);
        var w = D(new[] { 0.3, -1.0, 2.0, 0.5, -0.7, 1.1, 0.2, -0.4, 1.5, -0.3, 0.8, -1.2, 0.6, 0.9, -0.5, 0.25 }, 2, 8);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var y = Cpu.TensorLinearFilter(b, a, x, zi);
            grads = tape.ComputeGradients(Cpu.ReduceSum(Cpu.TensorMultiply(y, w), null), new[] { b, a, x, zi });
        }
        // Central differences of sum(w * lfilter(b, a, x, zi=zi)[0]) (SciPy 1.18.1, h = 1e-6), rounded to 1e-9.
        AssertClose(new[] { 2.455818367, -0.731849338, 0.898932625 }, grads[b].ToArray(), 2e-9, "db");
        AssertClose(new[] { -0.593657784, 0.368911585, -0.108499213, -0.459200625 }, grads[a].ToArray(), 2e-9, "da");
        AssertClose(new[]
        {
            0.10967899, -0.2973112, 0.482856, 0.14797, -0.2011, 0.268, 0.06, -0.1,
            0.3873346, -0.10191675, 0.232286875, -0.31010625, 0.12475, 0.23875, -0.13125, 0.0625,
        }, grads[x].ToArray(), 2e-9, "dx");
        AssertClose(new[] { -0.13778388, -0.3975856, 2.087328, 1.4556073, -0.064799, 0.3390575 }, grads[zi].ToArray(), 2e-9, "dzi");
    }

    /// <summary>A Random that replays a fixed sequence of NextDouble values (and counts the calls).</summary>
    private sealed class SequenceRandom : Random
    {
        private readonly double[] _values;

        // The base seed is irrelevant: every draw the ops make goes through NextDouble below.
        public SequenceRandom(params double[] values) : base(0) => _values = values;

        public int Calls { get; private set; }

        public override double NextDouble() => _values[Calls++];

        protected override double Sample() => NextDouble();
    }
}
