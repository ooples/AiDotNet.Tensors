using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// The typed float/double host paths of <c>BackwardFunctions.ELUBackward</c>, <c>CpuEngine.EluBackward</c>,
/// <c>ComplexMultiplyBackward</c> and <c>ComplexMagnitudeBackward</c> replaced per-element GetFlat/SetFlat plus a
/// virtual NumOps call. They must reproduce the scalar formula they replaced bit for bit (same operations, same order,
/// same precision), for edge values as well as ordinary ones, and still be the true derivative.
/// </summary>
/// <remarks>
/// NaN results are compared as "both NaN": the payload a NaN carries through an add or multiply depends on operand
/// order, which the JIT may commute for the scalar reference, and no consumer depends on NaN payloads.
/// </remarks>
public class ElementwiseBackwardTypedPathTests
{
    private readonly CpuEngine _engine = new();

    // Edge values: signed zeros, the ELU kink neighbourhood, large magnitudes, infinities and NaN.
    private static readonly double[] EdgeValues =
    {
        0.0, -0.0, 1e-30, -1e-30, 1e-7, -1e-7, 0.5, -0.5, 1.0, -1.0, -0.999999, 3.0, -3.0,
        1e6, -1e6, 1e30, -1e30, 3e38, -3e38, double.PositiveInfinity, double.NegativeInfinity, double.NaN,
    };

    private static double[] Mixed(int length, int seed)
    {
        var rng = new Random(seed);
        var v = new double[length];
        for (int i = 0; i < length; i++)
            v[i] = i < EdgeValues.Length ? EdgeValues[i] : (rng.NextDouble() - 0.5) * 8.0;
        return v;
    }

    // Shuffled against Mixed so edge values meet each other (NaN gradient on a positive output, and so on).
    private static double[] MixedShifted(int length, int seed)
    {
        var v = Mixed(length, seed);
        var shifted = new double[length];
        for (int i = 0; i < length; i++) shifted[i] = v[(i + 7) % length];
        return shifted;
    }

    private static Tensor<float> F(double[] v, int[] shape)
    {
        var data = new float[v.Length];
        for (int i = 0; i < v.Length; i++) data[i] = (float)v[i];
        return new Tensor<float>(data, shape);
    }

    private static Tensor<double> D(double[] v, int[] shape) => new((double[])v.Clone(), shape);

    private static void AssertSame(float expected, float actual, string where)
    {
        if (float.IsNaN(expected) && float.IsNaN(actual)) return;
        Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected) == TestHelpers.MathCompat.SingleToInt32Bits(actual),
            $"{where}: expected {expected:R} got {actual:R}");
    }

    private static void AssertSame(double expected, double actual, string where)
    {
        if (double.IsNaN(expected) && double.IsNaN(actual)) return;
        Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual),
            $"{where}: expected {expected:R} got {actual:R}");
    }

    // Lengths: an all-edge-value case, a non-multiple of every SIMD width (scalar tail), and one large enough to split
    // into parallel chunks.
    public static IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { new[] { 22 } },
        new object[] { new[] { 7, 149 } },
        new object[] { new[] { 408, 16, 30 } },
    };

    // ---------------------------------------------------------------- ELU, tape backward (derivative from the output)

    [Theory]
    [MemberData(nameof(Shapes))]
    public void EluTapeBackward_Float_MatchesScalarFormulaBitExactly(int[] shape)
    {
        int n = Length(shape);
        const double alpha = 1.3;
        var y = F(Mixed(n, 1), shape);
        var g = F(MixedShifted(n, 2), shape);
        var x = new Tensor<float>(shape);

        // First contribution: stored.
        var grads = new Dictionary<Tensor<float>, Tensor<float>>();
        BackwardFunctions<float>.ELUBackward(g, new[] { x }, y, new object[] { alpha }, _engine, grads);
        var first = grads[x];
        float a = (float)alpha;
        for (int i = 0; i < n; i++)
            AssertSame(g[i] * (y[i] > 0f ? 1f : y[i] + a), first[i], $"first[{i}] y={y[i]:R} g={g[i]:R}");

        // Second contribution: added to the existing accumulator (existing + contribution).
        var existing = new float[n];
        for (int i = 0; i < n; i++) existing[i] = first[i];
        BackwardFunctions<float>.ELUBackward(g, new[] { x }, y, new object[] { alpha }, _engine, grads);
        var second = grads[x];
        for (int i = 0; i < n; i++)
            AssertSame(existing[i] + g[i] * (y[i] > 0f ? 1f : y[i] + a), second[i], $"second[{i}]");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void EluTapeBackward_Double_MatchesScalarFormulaBitExactly(int[] shape)
    {
        int n = Length(shape);
        const double alpha = 0.7;
        var y = D(Mixed(n, 3), shape);
        var g = D(MixedShifted(n, 4), shape);
        var x = new Tensor<double>(shape);

        var grads = new Dictionary<Tensor<double>, Tensor<double>>();
        BackwardFunctions<double>.ELUBackward(g, new[] { x }, y, new object[] { alpha }, _engine, grads);
        var first = grads[x];
        for (int i = 0; i < n; i++)
            AssertSame(g[i] * (y[i] > 0d ? 1d : y[i] + alpha), first[i], $"first[{i}] y={y[i]:R} g={g[i]:R}");

        var existing = new double[n];
        for (int i = 0; i < n; i++) existing[i] = first[i];
        BackwardFunctions<double>.ELUBackward(g, new[] { x }, y, new object[] { alpha }, _engine, grads);
        var second = grads[x];
        for (int i = 0; i < n; i++)
            AssertSame(existing[i] + g[i] * (y[i] > 0d ? 1d : y[i] + alpha), second[i], $"second[{i}]");
    }

    [Fact]
    public void EluTapeBackward_NaNOutput_PropagatesNaN()
    {
        var y = new Tensor<float>(new[] { float.NaN, 2f, -0.5f }, new[] { 3 });
        var g = new Tensor<float>(new[] { 1f, float.NaN, 0f }, new[] { 3 });
        var x = new Tensor<float>(new[] { 3 });
        var grads = new Dictionary<Tensor<float>, Tensor<float>>();
        BackwardFunctions<float>.ELUBackward(g, new[] { x }, y, new object[] { 1.0 }, _engine, grads);
        Assert.True(float.IsNaN(grads[x][0]), "a NaN output takes the y + alpha branch and stays NaN");
        Assert.True(float.IsNaN(grads[x][1]), "a NaN upstream gradient stays NaN");
        Assert.Equal(0f, grads[x][2]);
    }

    [Fact]
    public void EluTapeBackward_StridedGradient_TakesGenericPathWithSameResult()
    {
        // A transposed (non-contiguous) upstream gradient is not eligible for the typed path; the generic path must
        // still produce g * derivative element by element in logical order.
        var yData = Mixed(6 * 5, 5);
        var y = D(yData, new[] { 6, 5 });
        var gBase = D(MixedShifted(5 * 6, 6), new[] { 5, 6 });
        var g = gBase.Transpose(new[] { 1, 0 });
        Assert.False(g.IsContiguous);
        var x = new Tensor<double>(new[] { 6, 5 });
        var grads = new Dictionary<Tensor<double>, Tensor<double>>();
        BackwardFunctions<double>.ELUBackward(g, new[] { x }, y, new object[] { 1.0 }, _engine, grads);
        for (int r = 0; r < 6; r++)
            for (int c = 0; c < 5; c++)
            {
                double yi = y[r, c], gi = g[r, c];
                AssertSame(gi * (yi > 0d ? 1d : yi + 1.0), grads[x][r, c], $"[{r},{c}]");
            }
    }

    [Fact]
    public void EluTape_Double_MatchesFiniteDifferences()
    {
        var rng = new Random(11);
        var shape = new[] { 4, 9 };
        int n = Length(shape);
        var xs = new double[n];
        var ws = new double[n];
        for (int i = 0; i < n; i++)
        {
            double v = (rng.NextDouble() - 0.5) * 6.0;
            xs[i] = Math.Abs(v) < 1e-3 ? 0.25 : v;   // keep finite-difference probes off the kink at 0
            ws[i] = rng.NextDouble() * 2.0 - 1.0;
        }
        var x = D(xs, shape);
        var w = D(ws, shape);
        const double alpha = 1.1;

        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            tape.BindEngineIfUnset(_engine);
            var loss = _engine.ReduceSum(_engine.TensorMultiply(_engine.ELU(x, alpha), w), null);
            grads = tape.ComputeGradients(loss, new[] { x });
        }
        var analytic = grads[x];

        const double eps = 1e-6;
        for (int i = 0; i < n; i++)
        {
            double orig = x[i];
            x[i] = orig + eps; double lp = _engine.TensorSum(_engine.TensorMultiply(_engine.ELU(x, alpha), w));
            x[i] = orig - eps; double lm = _engine.TensorSum(_engine.TensorMultiply(_engine.ELU(x, alpha), w));
            x[i] = orig;
            double numeric = (lp - lm) / (2 * eps);
            Assert.True(Math.Abs(analytic[i] - numeric) <= 1e-6 * Math.Max(1.0, Math.Abs(numeric)),
                $"x[{i}]={orig:R}: analytic {analytic[i]:R} numeric {numeric:R}");
        }
    }

    // ---------------------------------------------------------------- ELU, engine op (derivative from the input)

    [Theory]
    [MemberData(nameof(Shapes))]
    public void EngineEluBackward_Float_MatchesScalarFormulaBitExactly(int[] shape)
    {
        int n = Length(shape);
        const double alpha = 1.3;
        var x = F(Mixed(n, 7), shape);
        var y = F(MixedShifted(n, 8), shape);
        var g = F(Mixed(n, 9), shape);
        var result = _engine.EluBackward(g, x, y, alpha);
        for (int i = 0; i < n; i++)
        {
            double xi = x[i];
            double derivative = xi > 0 ? 1.0 : (double)y[i] + alpha;
            AssertSame((float)((double)g[i] * derivative), result[i], $"[{i}] x={x[i]:R} y={y[i]:R} g={g[i]:R}");
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void EngineEluBackward_Double_MatchesScalarFormulaBitExactly(int[] shape)
    {
        int n = Length(shape);
        const double alpha = 0.7;
        var x = D(Mixed(n, 10), shape);
        var y = D(MixedShifted(n, 11), shape);
        var g = D(Mixed(n, 12), shape);
        var result = _engine.EluBackward(g, x, y, alpha);
        for (int i = 0; i < n; i++)
            AssertSame(g[i] * (x[i] > 0 ? 1.0 : y[i] + alpha), result[i], $"[{i}]");
    }

    // ---------------------------------------------------------------- complex multiply / magnitude

    [Fact]
    public void ComplexMultiplyBackward_FloatAndDouble_MatchScalarFormulaBitExactly()
    {
        var shape = new[] { 3, 46 };   // 69 interleaved pairs, edge values included
        int n = Length(shape);
        double[] gv = Mixed(n, 13), av = MixedShifted(n, 14), bv = Mixed(n, 15);

        var gf = F(gv, shape); var af = F(av, shape); var bf = F(bv, shape);
        var gradsF = new Dictionary<Tensor<float>, Tensor<float>>();
        BackwardFunctions<float>.ComplexMultiplyBackward(gf, new[] { af, bf }, new Tensor<float>(shape), Array.Empty<object>(), _engine, gradsF);
        for (int p = 0; p < n / 2; p++)
        {
            int i = 2 * p;
            AssertSame(gf[i] * bf[i] + gf[i + 1] * bf[i + 1], gradsF[af][i], $"float dA re[{p}]");
            AssertSame(gf[i + 1] * bf[i] - gf[i] * bf[i + 1], gradsF[af][i + 1], $"float dA im[{p}]");
            AssertSame(af[i] * gf[i] + af[i + 1] * gf[i + 1], gradsF[bf][i], $"float dB re[{p}]");
            AssertSame(af[i] * gf[i + 1] - af[i + 1] * gf[i], gradsF[bf][i + 1], $"float dB im[{p}]");
        }

        var gd = D(gv, shape); var ad = D(av, shape); var bd = D(bv, shape);
        var gradsD = new Dictionary<Tensor<double>, Tensor<double>>();
        BackwardFunctions<double>.ComplexMultiplyBackward(gd, new[] { ad, bd }, new Tensor<double>(shape), Array.Empty<object>(), _engine, gradsD);
        for (int p = 0; p < n / 2; p++)
        {
            int i = 2 * p;
            AssertSame(gd[i] * bd[i] + gd[i + 1] * bd[i + 1], gradsD[ad][i], $"double dA re[{p}]");
            AssertSame(gd[i + 1] * bd[i] - gd[i] * bd[i + 1], gradsD[ad][i + 1], $"double dA im[{p}]");
            AssertSame(ad[i] * gd[i] + ad[i + 1] * gd[i + 1], gradsD[bd][i], $"double dB re[{p}]");
            AssertSame(ad[i] * gd[i + 1] - ad[i + 1] * gd[i], gradsD[bd][i + 1], $"double dB im[{p}]");
        }
    }

    [Fact]
    public void ComplexMagnitudeBackward_FloatAndDouble_MatchScalarFormulaBitExactly()
    {
        int pairs = 60;
        double[] av = Mixed(2 * pairs, 16), magv = MixedShifted(pairs, 17), gv = Mixed(pairs, 18);

        var af = F(av, new[] { 2 * pairs }); var magf = F(magv, new[] { pairs }); var gf = F(gv, new[] { pairs });
        var gradsF = new Dictionary<Tensor<float>, Tensor<float>>();
        BackwardFunctions<float>.ComplexMagnitudeBackward(gf, new[] { af }, magf, new object[] { af }, _engine, gradsF);
        float epsF = (float)1e-8;
        for (int p = 0; p < pairs; p++)
        {
            float safe = magf[p] + epsF;
            AssertSame(gf[p] * (af[2 * p] / safe), gradsF[af][2 * p], $"float re[{p}]");
            AssertSame(gf[p] * (af[2 * p + 1] / safe), gradsF[af][2 * p + 1], $"float im[{p}]");
        }

        var ad = D(av, new[] { 2 * pairs }); var magd = D(magv, new[] { pairs }); var gd = D(gv, new[] { pairs });
        var gradsD = new Dictionary<Tensor<double>, Tensor<double>>();
        BackwardFunctions<double>.ComplexMagnitudeBackward(gd, new[] { ad }, magd, new object[] { ad }, _engine, gradsD);
        for (int p = 0; p < pairs; p++)
        {
            double safe = magd[p] + 1e-8;
            AssertSame(gd[p] * (ad[2 * p] / safe), gradsD[ad][2 * p], $"double re[{p}]");
            AssertSame(gd[p] * (ad[2 * p + 1] / safe), gradsD[ad][2 * p + 1], $"double im[{p}]");
        }
    }

    [Fact]
    public void ComplexMultiplyAndMagnitude_Double_MatchFiniteDifferences()
    {
        var rng = new Random(21);
        var shape = new[] { 12 };   // 6 complex numbers
        var a = new Tensor<double>(shape);
        var b = new Tensor<double>(shape);
        var w = new Tensor<double>(new[] { 6 });
        for (int i = 0; i < 12; i++) { a[i] = rng.NextDouble() * 2 - 1; b[i] = rng.NextDouble() * 2 - 1; }
        for (int i = 0; i < 6; i++) w[i] = rng.NextDouble() * 2 - 1;

        // loss = sum(w * |a * b|): exercises both backward functions through the tape.
        Func<double> lossValue = () => _engine.TensorSum(
            _engine.TensorMultiply(_engine.TensorComplexMagnitude(_engine.TensorComplexMultiply(a, b)), w));

        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            tape.BindEngineIfUnset(_engine);
            var loss = _engine.ReduceSum(
                _engine.TensorMultiply(_engine.TensorComplexMagnitude(_engine.TensorComplexMultiply(a, b)), w), null);
            grads = tape.ComputeGradients(loss, new[] { a, b });
        }

        const double eps = 1e-6;
        foreach (var (t, name) in new[] { (a, "a"), (b, "b") })
        {
            var analytic = grads[t];
            for (int i = 0; i < t.Length; i++)
            {
                double orig = t[i];
                t[i] = orig + eps; double lp = lossValue();
                t[i] = orig - eps; double lm = lossValue();
                t[i] = orig;
                double numeric = (lp - lm) / (2 * eps);
                // |z| + 1e-8 in the backward biases it by ~1e-8 relative; well inside this tolerance.
                Assert.True(Math.Abs(analytic[i] - numeric) <= 1e-5 * Math.Max(1.0, Math.Abs(numeric)),
                    $"{name}[{i}]: analytic {analytic[i]:R} numeric {numeric:R}");
            }
        }
    }

    private static int Length(int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        return n;
    }
}
