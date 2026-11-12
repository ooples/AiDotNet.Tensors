using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The model-level ops (CRF log-partition, hyperbolic linear, fake quantization, surrogate spikes, top-k softmax,
/// time embeddings, soft split / k-NN / locally-weighted regression, standalone activations): forward values against
/// independent references (brute force, closed forms, AiDotNet's legacy autodiff formulas) and every gradient against
/// <see cref="GradCheck"/> in float and double.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class TorchModelOpsTests
{
    private readonly CpuEngine _engine = new();

    private static Tensor<T> Make<T>(double[] values, params int[] shape)
    {
        if (typeof(T) == typeof(double)) return (Tensor<T>)(object)new Tensor<double>((double[])values.Clone(), shape);
        if (typeof(T) == typeof(float)) return (Tensor<T>)(object)new Tensor<float>(values.Select(v => (float)v).ToArray(), shape);
        throw new NotSupportedException(typeof(T).Name);
    }

    private static double[] Rand(int seed, int count, double lo = -1, double hi = 1)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => lo + (hi - lo) * rng.NextDouble()).ToArray();
    }

    private static double[] D<T>(Tensor<T> t) => t.ToArray().Select(v => Convert.ToDouble(v)).ToArray();

    private static void Close(double[] expected, double[] actual, string what, double tolerance = 1e-10)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }

    private void AssertGradCheck<T>(Func<IReadOnlyList<Tensor<T>>, Tensor<T>> f, params Tensor<T>[] inputs)
    {
        var result = GradCheck.CheckMany(f, inputs, new GradCheckOptions { Engine = _engine });
        Assert.True(result.Passed, $"{typeof(T).Name}: {result}");
    }

    // ---------------------------------------------------------------- CRF

    // log Σ over every tag path of exp(score), enumerated.
    private static double CrfBruteForce(double[] emit, int offset, int seqLen, int k, double[] trans, double[]? start, double[]? end)
    {
        var scores = new List<double>();
        var path = new int[seqLen];
        int total = (int)Math.Pow(k, seqLen);
        for (int code = 0; code < total; code++)
        {
            int c = code;
            for (int t = 0; t < seqLen; t++) { path[t] = c % k; c /= k; }
            double s = (start?[path[0]] ?? 0) + (end?[path[seqLen - 1]] ?? 0);
            for (int t = 0; t < seqLen; t++) s += emit[offset + t * k + path[t]];
            for (int t = 1; t < seqLen; t++) s += trans[path[t - 1] * k + path[t]];
            scores.Add(s);
        }
        double max = scores.Max();
        return max + Math.Log(scores.Sum(s => Math.Exp(s - max)));
    }

    [Fact]
    public void CrfForward_MatchesPathEnumeration_UnbatchedAndBatched()
    {
        const int s = 4, k = 3, b = 2;
        var emit = Rand(1, b * s * k, -2, 2);
        var trans = Rand(2, k * k, -1, 1);
        var start = Rand(3, k);
        var end = Rand(4, k);

        var single = _engine.TensorCrfForward(Make<double>(emit.Take(s * k).ToArray(), s, k), Make<double>(trans, k, k),
            Make<double>(start, k), Make<double>(end, k));
        Assert.Equal(new[] { 1 }, single.Shape.ToArray());
        Close(new[] { CrfBruteForce(emit, 0, s, k, trans, start, end) }, D(single), "crf");

        var noBoundary = _engine.TensorCrfForward(Make<double>(emit.Take(s * k).ToArray(), s, k), Make<double>(trans, k, k));
        Close(new[] { CrfBruteForce(emit, 0, s, k, trans, null, null) }, D(noBoundary), "crf no start/end");

        var batched = _engine.TensorCrfForward(Make<double>(emit, b, s, k), Make<double>(trans, k, k), Make<double>(start, k), Make<double>(end, k));
        Assert.Equal(new[] { b }, batched.Shape.ToArray());
        Close(new[] { CrfBruteForce(emit, 0, s, k, trans, start, end), CrfBruteForce(emit, s * k, s, k, trans, start, end) },
            D(batched), "crf batched");
    }

    [Fact]
    public void CrfForward_EmissionGradientIsThePosteriorMarginal()
    {
        // Each step's marginals sum to one, so the emission gradient of one row sums to the sequence length.
        const int s = 3, k = 4;
        var emissions = Make<double>(Rand(5, s * k, -2, 2), s, k);
        using var tape = new GradientTape<double>();
        var logZ = _engine.TensorCrfForward(emissions, Make<double>(Rand(6, k * k), k, k));
        var grad = tape.ComputeGradients(logZ, new[] { emissions })[emissions];
        for (int t = 0; t < s; t++)
            Assert.Equal(1.0, Enumerable.Range(0, k).Sum(j => grad[t, j]), 12);
    }

    private void CrfGradCheck<T>()
    {
        const int s = 3, k = 3;
        AssertGradCheck<T>(x => _engine.TensorCrfForward(x[0], x[1], x[2], x[3]),
            Make<T>(Rand(7, 2 * s * k, -1, 1), 2, s, k), Make<T>(Rand(8, k * k), k, k), Make<T>(Rand(9, k), k), Make<T>(Rand(10, k), k));
        AssertGradCheck<T>(x => _engine.TensorCrfForward(x[0], x[1]), Make<T>(Rand(11, s * k), s, k), Make<T>(Rand(12, k * k), k, k));
    }

    [Fact] public void CrfForward_GradCheck_Double() => CrfGradCheck<double>();
    [Fact] public void CrfForward_GradCheck_Float() => CrfGradCheck<float>();

    // ---------------------------------------------------------------- Hyperbolic linear

    private static double[] HyperbolicReference(double[] x, int rows, int inF, double[] w, int outF, double[]? bias, double c)
    {
        double sqrtC = Math.Sqrt(c), maxNorm = (1 - 1e-5) / sqrtC;
        static double Norm(double[] v) => Math.Sqrt(v.Sum(a => a * a));
        double[] Project(double[] v)
        {
            double n = Norm(v);
            return n > maxNorm ? v.Select(a => a * maxNorm / n).ToArray() : v;
        }
        var result = new double[rows * outF];
        for (int r = 0; r < rows; r++)
        {
            var row = x.Skip(r * inF).Take(inF).ToArray();
            double xn = Norm(row);
            var u = row.Select(a => a * Atanh(sqrtC * xn) / (sqrtC * xn)).ToArray();
            var m = new double[outF];
            for (int o = 0; o < outF; o++) for (int i = 0; i < inF; i++) m[o] += w[o * inF + i] * u[i];
            double mn = Norm(m);
            var h = m.Select(a => a * Math.Tanh(sqrtC * mn) / (sqrtC * mn)).ToArray();
            if (bias is not null)
            {
                var bp = Project(bias);
                double xy = h.Zip(bp, (a, q) => a * q).Sum(), x2 = h.Sum(a => a * a), y2 = bp.Sum(a => a * a);
                double den = 1 + 2 * c * xy + c * c * x2 * y2;
                h = h.Select((a, i) => ((1 + 2 * c * xy + c * y2) * a + (1 - c * x2) * bp[i]) / den).ToArray();
            }
            h = Project(h);
            Array.Copy(h, 0, result, r * outF, outF);
        }
        return result;
    }

    private static double Atanh(double v) => 0.5 * Math.Log((1 + v) / (1 - v));

    [Fact]
    public void HyperbolicLinear_MatchesMobiusMatVecReference()
    {
        const int rows = 3, inF = 4, outF = 2;
        var x = Rand(20, rows * inF, -0.4, 0.4);
        var w = Rand(21, outF * inF, -1, 1);
        var b = Rand(22, outF, -0.3, 0.3);
        foreach (double c in new[] { 1.0, 0.5 })
        {
            var y = _engine.TensorHyperbolicLinear(Make<double>(x, rows, inF), Make<double>(w, outF, inF), Make<double>(b, outF), -c);
            Close(HyperbolicReference(x, rows, inF, w, outF, b, c), D(y), $"hyperbolic c={c}", 1e-9);
            var noBias = _engine.TensorHyperbolicLinear(Make<double>(x, rows, inF), Make<double>(w, outF, inF), null, c);
            Close(HyperbolicReference(x, rows, inF, w, outF, null, c), D(noBias), $"hyperbolic no bias c={c}", 1e-9);
        }

        // Leading axes fold into rows; the output keeps them.
        var batched = _engine.TensorHyperbolicLinear(Make<double>(x.Concat(x).ToArray(), 2, rows, inF), Make<double>(w, outF, inF), Make<double>(b, outF));
        Assert.Equal(new[] { 2, rows, outF }, batched.Shape.ToArray());

        // The Möbius-addition step agrees with the engine's own MobiusAdd.
        var h = HyperbolicReference(x, rows, inF, w, outF, null, 1.0).Take(outF).ToArray();
        var expected = _engine.MobiusAdd(new Vector<double>(h), new Vector<double>(b), 1.0);
        Close(expected.ToArray(), D(_engine.TensorHyperbolicLinear(Make<double>(x.Take(inF).ToArray(), 1, inF), Make<double>(w, outF, inF), Make<double>(b, outF))),
            "agrees with MobiusAdd", 1e-9);
    }

    private void HyperbolicGradCheck<T>()
    {
        AssertGradCheck<T>(v => _engine.TensorHyperbolicLinear(v[0], v[1], v[2], 0.8),
            Make<T>(Rand(23, 2 * 3, -0.4, 0.4), 2, 3), Make<T>(Rand(24, 4 * 3), 4, 3), Make<T>(Rand(25, 4, -0.3, 0.3), 4));
    }

    [Fact] public void HyperbolicLinear_GradCheck_Double() => HyperbolicGradCheck<double>();
    [Fact] public void HyperbolicLinear_GradCheck_Float() => HyperbolicGradCheck<float>();

    [Fact]
    public void HyperbolicLinear_SecondOrder_Double()
    {
        var result = GradCheck.GradGradCheckMany<double>(v => _engine.TensorHyperbolicLinear(v[0], v[1], v[2]),
            new[] { Make<double>(Rand(26, 2 * 3, -0.4, 0.4), 2, 3), Make<double>(Rand(27, 2 * 3), 2, 3), Make<double>(Rand(28, 2, -0.3, 0.3), 2) },
            new GradCheckOptions { Engine = _engine });
        Assert.True(result.Passed, result.ToString());
    }

    // ---------------------------------------------------------------- Fake quantization

    [Fact]
    public void FakeQuantize_MatchesLegacyFormulaAndTorchRangeMaskedGradient()
    {
        var x = Make<double>(new[] { -1.3, -0.26, -0.05, 0.0, 0.049, 0.51, 0.74, 2.0 }, 8);
        // 3-bit symmetric, scale 0.25: q in [-4, 3].
        var y = _engine.TensorFakeQuantize(x, 3, 0.25);
        Close(new[] { -1.0, -0.25, -0.0, 0.0, 0.0, 0.5, 0.75, 0.75 }, D(y), "symmetric");

        using (var tape = new GradientTape<double>())
        {
            var outT = _engine.TensorFakeQuantize(x, 3, 0.25);
            var g = tape.ComputeGradients(_engine.ReduceSum(outT, null), new[] { x })[x];
            // round(-1.3/0.25) = -5 and round(2/0.25) = 8 fall outside [-4, 3]: zero gradient, as torch.
            Close(new[] { 0.0, 1, 1, 1, 1, 1, 1, 0 }, D(g), "STE mask");
        }

        // 2-bit asymmetric with an explicit zero point 1: q in [0, 3].
        Close(new[] { -0.5, -0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0 }, D(_engine.TensorFakeQuantize(x, 2, 0.5, 1, symmetric: false)), "asymmetric");

        // Derived symmetric scale: max|x| / qMax (legacy): 8-bit, max |x| = 2 → step 2/127.
        double step = 2.0 / 127;
        Close(D(x).Select(v => Math.Round(v / step) * step).ToArray(), D(_engine.TensorFakeQuantize(x)), "derived scale", 1e-12);

        // Derived asymmetric scale and zero point: min maps to 0 and max to 255.
        // step 3.3/255, zero point round(1.3/step) = 100: every value is within half a step of its input.
        var asym = D(_engine.TensorFakeQuantize(x, 8, null, null, symmetric: false));
        double asymStep = 3.3 / 255;
        var inputs = D(x);
        for (int i = 0; i < inputs.Length; i++)
            Assert.True(Math.Abs(asym[i] - inputs[i]) <= asymStep / 2 + 1e-12, $"asymmetric[{i}]: {asym[i]} vs {inputs[i]}");
        Assert.Equal(-100 * asymStep, asym[0], 12);
    }

    [Fact]
    public void FakeQuantize_FloatMatchesDouble()
    {
        var values = Rand(30, 16, -2, 2);
        Close(D(_engine.TensorFakeQuantize(Make<double>(values, 16), 4, 0.2)), D(_engine.TensorFakeQuantize(Make<float>(values, 16), 4, 0.2)), "float", 1e-6);
    }

    // ---------------------------------------------------------------- Surrogate spike

    private void SurrogateSpikeCase<T>()
    {
        var v = Make<T>(new[] { -0.5, 0.2, 0.99, 1.0, 1.01, 2.5 }, 6);
        Close(new[] { 0.0, 0, 0, 0, 1, 1 }, D(_engine.TensorSurrogateSpike(v, 1.0, 4.0)), "spikes");
        using var tape = new GradientTape<T>();
        var spikes = _engine.TensorSurrogateSpike(v, 1.0, 4.0);
        var g = tape.ComputeGradients(_engine.ReduceSum(spikes, null), new[] { v })[v];
        var expected = D(v).Select(p =>
        {
            double s = 1 / (1 + Math.Exp(-4 * (p - 1)));
            return 4 * s * (1 - s);
        }).ToArray();
        Close(expected, D(g), "surrogate gradient", typeof(T) == typeof(float) ? 1e-6 : 1e-12);
    }

    [Fact] public void SurrogateSpike_HeavisideForwardSigmoidSurrogateBackward_Double() => SurrogateSpikeCase<double>();
    [Fact] public void SurrogateSpike_HeavisideForwardSigmoidSurrogateBackward_Float() => SurrogateSpikeCase<float>();

    // ---------------------------------------------------------------- Top-k softmax

    [Fact]
    public void TopKSoftmax_SoftmaxOverTheTopKZeroElsewhere()
    {
        var scores = Make<double>(new[] { 0.1, 2.0, -1.0, 1.0, 3.0, 0.5, 0.5, 0.2 }, 2, 4);
        var y = D(_engine.TensorTopKSoftmax(scores, 2));
        double e0 = Math.Exp(2.0), e1 = Math.Exp(1.0);
        double f0 = Math.Exp(3.0), f1 = Math.Exp(0.5);
        Close(new[] { 0, e0 / (e0 + e1), 0, e1 / (e0 + e1), f0 / (f0 + f1), f1 / (f0 + f1), 0, 0 }, y, "top-2");
        // k = width is the plain softmax.
        Close(D(_engine.Softmax(scores, -1)), D(_engine.TensorTopKSoftmax(scores, 4)), "k = width");
    }

    private void TopKGradCheck<T>()
        => AssertGradCheck<T>(v => _engine.TensorTopKSoftmax(v[0], 3),
            Make<T>(new[] { 0.3, -1.1, 2.0, 0.9, -0.2, 1.4, 0.7, -0.6, 1.9, 0.1, -1.5, 0.4 }, 2, 6));

    [Fact] public void TopKSoftmax_GradCheck_Double() => TopKGradCheck<double>();
    [Fact] public void TopKSoftmax_GradCheck_Float() => TopKGradCheck<float>();

    // ---------------------------------------------------------------- Sinusoidal time embedding

    [Fact]
    public void SinusoidalTimeEmbedding_MatchesLegacyLayout()
    {
        foreach (int dim in new[] { 8, 7 })
        {
            var t = new[] { 0.0, 1.0, 37.5 };
            var y = D(_engine.TensorSinusoidalTimeEmbedding(Make<double>(t, 3, 1), dim));
            int half = dim / 2;
            double logScale = Math.Log(10000.0) / Math.Max(half - 1, 1);
            var expected = new double[3 * dim];
            for (int b = 0; b < 3; b++)
            for (int i = 0; i < half; i++)
            {
                double freq = Math.Exp(-logScale * i);
                expected[b * dim + i] = Math.Sin(t[b] * freq);
                expected[b * dim + i + half] = Math.Cos(t[b] * freq);
            }
            Close(expected, y, $"dim {dim}");
        }
    }

    private void SinusoidalGradCheck<T>()
        => AssertGradCheck<T>(v => _engine.TensorSinusoidalTimeEmbedding(v[0], 6), Make<T>(new[] { 0.5, 1.7, 3.2 }, 3));

    [Fact] public void SinusoidalTimeEmbedding_GradCheck_Double() => SinusoidalGradCheck<double>();
    [Fact] public void SinusoidalTimeEmbedding_GradCheck_Float() => SinusoidalGradCheck<float>();

    // ---------------------------------------------------------------- Soft split / k-NN / locally weighted

    [Fact]
    public void SoftSplit_MatchesLegacyFormula()
    {
        var x = new[] { 0.3, -0.7, 1.2 };
        var left = new[] { 1.0, 2.0 };
        var right = new[] { -1.0, 0.5 };
        double p = 1 / (1 + Math.Exp(-((0.2 - (-0.7)) / 0.5)));
        var y = _engine.TensorSoftSplit(Make<double>(x, 3), Make<double>(left, 2), Make<double>(right, 2), 1, 0.2, 0.5);
        Close(left.Zip(right, (l, r) => p * l + (1 - p) * r).ToArray(), D(y), "soft split");
    }

    private void SoftSplitGradCheck<T>()
        => AssertGradCheck<T>(v => _engine.TensorSoftSplit(v[0], v[1], v[2], 2, 0.1, 0.7),
            Make<T>(Rand(40, 4), 2, 2), Make<T>(Rand(41, 3), 3), Make<T>(Rand(42, 3), 3));

    [Fact] public void SoftSplit_GradCheck_Double() => SoftSplitGradCheck<double>();
    [Fact] public void SoftSplit_GradCheck_Float() => SoftSplitGradCheck<float>();

    private static double[] SoftKnnReference(double[] query, double[] sv, int n, int f, double[] labels, double temperature)
    {
        int outputs = labels.Length / n;
        var d = Enumerable.Range(0, n).Select(i => -Enumerable.Range(0, f).Sum(j => Math.Pow(query[j] - sv[i * f + j], 2)) / temperature).ToArray();
        double max = d.Max();
        var e = d.Select(v => Math.Exp(v - max)).ToArray();
        double sum = e.Sum();
        return Enumerable.Range(0, outputs).Select(o => Enumerable.Range(0, n).Sum(i => e[i] / sum * labels[i * outputs + o])).ToArray();
    }

    [Fact]
    public void SoftKnnAndLocallyWeighted_MatchLegacyFormula_SingleAndBatched()
    {
        const int n = 5, f = 3;
        var sv = Rand(50, n * f);
        var labels1 = Rand(51, n);
        var labels2 = Rand(52, n * 2);
        var q = Rand(53, f);
        var q2 = Rand(54, f);

        var single = _engine.TensorSoftKnn(Make<double>(q, f), Make<double>(sv, n, f), Make<double>(labels1, n), 0.7);
        Assert.Equal(new[] { 1 }, single.Shape.ToArray());
        Close(SoftKnnReference(q, sv, n, f, labels1, 0.7), D(single), "soft knn [n] labels");

        var batch = _engine.TensorSoftKnn(Make<double>(q.Concat(q2).ToArray(), 2, f), Make<double>(sv, n, f), Make<double>(labels2, n, 2), 0.7);
        Assert.Equal(new[] { 2, 2 }, batch.Shape.ToArray());
        Close(SoftKnnReference(q, sv, n, f, labels2, 0.7).Concat(SoftKnnReference(q2, sv, n, f, labels2, 0.7)).ToArray(), D(batch), "soft knn batched");

        var lwr = _engine.TensorSoftLocallyWeighted(Make<double>(q, f), Make<double>(sv, n, f), Make<double>(labels2, n, 2), 1.3);
        Close(SoftKnnReference(q, sv, n, f, labels2, 1.3), D(lwr), "locally weighted");
    }

    [Fact]
    public void SoftKnn_QueryGradientIsNotZero()
    {
        // The legacy backward summed the softmax Jacobian's columns and returned exactly zero for the query; the
        // composed op gives the true gradient (the GradCheck tests pin its value).
        var q = Make<double>(Rand(55, 3), 3);
        using var tape = new GradientTape<double>();
        var y = _engine.TensorSoftKnn(q, Make<double>(Rand(56, 12), 4, 3), Make<double>(Rand(57, 4), 4));
        var g = tape.ComputeGradients(_engine.ReduceSum(y, null), new[] { q })[q];
        Assert.Contains(D(g), v => Math.Abs(v) > 1e-6);
    }

    private void SoftKnnGradCheck<T>()
    {
        AssertGradCheck<T>(v => _engine.TensorSoftKnn(v[0], v[1], v[2], 0.9),
            Make<T>(Rand(60, 2 * 3), 2, 3), Make<T>(Rand(61, 4 * 3), 4, 3), Make<T>(Rand(62, 4 * 2), 4, 2));
        AssertGradCheck<T>(v => _engine.TensorSoftLocallyWeighted(v[0], v[1], v[2], 1.5),
            Make<T>(Rand(63, 3), 3), Make<T>(Rand(64, 4 * 3), 4, 3), Make<T>(Rand(65, 4), 4));
    }

    [Fact] public void SoftKnnAndLocallyWeighted_GradCheck_Double() => SoftKnnGradCheck<double>();
    [Fact] public void SoftKnnAndLocallyWeighted_GradCheck_Float() => SoftKnnGradCheck<float>();

    [Fact]
    public void ComposedOps_SecondOrder_Double()
    {
        var options = new GradCheckOptions { Engine = _engine };
        var knn = GradCheck.GradGradCheckMany<double>(v => _engine.TensorSoftKnn(v[0], v[1], v[2]),
            new[] { Make<double>(Rand(66, 3), 3), Make<double>(Rand(67, 9), 3, 3), Make<double>(Rand(68, 3), 3) }, options);
        Assert.True(knn.Passed, knn.ToString());
        var split = GradCheck.GradGradCheckMany<double>(v => _engine.TensorSoftSplit(v[0], v[1], v[2], 0, 0.1),
            new[] { Make<double>(Rand(69, 2), 2), Make<double>(Rand(70, 2), 2), Make<double>(Rand(71, 2), 2) }, options);
        Assert.True(split.Passed, split.ToString());
        var logSoftmin = GradCheck.GradGradCheck<double>(v => _engine.TensorLogSoftmin(v), Make<double>(Rand(72, 6), 2, 3), options);
        Assert.True(logSoftmin.Passed, logSoftmin.ToString());
    }

    // ---------------------------------------------------------------- Activations

    private static readonly double[] ActivationInputs = { -2.5, -1.2, -0.4, -0.05, 0.3, 0.8, 1.6, 2.7 };

    [Fact]
    public void Activations_MatchClosedFormsAndLegacy()
    {
        var x = Make<double>(ActivationInputs, 8);
        Close(ActivationInputs.Select(v => v * Math.Tanh(v)).ToArray(), D(_engine.TensorLisht(x)), "lisht");
        Close(ActivationInputs.Select(v => (Math.Sqrt(v * v + 1) - 1) / 2 + v).ToArray(), D(_engine.TensorBentIdentity(x)), "bent identity");
        Close(ActivationInputs.Select(v => Math.Exp(-v * v)).ToArray(), D(_engine.TensorGaussian(x)), "gaussian");
        Close(ActivationInputs.Select(v => 1.5 * Math.Tanh(0.5 * v)).ToArray(), D(_engine.TensorScaledTanh(x, 1.5, 0.5)), "scaled tanh");
        // The legacy ScaledTanh(β) = (1 - e^(-βx)) / (1 + e^(-βx)) is alpha 1, beta β/2.
        Close(ActivationInputs.Select(v => (1 - Math.Exp(-2 * v)) / (1 + Math.Exp(-2 * v))).ToArray(), D(_engine.TensorScaledTanh(x, 1, 1)), "legacy scaled tanh β=2");
        Close(ActivationInputs.Select(v => v > 0.5 ? v : 0).ToArray(), D(_engine.TensorThresholdedRelu(x, 0.5)), "thresholded relu");
        Close(ActivationInputs.Select(v => v / Math.Sqrt(1 + 0.5 * v * v)).ToArray(), D(_engine.TensorIsru(x, 0.5)), "isru");
        Close(ActivationInputs.Select(v => Math.Exp(-0.5 * v * v)).ToArray(), D(_engine.TensorSqrbf(x, 0.5)), "sqrbf");

        var m = Make<double>(ActivationInputs, 2, 4);
        var expected = new double[8];
        for (int r = 0; r < 2; r++)
        {
            var row = ActivationInputs.Skip(4 * r).Take(4).Select(v => -v).ToArray();
            double lse = Math.Log(row.Sum(Math.Exp));
            for (int i = 0; i < 4; i++) expected[4 * r + i] = row[i] - lse;
        }
        Close(expected, D(_engine.TensorLogSoftmin(m)), "log softmin");
        Close(expected, D(_engine.TensorLogSoftmin(m, 1)), "log softmin axis 1");
    }

    private void ActivationGradChecks<T>()
    {
        // ThresholdedRelu's inputs stay away from its kink at the threshold.
        var x = Make<T>(ActivationInputs, 2, 4);
        AssertGradCheck<T>(v => _engine.TensorLisht(v[0]), x);
        AssertGradCheck<T>(v => _engine.TensorBentIdentity(v[0]), x);
        AssertGradCheck<T>(v => _engine.TensorGaussian(v[0]), x);
        AssertGradCheck<T>(v => _engine.TensorScaledTanh(v[0], 1.7159, 2.0 / 3), x);
        AssertGradCheck<T>(v => _engine.TensorThresholdedRelu(v[0], 0.5), x);
        AssertGradCheck<T>(v => _engine.TensorIsru(v[0], 0.5), x);
        AssertGradCheck<T>(v => _engine.TensorSqrbf(v[0], 0.5), x);
        AssertGradCheck<T>(v => _engine.TensorLogSoftmin(v[0], 1), x);
    }

    [Fact] public void Activations_GradCheck_Double() => ActivationGradChecks<double>();
    [Fact] public void Activations_GradCheck_Float() => ActivationGradChecks<float>();

    // ---------------------------------------------------------------- Compiled training

    [Fact]
    public void HostComputedOps_RecordACompiledNode()
    {
        // Under a training trace the host-computed ops are graph nodes, not trace-time constants: a compiled step's
        // gradient matches the eager tape's (checked for every Differentiable op by the gradcheck sweep); here a
        // plain eager run under an active tape must still give a gradient for each.
        var emissions = Make<double>(Rand(80, 6), 2, 3);
        using var tape = new GradientTape<double>();
        var loss = _engine.TensorAdd(
            _engine.ReduceSum(_engine.TensorCrfForward(emissions, Make<double>(Rand(81, 9), 3, 3)), null),
            _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorTopKSoftmax(emissions, 2), emissions), null));
        var g = tape.ComputeGradients(loss, new[] { emissions })[emissions];
        Assert.Equal(6, g.Length);
    }
}
