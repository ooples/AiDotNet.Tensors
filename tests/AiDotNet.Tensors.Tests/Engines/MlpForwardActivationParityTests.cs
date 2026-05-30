using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tensors #500: the FusedLinear/MlpForward pointwise-activation dispatch tables
/// (CpuFusedOperations._floatActivations/_doubleActivations) gained ELU, SELU,
/// Softplus, Mish, HardSwish, HardSigmoid, HardTanh — previously only None/ReLU/
/// GELU/Sigmoid/Tanh/LeakyReLU/Swish were supported, so MlpForward THREW for the
/// rest (the gap that forced AiDotNet's Mish/ELU activations off the fused path).
///
/// Each test runs the same linear pre-activation two ways — MlpForward with
/// activation=None (raw x·W) and MlpForward with the activation applied — and
/// checks the activated output equals the canonical scalar formula applied to the
/// raw output. This validates the table entry is dispatched AND numerically
/// correct (an independent textbook reference, not the table's own lambda).
/// </summary>
public class MlpForwardActivationParityTests
{
    private const double SeluAlpha = 1.6732632423543772, SeluScale = 1.0507009873554805;

    private static double Reference(FusedActivationType act, double x) => act switch
    {
        FusedActivationType.ELU => x > 0 ? x : Math.Exp(x) - 1.0,
        FusedActivationType.SELU => SeluScale * (x > 0 ? x : SeluAlpha * (Math.Exp(x) - 1.0)),
        FusedActivationType.Softplus => x > 20 ? x : Math.Log(1.0 + Math.Exp(x)),
        FusedActivationType.Mish => x * Math.Tanh(x > 20 ? x : Math.Log(1.0 + Math.Exp(x))),
        FusedActivationType.HardSwish => x * Math.Max(0.0, Math.Min(1.0, (x + 3.0) / 6.0)),
        FusedActivationType.HardSigmoid => Math.Max(0.0, Math.Min(1.0, (x + 3.0) / 6.0)),
        FusedActivationType.HardTanh => Math.Max(-1.0, Math.Min(1.0, x)),
        FusedActivationType.ReLU6 => Math.Max(0.0, Math.Min(6.0, x)),
        FusedActivationType.SoftSign => x / (1.0 + Math.Abs(x)),
        FusedActivationType.Sign => x < 0 ? -1.0 : (x > 0 ? 1.0 : 0.0),
        FusedActivationType.BentIdentity => 0.5 * (Math.Sqrt(x * x + 1.0) - 1.0) + x,
        FusedActivationType.Gaussian => Math.Exp(-x * x),
        FusedActivationType.LiSHT => x * Math.Tanh(x),
        FusedActivationType.ISRU => x / Math.Sqrt(1.0 + x * x),        // alpha=1 default
        FusedActivationType.SQRBF => Math.Exp(-x * x),                  // beta=1 default
        FusedActivationType.BinarySpiking => x >= 1.0 ? 1.0 : 0.0,      // threshold=1 default
        _ => throw new ArgumentOutOfRangeException(nameof(act)),
    };

    [Theory]
    [InlineData(FusedActivationType.ELU)]
    [InlineData(FusedActivationType.SELU)]
    [InlineData(FusedActivationType.Softplus)]
    [InlineData(FusedActivationType.Mish)]
    [InlineData(FusedActivationType.HardSwish)]
    [InlineData(FusedActivationType.HardSigmoid)]
    [InlineData(FusedActivationType.HardTanh)]
    [InlineData(FusedActivationType.ReLU6)]
    [InlineData(FusedActivationType.SoftSign)]
    [InlineData(FusedActivationType.Sign)]
    [InlineData(FusedActivationType.BentIdentity)]
    [InlineData(FusedActivationType.Gaussian)]
    [InlineData(FusedActivationType.LiSHT)]
    [InlineData(FusedActivationType.ISRU)]
    [InlineData(FusedActivationType.SQRBF)]
    [InlineData(FusedActivationType.BinarySpiking)]
    public void MlpForward_NewActivationFloat_MatchesCanonicalFormula(FusedActivationType act)
    {
        var engine = new CpuEngine();
        const int batch = 4, inF = 16, outF = 8;
        var rng = new Random(20260530);
        var wData = new float[inF * outF];
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var w = new Tensor<float>(wData, new[] { inF, outF });
        var xData = new float[batch * inF];
        for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
        var x = new Tensor<float>(xData, new[] { batch, inF });

        var weights = new System.Collections.Generic.List<Tensor<float>> { w };
        var noBias = new System.Collections.Generic.List<Tensor<float>?> { null };

        var raw = engine.MlpForward(x, weights, noBias, FusedActivationType.None, FusedActivationType.None);
        var fused = engine.MlpForward(x, weights, noBias, FusedActivationType.None, act);

        Assert.Equal(raw.Length, fused.Length);
        for (int i = 0; i < raw.Length; i++)
        {
            double expected = Reference(act, Convert.ToDouble(raw[i]));
            double actual = Convert.ToDouble(fused[i]);
            Assert.True(Math.Abs(expected - actual) < 1e-4,
                $"{act}: MlpForward {actual} != canonical {expected} at index {i} (raw={raw[i]}).");
        }
    }

    [Theory]
    [InlineData(FusedActivationType.ELU)]
    [InlineData(FusedActivationType.SELU)]
    [InlineData(FusedActivationType.Softplus)]
    [InlineData(FusedActivationType.Mish)]
    [InlineData(FusedActivationType.HardSwish)]
    [InlineData(FusedActivationType.HardSigmoid)]
    [InlineData(FusedActivationType.HardTanh)]
    [InlineData(FusedActivationType.ReLU6)]
    [InlineData(FusedActivationType.SoftSign)]
    [InlineData(FusedActivationType.Sign)]
    [InlineData(FusedActivationType.BentIdentity)]
    [InlineData(FusedActivationType.Gaussian)]
    [InlineData(FusedActivationType.LiSHT)]
    [InlineData(FusedActivationType.ISRU)]
    [InlineData(FusedActivationType.SQRBF)]
    [InlineData(FusedActivationType.BinarySpiking)]
    public void MlpForward_NewActivationDouble_MatchesCanonicalFormula(FusedActivationType act)
    {
        var engine = new CpuEngine();
        const int batch = 4, inF = 16, outF = 8;
        var rng = new Random(20260530);
        var wData = new double[inF * outF];
        for (int i = 0; i < wData.Length; i++) wData[i] = rng.NextDouble() * 2.0 - 1.0;
        var w = new Tensor<double>(wData, new[] { inF, outF });
        var xData = new double[batch * inF];
        for (int i = 0; i < xData.Length; i++) xData[i] = rng.NextDouble() * 4.0 - 2.0;
        var x = new Tensor<double>(xData, new[] { batch, inF });

        var weights = new System.Collections.Generic.List<Tensor<double>> { w };
        var noBias = new System.Collections.Generic.List<Tensor<double>?> { null };

        var raw = engine.MlpForward(x, weights, noBias, FusedActivationType.None, FusedActivationType.None);
        var fused = engine.MlpForward(x, weights, noBias, FusedActivationType.None, act);

        Assert.Equal(raw.Length, fused.Length);
        for (int i = 0; i < raw.Length; i++)
        {
            double expected = Reference(act, raw[i]);
            double actual = fused[i];
            Assert.True(Math.Abs(expected - actual) < 1e-9,
                $"{act}: MlpForward(double) {actual} != canonical {expected} at index {i} (raw={raw[i]}).");
        }
    }

    // ---- parametric activations (FusedActivationParams) -------------------

    private static double ParamRef(FusedActivationType act, double x, FusedActivationParams p) => act switch
    {
        FusedActivationType.LeakyReLU => x > 0 ? x : p.Alpha!.Value * x,
        FusedActivationType.RReLU => x > 0 ? x : p.Alpha!.Value * x,
        FusedActivationType.ELU => x > 0 ? x : p.Alpha!.Value * (Math.Exp(x) - 1.0),
        FusedActivationType.CELU => x >= 0 ? x : p.Alpha!.Value * (Math.Exp(x / p.Alpha!.Value) - 1.0),
        FusedActivationType.ThresholdedReLU => x > p.Theta!.Value ? x : 0.0,
        FusedActivationType.ScaledTanh => p.Alpha!.Value * Math.Tanh(p.Beta!.Value * x),
        _ => throw new ArgumentOutOfRangeException(nameof(act)),
    };

    public static TheoryData<FusedActivationType, FusedActivationParams> ParametricCases => new()
    {
        { FusedActivationType.LeakyReLU, new FusedActivationParams { Alpha = 0.2f } },   // non-default slope
        { FusedActivationType.RReLU, new FusedActivationParams { Alpha = 0.15f } },       // eval slope
        { FusedActivationType.ELU, new FusedActivationParams { Alpha = 2.0f } },          // non-default alpha
        { FusedActivationType.CELU, new FusedActivationParams { Alpha = 1.5f } },
        { FusedActivationType.ThresholdedReLU, new FusedActivationParams { Theta = 0.5f } },
        { FusedActivationType.ScaledTanh, new FusedActivationParams { Alpha = 1.7f, Beta = 0.66f } },
    };

    /// <summary>
    /// Parametric activations must honor the supplied FusedActivationParams through
    /// MlpForward — LeakyReLU(0.2)/ELU(2) use the given (non-default) parameter, and
    /// CELU/ThresholdedReLU/ScaledTanh fuse at all (they have no hardcoded entry).
    /// </summary>
    [Theory]
    [MemberData(nameof(ParametricCases))]
    public void MlpForward_ParametricActivationFloat_HonorsParams(FusedActivationType act, FusedActivationParams p)
    {
        var engine = new CpuEngine();
        const int batch = 4, inF = 16, outF = 8;
        var rng = new Random(20260601);
        var wData = new float[inF * outF];
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var w = new Tensor<float>(wData, new[] { inF, outF });
        var xData = new float[batch * inF];
        for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
        var x = new Tensor<float>(xData, new[] { batch, inF });

        var weights = new System.Collections.Generic.List<Tensor<float>> { w };
        var noBias = new System.Collections.Generic.List<Tensor<float>?> { null };

        var raw = engine.MlpForward(x, weights, noBias, FusedActivationType.None, FusedActivationType.None);
        var fused = engine.MlpForward(x, weights, noBias, FusedActivationType.None, act,
            hiddenActivationParams: null, outputActivationParams: p);

        for (int i = 0; i < raw.Length; i++)
        {
            double expected = ParamRef(act, Convert.ToDouble(raw[i]), p);
            double actual = Convert.ToDouble(fused[i]);
            Assert.True(Math.Abs(expected - actual) < 1e-4,
                $"{act} params: MlpForward {actual} != {expected} at {i} (raw={raw[i]}).");
        }
    }

    /// <summary>PReLU applies a per-output-channel (per-column) learned slope.</summary>
    [Fact]
    public void MlpForward_PReLU_AppliesPerChannelSlope()
    {
        var engine = new CpuEngine();
        const int batch = 4, inF = 16, outF = 8;
        var rng = new Random(20260602);
        var wData = new float[inF * outF];
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var w = new Tensor<float>(wData, new[] { inF, outF });
        var xData = new float[batch * inF];
        for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
        var x = new Tensor<float>(xData, new[] { batch, inF });
        var slope = new float[outF];
        for (int j = 0; j < outF; j++) slope[j] = 0.05f + 0.1f * j; // distinct per channel

        var weights = new System.Collections.Generic.List<Tensor<float>> { w };
        var noBias = new System.Collections.Generic.List<Tensor<float>?> { null };

        var raw = engine.MlpForward(x, weights, noBias, FusedActivationType.None, FusedActivationType.None);
        var fused = engine.MlpForward(x, weights, noBias, FusedActivationType.None, FusedActivationType.PReLU,
            outputActivationParams: new FusedActivationParams { PReluSlope = slope });

        for (int i = 0; i < batch; i++)
            for (int j = 0; j < outF; j++)
            {
                double v = Convert.ToDouble(raw[i * outF + j]);
                double expected = v > 0 ? v : slope[j] * v;
                double actual = Convert.ToDouble(fused[i * outF + j]);
                Assert.True(Math.Abs(expected - actual) < 1e-4,
                    $"PReLU col {j}: {actual} != {expected} (raw={v}).");
            }
    }

    /// <summary>Softmax / Softmin normalize across each row (feature dim).</summary>
    [Theory]
    [InlineData(FusedActivationType.Softmax)]
    [InlineData(FusedActivationType.Softmin)]
    public void MlpForward_SoftmaxSoftmin_NormalizesEachRow(FusedActivationType act)
    {
        var engine = new CpuEngine();
        const int batch = 4, inF = 16, outF = 8;
        var rng = new Random(20260603);
        var wData = new float[inF * outF];
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var w = new Tensor<float>(wData, new[] { inF, outF });
        var xData = new float[batch * inF];
        for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
        var x = new Tensor<float>(xData, new[] { batch, inF });

        var weights = new System.Collections.Generic.List<Tensor<float>> { w };
        var noBias = new System.Collections.Generic.List<Tensor<float>?> { null };

        var raw = engine.MlpForward(x, weights, noBias, FusedActivationType.None, FusedActivationType.None);
        var fused = engine.MlpForward(x, weights, noBias, FusedActivationType.None, act);

        bool min = act == FusedActivationType.Softmin;
        for (int i = 0; i < batch; i++)
        {
            // independent reference softmax over the row
            double mx = double.NegativeInfinity;
            for (int j = 0; j < outF; j++) { double v = min ? -raw[i * outF + j] : raw[i * outF + j]; if (v > mx) mx = v; }
            double sum = 0; var exp = new double[outF];
            for (int j = 0; j < outF; j++) { double v = min ? -raw[i * outF + j] : raw[i * outF + j]; exp[j] = Math.Exp(v - mx); sum += exp[j]; }
            double rowSum = 0;
            for (int j = 0; j < outF; j++)
            {
                double expected = exp[j] / sum;
                double actual = Convert.ToDouble(fused[i * outF + j]);
                Assert.True(Math.Abs(expected - actual) < 1e-5,
                    $"{act} row {i} col {j}: {actual} != {expected}.");
                rowSum += actual;
            }
            Assert.True(Math.Abs(rowSum - 1.0) < 1e-4, $"{act} row {i} does not sum to 1 ({rowSum}).");
        }
    }

    /// <summary>Independent reference for the remaining row-wise activations.</summary>
    private static double[] RowwiseRef(FusedActivationType act, double[] r)
    {
        int n = r.Length; var y = new double[n];
        switch (act)
        {
            case FusedActivationType.LogSoftmax:
            {
                double mx = double.NegativeInfinity; foreach (var v in r) if (v > mx) mx = v;
                double s = 0; foreach (var v in r) s += Math.Exp(v - mx);
                double lse = mx + Math.Log(s);
                for (int j = 0; j < n; j++) y[j] = r[j] - lse;
                break;
            }
            case FusedActivationType.LogSoftmin:
            {
                double mx = double.NegativeInfinity; foreach (var v in r) if (-v > mx) mx = -v;
                double s = 0; foreach (var v in r) s += Math.Exp(-v - mx);
                double lse = mx + Math.Log(s);
                for (int j = 0; j < n; j++) y[j] = -r[j] - lse;
                break;
            }
            case FusedActivationType.SphericalSoftmax:
            {
                double ss = 0; foreach (var v in r) ss += v * v; double norm = Math.Sqrt(ss);
                double s = 0; var e = new double[n];
                for (int j = 0; j < n; j++) { e[j] = Math.Exp(r[j] / norm); s += e[j]; }
                for (int j = 0; j < n; j++) y[j] = e[j] / s;
                break;
            }
            case FusedActivationType.TaylorSoftmax:
            {
                double s = 0; var t = new double[n];
                for (int j = 0; j < n; j++) { t[j] = 1 + r[j] + 0.5 * r[j] * r[j]; s += t[j]; }
                for (int j = 0; j < n; j++) y[j] = t[j] / s;
                break;
            }
            case FusedActivationType.GumbelSoftmax: // tau=1 ⇒ softmax(x)
            {
                double mx = double.NegativeInfinity; foreach (var v in r) if (v > mx) mx = v;
                double s = 0; var e = new double[n];
                for (int j = 0; j < n; j++) { e[j] = Math.Exp(r[j] - mx); s += e[j]; }
                for (int j = 0; j < n; j++) y[j] = e[j] / s;
                break;
            }
            case FusedActivationType.Squash:
            {
                double ss = 0; foreach (var v in r) ss += v * v; double norm = Math.Sqrt(ss);
                double k = norm > 0 ? (ss / (1 + ss)) / norm : 0;
                for (int j = 0; j < n; j++) y[j] = r[j] * k;
                break;
            }
            case FusedActivationType.Sparsemax:
            {
                var z = (double[])r.Clone(); Array.Sort(z); Array.Reverse(z); // descending
                double cum = 0; int k = 1; double cumK = 0;
                for (int i = 0; i < n; i++) { cum += z[i]; if (1 + (i + 1) * z[i] > cum) { k = i + 1; cumK = cum; } }
                double tau = (cumK - 1) / k;
                for (int j = 0; j < n; j++) y[j] = Math.Max(0, r[j] - tau);
                break;
            }
            default: throw new ArgumentOutOfRangeException(nameof(act));
        }
        return y;
    }

    [Theory]
    [InlineData(FusedActivationType.LogSoftmax)]
    [InlineData(FusedActivationType.LogSoftmin)]
    [InlineData(FusedActivationType.SphericalSoftmax)]
    [InlineData(FusedActivationType.TaylorSoftmax)]
    [InlineData(FusedActivationType.GumbelSoftmax)]
    [InlineData(FusedActivationType.Sparsemax)]
    [InlineData(FusedActivationType.Squash)]
    public void MlpForward_RowwiseActivation_MatchesReference(FusedActivationType act)
    {
        var engine = new CpuEngine();
        const int batch = 4, inF = 16, outF = 8;
        var rng = new Random(20260604);
        var wData = new float[inF * outF];
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var w = new Tensor<float>(wData, new[] { inF, outF });
        var xData = new float[batch * inF];
        for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
        var x = new Tensor<float>(xData, new[] { batch, inF });

        var weights = new System.Collections.Generic.List<Tensor<float>> { w };
        var noBias = new System.Collections.Generic.List<Tensor<float>?> { null };

        var raw = engine.MlpForward(x, weights, noBias, FusedActivationType.None, FusedActivationType.None);
        var fused = engine.MlpForward(x, weights, noBias, FusedActivationType.None, act);

        for (int i = 0; i < batch; i++)
        {
            var rowRaw = new double[outF];
            for (int j = 0; j < outF; j++) rowRaw[j] = Convert.ToDouble(raw[i * outF + j]);
            var expected = RowwiseRef(act, rowRaw);
            for (int j = 0; j < outF; j++)
            {
                double actual = Convert.ToDouble(fused[i * outF + j]);
                Assert.True(Math.Abs(expected[j] - actual) < 1e-4,
                    $"{act} row {i} col {j}: {actual} != {expected[j]}.");
            }
        }
    }
}
