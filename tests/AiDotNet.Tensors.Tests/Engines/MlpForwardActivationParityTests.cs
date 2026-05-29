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
}
