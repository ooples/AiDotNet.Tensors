using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Validates the fused tape-aware <b>double</b> LSTM backward
/// (CpuEngine.LstmSequenceForwardDoubleTrain + LstmSequenceBackwardDouble, the #478 follow-up that
/// lets a &lt;double&gt; LSTM train through ONE fused BPTT node instead of throwing) against central
/// finite differences for every differentiable input. Same structure as the float gradient test
/// (<see cref="LstmFusedBackwardGradientTests"/>) but with a much tighter tolerance — double FD with
/// central differences is accurate to ~1e-10, so a gate-coupling or transpose error in the BPTT
/// would show up far outside the threshold.
///
/// The loss is a random-weighted sum of the output (non-trivial upstream gradient), and every loss
/// evaluation runs under a tape so the FD forward uses the SAME exact-activation fused path as the
/// analytic gradient.
/// </summary>
public class LstmFusedBackwardGradientDoubleTests
{
    private static Tensor<double> Rand(int[] shape, Random rng, double scale = 0.4)
    {
        var t = new Tensor<double>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private static double Loss(CpuEngine eng, Tensor<double> input, Tensor<double>? h0, Tensor<double>? c0,
        Tensor<double> wIh, Tensor<double> wHh, Tensor<double>? bIh, Tensor<double>? bHh,
        Tensor<double> coeff, bool returnSequences)
    {
        using var tape = new GradientTape<double>();
        var outp = eng.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, returnSequences);
        var prod = eng.TensorMultiply(outp, coeff);
        var loss = eng.ReduceSum(prod, null);
        return loss.AsSpan()[0];
    }

    private static void CheckFiniteDiff(string name,
        Tensor<double> analytic, Func<double> lossFn, Tensor<double> paramRef)
    {
        const double eps = 1e-5;
        var pSpan = paramRef.AsWritableSpan();
        var aSpan = analytic.AsSpan();
        double maxAbsErr = 0.0, maxRel = 0.0;
        for (int i = 0; i < pSpan.Length; i++)
        {
            double orig = pSpan[i];
            pSpan[i] = orig + eps; double lp = lossFn();
            pSpan[i] = orig - eps; double lm = lossFn();
            pSpan[i] = orig;
            double num = (lp - lm) / (2 * eps);
            double ana = aSpan[i];
            double absErr = Math.Abs(num - ana);
            double rel = absErr / (1e-9 + Math.Abs(num) + Math.Abs(ana));
            maxAbsErr = Math.Max(maxAbsErr, absErr);
            maxRel = Math.Max(maxRel, rel);
        }
        // double central-difference FD is clean; require small absolute OR relative error per worst element.
        Assert.True(maxAbsErr < 1e-6 || maxRel < 1e-6,
            $"{name}: gradient mismatch vs finite-difference. maxAbsErr={maxAbsErr:E3}, maxRel={maxRel:E3}");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Gradients_MatchFiniteDifference_WithBiasesAndInitialStates(bool returnSequences)
    {
        var eng = new CpuEngine();
        var rng = new Random(20260611);
        int batch = 2, seq = 3, inF = 4, hidden = 5, G = 4 * hidden;

        var input = Rand(new[] { batch, seq, inF }, rng);
        var wIh = Rand(new[] { G, inF }, rng);
        var wHh = Rand(new[] { G, hidden }, rng);
        var bIh = Rand(new[] { G }, rng);
        var bHh = Rand(new[] { G }, rng);
        var h0 = Rand(new[] { batch, hidden }, rng);
        var c0 = Rand(new[] { batch, hidden }, rng);
        var coeff = Rand(returnSequences ? new[] { batch, seq, hidden } : new[] { batch, hidden }, rng);

        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var outp = eng.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, returnSequences);
            var loss = eng.ReduceSum(eng.TensorMultiply(outp, coeff), null);
            grads = tape.ComputeGradients(loss, new[] { input, wIh, wHh, bIh, bHh, h0, c0 });
        }

        Func<double> L = () => Loss(eng, input, h0, c0, wIh, wHh, bIh, bHh, coeff, returnSequences);
        CheckFiniteDiff("input", grads[input], L, input);
        CheckFiniteDiff("wIh", grads[wIh], L, wIh);
        CheckFiniteDiff("wHh", grads[wHh], L, wHh);
        CheckFiniteDiff("bIh", grads[bIh], L, bIh);
        CheckFiniteDiff("bHh", grads[bHh], L, bHh);
        CheckFiniteDiff("h0", grads[h0], L, h0);
        CheckFiniteDiff("c0", grads[c0], L, c0);
    }

    [Fact]
    public void Gradients_MatchFiniteDifference_NoBiasesNoInitialStates()
    {
        var eng = new CpuEngine();
        var rng = new Random(7);
        int batch = 3, seq = 4, inF = 3, hidden = 4, G = 4 * hidden;

        var input = Rand(new[] { batch, seq, inF }, rng);
        var wIh = Rand(new[] { G, inF }, rng);
        var wHh = Rand(new[] { G, hidden }, rng);
        var coeff = Rand(new[] { batch, seq, hidden }, rng);

        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var outp = eng.LstmSequenceForward(input, null, null, wIh, wHh, null, null, returnSequences: true);
            var loss = eng.ReduceSum(eng.TensorMultiply(outp, coeff), null);
            grads = tape.ComputeGradients(loss, new[] { input, wIh, wHh });
        }

        Func<double> L = () => Loss(eng, input, null, null, wIh, wHh, null, null, coeff, true);
        CheckFiniteDiff("input", grads[input], L, input);
        CheckFiniteDiff("wIh", grads[wIh], L, wIh);
        CheckFiniteDiff("wHh", grads[wHh], L, wHh);
    }
}
