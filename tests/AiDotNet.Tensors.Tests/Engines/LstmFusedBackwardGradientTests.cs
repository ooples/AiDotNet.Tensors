using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Validates the fused tape-aware LSTM backward (CpuEngine.LstmSequenceBackward.cs)
/// against central finite differences for every differentiable input. The loss is a
/// random-weighted sum of the output so the upstream gradient is non-trivial (not the
/// all-ones a sum-only loss would produce, which can mask gate-coupling errors).
///
/// The finite-difference loss must use the SAME exact-activation forward as the analytic
/// gradient, so every loss evaluation runs under a tape (the no-tape inference path uses
/// approximate FastSigmoid/FastTanh and would not match).
/// </summary>
public class LstmFusedBackwardGradientTests
{
    private static Tensor<float> Rand(int[] shape, Random rng, float scale = 0.4f)
    {
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private static float Loss(CpuEngine eng, Tensor<float> input, Tensor<float>? h0, Tensor<float>? c0,
        Tensor<float> wIh, Tensor<float> wHh, Tensor<float>? bIh, Tensor<float>? bHh,
        Tensor<float> coeff, bool returnSequences)
    {
        using var tape = new GradientTape<float>();
        var outp = eng.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, returnSequences);
        var prod = eng.TensorMultiply(outp, coeff);
        var loss = eng.ReduceSum(prod, null);
        return loss.AsSpan()[0];
    }

    private static void CheckFiniteDiff(string name, CpuEngine eng,
        Tensor<float> param, Tensor<float> analytic,
        Func<float> lossFn, Tensor<float> paramRef)
    {
        const float eps = 1e-2f;
        var pSpan = paramRef.AsWritableSpan();
        var aSpan = analytic.AsSpan();
        float maxAbsErr = 0f, maxRel = 0f;
        for (int i = 0; i < pSpan.Length; i++)
        {
            float orig = pSpan[i];
            pSpan[i] = orig + eps; float lp = lossFn();
            pSpan[i] = orig - eps; float lm = lossFn();
            pSpan[i] = orig;
            float num = (lp - lm) / (2 * eps);
            float ana = aSpan[i];
            float absErr = Math.Abs(num - ana);
            float rel = absErr / (1e-3f + Math.Abs(num) + Math.Abs(ana));
            maxAbsErr = Math.Max(maxAbsErr, absErr);
            maxRel = Math.Max(maxRel, rel);
        }
        // float FD is noisy; pass if EITHER absolute or relative error is small per the worst element.
        Assert.True(maxAbsErr < 2e-2f || maxRel < 2e-2f,
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

        Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            var outp = eng.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, returnSequences);
            var loss = eng.ReduceSum(eng.TensorMultiply(outp, coeff), null);
            grads = tape.ComputeGradients(loss, new[] { input, wIh, wHh, bIh, bHh, h0, c0 });
        }

        Func<float> L = () => Loss(eng, input, h0, c0, wIh, wHh, bIh, bHh, coeff, returnSequences);
        CheckFiniteDiff("input", eng, input, grads[input], L, input);
        CheckFiniteDiff("wIh", eng, wIh, grads[wIh], L, wIh);
        CheckFiniteDiff("wHh", eng, wHh, grads[wHh], L, wHh);
        CheckFiniteDiff("bIh", eng, bIh, grads[bIh], L, bIh);
        CheckFiniteDiff("bHh", eng, bHh, grads[bHh], L, bHh);
        CheckFiniteDiff("h0", eng, h0, grads[h0], L, h0);
        CheckFiniteDiff("c0", eng, c0, grads[c0], L, c0);
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

        Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            var outp = eng.LstmSequenceForward(input, null, null, wIh, wHh, null, null, returnSequences: true);
            var loss = eng.ReduceSum(eng.TensorMultiply(outp, coeff), null);
            grads = tape.ComputeGradients(loss, new[] { input, wIh, wHh });
        }

        Func<float> L = () => Loss(eng, input, null, null, wIh, wHh, null, null, coeff, true);
        CheckFiniteDiff("input", eng, input, grads[input], L, input);
        CheckFiniteDiff("wIh", eng, wIh, grads[wIh], L, wIh);
        CheckFiniteDiff("wHh", eng, wHh, grads[wHh], L, wHh);
    }
}
