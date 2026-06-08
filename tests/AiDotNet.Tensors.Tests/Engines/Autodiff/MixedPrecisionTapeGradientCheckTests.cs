using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Phase-1 end-to-end gate for industry-standard mixed-dtype autograd (Tensors #555,
/// docs/fp16-activation-storage-design.md). Builds a TWO-segment interleaved AMP graph —
/// FP32 input → cast → FP16 op → cast → FP32 interior → cast → FP16 op → cast → FP32 loss —
/// and verifies the gradients that flow ACROSS both cast boundaries (and through an FP32 interior
/// node that is itself only consumed by a down-cast) match the closed-form analytic gradient.
///
/// All inputs are chosen exactly FP16-representable (small integers / halves), so every cast is
/// lossless and the analytic gradient is the EXACT target — a defect in the cross-tape bridge or the
/// seeded backward shows up as a real mismatch, not tolerance slop. This is the multi-layer case the
/// two-pass shortcut cannot handle: grad at the FP32 input depends on the FP16 backward of the FIRST
/// segment, which is only reachable after the FP16 grad of the SECOND segment bridges back into the
/// FP32 tape and re-seeds it — i.e. it exercises the Gauss-Seidel sweep, not a single pass.
/// </summary>
public class MixedPrecisionTapeGradientCheckTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Vec(params float[] v) => new(v, new[] { v.Length });

    [Fact]
    public void TwoSegment_Interleaved_MixedPrecision_Backward_MatchesAnalytic()
    {
        // loss = sum_i( x_i * W1_i * c1_i * W2_i ), built as:
        //   d1 = cast16(x); h1 = d1 * W1 (fp16); y1 = cast32(h1)
        //   a1 = y1 * c1 (fp32, consumed ONLY by the next down-cast)
        //   d2 = cast16(a1); h2 = d2 * W2 (fp16); y2 = cast32(h2)
        //   loss = sum(y2)
        // All values FP16-exact ⇒ analytic grads are exact:
        //   dL/dx_i  = W1_i * c1_i * W2_i
        //   dL/dW1_i = x_i  * c1_i * W2_i
        //   dL/dW2_i = x_i  * W1_i * c1_i
        var x  = Vec(1f, 2f, 0.5f, -1f);
        var c1 = Vec(1f, 0.5f, 2f, -2f);

        // FP16 params (leaves on the FP16 sub-tape).
        var W1 = new Tensor<Half>(new[] { (Half)2f, (Half)(-1f), (Half)0.5f, (Half)1f }, new[] { 4 });
        var W2 = new Tensor<Half>(new[] { (Half)0.5f, (Half)2f, (Half)(-1f), (Half)1f }, new[] { 4 });

        MixedPrecisionTape.MixedGrads grads;
        using (var mp = new MixedPrecisionTape())
        {
            var d1 = mp.CastToFp16(x);
            var h1 = _engine.TensorMultiply(d1, W1);   // FP16 tape
            var y1 = mp.CastToFp32(h1);
            var a1 = _engine.TensorMultiply(y1, c1);   // FP32 tape, only consumed by the down-cast below
            var d2 = mp.CastToFp16(a1);
            var h2 = _engine.TensorMultiply(d2, W2);   // FP16 tape
            var y2 = mp.CastToFp32(h2);
            var loss = _engine.ReduceSum(y2);          // FP32 scalar
            grads = mp.ComputeGradients(loss);
        }

        // Analytic targets.
        var xa = x.ToArray(); var c1a = c1.ToArray();
        var w1a = W1.ToArray(); var w2a = W2.ToArray();
        var expGx = new float[4]; var expGw1 = new float[4]; var expGw2 = new float[4];
        for (int i = 0; i < 4; i++)
        {
            float w1 = (float)w1a[i], w2 = (float)w2a[i];
            expGx[i]  = w1 * c1a[i] * w2;
            expGw1[i] = xa[i] * c1a[i] * w2;
            expGw2[i] = xa[i] * w1 * c1a[i];
        }

        Assert.True(grads.Fp32.TryGetValue(x, out var gx), "no gradient produced for FP32 input x");
        Assert.True(grads.Fp16.TryGetValue(W1, out var gW1), "no gradient produced for FP16 param W1 (first segment — needs the full sweep)");
        Assert.True(grads.Fp16.TryGetValue(W2, out var gW2), "no gradient produced for FP16 param W2");

        AssertClose(expGx, gx.ToArray(), "dL/dx");
        AssertCloseHalf(expGw1, gW1.ToArray(), "dL/dW1");
        AssertCloseHalf(expGw2, gW2.ToArray(), "dL/dW2");
    }

    private static void AssertClose(float[] expected, float[] got, string name)
    {
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(got[i]) || float.IsInfinity(got[i]), $"{name}[{i}] non-finite");
            // FP16-exact inputs ⇒ exact; allow a hair for FP32 accumulate order.
            Assert.True(Math.Abs(got[i] - expected[i]) <= 1e-4f + 1e-3f * Math.Abs(expected[i]),
                $"{name}[{i}] mismatch: expected {expected[i]}, got {got[i]}");
        }
    }

    private static void AssertCloseHalf(float[] expected, Half[] got, string name)
    {
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float g = (float)got[i];
            Assert.False(float.IsNaN(g) || float.IsInfinity(g), $"{name}[{i}] non-finite");
            Assert.True(Math.Abs(g - expected[i]) <= 1e-3f + 2e-3f * Math.Abs(expected[i]),
                $"{name}[{i}] mismatch: expected {expected[i]}, got {g}");
        }
    }
}
