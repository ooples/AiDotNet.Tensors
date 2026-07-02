using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Phase-3 gate: dynamic loss scaling over the mixed-dtype tape (Tensors #555,
/// docs/fp16-activation-storage-design.md; Micikevicius et al. AMP). A gradient that is ~1e-8 where it
/// crosses into the FP16 sub-tape underflows FP16 (smallest subnormal ≈ 6e-8) and flushes to ZERO —
/// silently killing the update. <see cref="GradScaler"/> multiplies the backward seed so the FP16-
/// resident grad stays representable, then the result is unscaled in FP32. This test shows BOTH the
/// failure (unscaled → 0) and the recovery (scaled → correct), which is the whole point of loss scaling.
/// </summary>
[Collection(AiDotNet.Tensors.Tests.Engines.Compilation.MixedPrecisionTestCollection.Name)]
public class MixedPrecisionLossScalingTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private static Tensor<float> Vec(params float[] v) => new(v, new[] { v.Length });

    // loss = sum_i( x_i * W_i * t )  with a tiny FP32 multiplier t downstream of the up-cast.
    //   dL/dW_i = x_i * t   (this gradient lives in the FP16 sub-tape — it is what underflows)
    private (Tensor<float> loss, Tensor<float> x, Tensor<Half> W, Tensor<float> t) Build(MixedPrecisionTape mp)
    {
        var x = Vec(2f, 2f, 2f, 2f);
        var W = new Tensor<Half>(new[] { (Half)1f, (Half)1f, (Half)1f, (Half)1f }, new[] { 4 });
        var t = Vec(1e-8f, 1e-8f, 1e-8f, 1e-8f);

        var d = mp.CastToFp16(x);
        var m = _engine.TensorMultiply(d, W);    // FP16
        var y = mp.CastToFp32(m);
        var ys = _engine.TensorMultiply(y, t);   // FP32, scales the loss gradient down to ~1e-8
        var loss = _engine.ReduceSum(ys);
        return (loss, x, W, t);
    }

    [SkippableFact]
    public void Fp16ResidentGradient_Underflows_WithoutLossScaling()
    {
#if !NET5_0_OR_GREATER
        Skip.If(true, "The net471 Half compatibility shim stores float values, so IEEE FP16 underflow is not observable.");
#endif
        using var mp = new MixedPrecisionTape();
        var (loss, _, W, _) = Build(mp);
        var grads = mp.ComputeGradients(loss);

        Assert.True(grads.Fp16.TryGetValue(W, out var gW));
        // True dL/dW = x*t = 2e-8, but it underflows FP16 to exactly zero on the unscaled path.
        foreach (var g in gW.ToArray())
            Assert.Equal(0f, (float)g);
    }

    [Fact]
    public void LossScaling_Recovers_TheUnderflowingGradient()
    {
        using var mp = new MixedPrecisionTape();
        var (loss, x, W, _) = Build(mp);

        // Fixed scale 2^16 = 65536: 1e-8 * 65536 ≈ 6.6e-4, comfortably inside FP16's normal range.
        var scaler = new GradScaler(new MixedPrecisionConfig { LossScale = 65536f, DynamicLossScale = false });
        var grads = mp.ComputeGradients(loss, scaler);

        Assert.False(grads.FoundInfNan, "no overflow expected at this scale");

        Assert.True(grads.Fp16Params.TryGetValue(W, out var gW), "no FP16 param grad for W");
        foreach (var g in gW.ToArray())
        {
            Assert.False(float.IsNaN(g) || float.IsInfinity(g));
            // Recovered dL/dW = x*t = 2e-8 (was 0 unscaled). Allow FP16 rounding of the scaled intermediate.
            Assert.True(Math.Abs(g - 2e-8f) <= 2e-10f + 2e-2f * 2e-8f, $"expected ~2e-8, got {g}");
        }

        // FP32 input gradient dL/dx = W*t = 1e-8, also recovered through the bridge.
        Assert.True(grads.Fp32.TryGetValue(x, out var gx), "no FP32 grad for x");
        foreach (var g in gx.ToArray())
            Assert.True(Math.Abs(g - 1e-8f) <= 2e-10f + 2e-2f * 1e-8f, $"expected ~1e-8, got {g}");
    }
}
