using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32 Adam update (issue #848). Adam
/// mutates parameters and both moment tensors in place, so these assertions pin
/// the emitted arithmetic to the established adam_update kernel term for term
/// and pin the domain guards that protect the baked scalars.
/// </summary>
public class DirectPtxAdamUpdateTests
{
    private static DirectPtxAdamHyperparameters Defaults(
        float weightDecay = 0f, int step = 1) =>
        new(learningRate: 1e-3f, beta1: 0.9f, beta2: 0.999f,
            epsilon: 1e-8f, weightDecay: weightDecay, step: step);

    [Fact]
    public void Emitter_TakesScalarsAsLaunchParamsSoTheModuleKeyIsShapeOnly()
    {
        string ptx = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 262_144, hasWeightDecay: false);
        Assert.Contains(".visible .entry aidotnet_fused_adam_update_f32(", ptx);
        Assert.Contains("op=adam-update", ptx);

        // Four pointers plus eight scalars. The scalars used to be baked, which
        // put the step counter in the module key and forced a fresh JIT on every
        // training step; as launch parameters one module serves a whole run.
        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.Equal(8, Count(ptx, ".param .f32"));
        Assert.Contains("ld.param.f32 %f20, [lr_over_bc1];", ptx);
        Assert.Contains("ld.param.f32 %f25, [rsqrt_bc2];", ptx);

        // Reads param, gradient, m, v; writes param, m, v.
        Assert.Equal(4, Count(ptx, "ld.global."));
        Assert.Equal(3, Count(ptx, "st.global.v4.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void Emitter_IsIdenticalAcrossStepsAndLearningRates()
    {
        // The whole point of moving the scalars out: nothing that changes during
        // training can change the module. Same shape and decay presence => byte
        // identical PTX, so the cache holds exactly one module per shape.
        string a = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, hasWeightDecay: false);
        string b = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, hasWeightDecay: false);
        Assert.Equal(a, b);
        // Only the shape and the decay presence may change it.
        Assert.NotEqual(a, PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, hasWeightDecay: true));
        Assert.NotEqual(a, PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 262_144, hasWeightDecay: false));
    }

    [Fact]
    public void Emitter_LeavesExactlyOneDividePerElement()
    {
        string ptx = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, hasWeightDecay: false);

        // The reference divides three times per element: m/bc1, v/bc2, and the
        // final step. Both bias-correction divides are loop-invariant, so they
        // fold into host-precomputed multipliers - (lr/bc1)*m and
        // sqrt(v)*rsqrt(bc2) - leaving one genuine divide.
        Assert.Equal(4, Count(ptx, "div.rn.f32"));          // 4 elements, 1 each
        Assert.Equal(4, Count(ptx, "mul.rn.f32 %f16, %f20,"));   // (lr/bc1) * m
        // The denominator is a single fma: sqrt(v) * rsqrt(bc2) + eps.
        Assert.Equal(4, Count(ptx, "fma.rn.f32 %f17, %f17, %f25, %f26;"));
        Assert.Equal(4, Count(ptx, "sqrt.rn.f32"));
        // No powf and no on-device exp/log substitute for it.
        string body = StripComments(ptx);
        Assert.DoesNotContain("pow", body, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("lg2", body, StringComparison.Ordinal);
        Assert.DoesNotContain("ex2", body, StringComparison.Ordinal);
    }

    [Fact]
    public void Emitter_ReadsTheGradientThroughTheReadOnlyCache()
    {
        string ptx = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, hasWeightDecay: false);
        // param, m and v are read-modify-write and stay on the cached path; the
        // gradient is read once and never revisited, so it goes through the
        // read-only data cache and leaves L1 for the three resident tensors.
        Assert.Equal(1, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Equal(3, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Contains("ld.global.nc.v4.f32 {%f4, %f5, %f6, %f7}, [%rd7];", ptx);
    }

    [Fact]
    public void Emitter_KeepsTheLeftAssociationOfTheSecondMomentTerm()
    {
        string ptx = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, hasWeightDecay: false);
        for (int i = 0; i < 4; i++)
        {
            int g = 4 + i;
            // ((1 - beta2) * grad) * grad, not (1 - beta2) * (grad * grad).
            // %f24 carries the one-minus-beta2 launch parameter.
            Assert.Contains($"mul.rn.f32 %f17, %f24, %f{g};", ptx);
            Assert.Contains($"mul.rn.f32 %f17, %f17, %f{g};", ptx);
        }
    }

    [Fact]
    public void Emitter_OnlyEmitsTheWeightDecayTermWhenItIsRequested()
    {
        string without = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, hasWeightDecay: false);
        string with = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, hasWeightDecay: true);

        // Decay presence stays a MODULE property rather than a runtime branch,
        // so the decay-free module carries no extra instruction and neither
        // module branches. That keeps the key space finite - two modules per
        // shape - which is what lets these be precompiled.
        Assert.Equal(4, Count(with, "mul.rn.f32 %f16, %f27,"));
        Assert.DoesNotContain("%f27", without);
        Assert.DoesNotContain("bra", with, StringComparison.Ordinal);
        Assert.True(with.Length > without.Length);
    }

    [Fact]
    public void Hyperparameters_RejectValuesThatWouldPoisonEveryCachedUpdate()
    {
        // Epsilon is the only thing keeping the denominator away from zero.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DirectPtxAdamHyperparameters(1e-3f, 0.9f, 0.999f, 0f, 0f, 1).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DirectPtxAdamHyperparameters(1e-3f, 0.9f, 0.999f, -1e-8f, 0f, 1).Validate());
        // Betas must be strictly inside (0, 1) or the bias correction degenerates.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DirectPtxAdamHyperparameters(1e-3f, 1f, 0.999f, 1e-8f, 0f, 1).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DirectPtxAdamHyperparameters(1e-3f, 0.9f, 0f, 1e-8f, 0f, 1).Validate());
        // Step 0 leaves both bias corrections at zero -> divide by zero.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DirectPtxAdamHyperparameters(1e-3f, 0.9f, 0.999f, 1e-8f, 0f, 0).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DirectPtxAdamHyperparameters(float.NaN, 0.9f, 0.999f, 1e-8f, 0f, 1).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DirectPtxAdamHyperparameters(1e-3f, 0.9f, 0.999f, 1e-8f, -0.1f, 1).Validate());
        // The documented-good configuration must survive.
        Defaults(weightDecay: 0.01f, step: 7).Validate();
    }

    [Fact]
    public void Hyperparameters_KeyOnExactBitsSoDistinctStepsNeverShareAModule()
    {
        var a = Defaults(step: 1);
        var b = Defaults(step: 2);
        Assert.NotEqual(a, b);
        Assert.Equal(Defaults(step: 3), Defaults(step: 3));
        Assert.Equal(Defaults(step: 3).GetHashCode(), Defaults(step: 3).GetHashCode());
        // Two learning rates that differ only in the last ulp are distinct keys.
        var lo = new DirectPtxAdamHyperparameters(1e-3f, 0.9f, 0.999f, 1e-8f, 0f, 1);
        var hi = new DirectPtxAdamHyperparameters(
            MathCompat.Int32BitsToSingle(MathCompat.SingleToInt32Bits(1e-3f) + 1),
            0.9f, 0.999f, 1e-8f, 0f, 1);
        Assert.NotEqual(lo, hi);
    }

    [Fact]
    public void ShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedAdamUpdateF32Kernel.IsSupportedShape(65_536));
        Assert.True(PtxFusedAdamUpdateF32Kernel.IsSupportedShape(4_194_304));
        Assert.False(PtxFusedAdamUpdateF32Kernel.IsSupportedShape(65_535));
        Assert.False(PtxFusedAdamUpdateF32Kernel.IsPromotedShape(65_536));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 1_000, hasWeightDecay: false));
    }

    [Fact]
    public void ArchitectureGate_FailsClosedOutsideSm86()
    {
        Assert.True(DirectPtxArchitecture.HasValidatedAdamUpdate(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedAdamUpdate(8, 0));
        Assert.False(DirectPtxArchitecture.HasValidatedAdamUpdate(8, 7));
        Assert.False(DirectPtxArchitecture.HasValidatedAdamUpdate(8, 9));
        Assert.False(DirectPtxArchitecture.HasValidatedAdamUpdate(9, 0));
    }

    /// <summary>Drops // comment lines so assertions test emitted instructions.</summary>
    private static string StripComments(string ptx)
    {
        var builder = new System.Text.StringBuilder(ptx.Length);
        foreach (string line in ptx.Split('\n'))
            if (!line.TrimStart().StartsWith("//", StringComparison.Ordinal))
                builder.Append(line).Append('\n');
        return builder.ToString();
    }

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }
}
