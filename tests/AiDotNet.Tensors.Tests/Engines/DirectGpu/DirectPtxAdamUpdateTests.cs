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
    public void Emitter_BakesEveryScalarSoTheLaunchAbiIsFourPointers()
    {
        string ptx = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 262_144, Defaults());
        Assert.Contains(".visible .entry aidotnet_fused_adam_update_f32(", ptx);
        Assert.Contains("op=adam-update", ptx);
        // Four pointers and nothing else: no scalar reaches the launch ABI.
        Assert.Equal(4, Count(ptx, "ld.param.u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .f32", ptx, StringComparison.Ordinal);
        // Reads param, gradient, m, v; writes param, m, v.
        Assert.Equal(4, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(3, Count(ptx, "st.global.v4.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void Emitter_RemovesBothPowfCallsByBakingTheBiasCorrections()
    {
        var hyper = Defaults(step: 50);
        string ptx = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, hyper);

        // 1 - beta^step is loop-invariant, so it is a literal, not a computation.
        // Check instructions only: the header comment legitimately says "powf".
        string body = StripComments(ptx);
        Assert.DoesNotContain("pow", body, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("lg2", body, StringComparison.Ordinal);
        Assert.DoesNotContain("ex2", body, StringComparison.Ordinal);
        Assert.Contains($"baked step=50", ptx);

        // Computed in double, rounded once - so it matches the reference value.
        float expected1 = (float)(1.0 - Math.Pow(0.9f, 50));
        float expected2 = (float)(1.0 - Math.Pow(0.999f, 50));
        Assert.Equal(expected1, hyper.BiasCorrection1);
        Assert.Equal(expected2, hyper.BiasCorrection2);
        Assert.Contains($"0f{MathCompat.SingleToUInt32Bits(expected1):X8}", ptx);
        Assert.Contains($"0f{MathCompat.SingleToUInt32Bits(expected2):X8}", ptx);
        // One divide per bias correction per element, plus the final step divide.
        Assert.Equal(12, Count(ptx, "div.rn.f32"));
    }

    [Fact]
    public void Emitter_KeepsTheLeftAssociationOfTheSecondMomentTerm()
    {
        string ptx = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, Defaults());
        string oneMinusBeta2 = "0f" + MathCompat.SingleToUInt32Bits(1f - 0.999f).ToString("X8");
        for (int i = 0; i < 4; i++)
        {
            int g = 4 + i;
            // ((1 - beta2) * grad) * grad, not (1 - beta2) * (grad * grad).
            Assert.Contains($"mul.rn.f32 %f17, {oneMinusBeta2}, %f{g};", ptx);
            Assert.Contains($"mul.rn.f32 %f17, %f17, %f{g};", ptx);
        }
        // sqrt then add epsilon, never the reverse.
        Assert.Equal(4, Count(ptx, "sqrt.rn.f32 %f17, %f17;"));
    }

    [Fact]
    public void Emitter_OnlyEmitsTheWeightDecayTermWhenItIsPositive()
    {
        string without = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, Defaults(weightDecay: 0f));
        string with = PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 65_536, Defaults(weightDecay: 0.01f));
        string decayLiteral = "0f" + MathCompat.SingleToUInt32Bits(0.01f).ToString("X8");

        // The established kernel guards on weightDecay > 0. Because the value is
        // baked, that guard resolves at emit time - so the term is simply absent,
        // with no runtime branch in either build.
        Assert.DoesNotContain(decayLiteral, without);
        Assert.Equal(4, Count(with, $"mul.rn.f32 %f16, {decayLiteral}, %f"));
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
            PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, 1_000, Defaults()));
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
