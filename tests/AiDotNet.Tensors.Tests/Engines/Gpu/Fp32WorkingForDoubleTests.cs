using System;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Stage 9 (#415) parity tests for MixedPrecisionContext FP32-working /
/// FP64-master mode scaffolding. Verifies:
///  - the flag is silently set false for non-double T (cannot accidentally
///    flip behaviour for FP32 / FP16 / BF16 callers)
///  - CastDoubleToFloat / CastFloatToDouble round-trip preserves all
///    representable values within FP32's mantissa range
///  - GradientCosineSimilarity returns 1.0 for identical vectors,
///    monotonically lower for divergent ones
/// </summary>
public class Fp32WorkingForDoubleTests
{
    [Fact]
    public void Flag_HonoredForDouble_IgnoredForFloat()
    {
        using var ctxD = new MixedPrecisionContext<double>(
            defaultPrecision: PrecisionMode.Float32,
            fp32WorkingForDouble: true);
        Assert.True(ctxD.Fp32WorkingForDouble);

        using var ctxF = new MixedPrecisionContext<float>(
            defaultPrecision: PrecisionMode.Float16,
            fp32WorkingForDouble: true);
        // Silently set false because T != double — we don't want the flag
        // to accidentally flip the FP32-working / FP16-master path callers
        // may already rely on.
        Assert.False(ctxF.Fp32WorkingForDouble);
    }

    [Fact]
    public void CastDoubleToFloat_PreservesValuesWithinFp32Range()
    {
        var src = new double[] { 0.0, 1.0, -1.0, 1.5e10, -1.5e10, Math.PI, Math.E, 1e-30 };
        var dst = new float[src.Length];
        MixedPrecisionContext<double>.CastDoubleToFloat(src, dst);
        for (int i = 0; i < src.Length; i++)
        {
            float expected = (float)src[i];
            Assert.Equal(expected, dst[i]);
        }
    }

    [Fact]
    public void CastFloatToDouble_RoundTripsExactlyForFp32Values()
    {
        var src = new float[] { 0f, 1f, -1f, 1.5e10f, -1.5e10f, MathF.PI, MathF.E, 1e-30f };
        var dst = new double[src.Length];
        MixedPrecisionContext<double>.CastFloatToDouble(src, dst);
        for (int i = 0; i < src.Length; i++)
            Assert.Equal((double)src[i], dst[i]);
    }

    [Fact]
    public void CastDoubleToFloat_LengthMismatchThrows()
    {
        var src = new double[3];
        var dst = new float[4];
        Assert.Throws<ArgumentException>(() =>
            MixedPrecisionContext<double>.CastDoubleToFloat(src, dst));
    }

    [Fact]
    public void GradientCosineSimilarity_IdenticalReturnsOne()
    {
        var a = new double[] { 1, 2, 3, 4, 5 };
        var sim = MixedPrecisionContext<double>.GradientCosineSimilarity(a, a);
        Assert.True(Math.Abs(sim - 1.0) < 1e-12, $"sim={sim:F14}");
    }

    [Fact]
    public void GradientCosineSimilarity_OppositeReturnsNegativeOne()
    {
        var a = new double[] { 1, 2, 3 };
        var b = new double[] { -1, -2, -3 };
        var sim = MixedPrecisionContext<double>.GradientCosineSimilarity(a, b);
        Assert.True(Math.Abs(sim - (-1.0)) < 1e-12, $"sim={sim:F14}");
    }

    [Fact]
    public void GradientCosineSimilarity_NearlyAlignedSmallDriftHighSim()
    {
        // FP32-working drift simulation: small per-element perturbation,
        // overall direction preserved.
        var rng = new Random(42);
        int n = 1000;
        var refGrad = new double[n];
        var fp32Grad = new double[n];
        for (int i = 0; i < n; i++)
        {
            refGrad[i] = rng.NextDouble() - 0.5;
            // ~1e-6 relative perturbation — what an FP32 forward+backward
            // would produce vs FP64 reference.
            fp32Grad[i] = refGrad[i] * (1.0 + (rng.NextDouble() - 0.5) * 2e-6);
        }
        var sim = MixedPrecisionContext<double>.GradientCosineSimilarity(refGrad, fp32Grad);
        Assert.True(sim > 0.999999, $"sim={sim:F14} (expected > 0.999999 for ~1e-6 drift)");
    }
}
