using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Phase-1 gradient-bridge keystone gate for industry-standard mixed precision (Tensors #555,
/// docs/fp16-activation-storage-design.md). A cross-type cast has an identity local Jacobian, so the
/// forward round-trips to within FP16 rounding AND the gradient bridge round-trips to within FP16
/// rounding (the straight-through contract PyTorch uses). These are the invariants every later mixed-
/// precision op relies on; if the cast bridge drifts, ALL FP16 gradients would be wrong. A finite-
/// difference test is deliberately NOT used here — a down-cast is a rounding staircase (non-smooth),
/// so the correct gate is round-trip consistency, not numeric differentiation through the rounding.
/// </summary>
[Collection(AiDotNet.Tensors.Tests.Engines.Compilation.MixedPrecisionTestCollection.Name)]
public class MixedPrecisionCastTests
{
    private static Tensor<float> Rand(int n, int seed, double scale)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(d, new[] { n });
    }

    private const float Fp16Rel = 1.0f / 1024f; // ~2^-10, FP16 mantissa is 10 bits

    [Fact]
    public void Forward_RoundTrip_FP32_FP16_FP32_IsIdentityWithinFp16()
    {
        var x = Rand(257, 7, 4.0); // mix of magnitudes within FP16 normal range
        var back = MixedPrecisionCast.CastToFp32(MixedPrecisionCast.CastToFp16(x));

        var xa = x.ToArray();
        var ba = back.ToArray();
        Assert.Equal(xa.Length, ba.Length);
        for (int i = 0; i < xa.Length; i++)
        {
            Assert.False(float.IsNaN(ba[i]) || float.IsInfinity(ba[i]), $"non-finite at {i}");
            float denom = Math.Max(1e-3f, Math.Abs(xa[i]));
            Assert.True(Math.Abs(ba[i] - xa[i]) / denom <= Fp16Rel + 1e-6f,
                $"round-trip drift at {i}: {xa[i]} -> {ba[i]}");
        }
    }

    [Fact]
    public void Backward_Bridge_RoundTrips_Within_Fp16()
    {
        // A gradient that flows FP32 -> (down to FP16 grad space) -> (up to FP32) must return to itself
        // within FP16 rounding: CastToFp16Backward(CastToFp32Backward(g)) ≈ g. This is exactly the
        // path a gradient takes across an FP32-op -> FP16-op boundary and back.
        var g = Rand(257, 11, 2.0);
        var downThenUp = MixedPrecisionCast.CastToFp16Backward(
                             MixedPrecisionCast.CastToFp32Backward(g));

        var ga = g.ToArray();
        var ra = downThenUp.ToArray();
        for (int i = 0; i < ga.Length; i++)
        {
            Assert.False(float.IsNaN(ra[i]) || float.IsInfinity(ra[i]), $"non-finite grad at {i}");
            float denom = Math.Max(1e-3f, Math.Abs(ga[i]));
            Assert.True(Math.Abs(ra[i] - ga[i]) / denom <= Fp16Rel + 1e-6f,
                $"grad bridge drift at {i}: {ga[i]} -> {ra[i]}");
        }
    }

    [Fact]
    public void Backward_Cast_PreservesShape_And_IsExactIdentityOnRepresentableValues()
    {
        // On values already exactly representable in FP16 (small integers), the bridge is EXACT both
        // ways — the only error source is FP16 rounding, which these inputs avoid. Guards against a
        // wrong-axis / wrong-length cast (a shape bug would surface here, not just a tolerance miss).
        var g = new Tensor<float>(new float[] { 0f, 1f, -1f, 2f, -4f, 8f, 0.5f, -0.25f }, new[] { 8 });
        var roundTrip = MixedPrecisionCast.CastToFp16Backward(MixedPrecisionCast.CastToFp32Backward(g));
        Assert.Equal(g.Shape.ToArray(), roundTrip.Shape.ToArray());
        var ga = g.ToArray();
        var ra = roundTrip.ToArray();
        for (int i = 0; i < ga.Length; i++)
            Assert.Equal(ga[i], ra[i]); // exact — these values are FP16-representable
    }
}
