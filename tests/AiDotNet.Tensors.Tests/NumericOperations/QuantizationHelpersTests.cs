using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.NumericOperations;

/// <summary>
/// Tests for issue #207 B1 — <see cref="QuantizationHelpers"/>:
/// quantize / dequantize round trip for int4 (per-group) and int1
/// (BitNet absmean scale).
/// </summary>
public class QuantizationHelpersTests
{
    // ──────────── Int4 round trip ────────────

    [Fact]
    public void QuantizeInt4_RoundTrip_WithinPerGroupQuantError()
    {
        var rng = new Random(0xDE);
        int n = 64;
        var src = new float[n];
        for (int i = 0; i < n; i++) src[i] = (float)(rng.NextDouble() * 4 - 2); // [-2, 2]

        var packed = new PackedInt4[(n + 1) / 2];
        var scale = QuantizationHelpers.QuantizeInt4(src, packed, groupSize: 32);
        Assert.Equal(32, scale.GroupSize);
        Assert.Equal(2, scale.Scales.Length); // 64 / 32 = 2 groups

        var restored = new float[n];
        QuantizationHelpers.DequantizeInt4(packed, scale, restored);

        // Error bound: each group's step = absmax/7, so per-element error
        // is at most step/2 = absmax/14. That's ABSOLUTE error, not
        // relative — small values in a group with a large absmax can
        // suffer ~7% of the group absmax.
        for (int groupStart = 0; groupStart < n; groupStart += 32)
        {
            float groupAbsMax = 0f;
            int groupEnd = Math.Min(groupStart + 32, n);
            for (int i = groupStart; i < groupEnd; i++)
                groupAbsMax = Math.Max(groupAbsMax, Math.Abs(src[i]));
            float bound = groupAbsMax / 14f + 1e-5f;
            for (int i = groupStart; i < groupEnd; i++)
            {
                float err = Math.Abs(restored[i] - src[i]);
                Assert.True(err <= bound,
                    $"[{i}] src {src[i]} restored {restored[i]} err {err} bound {bound}");
            }
        }
    }

    [Fact]
    public void QuantizeInt4_PreservesZero()
    {
        var src = new float[] { 0f, 0f, 0f, 0f };
        var packed = new PackedInt4[2];
        var scale = QuantizationHelpers.QuantizeInt4(src, packed, groupSize: 4);
        var restored = new float[4];
        QuantizationHelpers.DequantizeInt4(packed, scale, restored);
        foreach (var v in restored) Assert.Equal(0f, v);
    }

    [Fact]
    public void QuantizeInt4_OddGroupSize_Throws()
    {
        var src = new float[16];
        var dst = new PackedInt4[8];
        Assert.Throws<ArgumentException>(() =>
            QuantizationHelpers.QuantizeInt4(src, dst, groupSize: 5));
    }

    [Fact]
    public void QuantizeInt4_DstTooSmall_Throws()
    {
        var src = new float[16];
        var dst = new PackedInt4[4]; // need 8
        Assert.Throws<ArgumentException>(() =>
            QuantizationHelpers.QuantizeInt4(src, dst, groupSize: 8));
    }

    [Fact]
    public void QuantizeInt4_SaturatesLargeValues()
    {
        // Ensure extreme magnitudes don't wrap — they should clamp to ±7×scale.
        var src = new float[] { 1e10f, -1e10f, 1f, -1f, 0f, 0f, 0f, 0f };
        var packed = new PackedInt4[4];
        var scale = QuantizationHelpers.QuantizeInt4(src, packed, groupSize: 8);
        var restored = new float[8];
        QuantizationHelpers.DequantizeInt4(packed, scale, restored);
        // Rel ordering preserved.
        Assert.True(restored[0] > 0);
        Assert.True(restored[1] < 0);
        Assert.Equal(restored[0], -restored[1], 3);
    }

    // ──────────── Int1 (BitNet) ────────────

    [Fact]
    public void QuantizeInt1_PerTensor_RoundTripSignIsExact()
    {
        var src = new float[] { 0.5f, -0.3f, 1.2f, -0.8f, 0f, -1e-5f, 2f, -2f };
        var packed = new PackedInt1[1];
        var scale = QuantizationHelpers.QuantizeInt1(src, packed);
        // BitNet scale = mean absolute = avg(|0.5|, 0.3, 1.2, 0.8, 0, 1e-5, 2, 2) ≈ 0.85
        Assert.Single(scale.Scales);
        Assert.InRange(scale.Scales[0], 0.8f, 0.9f);

        var restored = new float[8];
        QuantizationHelpers.DequantizeInt1(packed, scale, restored);
        // Sign preserved; magnitude is the single per-tensor scale.
        Assert.Equal(scale.Scales[0], restored[0], 3);
        Assert.Equal(-scale.Scales[0], restored[1], 3);
        Assert.Equal(scale.Scales[0], restored[2], 3);
        Assert.Equal(-scale.Scales[0], restored[3], 3);
    }

    [Fact]
    public void QuantizeInt1_PerGroup_MultipleGroups()
    {
        var src = new float[16];
        for (int i = 0; i < 8; i++) src[i] = 0.1f;      // group 0: absmean 0.1
        for (int i = 8; i < 16; i++) src[i] = 2.0f;     // group 1: absmean 2.0
        var packed = new PackedInt1[2];
        var scale = QuantizationHelpers.QuantizeInt1(src, packed, groupSize: 8);
        Assert.Equal(2, scale.Scales.Length);
        Assert.Equal(0.1f, scale.Scales[0], 3);
        Assert.Equal(2.0f, scale.Scales[1], 3);

        var restored = new float[16];
        QuantizationHelpers.DequantizeInt1(packed, scale, restored);
        for (int i = 0; i < 8; i++) Assert.Equal(0.1f, restored[i], 3);
        for (int i = 8; i < 16; i++) Assert.Equal(2.0f, restored[i], 3);
    }

    [Fact]
    public void QuantizeInt1_GroupSize_MustBeMultipleOf8()
    {
        var src = new float[16];
        var dst = new PackedInt1[2];
        Assert.Throws<ArgumentException>(() =>
            QuantizationHelpers.QuantizeInt1(src, dst, groupSize: 5));
    }

    [Fact]
    public void QuantizeInt1_DstTooSmall_Throws()
    {
        var src = new float[16];
        var dst = new PackedInt1[1];
        Assert.Throws<ArgumentException>(() =>
            QuantizationHelpers.QuantizeInt1(src, dst));
    }

    [Fact]
    public void QuantizationScale_Ctor_Null_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new QuantizationScale(null!, 32));
    }

    [Fact]
    public void QuantizationScale_Ctor_ZeroPointLengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new QuantizationScale(new float[] { 1f, 2f }, 32, new int[] { 0 }));
    }
}
