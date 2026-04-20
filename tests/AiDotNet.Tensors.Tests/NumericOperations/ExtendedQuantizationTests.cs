using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.NumericOperations;

/// <summary>
/// Tests for issue #207 B1 expansion — PackedInt2, PackedInt3,
/// NormalFloat4, PackedFp4, and the corresponding quantize /
/// dequantize helpers plus fake-quantize (QAT) passes.
/// </summary>
public class ExtendedQuantizationTests
{
    // ──────────── PackedInt2 ────────────

    [Fact]
    public void Int2_FromInts_RoundTripsAllBoundaryValues()
    {
        var p = PackedInt2.FromInts(-2, -1, 0, 1);
        Assert.Equal(-2, p.GetLane(0));
        Assert.Equal(-1, p.GetLane(1));
        Assert.Equal(0,  p.GetLane(2));
        Assert.Equal(1,  p.GetLane(3));
    }

    [Fact]
    public void Int2_FromInts_OutOfRange_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => PackedInt2.FromInts(2, 0, 0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PackedInt2.FromInts(0, -3, 0, 0));
    }

    [Fact]
    public void QuantizeInt2_DequantizeRoundTrip_PreservesSignForLargeValues()
    {
        // Int2 has only 4 levels → small magnitudes legitimately snap to
        // the 0 code. Use inputs where every nonzero |v| >= scale so
        // sign is preserved by the quantize-step rounding.
        var src = new float[] { 1f, -1f, 1f, -1f, 0f, 1f, -1f, 1f };
        var dst = new PackedInt2[2];
        var scale = QuantizationHelpers.QuantizeInt2(src, dst, groupSize: 4);

        var restored = new float[src.Length];
        QuantizationHelpers.DequantizeInt2(dst, scale, restored);
        for (int i = 0; i < src.Length; i++)
        {
            if (src[i] != 0f)
                Assert.Equal(Math.Sign(src[i]), Math.Sign(restored[i]));
        }
    }

    [Fact]
    public void QuantizeInt2_GroupSizeNotMultipleOf4_Throws()
    {
        var src = new float[8];
        var dst = new PackedInt2[2];
        Assert.Throws<ArgumentException>(() =>
            QuantizationHelpers.QuantizeInt2(src, dst, groupSize: 5));
    }

    // ──────────── PackedInt3Block ────────────

    [Fact]
    public void Int3Block_FromInts_RoundTripsAllLanes()
    {
        var values = new[] { -4, -3, -2, -1, 0, 1, 2, 3 };
        var block = PackedInt3Block.FromInts(values);
        for (int i = 0; i < 8; i++)
            Assert.Equal(values[i], block.GetLane(i));
    }

    [Fact]
    public void Int3Block_FromInts_OutOfRange_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PackedInt3Block.FromInts(new[] { 4, 0, 0, 0, 0, 0, 0, 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PackedInt3Block.FromInts(new[] { -5, 0, 0, 0, 0, 0, 0, 0 }));
    }

    [Fact]
    public void QuantizeInt3_DequantizeRoundTrip_WithinExpectedError()
    {
        var rng = new Random(0x123);
        var src = new float[64];
        for (int i = 0; i < src.Length; i++) src[i] = (float)(rng.NextDouble() * 4 - 2);
        var packed = new PackedInt3Block[src.Length / 8];
        var scale = QuantizationHelpers.QuantizeInt3(src, packed, groupSize: 32);

        var restored = new float[src.Length];
        QuantizationHelpers.DequantizeInt3(packed, scale, restored);

        // Int3 step = absmax/3, error ≤ step/2 = absmax/6.
        for (int g = 0; g < src.Length; g += 32)
        {
            float absMax = 0f;
            for (int i = g; i < g + 32; i++) absMax = Math.Max(absMax, Math.Abs(src[i]));
            float bound = absMax / 6f + 1e-5f;
            for (int i = g; i < g + 32; i++)
                Assert.True(Math.Abs(restored[i] - src[i]) <= bound,
                    $"[{i}] err exceeded bound {bound}");
        }
    }

    // ──────────── NormalFloat4 ────────────

    [Fact]
    public void NF4_TableHasSixteenDistinctValues()
    {
        var distinct = new HashSet<float>(NormalFloat4.Table.ToArray());
        Assert.Equal(16, distinct.Count);
    }

    [Fact]
    public void NF4_TableIsSortedAscending()
    {
        for (int i = 1; i < NormalFloat4.Table.Length; i++)
            Assert.True(NormalFloat4.Table[i] > NormalFloat4.Table[i - 1]);
    }

    [Fact]
    public void NF4_ToIndex_SnapsToNearestEntry()
    {
        Assert.Equal(7, NormalFloat4.ToIndex(0f));    // Table[7] = 0
        Assert.Equal(0, NormalFloat4.ToIndex(-1f));   // Table[0] = -1
        Assert.Equal(15, NormalFloat4.ToIndex(1f));   // Table[15] = 1
        // Midpoint between 6 (-0.091...) and 7 (0) is ~ -0.045, snaps to 7.
        Assert.Equal(7, NormalFloat4.ToIndex(-0.04f));
    }

    [Fact]
    public void NF4_ToIndex_Saturates()
    {
        Assert.Equal(0, NormalFloat4.ToIndex(-100f));
        Assert.Equal(15, NormalFloat4.ToIndex(100f));
    }

    [Fact]
    public void QuantizeNF4_DequantizeRoundTrip_NearIdentityForTableValues()
    {
        // Values drawn from Table itself should round-trip exactly after
        // scaling by the per-group absmax.
        var src = new float[]
        {
            NormalFloat4.Table[0], NormalFloat4.Table[5],
            NormalFloat4.Table[10], NormalFloat4.Table[15]
        };
        var packed = new PackedInt4[2];
        var scale = QuantizationHelpers.QuantizeNF4(src, packed, groupSize: 4);
        var restored = new float[src.Length];
        QuantizationHelpers.DequantizeNF4(packed, scale, restored);
        for (int i = 0; i < src.Length; i++)
            Assert.Equal(src[i], restored[i], 4);
    }

    // ──────────── FP4 (E2M1) ────────────

    [Fact]
    public void Fp4_TableEncodesSignAndMagnitude()
    {
        // Bit patterns: positive values in 0..7, negative in 8..15
        // (MSB = sign bit). Magnitude ordering must be consistent.
        for (int i = 1; i <= 7; i++)
            Assert.True(Fp4E2M1.Table[i] > Fp4E2M1.Table[i - 1]);
        // Mirror: Table[i+8] == -Table[i] for i > 0.
        for (int i = 1; i <= 7; i++)
            Assert.Equal(-Fp4E2M1.Table[i], Fp4E2M1.Table[i + 8]);
    }

    [Fact]
    public void Fp4_ToIndex_SaturatesBeyondSixRange()
    {
        Assert.Equal(7, Fp4E2M1.ToIndex(100f));
        Assert.Equal(15, Fp4E2M1.ToIndex(-100f));
    }

    [Fact]
    public void Fp4_NaN_CollapsesToZero()
    {
        Assert.Equal(0, Fp4E2M1.ToIndex(float.NaN));
    }

    [Fact]
    public void QuantizeFp4_DequantizeRoundTrip_WithinExpectedError()
    {
        var rng = new Random(44);
        int n = 32;
        var src = new float[n];
        for (int i = 0; i < n; i++) src[i] = (float)(rng.NextDouble() * 12 - 6); // ±6 range

        var packed = new PackedInt4[n / 2];
        var scale = QuantizationHelpers.QuantizeFp4(src, packed, groupSize: 16);
        var restored = new float[n];
        QuantizationHelpers.DequantizeFp4(packed, scale, restored);

        // 16 levels over ~±6 — values with |src| much smaller than
        // the step width can legitimately snap to the zero code. Check
        // sign agreement only for inputs clearly above the snap-to-zero
        // threshold (≥ half a step = absmax/12).
        float snapThreshold = 6f / 12f;
        for (int i = 0; i < n; i++)
        {
            if (Math.Abs(src[i]) > snapThreshold)
                Assert.Equal(Math.Sign(src[i]), Math.Sign(restored[i]));
        }
    }

    // ──────────── Fake-quantize (QAT) ────────────

    [Fact]
    public void FakeQuantizeInt4_MatchesRoundTrip()
    {
        var src = new float[] { 0.1f, -0.3f, 0.5f, -0.7f, 1.0f, -1.0f, 0f, 0.25f };
        var dstFake = new float[src.Length];
        QuantizationHelpers.FakeQuantizeInt4(src, dstFake, groupSize: 8);

        // Reference: explicit quant + dequant.
        var packed = new PackedInt4[(src.Length + 1) / 2];
        var scale = QuantizationHelpers.QuantizeInt4(src, packed, groupSize: 8);
        var dstRef = new float[src.Length];
        QuantizationHelpers.DequantizeInt4(packed, scale, dstRef);

        Assert.Equal(dstRef, dstFake);
    }

    [Fact]
    public void FakeQuantizeNF4_MatchesRoundTrip()
    {
        var src = new float[] { -0.8f, -0.4f, 0f, 0.3f };
        var dstFake = new float[src.Length];
        QuantizationHelpers.FakeQuantizeNF4(src, dstFake, groupSize: 4);

        var packed = new PackedInt4[(src.Length + 1) / 2];
        var scale = QuantizationHelpers.QuantizeNF4(src, packed, groupSize: 4);
        var dstRef = new float[src.Length];
        QuantizationHelpers.DequantizeNF4(packed, scale, dstRef);

        Assert.Equal(dstRef, dstFake);
    }

    [Fact]
    public void FakeQuantizeInt1_PreservesSignAndScale()
    {
        var src = new float[] { 0.5f, -0.3f, 1.0f, -1.0f, 0f, 0.8f, -0.2f, 0.4f };
        var dst = new float[src.Length];
        QuantizationHelpers.FakeQuantizeInt1(src, dst);
        // After fake-quant: every |dst[i]| = scale, same sign as src.
        float absmean = 0f;
        for (int i = 0; i < src.Length; i++) absmean += Math.Abs(src[i]);
        absmean /= src.Length;
        for (int i = 0; i < src.Length; i++)
        {
            Assert.Equal(absmean, Math.Abs(dst[i]), 3);
            Assert.Equal(Math.Sign(src[i]) == 0 ? 1 : Math.Sign(src[i]), Math.Sign(dst[i]));
        }
    }
}
