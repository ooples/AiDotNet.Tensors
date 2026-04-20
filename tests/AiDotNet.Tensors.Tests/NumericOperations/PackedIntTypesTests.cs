using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.NumericOperations;

/// <summary>
/// Tests for issue #207 B1 — <see cref="PackedInt1"/> and
/// <see cref="PackedInt4"/> storage types. Locks encoding correctness
/// + round-trip bounds + sign-extension behaviour.
/// </summary>
public class PackedIntTypesTests
{
    // ──────────── PackedInt1 ────────────

    [Fact]
    public void Int1_FromSigns_RoundTrips()
    {
        var signs = new sbyte[] { -1, 1, -1, 1, -1, 1, 1, -1 };
        var packed = PackedInt1.FromSigns(signs);
        for (int i = 0; i < 8; i++)
            Assert.Equal(signs[i], packed.GetLane(i));
    }

    [Fact]
    public void Int1_FromFloat_PositiveAndZeroMapToPlusOne()
    {
        // BitNet convention: sign(0) = +1.
        var values = new float[] { 0f, -0.1f, 1f, -1f, 0f, 100f, -100f, 0.5f };
        var packed = PackedInt1.FromFloat(values);
        Assert.Equal(1, packed.GetLane(0));   // 0 → +1
        Assert.Equal(-1, packed.GetLane(1));  // -0.1 → -1
        Assert.Equal(1, packed.GetLane(2));
        Assert.Equal(-1, packed.GetLane(3));
        Assert.Equal(1, packed.GetLane(4));
        Assert.Equal(1, packed.GetLane(5));
        Assert.Equal(-1, packed.GetLane(6));
        Assert.Equal(1, packed.GetLane(7));
    }

    [Fact]
    public void Int1_WrongLength_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            PackedInt1.FromSigns(new sbyte[] { 1, -1 }));
        Assert.Throws<ArgumentException>(() =>
            PackedInt1.FromFloat(new float[7]));
    }

    [Fact]
    public void Int1_GetLane_OutOfRange_Throws()
    {
        var p = new PackedInt1(0xFF);
        Assert.Throws<ArgumentOutOfRangeException>(() => p.GetLane(8));
    }

    [Fact]
    public void Int1_Equality_ByRaw()
    {
        var a = new PackedInt1(0b10101010);
        var b = new PackedInt1(0b10101010);
        Assert.True(a == b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    // ──────────── PackedInt4 ────────────

    [Fact]
    public void Int4_FromInts_RoundTripsPositive()
    {
        var p = PackedInt4.FromInts(lo: 3, hi: 5);
        Assert.Equal(3, p.LoNibble);
        Assert.Equal(5, p.HiNibble);
    }

    [Fact]
    public void Int4_FromInts_RoundTripsNegative()
    {
        var p = PackedInt4.FromInts(lo: -8, hi: -1);
        Assert.Equal(-8, p.LoNibble);
        Assert.Equal(-1, p.HiNibble);
    }

    [Fact]
    public void Int4_FromInts_BoundaryValues()
    {
        var p = PackedInt4.FromInts(lo: -8, hi: 7);
        Assert.Equal(-8, p.LoNibble);
        Assert.Equal(7, p.HiNibble);
    }

    [Fact]
    public void Int4_FromInts_OutOfRange_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => PackedInt4.FromInts(8, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PackedInt4.FromInts(-9, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PackedInt4.FromInts(0, 8));
    }

    [Fact]
    public void Int4_GetLane_InvalidIndex_Throws()
    {
        var p = PackedInt4.FromInts(1, 2);
        Assert.Throws<ArgumentOutOfRangeException>(() => p.GetLane(2));
    }

    [Fact]
    public void Int4_Equality_ByRaw()
    {
        var a = PackedInt4.FromInts(3, 5);
        var b = PackedInt4.FromInts(3, 5);
        Assert.True(a == b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }
}
