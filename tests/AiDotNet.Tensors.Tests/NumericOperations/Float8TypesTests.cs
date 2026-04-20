using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.NumericOperations;

/// <summary>
/// Tests for issue #197 — <see cref="Float8E4M3"/> and
/// <see cref="Float8E5M2"/>. Locks in OCP FP8 semantics: roundtrip
/// precision within representable range, saturating overflow on E4M3
/// (no Inf encoding), IEEE-like Inf+NaN on E5M2.
/// </summary>
public class Float8TypesTests
{
    [Theory]
    [InlineData(0f)]
    [InlineData(1f)]
    [InlineData(-1f)]
    [InlineData(0.5f)]
    [InlineData(-0.5f)]
    [InlineData(100f)]
    [InlineData(-100f)]
    [InlineData(2.5f)]
    public void E4M3_Roundtrip_WithinExpectedRelativeError(float v)
    {
        var packed = Float8E4M3.FromFloat(v);
        var back = packed.ToFloat();
        // E4M3 has 3-bit mantissa → ~2^-3 ≈ 12.5% worst-case relative error.
        // For representable range [~2^-9, ±448] we expect far better.
        if (v == 0f)
        {
            Assert.Equal(0f, back);
        }
        else
        {
            float relErr = Math.Abs(back - v) / Math.Abs(v);
            Assert.True(relErr < 0.14f, $"roundtrip rel err {relErr} for {v}");
        }
    }

    [Fact]
    public void E4M3_OverflowSaturates_NotInf()
    {
        // E4M3 has no Inf encoding — values beyond MaxFinite must clamp.
        var tooBig = Float8E4M3.FromFloat(1e10f);
        var tooNeg = Float8E4M3.FromFloat(-1e10f);
        Assert.Equal(Float8E4M3.MaxFinite.RawValue, tooBig.RawValue);
        Assert.Equal(Float8E4M3.MinFinite.RawValue, tooNeg.RawValue);
    }

    [Fact]
    public void E4M3_InfSaturates_NotInf()
    {
        // +Inf / -Inf inputs saturate on FromFloat.
        var posInf = Float8E4M3.FromFloat(float.PositiveInfinity);
        var negInf = Float8E4M3.FromFloat(float.NegativeInfinity);
        Assert.False(float.IsInfinity(posInf.ToFloat()));
        Assert.False(float.IsInfinity(negInf.ToFloat()));
        Assert.Equal(Float8E4M3.MaxFinite.RawValue, posInf.RawValue);
        Assert.Equal(Float8E4M3.MinFinite.RawValue, negInf.RawValue);
    }

    [Fact]
    public void E4M3_NaNPropagates()
    {
        var nan = Float8E4M3.FromFloat(float.NaN);
        Assert.True(nan.IsNaN);
        Assert.True(float.IsNaN(nan.ToFloat()));
    }

    [Fact]
    public void E4M3_Comparison_MatchesFloatOrder()
    {
        var a = Float8E4M3.FromFloat(-3f);
        var b = Float8E4M3.FromFloat(2f);
        var c = Float8E4M3.FromFloat(2f);
        Assert.True(a.CompareTo(b) < 0);
        Assert.Equal(0, b.CompareTo(c));
        Assert.True(b.CompareTo(a) > 0);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(1f)]
    [InlineData(-1f)]
    [InlineData(100f)]
    [InlineData(-100f)]
    // Sub-normal (< 2^-14 ≈ 6.1e-5) underflows to zero by design; over-max
    // (> 57344) saturates to Inf. Test values stay within [2^-13, 2^14].
    [InlineData(1e-3f)]
    [InlineData(1e3f)]
    public void E5M2_Roundtrip_WithinExpectedRelativeError(float v)
    {
        var packed = Float8E5M2.FromFloat(v);
        var back = packed.ToFloat();
        if (v == 0f)
        {
            Assert.Equal(0f, back);
        }
        else
        {
            // E5M2 has 2-bit mantissa → ~2^-2 = 25% worst-case relative error.
            float relErr = Math.Abs(back - v) / Math.Abs(v);
            Assert.True(relErr < 0.26f, $"roundtrip rel err {relErr} for {v}");
        }
    }

    [Fact]
    public void E5M2_PreservesInf()
    {
        // Unlike E4M3, E5M2 has an Inf encoding.
        var posInf = Float8E5M2.FromFloat(float.PositiveInfinity);
        var negInf = Float8E5M2.FromFloat(float.NegativeInfinity);
        Assert.True(posInf.IsInfinity);
        Assert.True(negInf.IsInfinity);
        Assert.Equal(float.PositiveInfinity, posInf.ToFloat());
        Assert.Equal(float.NegativeInfinity, negInf.ToFloat());
    }

    [Fact]
    public void E5M2_NaNPropagates()
    {
        var nan = Float8E5M2.FromFloat(float.NaN);
        Assert.True(nan.IsNaN);
        Assert.True(float.IsNaN(nan.ToFloat()));
    }

    [Fact]
    public void E4M3_Equality_ByRaw()
    {
        var a = Float8E4M3.FromFloat(1.25f);
        var b = Float8E4M3.FromFloat(1.25f);
        Assert.True(a == b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void ExplicitConversion_RoundTrips()
    {
        Float8E4M3 p = (Float8E4M3)2.0f;
        float back = (float)p;
        Assert.Equal(2.0f, back, 1);
    }
}
