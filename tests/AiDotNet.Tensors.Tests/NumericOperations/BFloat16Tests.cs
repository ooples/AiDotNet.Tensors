// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.NumericOperations;

/// <summary>
/// Issue #276 sub-feature 1: BFloat16 type, INumericOperations registry,
/// round-trip + arithmetic + edge cases.
/// </summary>
public class BFloat16Tests
{
    [Fact]
    public void RawBitPattern_OneFloat_Equals_Bf16OneEncoding()
    {
        // 1.0f in IEEE-754 = 0x3F800000; bf16 = upper 16 bits = 0x3F80.
        Assert.Equal((ushort)0x3F80, BFloat16.One.RawValue);
        Assert.Equal((ushort)0x0000, BFloat16.Zero.RawValue);
    }

    [Fact]
    public void RoundTrip_RepresentableValues_AreLossless()
    {
        // bf16 = upper 16 bits of float32; values whose lower 16 bits are
        // already zero must round-trip exactly.
        float[] exact = { 0f, 1f, -1f, 2f, 0.5f, 1024f, -1024f, float.PositiveInfinity, float.NegativeInfinity };
        foreach (var v in exact)
        {
            var bf = BFloat16.FromFloat(v);
            Assert.Equal(v, (float)bf);
        }
    }

    [Fact]
    public void RoundTrip_RandomFloats_StaysWithinHalfUlp()
    {
        var rng = new Random(42);
        for (int i = 0; i < 1000; i++)
        {
            float v = (float)(rng.NextDouble() * 2 - 1) * 1e6f;
            var bf = BFloat16.FromFloat(v);
            float back = (float)bf;
            // bf16 has 7 mantissa bits → max relative error ~2^-8 ≈ 0.4%.
            float relErr = Math.Abs(back - v) / Math.Max(1e-30f, Math.Abs(v));
            Assert.True(relErr < 5e-3f, $"v={v}, back={back}, rel={relErr}");
        }
    }

    [Fact]
    public void NaN_Preserves_Through_RoundTrip()
    {
        var bf = BFloat16.FromFloat(float.NaN);
        Assert.True(BFloat16.IsNaN(bf));
        Assert.True(float.IsNaN((float)bf));
    }

    [Fact]
    public void Infinity_Preserves_Through_RoundTrip()
    {
        var posInf = BFloat16.FromFloat(float.PositiveInfinity);
        var negInf = BFloat16.FromFloat(float.NegativeInfinity);
        Assert.True(BFloat16.IsInfinity(posInf));
        Assert.True(BFloat16.IsInfinity(negInf));
        Assert.Equal(float.PositiveInfinity, (float)posInf);
        Assert.Equal(float.NegativeInfinity, (float)negInf);
    }

    [Fact]
    public void Negate_FlipsSignBit()
    {
        var a = BFloat16.FromFloat(3.5f);
        var n = -a;
        Assert.Equal(-3.5f, (float)n);
        Assert.Equal((ushort)(a.RawValue ^ 0x8000), n.RawValue);
    }

    [Fact]
    public void Arithmetic_Operators_RoundTripViaFloat()
    {
        var a = BFloat16.FromFloat(2.0f);
        var b = BFloat16.FromFloat(0.5f);
        Assert.Equal(2.5f, (float)(a + b), 3);
        Assert.Equal(1.5f, (float)(a - b), 3);
        Assert.Equal(1.0f, (float)(a * b), 3);
        Assert.Equal(4.0f, (float)(a / b), 3);
    }

    [Fact]
    public void Comparison_FollowsFloatSemantics()
    {
        var a = BFloat16.FromFloat(1.5f);
        var b = BFloat16.FromFloat(2.5f);
        Assert.True(a < b);
        Assert.True(b > a);
        Assert.True(a <= a);
        Assert.True(a >= a);
        Assert.False(a == b);
    }

    [Fact]
    public void Range_MatchesFloat_NotHalf()
    {
        // bf16 has 8-bit exponent (same as float), so max value is ~3.39e38.
        // System.Half tops out at 65504 — bf16 must NOT.
        Assert.True((float)BFloat16.MaxValue > 1e30f);
        Assert.True((float)BFloat16.MinValue < -1e30f);
    }

    [Fact]
    public void MathHelper_ResolvesBFloat16Operations()
    {
        var ops = MathHelper.GetNumericOperations<BFloat16>();
        Assert.NotNull(ops);
        Assert.IsType<BFloat16Operations>(ops);
    }

    [Fact]
    public void INumericOperations_Add_Mul_RoundTripCorrect()
    {
        var ops = MathHelper.GetNumericOperations<BFloat16>();
        var a = ops.FromDouble(3.5);
        var b = ops.FromDouble(0.25);
        Assert.Equal(3.75, (double)ops.Add(a, b), 2);
        Assert.Equal(0.875, (double)ops.Multiply(a, b), 2);
    }

    [Fact]
    public void Vectorized_Add_MatchesScalar()
    {
        var ops = new BFloat16Operations();
        var x = new BFloat16[] { ops.FromFloat(1f), ops.FromFloat(2f), ops.FromFloat(3f), ops.FromFloat(4f) };
        var y = new BFloat16[] { ops.FromFloat(0.5f), ops.FromFloat(0.5f), ops.FromFloat(0.5f), ops.FromFloat(0.5f) };
        var dst = new BFloat16[4];
        ops.Add(x, y, dst);
        for (int i = 0; i < 4; i++)
            Assert.Equal((float)x[i] + (float)y[i], (float)dst[i], 3);
    }

    [Fact]
    public void IsFloatingPoint_RecognizesBFloat16()
    {
        Assert.True(MathHelper.IsFloatingPoint<BFloat16>());
    }
}
