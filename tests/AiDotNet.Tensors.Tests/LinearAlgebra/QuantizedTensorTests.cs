// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Issue #276 sub-feature 3: int8 / int4 QuantizedTensor calibration +
/// dequant round-trip. Acceptance: dequantize ≈ original within
/// per-group quant noise (≤ scale/2 absolute, well below 5% relative
/// for typical weight distributions).
/// </summary>
public class QuantizedTensorTests
{
    [Fact]
    public void Int8_FromFloat_RoundTrip_StaysWithinHalfScale()
    {
        var src = new Tensor<float>(new[] { 64 });
        var rng = new Random(123);
        var span = src.AsWritableSpan();
        for (int i = 0; i < 64; i++) span[i] = (float)(rng.NextDouble() * 4 - 2);

        var qt = QuantizedTensor<sbyte>.FromFloatInt8(src, groupSize: 32);
        Assert.Equal(QuantizationBits.Int8, qt.Bits);
        Assert.Equal(64, qt.Length);
        Assert.Equal(64, qt.Payload.Length);
        Assert.Equal(2, qt.Scale.Scales.Length); // 64 / 32 = 2 groups

        var back = qt.Dequantize().AsSpan();
        for (int i = 0; i < 64; i++)
        {
            int g = i / 32;
            float scale = qt.Scale.Scales[g];
            // Symmetric int8 quant: round-trip error ≤ scale/2 (one ULP at absmax/127).
            float err = Math.Abs(back[i] - span[i]);
            Assert.True(err <= scale * 0.51f,
                $"int8 round-trip error {err} exceeds scale/2 ({scale * 0.51f}) at i={i}");
        }
    }

    [Fact]
    public void Int4_FromFloat_RoundTrip_StaysWithinPerGroupScale()
    {
        var src = new Tensor<float>(new[] { 64 });
        var rng = new Random(99);
        var span = src.AsWritableSpan();
        for (int i = 0; i < 64; i++) span[i] = (float)(rng.NextDouble() * 4 - 2);

        var qt = QuantizedTensor<PackedInt4>.FromFloatInt4(src, groupSize: 32);
        Assert.Equal(QuantizationBits.Int4, qt.Bits);
        Assert.Equal(64, qt.Length);
        Assert.Equal(32, qt.Payload.Length); // 2 elements per byte

        var back = qt.Dequantize().AsSpan();
        for (int i = 0; i < 64; i++)
        {
            int g = i / 32;
            float scale = qt.Scale.Scales[g];
            // int4 has 4-bit symmetric range [-7, 7] → max round-trip error ≈ scale.
            float err = Math.Abs(back[i] - span[i]);
            Assert.True(err <= scale * 1.01f,
                $"int4 round-trip error {err} exceeds scale ({scale * 1.01f}) at i={i}");
        }
    }

    [Fact]
    public void Int8_AllZeros_HandlesGracefully()
    {
        var src = new Tensor<float>(new[] { 32 });
        var qt = QuantizedTensor<sbyte>.FromFloatInt8(src, groupSize: 32);
        var back = qt.Dequantize().AsSpan();
        for (int i = 0; i < 32; i++) Assert.Equal(0f, back[i]);
    }

    [Fact]
    public void AsymmetricScheme_NotYetWired_Throws()
    {
        var src = new Tensor<float>(new[] { 16 });
        Assert.Throws<NotSupportedException>(() =>
            QuantizedTensor<sbyte>.FromFloatInt8(src, scheme: QuantizationScheme.AsymmetricPerGroup));
    }
}
