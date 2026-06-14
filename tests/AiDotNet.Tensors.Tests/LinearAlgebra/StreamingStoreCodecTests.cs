// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Correctness of the bf16 streaming-store codec: round-trip error bounds,
/// deterministic-RNE agreement with <see cref="BFloat16"/>, special-value
/// handling, and — the key training-safety property — that STOCHASTIC rounding
/// is unbiased (the decoded mean of many quantizations equals the original).
/// </summary>
public class StreamingStoreCodecTests
{
    private struct Rng
    {
        private ulong _s;
        public Rng(ulong seed) { _s = seed | 1UL; }
        public double NextUnit() { ulong x = _s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; _s = x; return (x >> 11) * (1.0 / (1UL << 53)); }
        public float NextGaussian(double std) { double u1 = Math.Max(1e-12, NextUnit()), u2 = NextUnit(); return (float)(std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)); }
    }

    [Fact]
    public void EncodeDecodeFloat_RoundTrip_WithinBf16Precision()
    {
        var rng = new Rng(1);
        int n = 4096;
        var src = new float[n];
        for (int i = 0; i < n; i++) src[i] = rng.NextGaussian(0.05);

        var enc = new byte[n * StreamingStoreCodec.Bf16ElementSize];
        StreamingStoreCodec.EncodeFloat(src, enc, stochastic: false);
        Assert.Equal(n * 2, enc.Length); // exactly 2x smaller than fp32's 4n

        var dec = new float[n];
        StreamingStoreCodec.DecodeFloat(enc, dec);

        double sum2 = 0, ref2 = 0;
        for (int i = 0; i < n; i++)
        {
            // bf16 has 8 mantissa bits → relative error ≤ 2^-8 per element.
            if (Math.Abs(src[i]) > 1e-9)
                Assert.True(Math.Abs(dec[i] - src[i]) / Math.Abs(src[i]) <= 1.0 / 256 + 1e-6,
                    $"elem {i}: {src[i]} -> {dec[i]} exceeds bf16 relative precision");
            sum2 += (dec[i] - src[i]) * (double)(dec[i] - src[i]); ref2 += (double)src[i] * src[i];
        }
        Assert.True(Math.Sqrt(sum2 / ref2) < 0.005, "RMS relative error should be well under 0.5%");
    }

    [Fact]
    public void DeterministicRounding_MatchesBFloat16FromFloat()
    {
        var rng = new Rng(2);
        for (int i = 0; i < 2000; i++)
        {
            float v = rng.NextGaussian(1.0);
            var enc = new byte[2];
            StreamingStoreCodec.EncodeFloat(new[] { v }, enc, stochastic: false);
            ushort raw = (ushort)(enc[0] | (enc[1] << 8));
            Assert.Equal(BFloat16.FromFloat(v).RawValue, raw);
        }
    }

    [Fact]
    public void SpecialValues_SurviveRoundTrip()
    {
        var src = new[] { 0f, -0f, 1f, -1f, 123456f, -0.0001f, float.PositiveInfinity, float.NegativeInfinity };
        var enc = new byte[src.Length * 2];
        StreamingStoreCodec.EncodeFloat(src, enc, stochastic: false);
        var dec = new float[src.Length];
        StreamingStoreCodec.DecodeFloat(enc, dec);
        Assert.Equal(0f, dec[0]); Assert.Equal(1f, dec[2]); Assert.Equal(-1f, dec[3]);
        Assert.True(float.IsPositiveInfinity(dec[6]));
        Assert.True(float.IsNegativeInfinity(dec[7]));

        // NaN stays NaN.
        var nanEnc = new byte[2];
        StreamingStoreCodec.EncodeFloat(new[] { float.NaN }, nanEnc, stochastic: false);
        var nanDec = new float[1];
        StreamingStoreCodec.DecodeFloat(nanEnc, nanDec);
        Assert.True(float.IsNaN(nanDec[0]));
    }

    [Fact]
    public void StochasticRounding_IsUnbiased()
    {
        // A value strictly between two bf16 grid points: quantizing it MANY times
        // stochastically and averaging the decoded results must converge back to
        // the original (unbiased). Deterministic rounding would always give the
        // same single grid point (biased away from the true value).
        float v = 0.1234567f; // not representable in bf16
        const int trials = 200_000;
        double sum = 0;
        var enc = new byte[2];
        var dec = new float[1];
        for (int i = 0; i < trials; i++)
        {
            StreamingStoreCodec.EncodeFloat(new[] { v }, enc, stochastic: true);
            StreamingStoreCodec.DecodeFloat(enc, dec);
            sum += dec[0];
        }
        double mean = sum / trials;
        // bf16 step near 0.12 is ~2^-3 * 2^-7 ≈ 0.0009; the unbiased mean should be
        // within a few x10^-5 of v after 200k trials.
        Assert.True(Math.Abs(mean - v) < 2e-4,
            $"Stochastic-rounding mean {mean} should converge to {v} (unbiased); |diff|={Math.Abs(mean - v)}");

        // Sanity: stochastic rounding actually produces BOTH neighbours (not a
        // constant), i.e. it's genuinely stochastic.
        var seen = new System.Collections.Generic.HashSet<float>();
        for (int i = 0; i < 1000; i++)
        {
            StreamingStoreCodec.EncodeFloat(new[] { v }, enc, stochastic: true);
            StreamingStoreCodec.DecodeFloat(enc, dec);
            seen.Add(dec[0]);
        }
        Assert.True(seen.Count >= 2, "stochastic rounding must straddle both bf16 neighbours");
    }

    [Fact]
    public void EncodeDecodeDouble_RoundTrip_WithinBf16Precision()
    {
        var rng = new Rng(3);
        int n = 2048;
        var src = new double[n];
        for (int i = 0; i < n; i++) src[i] = rng.NextGaussian(0.05);

        var enc = new byte[n * 2]; // 4x smaller than fp64's 8n
        StreamingStoreCodec.EncodeDouble(src, enc, stochastic: false);
        var dec = new double[n];
        StreamingStoreCodec.DecodeDouble(enc, dec);

        double sum2 = 0, ref2 = 0;
        for (int i = 0; i < n; i++) { sum2 += (dec[i] - src[i]) * (dec[i] - src[i]); ref2 += src[i] * src[i]; }
        Assert.True(Math.Sqrt(sum2 / ref2) < 0.005, "fp64→bf16 RMS relative error should be under 0.5%");
    }
}
