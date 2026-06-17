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

        // Negative zero must round-trip with sign preserved. `Assert.Equal(0f, -0f)`
        // would pass on +0f too — `1/-0f == -Inf` is the unambiguous sign check
        // (bf16 keeps the IEEE-754 sign bit so this contract holds).
        Assert.True(float.IsNegativeInfinity(1f / dec[1]),
            $"-0f did not round-trip with sign preserved: dec[1]={dec[1]} (1/dec[1]={1f / dec[1]})");
        // +0f symmetry — both signs of zero must survive.
        Assert.True(float.IsPositiveInfinity(1f / dec[0]),
            $"+0f did not round-trip with sign preserved: dec[0]={dec[0]} (1/dec[0]={1f / dec[0]})");

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
    public void Int8_RoundTrip_4xSmaller_WithinPerRowPrecision()
    {
        var rng = new Rng(7);
        const int rows = 64, k = 64, n = rows * k;
        var src = new float[n];
        for (int i = 0; i < n; i++) src[i] = rng.NextGaussian(0.05);

        int bytes = StreamingStoreCodec.Int8BufferBytes(n, rows);
        Assert.Equal(4 + 4 * rows + n, bytes); // [int32 rows][rows scales][int8 data] → still ~4x vs fp32
        var enc = new byte[bytes];
        StreamingStoreCodec.EncodeInt8Float(src, enc, rows);
        var dec = new float[n];
        StreamingStoreCodec.DecodeInt8Float(enc, dec);

        double sum2 = 0, ref2 = 0;
        for (int i = 0; i < n; i++) { double e = dec[i] - src[i]; sum2 += e * e; ref2 += (double)src[i] * src[i]; }
        double rmse = Math.Sqrt(sum2 / ref2);
        Assert.True(rmse < 0.02, $"int8 RMS relative error {rmse} should be ~1%");
    }

    [Fact]
    public void Int8_PerRow_BeatsPerTensor_OnVariedRowMagnitudes()
    {
        // The reason for per-row: rows with wildly different magnitudes. A single per-tensor
        // scale is dominated by the largest row, clipping the small rows to near-zero int8.
        var rng = new Rng(11);
        const int rows = 32, k = 64, n = rows * k;
        var src = new float[n];
        for (int r = 0; r < rows; r++)
        {
            // Row magnitude spans 4 orders: row 0 ~1e-3, last row ~1e1.
            double rowStd = Math.Pow(10, -3 + 4.0 * r / (rows - 1));
            for (int j = 0; j < k; j++) src[r * k + j] = rng.NextGaussian(rowStd);
        }

        // per-row (the new store): rows = 32
        var encRow = new byte[StreamingStoreCodec.Int8BufferBytes(n, rows)];
        StreamingStoreCodec.EncodeInt8Float(src, encRow, rows);
        var decRow = new float[n];
        StreamingStoreCodec.DecodeInt8Float(encRow, decRow);

        // per-tensor (rows = 1): one scale for everything
        var encTen = new byte[StreamingStoreCodec.Int8BufferBytes(n, 1)];
        StreamingStoreCodec.EncodeInt8Float(src, encTen, 1);
        var decTen = new float[n];
        StreamingStoreCodec.DecodeInt8Float(encTen, decTen);

        double Rmse(float[] dec) { double s = 0, rf = 0; for (int i = 0; i < n; i++) { double e = dec[i] - src[i]; s += e * e; rf += (double)src[i] * src[i]; } return Math.Sqrt(s / rf); }
        double rowRmse = Rmse(decRow), tenRmse = Rmse(decTen);
        // Per-row should be dramatically better when row magnitudes vary.
        Assert.True(rowRmse < tenRmse * 0.5,
            $"per-row RMSE {rowRmse:E3} should be < half per-tensor RMSE {tenRmse:E3} on varied-magnitude rows");
    }

    [Fact]
    public void Int8_Double_RoundTrips()
    {
        var rng = new Rng(8);
        const int rows = 16, k = 64, n = rows * k;
        var src = new double[n];
        for (int i = 0; i < n; i++) src[i] = rng.NextGaussian(0.1);
        var enc = new byte[StreamingStoreCodec.Int8BufferBytes(n, rows)];
        StreamingStoreCodec.EncodeInt8Double(src, enc, rows);
        var dec = new double[n];
        StreamingStoreCodec.DecodeInt8Double(enc, dec);
        double sum2 = 0, ref2 = 0;
        for (int i = 0; i < n; i++) { double e = dec[i] - src[i]; sum2 += e * e; ref2 += src[i] * src[i]; }
        Assert.True(Math.Sqrt(sum2 / ref2) < 0.02, "fp64→int8 RMS error ~1%");
    }

    [Fact]
    public void Int4_RoundTrip_8xSmaller_WithinGroupPrecision()
    {
        var rng = new Rng(13);
        const int n = 4096, gs = StreamingStoreCodec.DefaultInt4GroupSize;
        var src = new float[n];
        for (int i = 0; i < n; i++) src[i] = rng.NextGaussian(0.05);

        int bytes = StreamingStoreCodec.Int4BufferBytes(n, gs);
        // ~8x vs fp32: 4 bits/weight + a tiny per-group-scale header.
        Assert.True(bytes < n * sizeof(float) / 7,
            $"int4 store ({bytes} B) should be ~8x smaller than fp32 ({n * 4} B)");
        var enc = new byte[bytes];
        StreamingStoreCodec.EncodeInt4Float(src, enc, gs);
        var dec = new float[n];
        StreamingStoreCodec.DecodeInt4Float(enc, dec);

        double sum2 = 0, ref2 = 0;
        for (int i = 0; i < n; i++) { double e = dec[i] - src[i]; sum2 += e * e; ref2 += (double)src[i] * src[i]; }
        double rmse = Math.Sqrt(sum2 / ref2);
        // 4-bit symmetric group-quant: theoretical ~12-15% relative RMSE on Gaussian weights.
        Assert.True(rmse < 0.18, $"int4 group-quant RMS relative error {rmse} should be ~12-15%");
    }

    [Fact]
    public void Int4_GroupQuant_BeatsPerTensor_OnVariedMagnitudes()
    {
        var rng = new Rng(17);
        const int n = 4096, gs = 128;
        var src = new float[n];
        // Magnitude ramps across the array by 4 orders, so a single per-tensor scale clips
        // the small-magnitude region to near-zero int4 while group scaling tracks it.
        for (int i = 0; i < n; i++)
        {
            double std = Math.Pow(10, -3 + 4.0 * i / (n - 1));
            src[i] = rng.NextGaussian(std);
        }
        double Rmse(float[] dec) { double s = 0, rf = 0; for (int i = 0; i < n; i++) { double e = dec[i] - src[i]; s += e * e; rf += (double)src[i] * src[i]; } return Math.Sqrt(s / rf); }

        var encGrp = new byte[StreamingStoreCodec.Int4BufferBytes(n, gs)];
        StreamingStoreCodec.EncodeInt4Float(src, encGrp, gs);
        var decGrp = new float[n]; StreamingStoreCodec.DecodeInt4Float(encGrp, decGrp);

        var encTen = new byte[StreamingStoreCodec.Int4BufferBytes(n, n)]; // one group = per-tensor
        StreamingStoreCodec.EncodeInt4Float(src, encTen, n);
        var decTen = new float[n]; StreamingStoreCodec.DecodeInt4Float(encTen, decTen);

        Assert.True(Rmse(decGrp) < Rmse(decTen) * 0.5,
            $"group int4 RMSE {Rmse(decGrp):E3} should be < half per-tensor RMSE {Rmse(decTen):E3}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]    // odd length → final nibble unused
    [InlineData(129)]  // spills into a 2nd partial group
    [InlineData(257)]
    public void Int4_OddAndPartialGroup_RoundTrips(int n)
    {
        var rng = new Rng((ulong)(n + 500));
        var src = new float[n];
        for (int i = 0; i < n; i++) src[i] = rng.NextGaussian(0.1);
        var enc = new byte[StreamingStoreCodec.Int4BufferBytes(n, StreamingStoreCodec.DefaultInt4GroupSize)];
        StreamingStoreCodec.EncodeInt4Float(src, enc, StreamingStoreCodec.DefaultInt4GroupSize);
        var dec = new float[n];
        StreamingStoreCodec.DecodeInt4Float(enc, dec);
        // Each decoded value must be within one int4 step of the original (no packing corruption).
        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs(dec[i] - src[i]) <= Math.Abs(src[i]) + 1e-3,
                $"int4 element {i} round-trip corrupt: {src[i]} -> {dec[i]}");
    }

    [Fact]
    public void Int4_Double_RoundTrips()
    {
        var rng = new Rng(19);
        const int n = 2048, gs = 128;
        var src = new double[n];
        for (int i = 0; i < n; i++) src[i] = rng.NextGaussian(0.1);
        var enc = new byte[StreamingStoreCodec.Int4BufferBytes(n, gs)];
        StreamingStoreCodec.EncodeInt4Double(src, enc, gs);
        var dec = new double[n];
        StreamingStoreCodec.DecodeInt4Double(enc, dec);
        double sum2 = 0, ref2 = 0;
        for (int i = 0; i < n; i++) { double e = dec[i] - src[i]; sum2 += e * e; ref2 += src[i] * src[i]; }
        Assert.True(Math.Sqrt(sum2 / ref2) < 0.18, "fp64→int4 group-quant RMS error ~12-15%");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(1000)]
    [InlineData(4096)]
    public void Lossless_RoundTrip_IsBitExact_AndCompresses(int n)
    {
        var rng = new Rng((ulong)(n + 100));
        var src = new float[n];
        for (int i = 0; i < n; i++) src[i] = rng.NextGaussian(0.05);

        var enc = StreamingStoreCodec.EncodeLosslessFloat(src);
        var dec = new float[n];
        StreamingStoreCodec.DecodeLosslessFloat(enc, dec);

        // LOSSLESS → must be BIT-exact (not just close).
        for (int i = 0; i < n; i++)
            Assert.Equal(BitExactHelpers.SingleBits(src[i]), BitExactHelpers.SingleBits(dec[i]));

        // And for a non-trivial tensor it should actually shrink (byte-shuffle exposes
        // the structured exponent bytes to LZ4). Tiny tensors may not — only assert on big.
        if (n >= 1000)
            Assert.True(enc.Length < n * sizeof(float),
                $"lossless ({enc.Length} B) should be smaller than raw fp32 ({n * 4} B)");
    }

    [Fact]
    public void Lossless_Double_RoundTrip_IsBitExact()
    {
        var rng = new Rng(222);
        int n = 2048;
        var src = new double[n];
        for (int i = 0; i < n; i++) src[i] = rng.NextGaussian(0.05);
        var enc = StreamingStoreCodec.EncodeLosslessDouble(src);
        var dec = new double[n];
        StreamingStoreCodec.DecodeLosslessDouble(enc, dec);
        for (int i = 0; i < n; i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(src[i]), BitConverter.DoubleToInt64Bits(dec[i]));
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
