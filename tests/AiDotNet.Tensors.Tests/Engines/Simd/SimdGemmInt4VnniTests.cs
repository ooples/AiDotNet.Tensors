// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Correctness of the int4-native VNNI GEMM (Phase A / #1622). Validated two ways: (1) against an
/// independent scalar reimplementation of the exact u8-activation × s8-weight per-group integer
/// math (must match the kernel — VNNI or scalar fallback — to float rounding), and (2) bounded
/// against the fp32 weight-only int4 GEMM (the two differ only by the int8 activation quantization,
/// ~1%). Covers batch=1 and K not a multiple of the group size (forces the run-splitting path).
/// </summary>
public class SimdGemmInt4VnniTests
{
    private struct Rng
    {
        private ulong _s;
        public Rng(ulong seed) { _s = seed | 1UL; }
        public double NextUnit() { ulong x = _s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; _s = x; return (x >> 11) * (1.0 / (1UL << 53)); }
        public float NextGaussian(double std) { double u1 = Math.Max(1e-12, NextUnit()), u2 = NextUnit(); return (float)(std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)); }
    }

    private static (sbyte[] data, float[] scales, int gs) Quantize(float[] w, int n, int k, int gs)
    {
        var enc = new byte[StreamingStoreCodec.Int4BufferBytes(n * k, gs)];
        StreamingStoreCodec.EncodeInt4Float(w, enc, gs);
        return (StreamingStoreCodec.Int4DataOf(enc, n * k), StreamingStoreCodec.Int4ScalesOf(enc), gs);
    }

    // Independent scalar reference for the EXACT quantized math the VNNI kernel performs:
    // quantize activations to uint8 per row (same quantizer), then per-group integer dot with the
    // -128*Σw zero-point correction and per-group + per-row scales.
    private static float[] ScalarQuantReference(float[] a, sbyte[] wData, float[] scales, int gs, int m, int k, int n)
    {
        var aU8 = new byte[m * k];
        var actScale = new float[m];
        Int8Quantizer.QuantizeActivationsPerRowToUint8(a, m, k, aU8, actScale);
        var c = new float[m * n];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
            {
                double acc = 0.0;
                long wBase = (long)j * k, aBase = (long)i * k;
                int p = 0;
                while (p < k)
                {
                    long flat = wBase + p;
                    int g = (int)(flat / gs);
                    int len = Math.Min((int)((long)(g + 1) * gs - flat), k - p);
                    int dot = 0, wsum = 0;
                    for (int t = 0; t < len; t++)
                    {
                        dot += aU8[aBase + p + t] * wData[wBase + p + t];
                        wsum += wData[wBase + p + t];
                    }
                    acc += (dot - 128 * wsum) * (double)scales[g];
                    p += len;
                }
                c[i * n + j] = (float)(acc * actScale[i]);
            }
        return c;
    }

    private static float[] Fp32WeightOnlyReference(float[] a, sbyte[] wData, float[] scales, int gs, int m, int k, int n)
    {
        var c = new float[m * n];
        SimdGemm.SgemmWithInt4GroupScaled(a, wData, scales, gs, c, m, k, n);
        return c;
    }

    private static double RelFroErr(float[] x, float[] y)
    {
        double num = 0, den = 0;
        for (int i = 0; i < x.Length; i++) { double d = (double)x[i] - y[i]; num += d * d; den += (double)y[i] * y[i]; }
        return Math.Sqrt(num / Math.Max(1e-30, den));
    }

    [Theory]
    [InlineData(1, 256, 512, 128)]    // batch=1, group-aligned K (VNNI fast path)
    [InlineData(4, 512, 1024, 128)]
    [InlineData(2, 130, 257, 128)]    // K not a multiple of group size → run-splitting
    public void Int4Vnni_MatchesScalarQuantReference(int m, int k, int n, int gs)
    {
        var rng = new Rng((ulong)(m * 7 + k * 13 + n + gs));
        var a = new float[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextGaussian(0.1);
        var w = new float[n * k];
        for (int i = 0; i < w.Length; i++) w[i] = rng.NextGaussian(0.05);
        var (data, scales, groupSize) = Quantize(w, n, k, gs);

        var c = new float[m * n];
        SimdGemm.SgemmWithInt4GroupScaledVnni(a, data, scales, groupSize, c, m, k, n);
        var cRef = ScalarQuantReference(a, data, scales, groupSize, m, k, n);

        // Same integer math (VNNI path or scalar fallback) → equal to float-rounding.
        Assert.True(RelFroErr(c, cRef) < 1e-4,
            $"int4 VNNI vs scalar-quant reference rel err {RelFroErr(c, cRef):E3} (m={m},k={k},n={n})");
    }

    [Fact]
    public void Int4Vnni_CloseToFp32WeightOnly_WithinActivationQuantError()
    {
        const int m = 4, k = 512, n = 1024, gs = 128;
        var rng = new Rng(2024);
        var a = new float[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextGaussian(0.1);
        var w = new float[n * k];
        for (int i = 0; i < w.Length; i++) w[i] = rng.NextGaussian(0.05);
        var (data, scales, groupSize) = Quantize(w, n, k, gs);

        var cVnni = new float[m * n];
        SimdGemm.SgemmWithInt4GroupScaledVnni(a, data, scales, groupSize, cVnni, m, k, n);
        var cFp32 = Fp32WeightOnlyReference(a, data, scales, groupSize, m, k, n);

        // The two paths share the int4 weight; they differ only by the int8 activation quant.
        Assert.True(RelFroErr(cVnni, cFp32) < 0.05,
            $"int4 VNNI vs fp32 weight-only rel err {RelFroErr(cVnni, cFp32):E3} should be within ~int8 activation quant (~1-3%)");
    }
}
