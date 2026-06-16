// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Correctness of the weight-only int4 group-quant GEMM (Phase A / #1622). The kernel
/// dequantizes the int4 weight on the fly; its result must match a reference fp32 GEMM that
/// uses the SAME dequantized weights (so the only difference is float accumulation order).
/// Covers batch=1 (the N-parallel latency case) and the parallel path (large N).
/// </summary>
public class SimdGemmInt4GroupScaledTests
{
    private struct Rng
    {
        private ulong _s;
        public Rng(ulong seed) { _s = seed | 1UL; }
        public double NextUnit() { ulong x = _s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; _s = x; return (x >> 11) * (1.0 / (1UL << 53)); }
        public float NextGaussian(double std) { double u1 = Math.Max(1e-12, NextUnit()), u2 = NextUnit(); return (float)(std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)); }
    }

    // Quantize W[n,k] to int4 group-quant, then return (sign-extended data, scales, groupSize)
    // and the dequantized fp32 weight used for the reference.
    private static (sbyte[] data, float[] scales, int gs, float[] wDeq) QuantizeWeight(float[] w, int n, int k, int groupSize)
    {
        var enc = new byte[StreamingStoreCodec.Int4BufferBytes(n * k, groupSize)];
        StreamingStoreCodec.EncodeInt4Float(w, enc, groupSize);
        var data = StreamingStoreCodec.Int4DataOf(enc, n * k);
        var scales = StreamingStoreCodec.Int4ScalesOf(enc);
        var wDeq = new float[n * k];
        StreamingStoreCodec.DecodeInt4Float(enc, wDeq);
        return (data, scales, groupSize, wDeq);
    }

    private static float[] ReferenceGemm(float[] a, float[] wDeq, int m, int k, int n)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int p = 0; p < k; p++) acc += a[i * k + p] * wDeq[j * k + p];
                c[i * n + j] = acc;
            }
        return c;
    }

    // Relative Frobenius norm ‖x-y‖ / ‖y‖ — the standard GEMM-correctness metric. Per-element
    // relative error is meaningless when an individual output is ~0 from cancellation; the norm
    // measures the aggregate deviation, which is what "the kernel computes the same product"
    // actually means.
    private static double RelFroErr(float[] x, float[] y)
    {
        double num = 0, den = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double d = (double)x[i] - y[i];
            num += d * d;
            den += (double)y[i] * y[i];
        }
        return Math.Sqrt(num / Math.Max(1e-30, den));
    }

    [Theory]
    [InlineData(1, 256, 512, 128)]    // batch=1 (latency case), large N → parallel path
    [InlineData(1, 4096, 64, 128)]    // batch=1, huge K (FFN-like)
    [InlineData(4, 320, 1280, 128)]   // small batch, wide N
    [InlineData(8, 130, 257, 128)]    // K and N not multiples of groupSize / vector width
    public void Int4Gemm_MatchesReferenceDequantGemm(int m, int k, int n, int gs)
    {
        var rng = new Rng((ulong)(m * 131 + k * 17 + n + gs));
        var a = new float[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextGaussian(0.1);
        var w = new float[n * k];
        for (int i = 0; i < w.Length; i++) w[i] = rng.NextGaussian(0.05);

        var (data, scales, groupSize, wDeq) = QuantizeWeight(w, n, k, gs);
        var cRef = ReferenceGemm(a, wDeq, m, k, n);

        var c = new float[m * n];
        SimdGemm.SgemmWithInt4GroupScaled(a, data, scales, groupSize, c, m, k, n);

        // Same dequantized weights → only float accumulation-order differences remain.
        Assert.True(RelFroErr(c, cRef) < 1e-4,
            $"int4 GEMM rel Frobenius err {RelFroErr(c, cRef):E3} vs dequant reference should be ~0 (m={m},k={k},n={n})");
    }

    [Fact]
    public void Int4Gemm_ParallelMatchesSerial()
    {
        // Large enough N to exercise the N-parallel path; compare against the serial path
        // by toggling the global parallel flag.
        const int m = 2, k = 512, n = 2048, gs = 128;
        var rng = new Rng(99);
        var a = new float[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextGaussian(0.1);
        var w = new float[n * k];
        for (int i = 0; i < w.Length; i++) w[i] = rng.NextGaussian(0.05);
        var (data, scales, groupSize, _) = QuantizeWeight(w, n, k, gs);

        bool saved = SimdGemm.UseParallelGemm;
        try
        {
            SimdGemm.UseParallelGemm = false;
            var cSerial = new float[m * n];
            SimdGemm.SgemmWithInt4GroupScaled(a, data, scales, groupSize, cSerial, m, k, n);

            SimdGemm.UseParallelGemm = true;
            var cParallel = new float[m * n];
            SimdGemm.SgemmWithInt4GroupScaled(a, data, scales, groupSize, cParallel, m, k, n);

            // N-partitioning does not change per-output accumulation order → bit-identical.
            for (int i = 0; i < cSerial.Length; i++)
                Assert.Equal(cSerial[i], cParallel[i]);
        }
        finally { SimdGemm.UseParallelGemm = saved; }
    }
}
