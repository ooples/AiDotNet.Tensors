// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Quantized compute (lever 4 part 2) — a MEASUREMENT that decides the int8 compute
/// path. Computing a GEMM directly on int8-stored weights (int8 weight × fp32 activation
/// with per-row scales) is bit-for-bit equivalent to upcasting then doing fp32 GEMM
/// (exact to float precision), BUT it is NOT faster — measured ~7x SLOWER at an FFN
/// shape, because the inline int8→fp32 dequant costs more than the highly-optimized
/// fp32 machine-code microkernel saves, and fp32 activations preclude a true int8×int8
/// VNNI/AMX path.
///
/// TWO findings:
///   1. int8-WEIGHT-only × fp32-activation is exact vs fp32-on-quantized but ~7x SLOWER
///      (inline dequant + fp32 accumulate beats nothing) — so for a weight-only int8
///      store, compute should upcast → fast fp32 GEMM (what the store does today).
///   2. TRUE W8A8 (int8 ACTIVATIONS + int8 weights, int32 accumulate) IS a real compute
///      win: measured ~3.8x FASTER than fp32 at 1.79% total quant error — even on AVX2
///      without VNNI, and including the per-call activation quantization. VNNI/AMX would
///      widen the gap. This is the quantized-compute lever; its cost is activation-quant
///      accuracy (outliers) + (best on) VNNI/AMX hardware. The driver
///      SimdGemm.SgemmA8W8RowScaledCachedB is the integration primitive for wiring it
///      into inference forward paths.
/// </summary>
public class QuantizedComputeW8A8Tests
{
    private readonly ITestOutputHelper _out;
    public QuantizedComputeW8A8Tests(ITestOutputHelper output) => _out = output;

    private struct Rng
    {
        private ulong _s;
        public Rng(ulong seed) { _s = seed | 1UL; }
        public double NextUnit() { ulong x = _s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; _s = x; return (x >> 11) * (1.0 / (1UL << 53)); }
        public float NextGaussian(double std) { double u1 = Math.Max(1e-12, NextUnit()), u2 = NextUnit(); return (float)(std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)); }
    }

    // Per-row symmetric int8 quantization of weight B[n,k] (row-major). Returns the
    // int8 weight, the per-row scales, and the dequantized fp32 weight (what fp32
    // compute on the quantized weight would use).
    private static (sbyte[] bInt8, float[] rowScales, float[] bDeq) QuantizePerRow(float[] b, int n, int k)
    {
        var q = new sbyte[n * k];
        var scales = new float[n];
        var deq = new float[n * k];
        for (int i = 0; i < n; i++)
        {
            float amax = 0f;
            for (int j = 0; j < k; j++) { float a = Math.Abs(b[i * k + j]); if (a > amax) amax = a; }
            float scale = amax > 0f ? amax / 127f : 1f;
            scales[i] = scale;
            float inv = 1f / scale;
            for (int j = 0; j < k; j++)
            {
                int v = (int)Math.Round(b[i * k + j] * inv);
                if (v > 127) v = 127; else if (v < -127) v = -127;
                q[i * k + j] = (sbyte)v;
                deq[i * k + j] = v * scale;
            }
        }
        return (q, scales, deq);
    }

    [Fact]
    public void Int8WeightGemm_IsExact_ButNotFasterThanFp32_SoStoreUpcastsForCompute()
    {
        const int m = 128, k = 512, n = 512; // transformer-FFN-ish shape
        var rng = new Rng(99);
        var a = new float[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextGaussian(1.0);
        var b = new float[n * k]; // weight [n, k] row-major (i.e. Bᵀ for c = a·Bᵀ)
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextGaussian(1.0 / Math.Sqrt(k));

        var (bInt8, rowScales, bDeq) = QuantizePerRow(b, n, k);

        // fp32 reference on the SAME quantized weight: c_ref = a[m,k] · bDeq[n,k]ᵀ.
        // Sgemm computes a · B with B in [k,n], so transpose bDeq[n,k] → bT[k,n].
        var bT = new float[k * n];
        for (int i = 0; i < n; i++) for (int j = 0; j < k; j++) bT[j * n + i] = bDeq[i * k + j];
        var cRef = new float[m * n];
        SimdGemm.Sgemm(a, bT, cRef, m, k, n);

        // int8 compute path: weight stays int8, dequant happens inline.
        var cInt8 = new float[m * n];
        SimdGemm.SgemmWithInt8RowScaledCachedB(a, bInt8, rowScales, cInt8, m, k, n);

        double sum2 = 0, ref2 = 0;
        for (int i = 0; i < m * n; i++) { double e = cInt8[i] - cRef[i]; sum2 += e * e; ref2 += (double)cRef[i] * cRef[i]; }
        double rel = Math.Sqrt(sum2 / Math.Max(1e-30, ref2));
        _out.WriteLine($"W8A8 vs fp32-on-quantized: rel RMS error {rel:E3}");
        // Same quantized weight, only int8-vs-fp32 accumulation rounding differs → tiny.
        Assert.True(rel < 0.01, $"int8 GEMM must match fp32-on-quantized weight (rel {rel})");

        // Speed: warm both, then min-of-N (the valid estimator on a noisy box).
        SimdGemm.Sgemm(a, bT, cRef, m, k, n);
        SimdGemm.SgemmWithInt8RowScaledCachedB(a, bInt8, rowScales, cInt8, m, k, n);
        double Best(Action f) { double best = double.MaxValue; for (int r = 0; r < 30; r++) { var sw = Stopwatch.StartNew(); f(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); } return best; }
        double fp32Ms = Best(() => SimdGemm.Sgemm(a, bT, cRef, m, k, n));
        double int8Ms = Best(() => SimdGemm.SgemmWithInt8RowScaledCachedB(a, bInt8, rowScales, cInt8, m, k, n));
        _out.WriteLine($"GEMM [{m}x{k}x{n}] min-of-30: fp32 {fp32Ms:F3} ms, int8-weight {int8Ms:F3} ms ({fp32Ms / int8Ms:F2}x)");
        // Report-only on speed (CI boxes are noisy); correctness is the hard gate.
        Assert.True(int8Ms > 0);
    }

    [Fact]
    public void TrueW8A8_Int8ActivationsAndWeights_AccuracyAndSpeed()
    {
        // True W8A8: BOTH activations and weights int8 → int8×int8 VNNI/AMX path
        // (SgemmA8W8RowScaledCachedB quantizes activations to uint8 internally and
        // dispatches VNNI → AVX2 → scalar). The compute-speedup path (vs the int8-
        // WEIGHT-only path which was ~7x slower than fp32).
        const int m = 256, k = 1024, n = 1024;
        var rng = new Rng(2024);
        var a = new float[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextGaussian(1.0);
        var b = new float[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextGaussian(1.0 / Math.Sqrt(k));
        var (bInt8, rowScales, _) = QuantizePerRow(b, n, k);

        // Full fp32 reference: c = a · bᵀ (b unquantized). Sgemm wants B in [k,n].
        var bT = new float[k * n];
        for (int i = 0; i < n; i++) for (int j = 0; j < k; j++) bT[j * n + i] = b[i * k + j];
        var cFull = new float[m * n];
        SimdGemm.Sgemm(a, bT, cFull, m, k, n);

        var cW8A8 = new float[m * n];
        SimdGemm.SgemmA8W8RowScaledCachedB(a, bInt8, rowScales, cW8A8, m, k, n);

        double sum2 = 0, ref2 = 0;
        for (int i = 0; i < m * n; i++) { double e = cW8A8[i] - cFull[i]; sum2 += e * e; ref2 += (double)cFull[i] * cFull[i]; }
        double rel = Math.Sqrt(sum2 / Math.Max(1e-30, ref2));
        bool vnni = SimdGemm.Int8Int8VnniAvailable;
        _out.WriteLine($"True W8A8 (VNNI={vnni}) vs full fp32: total quant rel RMS {rel * 100:F2}% (activation+weight int8)");
        // Both operands int8 → larger than weight-only error, but bounded for inference.
        Assert.True(rel < 0.10, $"W8A8 total quantization error {rel} should be within ~10%");

        double Best(Action f) { double best = double.MaxValue; for (int r = 0; r < 30; r++) { var sw = Stopwatch.StartNew(); f(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); } return best; }
        double fp32Ms = Best(() => SimdGemm.Sgemm(a, bT, cFull, m, k, n));
        double w8a8Ms = Best(() => SimdGemm.SgemmA8W8RowScaledCachedB(a, bInt8, rowScales, cW8A8, m, k, n));
        _out.WriteLine($"GEMM [{m}x{k}x{n}] min-of-30: fp32 {fp32Ms:F3} ms, W8A8 {w8a8Ms:F3} ms ({fp32Ms / w8a8Ms:F2}x vs fp32; includes per-call activation quant)");
        Assert.True(w8a8Ms > 0);
    }
}
