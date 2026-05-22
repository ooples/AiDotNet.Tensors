// Copyright (c) AiDotNet. All rights reserved.
// Tensors#401: regression coverage for the per-row-scaled INT8 weight-only
// GEMM. The kernel mirrors SgemmWithInt8CachedB's tile pipeline but skips
// the FP32→INT8 quantize step (consumer hands us pre-quantized sbyte[])
// and folds per-row scales into the per-tile dequant instead of using a
// single per-tensor scale.
//
// The matmul contract:
//   C[r, o] += A[r, i] * (B_int8[o, i] * rowScales[o])
// where B is laid out [n, k] row-major.
//
// Correctness is measured as SNR (signal-to-noise ratio) vs a scalar
// reference. INT8 quantization introduces quantization noise; we assert
// ≥ 30 dB SNR (≈30 dB ↔ 31× signal-to-noise ratio) which matches the
// existing per-tensor INT8 path's tolerance and is comfortable inside the
// 35-45 dB typical for transformer weights.

#if NET5_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

public class SgemmWithInt8RowScaledCachedBTests
{
    private static double SnrDb(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual)
    {
        Assert.Equal(reference.Length, actual.Length);
        double signal = 0, noise = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            signal += (double)reference[i] * reference[i];
            double diff = reference[i] - actual[i];
            noise += diff * diff;
        }
        if (noise == 0) return double.PositiveInfinity;
        return 10.0 * Math.Log10(signal / noise);
    }

    private static void ReferenceMatMul(
        ReadOnlySpan<float> a, sbyte[] bInt8, ReadOnlySpan<float> rowScales,
        Span<float> c, int m, int k, int n)
    {
        c.Clear();
        for (int r = 0; r < m; r++)
        for (int o = 0; o < n; o++)
        {
            double acc = 0;
            float scaleO = rowScales[o];
            for (int i = 0; i < k; i++)
                acc += (double)a[r * k + i] * (bInt8[o * k + i] * scaleO);
            c[r * n + o] = (float)acc;
        }
    }

    private static (float[] a, sbyte[] bInt8, float[] rowScales, float[] cRef, int m, int k, int n) BuildRandomCase(
        int m, int k, int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);

        var bInt8 = new sbyte[n * k];
        var rowScales = new float[n];
        for (int o = 0; o < n; o++)
        {
            // Per-row scale magnitude varies across rows so a correctness
            // regression that drops per-row scales (and falls back to one
            // scale) would manifest as a much larger error.
            rowScales[o] = (float)(0.01 + 0.05 * rng.NextDouble());
            for (int i = 0; i < k; i++)
                bInt8[o * k + i] = (sbyte)rng.Next(-127, 128);
        }

        var cRef = new float[m * n];
        ReferenceMatMul(a, bInt8, rowScales, cRef, m, k, n);
        return (a, bInt8, rowScales, cRef, m, k, n);
    }

    [Theory]
    [InlineData(4, 64, 64, 1)]      // small square
    [InlineData(16, 64, 16, 2)]     // m > Nr, n == Nr — exactly one panel
    [InlineData(8, 32, 24, 3)]      // m and n not multiples of micro-kernel
    [InlineData(1, 64, 32, 4)]      // gemv corner (m=1)
    [InlineData(32, 256, 64, 5)]    // medium k inside Kc
    [InlineData(8, 1024, 16, 6)]    // k > Kc — exercise pcIter loop
    [InlineData(8, 600, 32, 7)]     // k % Kc != 0 — tail Kc panel
    [InlineData(4, 32, 5000, 8)]    // n > Nc — large-n fallback path
    // CodeRabbit #427 review comment regression: canParallelize=true AND
    // NumColSubBlocks=1 (n < Nr*4 ≈ 64). m*k*n = 64*1024*32 = 2 MiB hits
    // ParallelWorkThreshold's 2 MiB floor on any core count; n=32 < 64 forces
    // NumColSubBlocks=1. Pre-fix this NRE'd at packedABuf!.
    [InlineData(64, 1024, 32, 9)]   // parallel-work threshold met + single col sub-block
    public void Correctness_VsScalarReference_AtSnr30Db(int m, int k, int n, int seed)
    {
        var tc = BuildRandomCase(m, k, n, seed);
        var c = new float[m * n];

        SimdGemm.SgemmWithInt8RowScaledCachedB(
            tc.a.AsSpan(), tc.bInt8, tc.rowScales.AsSpan(),
            c.AsSpan(), m, k, n);

        double snr = SnrDb(tc.cRef, c);
        Assert.True(snr >= 30.0, $"SNR {snr:F1} dB below 30 dB threshold (m={m}, k={k}, n={n})");
    }

    [Fact]
    public void CacheHit_SecondCallSameBReference_BitIdenticalOutput()
    {
        // The cache is keyed on the sbyte[] reference; if the second call
        // hits the cache, the packed tiles + scales are reused. Output
        // must be bit-identical to the first call (same packed layout,
        // same kernel, same input A — only the pack step is skipped).
        var tc = BuildRandomCase(8, 128, 32, 100);
        var c1 = new float[8 * 32];
        var c2 = new float[8 * 32];

        SimdGemm.SgemmWithInt8RowScaledCachedB(
            tc.a.AsSpan(), tc.bInt8, tc.rowScales.AsSpan(), c1.AsSpan(), 8, 128, 32);
        SimdGemm.SgemmWithInt8RowScaledCachedB(
            tc.a.AsSpan(), tc.bInt8, tc.rowScales.AsSpan(), c2.AsSpan(), 8, 128, 32);

        // Bit-identical: compare IEEE-754 bit patterns, not numeric equality.
        // Assert.Equal(float, float) treats NaN-with-different-payloads / -0 vs +0
        // as equal — neither should occur here, but the test's name promises
        // bit-identity so the assertion should enforce it (CodeRabbit #427).
        for (int i = 0; i < c1.Length; i++)
            Assert.Equal(
                BitConverter.SingleToInt32Bits(c1[i]),
                BitConverter.SingleToInt32Bits(c2[i]));
    }

    [Fact]
    public void ScaleChange_ForcesRebuild_OutputReflectsNewScales()
    {
        // The cache compares cached rowScales against current; if they
        // differ, the cached pack is discarded and rebuilt. Verify that
        // a scale change after the first call DOES change the output
        // (i.e., we are not silently using the stale cached scales).
        var tc = BuildRandomCase(4, 64, 16, 200);
        var c1 = new float[4 * 16];
        var c2 = new float[4 * 16];

        SimdGemm.SgemmWithInt8RowScaledCachedB(
            tc.a.AsSpan(), tc.bInt8, tc.rowScales.AsSpan(), c1.AsSpan(), 4, 64, 16);

        var newScales = (float[])tc.rowScales.Clone();
        for (int i = 0; i < newScales.Length; i++) newScales[i] *= 2.0f;
        SimdGemm.SgemmWithInt8RowScaledCachedB(
            tc.a.AsSpan(), tc.bInt8, newScales.AsSpan(), c2.AsSpan(), 4, 64, 16);

        // c2 should be approximately 2x c1 (same A, same B, scales doubled).
        // Use a relative-error check to allow for quantization-noise effects;
        // any silent stale-cache reuse would give c2 == c1, far outside this.
        double maxRelDiff = 0;
        for (int i = 0; i < c1.Length; i++)
        {
            float expected = 2.0f * c1[i];
            if (Math.Abs(expected) < 1e-6) continue;
            double rel = Math.Abs(c2[i] - expected) / Math.Abs(expected);
            if (rel > maxRelDiff) maxRelDiff = rel;
        }
        Assert.True(maxRelDiff < 0.01,
            $"max relative diff {maxRelDiff:E3} too large — cache may have returned stale-scale output");
    }

    [Fact]
    public void ZeroDims_NoThrow_ZeroedOutput()
    {
        // The kernel must early-return cleanly for any zero dim. Output
        // should be zeroed (the standard GEMM contract: C is cleared on
        // entry regardless of dims).
        var a = new float[0];
        var b = new sbyte[0];
        var scales = new float[0];
        var c = new float[0];
        SimdGemm.SgemmWithInt8RowScaledCachedB(
            a.AsSpan(), b, scales.AsSpan(), c.AsSpan(), 0, 0, 0);
        // No exception is the assertion. (c.Length is 0, nothing to check.)
    }

    [Fact]
    public void ZeroK_NonEmptyC_PrefilledOutputIsCleared()
    {
        // CodeRabbit #427: the zero-dim test above only exercises m=k=n=0
        // (c.Length == 0), so the clear-on-entry contract is never observed.
        // m>0 ∧ n>0 ∧ k=0 still hits the early-return branch but C is non-empty;
        // the contract says C must be cleared regardless of dims.
        int m = 3, k = 0, n = 4;
        var a = new float[m * k];
        var b = new sbyte[n * k];
        var scales = new float[n];
        var c = new float[m * n];
        Array.Fill(c, 123.0f); // prefill with garbage to detect clear

        SimdGemm.SgemmWithInt8RowScaledCachedB(
            a.AsSpan(), b, scales.AsSpan(), c.AsSpan(), m, k, n);

        for (int i = 0; i < c.Length; i++)
            Assert.Equal(0f, c[i]);
    }

    [Fact]
    public void RowScalesTooShort_Throws()
    {
        var a = new float[4 * 8];
        var b = new sbyte[16 * 8];
        var scales = new float[15];  // one short of n=16
        var c = new float[4 * 16];
        Assert.Throws<ArgumentException>(() =>
            SimdGemm.SgemmWithInt8RowScaledCachedB(
                a.AsSpan(), b, scales.AsSpan(), c.AsSpan(), 4, 8, 16));
    }

    [Fact]
    public void BInt8TooShort_Throws()
    {
        var a = new float[4 * 8];
        var b = new sbyte[16 * 8 - 1];  // one short of n*k
        var scales = new float[16];
        var c = new float[4 * 16];
        Assert.Throws<ArgumentException>(() =>
            SimdGemm.SgemmWithInt8RowScaledCachedB(
                a.AsSpan(), b, scales.AsSpan(), c.AsSpan(), 4, 8, 16));
    }
}

#endif
