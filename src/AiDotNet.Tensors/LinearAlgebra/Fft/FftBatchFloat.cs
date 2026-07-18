// Copyright (c) AiDotNet. All rights reserved.
//
// Float-native, batch-vectorized radix-2 Cooley-Tukey FFT.
//
// The generic TransformComplex1D reference does one length-N FFT per row in
// DOUBLE (float input -> per-element ToDouble, a per-row double[2N] scratch, a
// per-call twiddle/bit-reversal table inside FftKernels.Transform1D, then
// FromDouble back). On a batched float workload ([B, 2N]) that means B tiny
// transforms, B scratch allocations, and full float<->double conversion — it
// loses to a single cache-blocked BLAS DFT matmul by ~30x and allocates ~6x the
// output size per call.
//
// This kernel instead vectorizes ACROSS the batch. Data is transposed into SoA
// planes re/im laid out [position, batch] so that for a fixed transform position
// the B batch values are contiguous. Every radix-2 butterfly is then a pure
// SVec op across the batch lanes with NO cross-lane shuffle; the twiddle
// for a butterfly is a scalar broadcast to all lanes. Twiddle + bit-reversal
// tables are computed once and cached. Work is split into batch-contiguous chunks
// across cores (each chunk is an independent set of transforms, cache-local
// through all log2(N) stages). End-to-end float, pooled scratch, single output
// allocation.

using System;
using System.Buffers;
using System.Collections.Generic;
using SVec = System.Numerics.Vector<float>;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

internal static class FftBatchFloat
{
    public static bool IsPowerOfTwo(int x) => x > 0 && (x & (x - 1)) == 0;

    // Cached (bit-reversal perm, twiddle-real, twiddle-imag) per (n, inverse).
    // Twiddle sign depends on direction; the bit-reversal perm does not, but we
    // key the whole triple together for a single lookup.
    private static readonly Dictionary<long, (int[] brev, float[] twR, float[] twI)> _cache = new();
    private static readonly object _cacheLock = new();

    private static (int[] brev, float[] twR, float[] twI) GetTables(int n, bool inverse)
    {
        long key = ((long)n << 1) | (inverse ? 1L : 0L);
        lock (_cacheLock)
        {
            if (_cache.TryGetValue(key, out var t)) return t;

            int logn = 0; while ((1 << logn) < n) logn++;   // n is pow2 → exact log2 (net471-portable)
            var brev = new int[n];
            for (int i = 0; i < n; i++)
            {
                int r = 0, x = i;
                for (int b = 0; b < logn; b++) { r = (r << 1) | (x & 1); x >>= 1; }
                brev[i] = r;
            }
            // Twiddles for stage lengths 2,4,...,n; k in [0, len/2). Total n-1 entries.
            var twR = new float[n];
            var twI = new float[n];
            double sign = inverse ? 1.0 : -1.0;
            int off = 0;
            for (int len = 2; len <= n; len <<= 1)
            {
                int half = len >> 1;
                for (int k = 0; k < half; k++)
                {
                    double ang = sign * 2.0 * Math.PI * k / len;
                    twR[off + k] = (float)Math.Cos(ang);
                    twI[off + k] = (float)Math.Sin(ang);
                }
                off += half;
            }
            var tables = (brev, twR, twI);
            _cache[key] = tables;
            return tables;
        }
    }

    /// <summary>
    /// Transform interleaved-complex rows [batch, 2*nIn] into [batch, 2*n] in
    /// float, vectorized across the batch. Zero-pads (n &gt; nIn) or crops
    /// (n &lt; nIn) like the reference. Returns false (handles nothing) if n is
    /// not a power of two — caller falls back to the generic double path.
    /// </summary>
    public static bool TryTransform(
        float[] inData, int inStride,
        float[] outData, int outStride,
        int batch, int nIn, int n, bool inverse, FftNorm norm)
    {
        if (!IsPowerOfTwo(n)) return false;
        if (batch <= 0) return true;

        var (brev, twR, twI) = GetTables(n, inverse);

        float scale = norm switch
        {
            FftNorm.Backward => inverse ? 1f / n : 1f,
            FftNorm.Forward => inverse ? 1f : 1f / n,
            FftNorm.Ortho => (float)(1.0 / Math.Sqrt(n)),
            _ => 1f,
        };

        int W = SVec.Count;
        int copy = Math.Min(nIn, n);
        bool pad = copy < n;

        int procs = Math.Max(1, Environment.ProcessorCount);
        int numChunks = Math.Min(procs, batch);
        int chunk = (batch + numChunks - 1) / numChunks;

        Parallel.For(0, numChunks, c =>
        {
            int b0 = c * chunk;
            int b1 = Math.Min(batch, b0 + chunk);
            int lb = b1 - b0;
            if (lb <= 0) return;

            var pool = ArrayPool<float>.Shared;
            float[] re = pool.Rent(n * lb);
            float[] im = pool.Rent(n * lb);
            try
            {
                if (pad)
                {
                    Array.Clear(re, 0, n * lb);
                    Array.Clear(im, 0, n * lb);
                }

                // ---- transpose + bit-reversal IN: re[brev(i)*lb + bb] = row-b pos-i ----
                for (int bb = 0; bb < lb; bb++)
                {
                    int inBase = (b0 + bb) * inStride;
                    for (int i = 0; i < copy; i++)
                    {
                        int j = brev[i];
                        re[j * lb + bb] = inData[inBase + 2 * i];
                        im[j * lb + bb] = inData[inBase + 2 * i + 1];
                    }
                }

                // ---- radix-2 stages, SIMD across the batch lanes ----
                int off = 0;
                for (int len = 2; len <= n; len <<= 1)
                {
                    int half = len >> 1;
                    for (int k = 0; k < half; k++)
                    {
                        float wrS = twR[off + k], wiS = twI[off + k];
                        var wr = new SVec(wrS);
                        var wi = new SVec(wiS);
                        for (int p0 = 0; p0 < n; p0 += len)
                        {
                            int baseA = (p0 + k) * lb;
                            int baseB = (p0 + k + half) * lb;
                            int bb = 0;
                            for (; bb + W <= lb; bb += W)
                            {
                                var ar = new SVec(re, baseA + bb);
                                var ai = new SVec(im, baseA + bb);
                                var br = new SVec(re, baseB + bb);
                                var bi = new SVec(im, baseB + bb);
                                var tr = wr * br - wi * bi;
                                var ti = wr * bi + wi * br;
                                (ar + tr).CopyTo(re, baseA + bb);
                                (ai + ti).CopyTo(im, baseA + bb);
                                (ar - tr).CopyTo(re, baseB + bb);
                                (ai - ti).CopyTo(im, baseB + bb);
                            }
                            for (; bb < lb; bb++)
                            {
                                float ar = re[baseA + bb], ai = im[baseA + bb];
                                float br = re[baseB + bb], bi = im[baseB + bb];
                                float tr = wrS * br - wiS * bi;
                                float ti = wrS * bi + wiS * br;
                                re[baseA + bb] = ar + tr; im[baseA + bb] = ai + ti;
                                re[baseB + bb] = ar - tr; im[baseB + bb] = ai - ti;
                            }
                        }
                    }
                    off += half;
                }

                // ---- transpose OUT + norm ----
                for (int bb = 0; bb < lb; bb++)
                {
                    int outBase = (b0 + bb) * outStride;
                    for (int i = 0; i < n; i++)
                    {
                        outData[outBase + 2 * i] = re[i * lb + bb] * scale;
                        outData[outBase + 2 * i + 1] = im[i * lb + bb] * scale;
                    }
                }
            }
            finally
            {
                pool.Return(re);
                pool.Return(im);
            }
        });
        return true;
    }
}
