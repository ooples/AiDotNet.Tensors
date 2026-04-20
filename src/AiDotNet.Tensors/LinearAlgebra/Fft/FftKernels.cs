// Copyright (c) AiDotNet. All rights reserved.
// Self-contained, allocation-lean FFT kernels:
//   - IterativeRadix2 — Cooley-Tukey in-place for n = 2^k
//   - Bluestein      — chirp-z for arbitrary n via a length-M ≥ 2n-1 power-of-2 FFT
//   - ApplyScale     — per-FftNorm pre/post scaling
//
// Layout convention: interleaved real/imag pairs in `double[2 * n]`. Stride-free,
// cache-friendly, zero generic overhead. Upper tiers (Tensor-level, per-dtype)
// convert to/from this double layout once per call.
//
// Supply-chain note: no external FFT library is called. Twiddles are materialized
// from sin/cos; Bluestein chirps from the same. This is the entire numerical
// FFT surface for the library.

using System;

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

internal static class FftKernels
{
    /// <summary>
    /// In-place 1D FFT (arbitrary length). <paramref name="buf"/> is interleaved
    /// real/imag with <c>buf.Length == 2 * n</c>. On return, <paramref name="buf"/>
    /// holds the transformed spectrum; the chosen <see cref="FftNorm"/> scaling is applied.
    /// </summary>
    /// <param name="buf">Interleaved complex buffer, modified in place.</param>
    /// <param name="n">Logical transform length (<c>buf.Length / 2</c>).</param>
    /// <param name="inverse">True for IFFT (conjugate twiddles), false for FFT.</param>
    /// <param name="norm">Normalization convention; see <see cref="FftNorm"/>.</param>
    internal static void Transform1D(Span<double> buf, int n, bool inverse, FftNorm norm)
    {
        if (n <= 0) return;
        if (buf.Length < 2 * n) throw new ArgumentException("buf must hold at least 2*n doubles.", nameof(buf));
        if (n == 1)
        {
            ApplyScale(buf, n, inverse, norm);
            return;
        }
        if (IsPowerOfTwo(n))
        {
            IterativeRadix2(buf, n, inverse);
        }
        else
        {
            Bluestein(buf, n, inverse);
        }
        ApplyScale(buf, n, inverse, norm);
    }

    /// <summary>
    /// Radix-2 entrypoint that bypasses the plan-cache machinery. Used by
    /// <see cref="BluesteinPlan"/> to pre-FFT the chirp kernel at plan
    /// construction time (recursive-safe: avoids cache lookup inside
    /// Bluestein which is itself the cache population path).
    /// </summary>
    internal static void IterativeRadix2NoCache(Span<double> buf, int n, bool inverse)
        => IterativeRadix2(buf, n, inverse);

    // ── Cooley-Tukey radix-2, iterative, in place ───────────────────────────
    // Standard decimation-in-time formulation: bit-reverse permute, then
    // log₂ n stages of butterflies. Each stage is dispatched to the SIMD
    // kernel (AVX-512 / AVX2 when available) or falls through to the scalar
    // reference loop when either the lane width doesn't fit (small stages)
    // or the runtime doesn't expose the intrinsics (e.g. ARM / net471).
    private static unsafe void IterativeRadix2(Span<double> buf, int n, bool inverse)
    {
        BitReverseShuffle(buf, n);
        double sign = inverse ? 1.0 : -1.0;
        fixed (double* bufPtr = buf)
        {
            for (int size = 2; size <= n; size <<= 1)
            {
#if NET7_0_OR_GREATER
                if (FftSimdKernels.TryRadix2Stage(bufPtr, n, size, inverse))
                    continue;
#endif
                ScalarRadix2Stage(buf, n, size, sign);
            }
        }
    }

    // Scalar radix-2 stage (same math as before, extracted so the SIMD
    // dispatcher can fall through to it for tiny stages where the vector
    // lane count exceeds `half`).
    private static void ScalarRadix2Stage(Span<double> buf, int n, int size, double sign)
    {
        int half = size >> 1;
        double theta = sign * 2.0 * Math.PI / size;
        double wStepReal = Math.Cos(theta);
        double wStepImag = Math.Sin(theta);
        for (int start = 0; start < n; start += size)
        {
            double wReal = 1.0, wImag = 0.0;
            for (int k = 0; k < half; k++)
            {
                int eIdx = 2 * (start + k);
                int oIdx = 2 * (start + k + half);
                double eRe = buf[eIdx], eIm = buf[eIdx + 1];
                double oRe = buf[oIdx], oIm = buf[oIdx + 1];
                double tRe = wReal * oRe - wImag * oIm;
                double tIm = wReal * oIm + wImag * oRe;
                buf[eIdx] = eRe + tRe;
                buf[eIdx + 1] = eIm + tIm;
                buf[oIdx] = eRe - tRe;
                buf[oIdx + 1] = eIm - tIm;
                double nwRe = wReal * wStepReal - wImag * wStepImag;
                double nwIm = wReal * wStepImag + wImag * wStepReal;
                wReal = nwRe;
                wImag = nwIm;
            }
        }
    }

    // ── Bluestein chirp-z for arbitrary n ──────────────────────────────────
    // Identity: n*k = (n² + k² − (k−n)²) / 2, so the DFT
    //   X[k] = Σ x[n] · e^{−iπk²/N} · e^{iπ(k−n)²/N} · e^{−iπn²/N}
    //        = e^{−iπk²/N} · (a ⊛ b)[k]   with
    //   a[n] = x[n] · e^{−iπn²/N},    b[n] = e^{iπn²/N}
    // where ⊛ is aperiodic convolution. Compute the convolution with a length
    // M ≥ 2N − 1 power-of-2 radix-2 FFT; "wrap" b into the negative-index half
    // of the length-M buffer so the circular convolution mod M realizes the
    // desired aperiodic convolution for indices 0..N−1.
    //
    // For the inverse transform we conjugate: a[n] = x[n] · e^{+iπn²/N} etc.
    private static void Bluestein(Span<double> buf, int n, bool inverse)
    {
        var plan = FftPlanCache.GetOrCreateBluestein(n, inverse);
        int m = plan.M;
        var cRe = plan.ChirpRe;
        var cIm = plan.ChirpIm;

        // a[i] = x[i] * c[i]  for i in [0, n), zero-padded to M.
        Span<double> a = new double[2 * m];
        for (int i = 0; i < n; i++)
        {
            double xRe = buf[2 * i];
            double xIm = buf[2 * i + 1];
            a[2 * i] = xRe * cRe[i] - xIm * cIm[i];
            a[2 * i + 1] = xRe * cIm[i] + xIm * cRe[i];
        }

        // Convolve: A = FFT(a); multiply by pre-transformed B spectrum from the
        // cached plan (plan.BSpectrum{Re,Im}); IFFT; final chirp multiply.
        IterativeRadix2(a, m, inverse: false);
        var bRe = plan.BSpectrumRe;
        var bIm = plan.BSpectrumIm;
        for (int i = 0; i < m; i++)
        {
            double aRe = a[2 * i];
            double aIm = a[2 * i + 1];
            a[2 * i] = aRe * bRe[i] - aIm * bIm[i];
            a[2 * i + 1] = aRe * bIm[i] + aIm * bRe[i];
        }
        IterativeRadix2(a, m, inverse: true);
        double invM = 1.0 / m;
        for (int i = 0; i < 2 * m; i++) a[i] *= invM;

        // Final chirp multiply: X[k] = c[k] · (a ⊛ b)[k].
        for (int k = 0; k < n; k++)
        {
            double rRe = a[2 * k];
            double rIm = a[2 * k + 1];
            buf[2 * k] = rRe * cRe[k] - rIm * cIm[k];
            buf[2 * k + 1] = rRe * cIm[k] + rIm * cRe[k];
        }
    }

    /// <summary>
    /// In-place bit-reversal permutation for a length-n buffer (n must be a
    /// power of two). Pairs of interleaved complex elements are swapped
    /// according to the reversed binary representation of their index.
    /// </summary>
    internal static void BitReverseShuffle(Span<double> buf, int n)
    {
        int j = 0;
        for (int i = 1; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0)
            {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if (i < j)
            {
                int ii = 2 * i;
                int jj = 2 * j;
                (buf[ii], buf[jj]) = (buf[jj], buf[ii]);
                (buf[ii + 1], buf[jj + 1]) = (buf[jj + 1], buf[ii + 1]);
            }
        }
    }

    /// <summary>
    /// Apply the <see cref="FftNorm"/> scale factor to an in-place transform
    /// result. <paramref name="inverse"/> selects which side of the norm
    /// receives the extra factor.
    /// </summary>
    internal static void ApplyScale(Span<double> buf, int n, bool inverse, FftNorm norm)
    {
        double scale = ScaleFor(n, inverse, norm);
        if (scale == 1.0) return;
        for (int i = 0; i < 2 * n; i++) buf[i] *= scale;
    }

    /// <summary>
    /// Returns the scalar applied to a length-<paramref name="n"/> transform
    /// under the requested normalization convention.
    /// </summary>
    internal static double ScaleFor(int n, bool inverse, FftNorm norm) => norm switch
    {
        FftNorm.Backward => inverse ? 1.0 / n : 1.0,
        FftNorm.Forward => inverse ? 1.0 : 1.0 / n,
        FftNorm.Ortho => 1.0 / Math.Sqrt(n),
        _ => throw new ArgumentOutOfRangeException(nameof(norm), norm, "Unknown FftNorm."),
    };

    internal static bool IsPowerOfTwo(int n) => n > 0 && (n & (n - 1)) == 0;
}
