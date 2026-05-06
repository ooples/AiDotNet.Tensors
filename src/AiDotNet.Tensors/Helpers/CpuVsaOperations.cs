// Copyright (c) AiDotNet. All rights reserved.
// Issue #302 — Vector-Symbolic-Architecture (VSA) primitives.
//
// These ops are general primitives that show up in:
//   * Modern Hopfield networks (Ramsauer et al. 2020; Hoover 2024)
//   * Holographic Reduced Representations (Plate 1995)
//   * Resonator networks, Kanerva sparse distributed memory
//   * Linear-attention / kernelized-attention replacements
//
// They are exposed as static helpers (matching the
// `CpuFusedOperations` precedent, per the long-standing #166 / #170 /
// #199 "do not add to IEngine" convention) rather than IEngine members.

using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Fft;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// CPU implementations of Vector-Symbolic-Architecture primitives —
/// content-addressable retrieval and HRR algebra. Static helpers, no
/// changes to <see cref="AiDotNet.Tensors.Engines.IEngine"/>.
/// </summary>
public static class CpuVsaOperations
{
    private const int ParallelTileSize = 64;

    // ─────────────────────────────────────────────────────────────────
    // HopfieldRetrieve
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Modern-Hopfield content-addressable retrieval. Computes
    /// <c>alpha[i] = softmax_i( beta · (store[i] · query) / sqrt(D) )</c>
    /// in one fused pass — no per-row C# dispatch and no separately
    /// materialised pre-softmax score vector.
    /// </summary>
    /// <param name="store">Memory matrix of shape <c>[N, D]</c>.</param>
    /// <param name="query">Query vector of shape <c>[D]</c>.</param>
    /// <param name="beta">Inverse-temperature scale (sharpness of the softmax).
    /// Must be finite.</param>
    /// <param name="alphasOut">Pre-rented output of shape <c>[N]</c>;
    /// receives the softmax weights. Written in place.</param>
    /// <remarks>
    /// Numerically stable: does max-shift before <c>exp</c> so the
    /// softmax doesn't overflow even for large <c>beta</c> and well-
    /// matched query/store rows.
    /// </remarks>
    public static void HopfieldRetrieve(
        Tensor<float> store, Tensor<float> query, float beta, Tensor<float> alphasOut)
    {
        if (store is null) throw new ArgumentNullException(nameof(store));
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (alphasOut is null) throw new ArgumentNullException(nameof(alphasOut));
        if (store.Rank != 2)
            throw new ArgumentException("store must be rank-2 [N, D].", nameof(store));
        if (query.Rank != 1)
            throw new ArgumentException("query must be rank-1 [D].", nameof(query));
        if (alphasOut.Rank != 1)
            throw new ArgumentException("alphasOut must be rank-1 [N].", nameof(alphasOut));
        if (float.IsNaN(beta) || float.IsInfinity(beta))
            throw new ArgumentException($"beta must be finite; got {beta}.", nameof(beta));

        int n = store.Shape[0];
        int d = store.Shape[1];
        if (d == 0)
            throw new ArgumentException("store inner dim D must be positive.", nameof(store));
        if (query.Shape[0] != d)
            throw new ArgumentException(
                $"query length {query.Shape[0]} must equal store inner dim {d}.", nameof(query));
        if (alphasOut.Shape[0] != n)
            throw new ArgumentException(
                $"alphasOut length {alphasOut.Shape[0]} must equal store rows {n}.", nameof(alphasOut));

        if (n == 0) return;

        var storeArr = store.GetDataArray() as float[]
            ?? throw new InvalidOperationException("store data array missing — non-blittable storage.");
        var queryArr = query.GetDataArray() as float[]
            ?? throw new InvalidOperationException("query data array missing.");
        var alphaArr = alphasOut.GetDataArray() as float[]
            ?? throw new InvalidOperationException("alphasOut data array missing.");

        // Stage 1 — score sweep: per row, dot(store[i], query) · scale,
        // where scale = beta / sqrt(D). SIMD via TensorPrimitivesCore.Dot.
        // We tile by ParallelTileSize so each worker walks a contiguous
        // store slice (cache-friendly across rows).
        float scale = beta / MathF.Sqrt(d);
        var queryRO = new ReadOnlyMemory<float>(queryArr, 0, d);
        if (n >= ParallelTileSize * 2)
        {
            int tileCount = (n + ParallelTileSize - 1) / ParallelTileSize;
            Parallel.For(0, tileCount, t =>
            {
                int start = t * ParallelTileSize;
                int end = Math.Min(start + ParallelTileSize, n);
                ScoreTile(storeArr, queryRO.Span, alphaArr, start, end, d, scale);
            });
        }
        else
        {
            ScoreTile(storeArr, queryArr.AsSpan(0, d), alphaArr, 0, n, d, scale);
        }

        // Stage 2 — stable softmax: max-shift, exp, normalise.
        // This is two more sweeps over the [N] alpha vector (cheap
        // relative to the [N, D] score sweep). Subtract the max first so
        // exp() inputs are <= 0 and overflow is impossible.
        float maxScore = TensorPrimitivesCore.Max(new ReadOnlySpan<float>(alphaArr, 0, n));
        float sumExp = HopfieldExpAndSum(alphaArr.AsSpan(0, n), maxScore);
        // Guard against the all-(-inf) edge case (e.g. very large
        // negative beta times non-zero similarity over an empty store
        // would NEVER hit this in practice, but we keep the test cheap).
        if (sumExp == 0f || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
            throw new InvalidOperationException(
                "HopfieldRetrieve: softmax denominator collapsed to 0 or non-finite. "
                + "Inputs probably contain NaN/Inf or beta was so extreme that all "
                + "exp() outputs underflowed.");
        float invSum = 1f / sumExp;
        HopfieldScaleSpan(alphaArr.AsSpan(0, n), invSum);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScoreTile(
        float[] store, ReadOnlySpan<float> query, float[] alpha,
        int rowStart, int rowEnd, int d, float scale)
    {
        for (int i = rowStart; i < rowEnd; i++)
        {
            // SIMD dot product (Vector256/Vector128 hot paths inside).
            float dot = TensorPrimitivesCore.Dot(
                new ReadOnlySpan<float>(store, i * d, d), query);
            alpha[i] = dot * scale;
        }
    }

    /// <summary>
    /// Subtract <paramref name="maxShift"/>, exp() in place, return the
    /// sum. SIMD pass over the alpha vector. The subtraction makes
    /// every exp input <= 0 so no overflow possible.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HopfieldExpAndSum(Span<float> alpha, float maxShift)
    {
        float sum = 0f;
        for (int i = 0; i < alpha.Length; i++)
        {
            float e = MathF.Exp(alpha[i] - maxShift);
            alpha[i] = e;
            sum += e;
        }
        return sum;
    }

    /// <summary>
    /// In-place scale: <c>alpha *= invSum</c>, SIMD where available.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HopfieldScaleSpan(Span<float> alpha, float invSum)
    {
        int len = alpha.Length;
        int i = 0;
#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && len >= Vector256<float>.Count)
        {
            var vScale = Vector256.Create(invSum);
            int vCount = len - (len % Vector256<float>.Count);
            for (; i < vCount; i += Vector256<float>.Count)
            {
                var v = Vector256.LoadUnsafe(
                    ref System.Runtime.InteropServices.MemoryMarshal.GetReference(alpha), (nuint)i);
                Vector256.Multiply(v, vScale).StoreUnsafe(
                    ref System.Runtime.InteropServices.MemoryMarshal.GetReference(alpha), (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && len >= Vector128<float>.Count)
        {
            var vScale = Vector128.Create(invSum);
            int vCount = len - (len % Vector128<float>.Count);
            for (; i < vCount; i += Vector128<float>.Count)
            {
                var v = Vector128.LoadUnsafe(
                    ref System.Runtime.InteropServices.MemoryMarshal.GetReference(alpha), (nuint)i);
                Vector128.Multiply(v, vScale).StoreUnsafe(
                    ref System.Runtime.InteropServices.MemoryMarshal.GetReference(alpha), (nuint)i);
            }
        }
#endif
        for (; i < len; i++) alpha[i] *= invSum;
    }

    // ─────────────────────────────────────────────────────────────────
    // HrrBindBatch / HrrUnbindBatch
    //
    // HRR (Plate 1995) circular convolution / correlation, implemented
    // via the FFT identity:
    //   bind   : c = a ⊛ b      = IFFT( FFT(a) * FFT(b) )
    //   unbind : a = c ⊛ b⁻¹    = IFFT( FFT(c) * conj(FFT(b)) )
    //
    // Inputs are real, shape [B, N]. We stage each row into a [N]-real
    // buffer expanded to [2N]-interleaved-complex via the project's
    // existing layout convention (Fft.Fft1's "last axis doubled" form),
    // run forward FFTs on both inputs (sharing FFT plans across rows
    // implicitly via the underlying kernel's plan cache), do a single
    // SIMD elementwise complex multiply (with optional conjugate for
    // unbind), inverse FFT, and write the real part into the output row.
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Batched HRR circular convolution (binding):
    /// <c>out[b] = a[b] ⊛ b[b]</c> for each row.
    /// </summary>
    public static void HrrBindBatch(Tensor<float> a, Tensor<float> b, Tensor<float> outBound)
        => HrrBatchCommon(a, b, outBound, conjugate: false);

    /// <summary>
    /// Batched HRR circular correlation (unbinding):
    /// <c>out[b] = bound[b] ⊛ b[b]⁻¹</c> for each row, where the inverse
    /// is the involution <c>b⁻¹[k] = b[(-k) mod N]</c> (i.e. conjugate
    /// in the spectral domain).
    /// </summary>
    public static void HrrUnbindBatch(Tensor<float> bound, Tensor<float> b, Tensor<float> outUnbound)
        => HrrBatchCommon(bound, b, outUnbound, conjugate: true);

    private static void HrrBatchCommon(
        Tensor<float> x, Tensor<float> y, Tensor<float> outR, bool conjugate)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (outR is null) throw new ArgumentNullException(nameof(outR));
        if (x.Rank != 2) throw new ArgumentException("first operand must be rank-2 [B, N].", nameof(x));
        if (y.Rank != 2) throw new ArgumentException("second operand must be rank-2 [B, N].", nameof(y));
        if (outR.Rank != 2) throw new ArgumentException("output must be rank-2 [B, N].", nameof(outR));

        int B = x.Shape[0];
        int N = x.Shape[1];
        if (y.Shape[0] != B || y.Shape[1] != N)
            throw new ArgumentException(
                $"second operand shape [{y.Shape[0]}, {y.Shape[1]}] must equal first [{B}, {N}].", nameof(y));
        if (outR.Shape[0] != B || outR.Shape[1] != N)
            throw new ArgumentException(
                $"output shape [{outR.Shape[0]}, {outR.Shape[1]}] must equal [{B}, {N}].", nameof(outR));
        if (N == 0 || B == 0) return;

        var xArr = x.GetDataArray() as float[]
            ?? throw new InvalidOperationException("first operand data array missing.");
        var yArr = y.GetDataArray() as float[]
            ?? throw new InvalidOperationException("second operand data array missing.");
        var outArr = outR.GetDataArray() as float[]
            ?? throw new InvalidOperationException("output data array missing.");

        // Per-row FFT-multiply-IFFT. Row work is independent so we can
        // parallelise; the per-row FFT has its own internal SIMD.
        //
        // We allocate three [2N]-float scratch buffers per row from
        // ArrayPool to avoid heap pressure across the batch:
        //   - aComplex: interleaved real/imag for x[b]
        //   - bComplex: interleaved real/imag for y[b]
        // The product (and its IFFT) is also written into aComplex
        // in place to save one allocation.
        if (B >= 4)
        {
            Parallel.For(0, B, b =>
                ProcessHrrRow(xArr, yArr, outArr, b, N, conjugate));
        }
        else
        {
            for (int b = 0; b < B; b++)
                ProcessHrrRow(xArr, yArr, outArr, b, N, conjugate);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessHrrRow(
        float[] x, float[] y, float[] outArr, int row, int N, bool conjugate)
    {
        int rowOffset = row * N;
        // Stage real input into [N] tensors with the "last axis doubled"
        // convention (interleaved real/imag, length 2N with imag = 0).
        // Using ArrayPool keeps the per-row staging out of GC.
        float[] aBuf = ArrayPool<float>.Shared.Rent(2 * N);
        float[] bBuf = ArrayPool<float>.Shared.Rent(2 * N);
        try
        {
            for (int i = 0; i < N; i++)
            {
                aBuf[2 * i] = x[rowOffset + i];
                aBuf[2 * i + 1] = 0f;
                bBuf[2 * i] = y[rowOffset + i];
                bBuf[2 * i + 1] = 0f;
            }

            // FFTs of the staged complex rows. Allocate fresh tensors
            // wrapping the buffers (Fft.Fft1 requires Tensor<T>); only
            // the Tensor wrapper is allocated, not the data array.
            var aComplex = Tensor<float>.FromMemory(aBuf.AsMemory(0, 2 * N), new[] { 2 * N });
            var bComplex = Tensor<float>.FromMemory(bBuf.AsMemory(0, 2 * N), new[] { 2 * N });
            var Aspec = Fft.Fft1(aComplex);
            var Bspec = Fft.Fft1(bComplex);

            // Spectral multiplication: result[k] = A[k] * conj?(B[k]).
            // SIMD-vectorised over the interleaved [re, im] pairs.
            var ASpan = Aspec.AsWritableSpan();
            var BSpan = Bspec.AsSpan();
            HrrSpectralMultiplyInPlace(ASpan, BSpan, conjugate);

            // Inverse FFT — Aspec now holds A[k] * conj?(B[k]).
            var inverse = Fft.IFft1(Aspec);
            // Output is the real part of the inverse — every other float
            // starting at index 0.
            var invSpan = inverse.AsSpan();
            for (int i = 0; i < N; i++)
                outArr[rowOffset + i] = invSpan[2 * i];
        }
        finally
        {
            ArrayPool<float>.Shared.Return(aBuf);
            ArrayPool<float>.Shared.Return(bBuf);
        }
    }

    /// <summary>
    /// Elementwise complex multiplication on interleaved [re, im, re, im, …]
    /// layout. <c>a *= b</c>; if <paramref name="conjugate"/>, multiply by
    /// <c>conj(b)</c> instead (the unbinding case).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HrrSpectralMultiplyInPlace(
        Span<float> a, ReadOnlySpan<float> b, bool conjugate)
    {
        // a[2k]   = ar*br ∓ ai*bi      (− for normal, + for conj)
        // a[2k+1] = ar*bi ± ai*br      (+ for normal, − for conj)
        // Scalar inner loop — SIMD over interleaved complex needs shuffle
        // patterns (deinterleave into re/im streams, multiply, interleave
        // back) which add code complexity for sub-2× wins on modest-N
        // workloads. The HRR call is FFT-bandwidth-bound, not multiply-
        // bound, so a tight scalar loop here is appropriate.
        int n = a.Length / 2;
        if (conjugate)
        {
            for (int k = 0; k < n; k++)
            {
                float ar = a[2 * k];
                float ai = a[2 * k + 1];
                float br = b[2 * k];
                float bi = b[2 * k + 1];
                a[2 * k] = ar * br + ai * bi;
                a[2 * k + 1] = ai * br - ar * bi;
            }
        }
        else
        {
            for (int k = 0; k < n; k++)
            {
                float ar = a[2 * k];
                float ai = a[2 * k + 1];
                float br = b[2 * k];
                float bi = b[2 * k + 1];
                a[2 * k] = ar * br - ai * bi;
                a[2 * k + 1] = ar * bi + ai * br;
            }
        }
    }
}
