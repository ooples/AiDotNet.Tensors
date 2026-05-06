// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using AiDotNet.Tensors.Engines.Simd;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// AVX2/FMA SIMD inner kernel helpers for
/// <see cref="FlashAttention{T}"/>'s <c>T == float</c> path.
/// Replaces the scalar Q·K^T, P·V, online-softmax-exp, and final
/// normalize loops with <see cref="Vector256{T}"/>-vectorized
/// equivalents that match what <see cref="FusedAttention"/>'s
/// rank-3 fast path delivers — but factored so the FlashAttention
/// generic-T outer loop can call them per-batch while preserving
/// LogSumExp tracking, attention bias, and queryOffset (KV-cache
/// decode), none of which the existing
/// <see cref="FusedAttention.FlashAttentionForwardPtr"/> supports.
///
/// <para>On net471 (no <see cref="Vector256{T}"/> intrinsic surface),
/// each method's SIMD bulk loop is <c>#if</c>'d out and the call
/// devolves to the scalar tail — same numerics, lower throughput.
/// On net5+ the SIMD path runs when AVX is supported (every x86-64
/// CPU made since 2011); FMA path adds Haswell+ FMA3.</para>
///
/// <para><b>Algorithm preserved verbatim from the scalar reference</b>
/// — these helpers swap arithmetic primitives, not the loop shape.
/// Online softmax (Dao 2023) and the alpha-rescale-then-accumulate
/// invariants are unchanged.</para>
/// </summary>
internal static class FlashAttentionFloatSimd
{
    /// <summary>
    /// AVX2/FMA dot product: <c>sum_{d=0..len-1} a[d] * b[d]</c>.
    /// 8-wide bulk loop with FMA when supported; scalar tail.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe float DotProduct(float* a, float* b, int len)
    {
        int d = 0;
        float scalar = 0f;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && len >= 8)
        {
            var acc = Vector256<float>.Zero;
            int bulkEnd = len - (len & 7);
            if (Fma.IsSupported)
            {
                for (; d < bulkEnd; d += 8)
                {
                    var va = Avx.LoadVector256(a + d);
                    var vb = Avx.LoadVector256(b + d);
                    acc = Fma.MultiplyAdd(va, vb, acc);
                }
            }
            else
            {
                for (; d < bulkEnd; d += 8)
                {
                    var va = Avx.LoadVector256(a + d);
                    var vb = Avx.LoadVector256(b + d);
                    acc = Avx.Add(acc, Avx.Multiply(va, vb));
                }
            }
            scalar = SimdKernels.HorizontalSum(acc);
        }
#endif
        for (; d < len; d++) scalar += a[d] * b[d];
        return scalar;
    }

    /// <summary>
    /// Online-softmax row update on the score row <paramref name="sRow"/>:
    /// vector-scan to find <c>rowMax</c>, then in-place
    /// <c>sRow[j] := exp(sRow[j] - mNew)</c> while accumulating the new
    /// partial denominator <c>lUpdate</c>. Caller combines
    /// <c>l_new = alpha · l_old + lUpdate</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void RowMaxAndExp(
        float* sRow, int kLen, float mPrev,
        out float mNew, out float alpha, out float lUpdate)
    {
        // 1. Row max — Vector256 max-reduce + scalar tail.
        int j = 0;
        float scalarMax = float.NegativeInfinity;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && kLen >= 8)
        {
            var vMax = Vector256.Create(float.NegativeInfinity);
            int bulkEnd = kLen - (kLen & 7);
            for (; j < bulkEnd; j += 8)
                vMax = Avx.Max(vMax, Avx.LoadVector256(sRow + j));
            scalarMax = SimdKernels.HorizontalMax(vMax);
        }
#endif
        for (; j < kLen; j++)
            if (sRow[j] > scalarMax) scalarMax = sRow[j];

        mNew = mPrev > scalarMax ? mPrev : scalarMax;
        alpha = float.IsNegativeInfinity(mPrev) ? 0f : MathF.Exp(mPrev - mNew);

        // 2. In-place exp(S - mNew) and accumulate the new partial l.
        // Using SimdKernels.FastExp256 — same approximation FusedAttention
        // uses for online softmax. Saturates at ~88.7; correct for
        // softmax inputs which are typically bounded by definition.
        j = 0;
        float lLocal = 0f;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && kLen >= 8)
        {
            var vMNew = Vector256.Create(mNew);
            var vAcc = Vector256<float>.Zero;
            int bulkEnd = kLen - (kLen & 7);
            for (; j < bulkEnd; j += 8)
            {
                var s = Avx.LoadVector256(sRow + j);
                var p = SimdKernels.FastExp256(Avx.Subtract(s, vMNew));
                Avx.Store(sRow + j, p);
                vAcc = Avx.Add(vAcc, p);
            }
            lLocal = SimdKernels.HorizontalSum(vAcc);
        }
#endif
        for (; j < kLen; j++)
        {
            float p = MathF.Exp(sRow[j] - mNew);
            sRow[j] = p;
            lLocal += p;
        }
        lUpdate = lLocal;
    }

    /// <summary>
    /// Output-block update for one Q row: <c>oBlock[d] = alpha · oBlock[d]
    /// + sum_j p[j] · v[j, d]</c> for d in [0, Dv).
    /// AVX2/FMA-vectorized over the Dv axis; scanned over the kLen axis.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void RescaleAndAccumulatePV(
        float* oRow, float alpha,
        float* sRow, float* vBase, int kStart, int kLen, int Dv,
        int vStrideElements)
    {
        // First step: rescale o by alpha (skipped when alpha == 1).
        if (alpha != 1f)
        {
            int d = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && Dv >= 8)
            {
                var vAlpha = Vector256.Create(alpha);
                int bulkEnd = Dv - (Dv & 7);
                for (; d < bulkEnd; d += 8)
                {
                    var oVec = Avx.LoadVector256(oRow + d);
                    Avx.Store(oRow + d, Avx.Multiply(oVec, vAlpha));
                }
            }
#endif
            for (; d < Dv; d++) oRow[d] *= alpha;
        }

        // Accumulate: o[d] += p[j] * v[j, d] for j in [0, kLen).
        // Inner SIMD axis is d; kLen iterations broadcast p[j] and FMA.
        for (int jj = 0; jj < kLen; jj++)
        {
            float p = sRow[jj];
            if (p == 0f) continue;
            float* vRow = vBase + (kStart + jj) * vStrideElements;
            int d = 0;
#if NET5_0_OR_GREATER
            if (Avx.IsSupported && Dv >= 8)
            {
                var vP = Vector256.Create(p);
                int bulkEnd = Dv - (Dv & 7);
                if (Fma.IsSupported)
                {
                    for (; d < bulkEnd; d += 8)
                    {
                        var oVec = Avx.LoadVector256(oRow + d);
                        var vVec = Avx.LoadVector256(vRow + d);
                        Avx.Store(oRow + d, Fma.MultiplyAdd(vP, vVec, oVec));
                    }
                }
                else
                {
                    for (; d < bulkEnd; d += 8)
                    {
                        var oVec = Avx.LoadVector256(oRow + d);
                        var vVec = Avx.LoadVector256(vRow + d);
                        Avx.Store(oRow + d, Avx.Add(oVec, Avx.Multiply(vP, vVec)));
                    }
                }
            }
#endif
            for (; d < Dv; d++) oRow[d] += p * vRow[d];
        }
    }

    /// <summary>
    /// Final per-row normalize: <c>o[d] *= invL</c> for d in [0, Dv).
    /// 8-wide multiply-by-broadcast.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void NormalizeRow(float* oRow, float invL, int Dv)
    {
        int d = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && Dv >= 8)
        {
            var vInv = Vector256.Create(invL);
            int bulkEnd = Dv - (Dv & 7);
            for (; d < bulkEnd; d += 8)
            {
                var oVec = Avx.LoadVector256(oRow + d);
                Avx.Store(oRow + d, Avx.Multiply(oVec, vInv));
            }
        }
#endif
        for (; d < Dv; d++) oRow[d] *= invL;
    }

    /// <summary>
    /// AXPY: <c>dst[d] += scale · src[d]</c>. Used by backward dV/dQ/dK
    /// accumulations where each (ii, jj) pair contributes a scaled
    /// row to the gradient buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AxpyAccumulate(float* dst, float* src, float scale, int len)
    {
        int d = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && len >= 8)
        {
            var vScale = Vector256.Create(scale);
            int bulkEnd = len - (len & 7);
            if (Fma.IsSupported)
            {
                for (; d < bulkEnd; d += 8)
                {
                    var dstVec = Avx.LoadVector256(dst + d);
                    var srcVec = Avx.LoadVector256(src + d);
                    Avx.Store(dst + d, Fma.MultiplyAdd(vScale, srcVec, dstVec));
                }
            }
            else
            {
                for (; d < bulkEnd; d += 8)
                {
                    var dstVec = Avx.LoadVector256(dst + d);
                    var srcVec = Avx.LoadVector256(src + d);
                    Avx.Store(dst + d, Avx.Add(dstVec, Avx.Multiply(vScale, srcVec)));
                }
            }
        }
#endif
        for (; d < len; d++) dst[d] += scale * src[d];
    }
}
