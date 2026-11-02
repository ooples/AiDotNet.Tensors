// Copyright (c) CrowdTrainer LLC. All rights reserved.
//
// Narrow-N direct GEMM on the 512-bit registers: a classifier head's [rows x 256]·[256 x 10] has fewer output
// columns than one 16-wide tile. Broadcasting A against a row of B (the 256-bit narrow kernel's form) left six of
// sixteen lanes idle and spent a load per FMA; here each output is a 16-lane dot product of an A row with a column of
// B (B transposed once), every lane busy, nine loads per twenty FMAs.

#if NET9_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    private const int NarrowDotRows = 4;
    private const int NarrowDotCols = 5;
    private const int NarrowDotMinK = 32;

    /// <summary>A/B switch: AIDOTNET_NARROW512=0 keeps narrow products on the 256-bit kernel.</summary>
    internal static bool s_narrow512 = System.Environment.GetEnvironmentVariable("AIDOTNET_NARROW512") != "0";

    [ThreadStatic] private static float[]? t_narrowBt;

    // Single-threaded narrow products above this many FMAs leave for the row-parallel dispatcher.
    private const long NarrowRouteMaxWork = 4L * 1024 * 1024;

    /// <summary>
    /// Whether a plain row-major <c>[m, k]·[k, n]</c> product with fewer output columns than one 8-wide vector block
    /// past a multiple of eight (a classifier head's n = 10) should take <see cref="Sgemm(ReadOnlySpan{float}, int, bool, ReadOnlySpan{float}, int, bool, Span{float}, int, int, int)"/>,
    /// whose narrow route runs <see cref="TrySgemmNarrowN512"/>, instead of the general dispatcher (no JIT or
    /// direct-tile route for n % 8 != 0). Only where the 512-bit tile runs.
    /// </summary>
    internal static bool PrefersNarrowN512(int m, int k, int n)
        => s_narrow512 && UseVector512Direct && n > 0 && n < Nr && (n & 7) != 0
           && m >= NarrowDotRows && k >= NarrowDotMinK && k <= SmallMatmulKThreshold && (long)m * k * n <= NarrowRouteMaxWork;

    /// <summary>
    /// <c>C[m, n] (=|+=) A[m, k] · B[k, n]</c> for n &lt;= 16 as dot products: B's columns copied once into Bt
    /// [n][kVec] (kVec = k rounded down to 16), then four rows of A by five columns of Bt a step, 16 lanes of k each,
    /// summed across lanes; k's last k % 16 terms are added after the lane sum, in order. Deterministic (fixed order, no
    /// threads). False (nothing touched) where the host does not run the 512-bit tile or the shape does not fit.
    /// </summary>
    private static unsafe bool TrySgemmNarrowN512(
        float* pA, int lda, float* pB, int ldb, float* pC, int ldc, int m, int k, int n, bool clearedOutput)
    {
        if (!s_narrow512 || !UseVector512Direct || n <= 0 || n > 16 || m <= 0 || k < NarrowDotMinK) return false;
        int kVec = k & ~15;
        int groups = (n + NarrowDotCols - 1) / NarrowDotCols;
        // One zero row past the last group: a group of fewer than five columns points its spare slots at it.
        int need = (groups * NarrowDotCols + 1) * kVec;
        var buffer = t_narrowBt;
        if (buffer is null || buffer.Length < need) t_narrowBt = buffer = new float[Math.Max(need, 4096)];
        fixed (float* bt = buffer)
        {
            for (int j = 0; j < n; j++)
            {
                float* src = pB + j, dst = bt + (long)j * kVec;
                for (int p = 0; p < kVec; p++) dst[p] = src[(long)p * ldb];
            }
            float* zero = bt + (long)groups * NarrowDotCols * kVec;
            new Span<float>(zero, kVec).Clear();

            int mFull = m & ~(NarrowDotRows - 1);
            for (int j0 = 0; j0 < n; j0 += NarrowDotCols)
            {
                int cols = Math.Min(NarrowDotCols, n - j0);
                float* b0 = bt + (long)j0 * kVec;
                float* b1 = cols > 1 ? b0 + kVec : zero, b2 = cols > 2 ? b0 + 2L * kVec : zero;
                float* b3 = cols > 3 ? b0 + 3L * kVec : zero, b4 = cols > 4 ? b0 + 4L * kVec : zero;
                for (int i = 0; i < mFull; i += NarrowDotRows)
                    NarrowDot4x5(pA + (long)i * lda, lda, b0, b1, b2, b3, b4, pB + j0, ldb, pC + (long)i * ldc + j0, ldc,
                        k, kVec, cols, clearedOutput);
                for (int i = mFull; i < m; i++)
                    NarrowDot1x5(pA + (long)i * lda, b0, b1, b2, b3, b4, pB + j0, ldb, pC + (long)i * ldc + j0,
                        k, kVec, cols, clearedOutput);
            }
        }
        return true;
    }

    [MethodImpl(Hot)]
    private static unsafe void NarrowDot4x5(float* a, int lda, float* b0, float* b1, float* b2, float* b3, float* b4,
        float* bTail, int ldb, float* c, int ldc, int k, int kVec, int cols, bool clearedOutput)
    {
        float* a0 = a, a1 = a + lda, a2 = a + 2L * lda, a3 = a + 3L * lda;
        Vector512<float> s00 = default, s01 = default, s02 = default, s03 = default, s04 = default;
        Vector512<float> s10 = default, s11 = default, s12 = default, s13 = default, s14 = default;
        Vector512<float> s20 = default, s21 = default, s22 = default, s23 = default, s24 = default;
        Vector512<float> s30 = default, s31 = default, s32 = default, s33 = default, s34 = default;
        for (int p = 0; p < kVec; p += 16)
        {
            var x0 = Vector512.Load(a0 + p); var x1 = Vector512.Load(a1 + p);
            var x2 = Vector512.Load(a2 + p); var x3 = Vector512.Load(a3 + p);
            var w = Vector512.Load(b0 + p);
            s00 = Vector512.FusedMultiplyAdd(x0, w, s00); s10 = Vector512.FusedMultiplyAdd(x1, w, s10);
            s20 = Vector512.FusedMultiplyAdd(x2, w, s20); s30 = Vector512.FusedMultiplyAdd(x3, w, s30);
            w = Vector512.Load(b1 + p);
            s01 = Vector512.FusedMultiplyAdd(x0, w, s01); s11 = Vector512.FusedMultiplyAdd(x1, w, s11);
            s21 = Vector512.FusedMultiplyAdd(x2, w, s21); s31 = Vector512.FusedMultiplyAdd(x3, w, s31);
            w = Vector512.Load(b2 + p);
            s02 = Vector512.FusedMultiplyAdd(x0, w, s02); s12 = Vector512.FusedMultiplyAdd(x1, w, s12);
            s22 = Vector512.FusedMultiplyAdd(x2, w, s22); s32 = Vector512.FusedMultiplyAdd(x3, w, s32);
            w = Vector512.Load(b3 + p);
            s03 = Vector512.FusedMultiplyAdd(x0, w, s03); s13 = Vector512.FusedMultiplyAdd(x1, w, s13);
            s23 = Vector512.FusedMultiplyAdd(x2, w, s23); s33 = Vector512.FusedMultiplyAdd(x3, w, s33);
            w = Vector512.Load(b4 + p);
            s04 = Vector512.FusedMultiplyAdd(x0, w, s04); s14 = Vector512.FusedMultiplyAdd(x1, w, s14);
            s24 = Vector512.FusedMultiplyAdd(x2, w, s24); s34 = Vector512.FusedMultiplyAdd(x3, w, s34);
        }
        NarrowDotStore(a0, c, s00, s01, s02, s03, s04, bTail, ldb, k, kVec, cols, clearedOutput);
        NarrowDotStore(a1, c + ldc, s10, s11, s12, s13, s14, bTail, ldb, k, kVec, cols, clearedOutput);
        NarrowDotStore(a2, c + 2L * ldc, s20, s21, s22, s23, s24, bTail, ldb, k, kVec, cols, clearedOutput);
        NarrowDotStore(a3, c + 3L * ldc, s30, s31, s32, s33, s34, bTail, ldb, k, kVec, cols, clearedOutput);
    }

    [MethodImpl(Hot)]
    private static unsafe void NarrowDot1x5(float* a, float* b0, float* b1, float* b2, float* b3, float* b4,
        float* bTail, int ldb, float* c, int k, int kVec, int cols, bool clearedOutput)
    {
        Vector512<float> s0 = default, s1 = default, s2 = default, s3 = default, s4 = default;
        for (int p = 0; p < kVec; p += 16)
        {
            var x = Vector512.Load(a + p);
            s0 = Vector512.FusedMultiplyAdd(x, Vector512.Load(b0 + p), s0);
            s1 = Vector512.FusedMultiplyAdd(x, Vector512.Load(b1 + p), s1);
            s2 = Vector512.FusedMultiplyAdd(x, Vector512.Load(b2 + p), s2);
            s3 = Vector512.FusedMultiplyAdd(x, Vector512.Load(b3 + p), s3);
            s4 = Vector512.FusedMultiplyAdd(x, Vector512.Load(b4 + p), s4);
        }
        NarrowDotStore(a, c, s0, s1, s2, s3, s4, bTail, ldb, k, kVec, cols, clearedOutput);
    }

    /// <summary>One row's <paramref name="cols"/> outputs: each lane sum, then k's tail terms in order, stored or added.</summary>
    [MethodImpl(HotInline)]
    private static unsafe void NarrowDotStore(float* aRow, float* cRow,
        Vector512<float> s0, Vector512<float> s1, Vector512<float> s2, Vector512<float> s3, Vector512<float> s4,
        float* bTail, int ldb, int k, int kVec, int cols, bool clearedOutput)
    {
        NarrowDotStoreOne(aRow, cRow, Vector512.Sum(s0), bTail, ldb, k, kVec, clearedOutput);
        if (cols > 1) NarrowDotStoreOne(aRow, cRow + 1, Vector512.Sum(s1), bTail + 1, ldb, k, kVec, clearedOutput);
        if (cols > 2) NarrowDotStoreOne(aRow, cRow + 2, Vector512.Sum(s2), bTail + 2, ldb, k, kVec, clearedOutput);
        if (cols > 3) NarrowDotStoreOne(aRow, cRow + 3, Vector512.Sum(s3), bTail + 3, ldb, k, kVec, clearedOutput);
        if (cols > 4) NarrowDotStoreOne(aRow, cRow + 4, Vector512.Sum(s4), bTail + 4, ldb, k, kVec, clearedOutput);
    }

    [MethodImpl(HotInline)]
    private static unsafe void NarrowDotStoreOne(float* aRow, float* c, float sum, float* bColumn, int ldb, int k, int kVec,
        bool clearedOutput)
    {
        for (int p = kVec; p < k; p++) sum = MathF.FusedMultiplyAdd(aRow[p], bColumn[(long)p * ldb], sum);
        *c = clearedOutput ? sum : *c + sum;
    }
}
#endif
