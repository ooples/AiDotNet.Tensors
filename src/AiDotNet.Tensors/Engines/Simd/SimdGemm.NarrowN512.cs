// Copyright (c) CrowdTrainer LLC. All rights reserved.
//
// Narrow-N direct GEMM on the 512-bit registers: a classifier head's [rows x 256]·[256 x 10] has fewer output
// columns than one 16-wide tile, so the 256-bit narrow kernel spent two FMAs per row per k step (the second
// mostly masked off). One zmm holds all sixteen columns: eight rows of one accumulator each.

#if NET9_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    private const int NarrowRows512 = 8;

    /// <summary>A/B switch: AIDOTNET_NARROW512=0 keeps narrow products on the 256-bit kernel.</summary>
    internal static bool s_narrow512 = System.Environment.GetEnvironmentVariable("AIDOTNET_NARROW512") != "0";

    [ThreadStatic] private static float[]? t_narrowPaddedB;

    /// <summary>
    /// <c>C[m, n] (=|+=) A[m, k] · B[k, n]</c> for n &lt;= 16 on the 512-bit tile: B copied once into a zero-padded
    /// [k][16] buffer (every load unmasked), eight rows of A per block, each element the same FMA chain over k the
    /// 256-bit narrow kernel runs, so the result is bit-identical to it. Rows past the last block of eight take that
    /// kernel. False (nothing touched) where the host does not run the 512-bit tile or the shape does not fit.
    /// </summary>
    private static unsafe bool TrySgemmNarrowN512(
        float* pA, int lda, float* pB, int ldb, float* pC, int ldc, int m, int k, int n, bool clearedOutput)
    {
        if (!s_narrow512 || !UseVector512Direct || n <= 0 || n > 16 || m < NarrowRows512 || k <= 0) return false;
        int need = k * 16;
        var buffer = t_narrowPaddedB;
        if (buffer is null || buffer.Length < need) t_narrowPaddedB = buffer = new float[Math.Max(need, 4096)];

        int lane0N = n >= 8 ? 8 : n;
        int lane1N = n > 8 ? n - 8 : 0;
        var mask0 = _partialNrMasks[lane0N].AsSingle();
        var mask1 = _partialNrMasks[lane1N].AsSingle();
        bool hasTail = lane1N > 0;
        fixed (float* bp = buffer)
        {
            // Masked loads read only B's n columns; the lanes past them load as zero.
            float* src = pB, dst = bp;
            for (int p = 0; p < k; p++, src += ldb, dst += 16)
            {
                Vector256.Store(lane0N == 8 ? Vector256.Load(src) : System.Runtime.Intrinsics.X86.Avx.MaskLoad(src, mask0), dst);
                Vector256.Store(hasTail ? System.Runtime.Intrinsics.X86.Avx.MaskLoad(src + 8, mask1) : Vector256<float>.Zero, dst + 8);
            }

            int mFull = (m / NarrowRows512) * NarrowRows512;
            for (int i = 0; i < mFull; i += NarrowRows512)
                NarrowKernel8x16(pA + (long)i * lda, lda, bp, pC + (long)i * ldc, ldc, k, mask0, mask1, hasTail, clearedOutput);
            for (int i = mFull; i < m; i += Mr)
                DirectKernelMxNarrow(pA + (long)i * lda, lda, pB, ldb, pC + (long)i * ldc, ldc, k, Math.Min(Mr, m - i), n, clearedOutput);
        }
        return true;
    }

    [MethodImpl(HotInline)]
    private static unsafe void NarrowKernel8x16(
        float* pA, int lda, float* bp, float* pC, int ldc, int k,
        Vector256<float> mask0, Vector256<float> mask1, bool hasTail, bool clearedOutput)
    {
        var c0 = Vector512<float>.Zero; var c1 = Vector512<float>.Zero;
        var c2 = Vector512<float>.Zero; var c3 = Vector512<float>.Zero;
        var c4 = Vector512<float>.Zero; var c5 = Vector512<float>.Zero;
        var c6 = Vector512<float>.Zero; var c7 = Vector512<float>.Zero;
        float* a0 = pA, a1 = pA + lda, a2 = pA + 2L * lda, a3 = pA + 3L * lda;
        float* a4 = pA + 4L * lda, a5 = pA + 5L * lda, a6 = pA + 6L * lda, a7 = pA + 7L * lda;
        float* b = bp;
        for (int p = 0; p < k; p++, b += 16)
        {
            var bv = Vector512.Load(b);
            c0 = Vector512.FusedMultiplyAdd(Vector512.Create(a0[p]), bv, c0);
            c1 = Vector512.FusedMultiplyAdd(Vector512.Create(a1[p]), bv, c1);
            c2 = Vector512.FusedMultiplyAdd(Vector512.Create(a2[p]), bv, c2);
            c3 = Vector512.FusedMultiplyAdd(Vector512.Create(a3[p]), bv, c3);
            c4 = Vector512.FusedMultiplyAdd(Vector512.Create(a4[p]), bv, c4);
            c5 = Vector512.FusedMultiplyAdd(Vector512.Create(a5[p]), bv, c5);
            c6 = Vector512.FusedMultiplyAdd(Vector512.Create(a6[p]), bv, c6);
            c7 = Vector512.FusedMultiplyAdd(Vector512.Create(a7[p]), bv, c7);
        }
        StoreNarrowRow(pC, mask0, mask1, c0.GetLower(), c0.GetUpper(), hasTail, clearedOutput);
        StoreNarrowRow(pC + ldc, mask0, mask1, c1.GetLower(), c1.GetUpper(), hasTail, clearedOutput);
        StoreNarrowRow(pC + 2L * ldc, mask0, mask1, c2.GetLower(), c2.GetUpper(), hasTail, clearedOutput);
        StoreNarrowRow(pC + 3L * ldc, mask0, mask1, c3.GetLower(), c3.GetUpper(), hasTail, clearedOutput);
        StoreNarrowRow(pC + 4L * ldc, mask0, mask1, c4.GetLower(), c4.GetUpper(), hasTail, clearedOutput);
        StoreNarrowRow(pC + 5L * ldc, mask0, mask1, c5.GetLower(), c5.GetUpper(), hasTail, clearedOutput);
        StoreNarrowRow(pC + 6L * ldc, mask0, mask1, c6.GetLower(), c6.GetUpper(), hasTail, clearedOutput);
        StoreNarrowRow(pC + 7L * ldc, mask0, mask1, c7.GetLower(), c7.GetUpper(), hasTail, clearedOutput);
    }
}
#endif
