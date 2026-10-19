// Copyright (c) AiDotNet. All rights reserved.
//
// The 512-bit counterpart of the direct 6x16 kernel: a 6x32 output tile in twelve Vector512 accumulators. On a core
// with two 512-bit FMA ports (Xeon Sapphire/Emerald/Granite Rapids) it does twice the 256-bit kernel's FLOPs per cycle;
// on Zen 4, whose 512-bit FMAs are double-pumped, it halves the instruction count for the same FLOPs.
//
// Each output element is still one fused multiply-add chain over k in ascending order, exactly as the 256-bit kernel
// computes it, so a GEMM gives the same bits on either kernel: a model's results do not depend on the host's vector
// width.

#if NET9_0_OR_GREATER
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>Output columns of the 512-bit direct tile.</summary>
    internal const int Nr512 = 32;

    /// <summary>
    /// Whether the direct GEMM paths run full 32-column strips on <see cref="DirectKernel6x32Store"/>: the runtime
    /// accelerates 512-bit vectors here (it reports false on CPUs where they downclock the core).
    /// </summary>
    internal static bool UseVector512Direct => s_forceVector512Direct ?? Vector512.IsHardwareAccelerated;

    /// <summary>Test override of <see cref="UseVector512Direct"/>: true runs the 512-bit tiles in software on any CPU (same
    /// results, slowly), false keeps them off. Null follows the hardware.</summary>
    internal static bool? s_forceVector512Direct;

    /// <summary><c>C[6,32] = A[6,k] · B[k,32]</c> (row-major, C overwritten): the 6x16 direct kernel at twice the width.</summary>
    [MethodImpl(HotInline)]
    internal static unsafe void DirectKernel6x32Store(float* pA, int lda, float* pB, int ldb, float* pC, int ldc, int k)
    {
        var c00 = Vector512<float>.Zero; var c01 = Vector512<float>.Zero;
        var c10 = Vector512<float>.Zero; var c11 = Vector512<float>.Zero;
        var c20 = Vector512<float>.Zero; var c21 = Vector512<float>.Zero;
        var c30 = Vector512<float>.Zero; var c31 = Vector512<float>.Zero;
        var c40 = Vector512<float>.Zero; var c41 = Vector512<float>.Zero;
        var c50 = Vector512<float>.Zero; var c51 = Vector512<float>.Zero;

        float* pA0 = pA;
        float* pA1 = pA + lda;
        float* pA2 = pA + lda * 2;
        float* pA3 = pA + lda * 3;
        float* pA4 = pA + lda * 4;
        float* pA5 = pA + lda * 5;

        for (int p = 0; p < k; p++)
        {
            var b0 = Vector512.Load(pB);
            var b1 = Vector512.Load(pB + 16);

            var a0 = Vector512.Create(pA0[p]);
            c00 = Vector512.FusedMultiplyAdd(a0, b0, c00); c01 = Vector512.FusedMultiplyAdd(a0, b1, c01);

            var a1 = Vector512.Create(pA1[p]);
            c10 = Vector512.FusedMultiplyAdd(a1, b0, c10); c11 = Vector512.FusedMultiplyAdd(a1, b1, c11);

            var a2 = Vector512.Create(pA2[p]);
            c20 = Vector512.FusedMultiplyAdd(a2, b0, c20); c21 = Vector512.FusedMultiplyAdd(a2, b1, c21);

            var a3 = Vector512.Create(pA3[p]);
            c30 = Vector512.FusedMultiplyAdd(a3, b0, c30); c31 = Vector512.FusedMultiplyAdd(a3, b1, c31);

            var a4 = Vector512.Create(pA4[p]);
            c40 = Vector512.FusedMultiplyAdd(a4, b0, c40); c41 = Vector512.FusedMultiplyAdd(a4, b1, c41);

            var a5 = Vector512.Create(pA5[p]);
            c50 = Vector512.FusedMultiplyAdd(a5, b0, c50); c51 = Vector512.FusedMultiplyAdd(a5, b1, c51);

            pB += ldb;
        }

        c00.Store(pC); c01.Store(pC + 16); pC += ldc;
        c10.Store(pC); c11.Store(pC + 16); pC += ldc;
        c20.Store(pC); c21.Store(pC + 16); pC += ldc;
        c30.Store(pC); c31.Store(pC + 16); pC += ldc;
        c40.Store(pC); c41.Store(pC + 16); pC += ldc;
        c50.Store(pC); c51.Store(pC + 16);
    }

    /// <summary>The 512-bit <see cref="TransAKernel6x16"/>: six rows x 32 columns of C = At*B, overwriting C; pA points at
    /// row i0 of At (A + i0), lda = m, and C's row stride equals B's (n).</summary>
    [MethodImpl(HotInline)]
    internal static unsafe void TransAKernel6x32(float* pA, int lda, float* pB, int ldb, float* pC, int k)
    {
        var c00 = Vector512<float>.Zero; var c01 = Vector512<float>.Zero;
        var c10 = Vector512<float>.Zero; var c11 = Vector512<float>.Zero;
        var c20 = Vector512<float>.Zero; var c21 = Vector512<float>.Zero;
        var c30 = Vector512<float>.Zero; var c31 = Vector512<float>.Zero;
        var c40 = Vector512<float>.Zero; var c41 = Vector512<float>.Zero;
        var c50 = Vector512<float>.Zero; var c51 = Vector512<float>.Zero;
        for (int p = 0; p < k; p++)
        {
            var b0 = Vector512.Load(pB);
            var b1 = Vector512.Load(pB + 16);
            var a0 = Vector512.Create(pA[0]);
            c00 = Vector512.FusedMultiplyAdd(a0, b0, c00); c01 = Vector512.FusedMultiplyAdd(a0, b1, c01);
            var a1 = Vector512.Create(pA[1]);
            c10 = Vector512.FusedMultiplyAdd(a1, b0, c10); c11 = Vector512.FusedMultiplyAdd(a1, b1, c11);
            var a2 = Vector512.Create(pA[2]);
            c20 = Vector512.FusedMultiplyAdd(a2, b0, c20); c21 = Vector512.FusedMultiplyAdd(a2, b1, c21);
            var a3 = Vector512.Create(pA[3]);
            c30 = Vector512.FusedMultiplyAdd(a3, b0, c30); c31 = Vector512.FusedMultiplyAdd(a3, b1, c31);
            var a4 = Vector512.Create(pA[4]);
            c40 = Vector512.FusedMultiplyAdd(a4, b0, c40); c41 = Vector512.FusedMultiplyAdd(a4, b1, c41);
            var a5 = Vector512.Create(pA[5]);
            c50 = Vector512.FusedMultiplyAdd(a5, b0, c50); c51 = Vector512.FusedMultiplyAdd(a5, b1, c51);
            pA += lda;
            pB += ldb;
        }
        c00.Store(pC); c01.Store(pC + 16); pC += ldb;
        c10.Store(pC); c11.Store(pC + 16); pC += ldb;
        c20.Store(pC); c21.Store(pC + 16); pC += ldb;
        c30.Store(pC); c31.Store(pC + 16); pC += ldb;
        c40.Store(pC); c41.Store(pC + 16); pC += ldb;
        c50.Store(pC); c51.Store(pC + 16);
    }
}
#endif
