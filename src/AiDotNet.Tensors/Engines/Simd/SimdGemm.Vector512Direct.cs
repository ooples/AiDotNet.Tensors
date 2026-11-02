// Copyright (c) CrowdTrainer LLC. All rights reserved.
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

    // Keep small-k calls from repeatedly growing the thread-local strip buffer.
    private const int MinPackedStripRows = 64;
    // 256 rows of 32 floats occupy 32 KB, leaving the packed tile in L1.
    private const int DefaultPackedStripRows = 256;

    /// <summary>
    /// Whether the direct GEMM paths run full 32-column strips on <see cref="DirectKernel6x32Store"/>: the runtime
    /// accelerates 512-bit vectors here (it reports false on CPUs where they downclock the core).
    /// </summary>
    internal static bool UseVector512Direct => s_forceVector512Direct ?? Vector512.IsHardwareAccelerated;

    /// <summary>Test override of <see cref="UseVector512Direct"/>: true runs the 512-bit tiles in software on any CPU (same
    /// results, slowly), false keeps them off. Null follows the hardware.</summary>
    internal static bool? s_forceVector512Direct =
        System.Environment.GetEnvironmentVariable("AIDOTNET_V512_DIRECT") == "0" ? false : null;   // A/B switch on AVX-512 hosts

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

    /// <summary>
    /// <see cref="DirectKernel6x32Store"/> resuming each element's FMA chain from the value C holds: the accumulators
    /// start at C instead of zero, so splitting k into blocks (each block's partial sums parked in C) runs exactly the
    /// chain the unsplit kernel runs, bit for bit.
    /// </summary>
    [MethodImpl(HotInline)]
    internal static unsafe void DirectKernel6x32Resume(float* pA, int lda, float* pB, int ldb, float* pC, int ldc, int k)
    {
        float* r0 = pC, r1 = pC + ldc, r2 = pC + 2 * ldc, r3 = pC + 3 * ldc, r4 = pC + 4 * ldc, r5 = pC + 5 * ldc;
        var c00 = Vector512.Load(r0); var c01 = Vector512.Load(r0 + 16);
        var c10 = Vector512.Load(r1); var c11 = Vector512.Load(r1 + 16);
        var c20 = Vector512.Load(r2); var c21 = Vector512.Load(r2 + 16);
        var c30 = Vector512.Load(r3); var c31 = Vector512.Load(r3 + 16);
        var c40 = Vector512.Load(r4); var c41 = Vector512.Load(r4 + 16);
        var c50 = Vector512.Load(r5); var c51 = Vector512.Load(r5 + 16);

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

        c00.Store(r0); c01.Store(r0 + 16);
        c10.Store(r1); c11.Store(r1 + 16);
        c20.Store(r2); c21.Store(r2 + 16);
        c30.Store(r3); c31.Store(r3 + 16);
        c40.Store(r4); c41.Store(r4 + 16);
        c50.Store(r5); c51.Store(r5 + 16);
    }

    /// <summary>
    /// Rows of k per block when the 512-bit direct tile reads a packed strip: 256 x 32 floats (32 KB) stays in L1 across
    /// the chunk's row blocks, where the whole strip (784 x 32 at a dense layer's input) streamed from L2 for every row
    /// block, and the A panel's lines with it. AIDOTNET_GEMM_KC overrides (0 = no blocking).
    /// </summary>
    internal static int s_stripKc =
        int.TryParse(System.Environment.GetEnvironmentVariable("AIDOTNET_GEMM_KC"), out int kc) && kc >= 0 ? kc : DefaultPackedStripRows;

    /// <summary>The 512-bit <see cref="TransAKernel6x16"/>: six rows x 32 columns of C = At*B, overwriting C; pA points at
    /// row i0 of At (A + i0), lda = m; C's rows are <paramref name="ldc"/> apart (B may be a packed strip, ldb = 32).</summary>
    [MethodImpl(HotInline)]
    internal static unsafe void TransAKernel6x32(float* pA, int lda, float* pB, int ldb, float* pC, int ldc, int k)
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
        c00.Store(pC); c01.Store(pC + 16); pC += ldc;
        c10.Store(pC); c11.Store(pC + 16); pC += ldc;
        c20.Store(pC); c21.Store(pC + 16); pC += ldc;
        c30.Store(pC); c31.Store(pC + 16); pC += ldc;
        c40.Store(pC); c41.Store(pC + 16); pC += ldc;
        c50.Store(pC); c51.Store(pC + 16);
    }

    /// <summary>
    /// Whether the 512-bit direct tiles read a 32-column strip of B from a contiguous copy (AIDOTNET_GEMM_PACK_STRIP=0
    /// turns it off). B's rows sit ldb floats apart, and at a power-of-two row length (n = 512: 2 KB) every row of a strip
    /// lands in the same few L1 sets, so the strip evicts itself and streams from L2 for each row block; the copy (k x 32
    /// floats) spreads over every set and is reused by all the row blocks of a chunk. Same FMA chains: bit-identical.
    /// </summary>
    internal static bool s_packStrip512 = System.Environment.GetEnvironmentVariable("AIDOTNET_GEMM_PACK_STRIP") != "0";

    /// <summary>Most rows (k) a packed strip takes: 1024 x 32 floats, 128 KB.</summary>
    internal const int MaxPackedStripK = 1024;

    [ThreadStatic] private static float[]? t_packedStrip;

    /// <summary>A thread's strip buffer, at least <paramref name="k"/> x 32 floats.</summary>
    internal static float[] PackedStripBuffer(int k)
    {
        var buffer = t_packedStrip;
        if (buffer is null || buffer.Length < k * Nr512) t_packedStrip = buffer = new float[Math.Max(k, MinPackedStripRows) * Nr512];
        return buffer;
    }

    /// <summary>Copies B's columns [0, 32) of rows [0, k) (rows ldb apart) to <paramref name="dst"/>, 32 floats a row.</summary>
    [MethodImpl(HotInline)]
    internal static unsafe void PackStrip32(float* src, int ldb, float* dst, int k)
    {
        for (int p = 0; p < k; p++, src += ldb, dst += Nr512)
        {
            Vector512.Store(Vector512.Load(src), dst);
            Vector512.Store(Vector512.Load(src + 16), dst + 16);
        }
    }
}
#endif
