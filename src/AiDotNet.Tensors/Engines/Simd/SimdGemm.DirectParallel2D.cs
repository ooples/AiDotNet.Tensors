// Copyright (c) CrowdTrainer LLC. All rights reserved.
//
// Small-M direct GEMM partitioned over BOTH output axes.
//
// A training batch (m = 32..192) is too few rows for the existing parallel paths to fill a many-core box:
// SgemmDirectParallelM splits only the Mr=6 row blocks (a 64-row batch yields ~5 usable chunks) and is gated
// to k <= 512 and >= 8M FMAs, so the parity MLP's [64x512]x[512x128] ran single-threaded; and the packed
// SgemmTiled path, which takes every k > 512, put a 64-row [64x784]x[784x512] through ONE packed-A task per Kc
// panel and capped the column split at 8 tiles, reaching ~130 GFLOP/s on 16 threads while the direct 6x16
// kernel alone runs ~105 GFLOP/s on ONE.
//
// This path keeps that direct kernel (no packing; each output element is one register FMA chain over the
// full k, so the result does not depend on how the output is partitioned) and fans disjoint output
// rectangles across the pool: columns first, so each worker's B column block stays L2-resident while every
// 6-row A panel streams through L1, then rows when the column strips alone cannot supply enough chunks.

#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>Largest M routed to <see cref="TrySgemmDirectParallel2D"/>. Above it the row blocks alone
    /// supply enough parallelism and the M-parallel / packed paths keep their tuned behaviour.</summary>
    internal const int DirectParallel2DMaxM = 192;

    /// <summary>Largest K routed to <see cref="TrySgemmDirectParallel2D"/>: a 6-row A panel of K floats must
    /// stay L1-resident across the column strips it is reused for (6 x 1024 x 4 B = 24 KB of a 32 KB L1d).</summary>
    internal const int DirectParallel2DMaxK = 1024;

    /// <summary>A/B and test toggle (like <see cref="UseParallelGemm"/>): false sends these GEMMs back to the
    /// paths below the 2D gate. Not a production setting.</summary>
    internal static bool UseDirectParallel2D = true;

    /// <summary>GEMMs the 2D path computed on this thread; a test probe that a gated shape actually took it.</summary>
    [ThreadStatic] internal static int t_directParallel2DRuns;
    internal static bool s_zzPackA2D; // probe A/B only
    internal static bool s_zzPack16; // probe A/B only

    /// <summary>
    /// <c>C[m,n] = A[m,k] · B[k,n]</c> (row-major, no transpose, ldc = n, C overwritten) over disjoint output
    /// rectangles in parallel. Every output element is one direct-kernel FMA chain over the full k, so the result
    /// is bit-identical for every partition -- including the serial one this runs when the thread count or the
    /// work allows only one chunk, which keeps a gated shape reproducible across thread counts (deterministic
    /// mode) and across <paramref name="allowParallel"/> (false runs the same kernels on the calling thread, for
    /// callers already inside a parallel region). Returns false, having touched nothing, only for shapes the 6x16
    /// kernels do not tile (m &lt; Mr, n &lt; Nr).
    /// </summary>
    internal static unsafe bool TrySgemmDirectParallel2D(
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c,
        int m, int k, int n, bool allowParallel)
    {
        if (m < Mr || n < Nr || k <= 0) return false;
        int cores = allowParallel ? Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism) : 1;
        int chunks = CapDirectChunksByWork(cores, (long)m * k * n);

        // Full 32-column strips run the 512-bit tile where the host accelerates it (same bits as two 16-column tiles).
#if NET9_0_OR_GREATER
        int strip = UseVector512Direct ? Nr512 : Nr;
#else
        const int strip = Nr;
#endif
        int rowBlocks = (m + Mr - 1) / Mr;    // the last block may be a partial (masked) one
        int colStrips = (n + strip - 1) / strip; // likewise the last strip
        int colChunks = Math.Max(1, Math.Min(chunks, colStrips));
        int rowChunks = Math.Min(rowBlocks, Math.Max(1, chunks / colChunks));
        int items = colChunks * rowChunks;

        fixed (float* pAroot = a, pBroot = b, pCroot = c)
        {
            IntPtr ipA = (IntPtr)pAroot, ipB = (IntPtr)pBroot, ipC = (IntPtr)pCroot;
            int mCap = m, kCap = k, nCap = n, ldaCap = lda, ldbCap = ldb;
            int colChunksCap = colChunks, rowChunksCap = rowChunks, rowBlocksCap = rowBlocks, colStripsCap = colStrips, stripCap = strip;

            Action<int> body = item =>
            {
                int rc = item / colChunksCap, cc = item % colChunksCap;
                // Balanced split of whole row blocks / column strips: every chunk gets floor or ceil of the share.
                int rb0 = (int)((long)rc * rowBlocksCap / rowChunksCap), rb1 = (int)((long)(rc + 1) * rowBlocksCap / rowChunksCap);
                int cs0 = (int)((long)cc * colStripsCap / colChunksCap), cs1 = (int)((long)(cc + 1) * colStripsCap / colChunksCap);
                float* pA = (float*)ipA, pB = (float*)ipB, pC = (float*)ipC;

                // Column strip outer: one strip of B (k x strip floats) stays in L2 across the chunk's row blocks. Row
                // blocks outer re-streamed the chunk's whole B slice from L3 per row block (mlp 128x784x512: 1.12x
                // scaling at 2 threads against PyTorch's 1.84x).
#if NET9_0_OR_GREATER
                // A full 32-column strip read by two or more full row blocks is copied once into a contiguous buffer
                // (see s_packStrip512): B's rows ldb apart thrash a few L1 sets at power-of-two row lengths.
                bool pack = stripCap == Nr512 && s_packStrip512 && kCap <= MaxPackedStripK
                    && Math.Min(rb1, mCap / Mr) - rb0 >= 2;
                float[]? stripBuffer = pack ? PackedStripBuffer(kCap) : null;
#endif
                for (int cs = cs0; cs < cs1; cs++)
                {
                    int stripStart = cs * stripCap;
                    int stripEnd = Math.Min(nCap, stripStart + stripCap);
#if NET9_0_OR_GREATER
                    if (stripBuffer is not null && stripEnd - stripStart == Nr512 && s_zzPackA2D)
                    {
                        int fullEnd = Math.Min(rb1, mCap / Mr);
                        int fullBlocks = fullEnd - rb0;
                        float[] aBuf = TransAPackABuffer(Math.Max(1, fullBlocks) * kCap * Mr + 8);
                        fixed (float* ps = stripBuffer)
                        fixed (float* ap = aBuf)
                        {
                            if (cs == cs0)
                                for (int b = 0; b < fullBlocks; b++)
                                {
                                    float* src = pA + (long)(rb0 + b) * Mr * ldaCap;
                                    float* dst = ap + (long)b * kCap * Mr;
                                    for (int pp = 0; pp < kCap; pp++, dst += Mr)
                                    {
                                        dst[0] = src[pp]; dst[1] = src[ldaCap + pp]; dst[2] = src[2 * ldaCap + pp];
                                        dst[3] = src[3 * ldaCap + pp]; dst[4] = src[4 * ldaCap + pp]; dst[5] = src[5 * ldaCap + pp];
                                    }
                                }
                            PackStrip32(pB + stripStart, ldbCap, ps, kCap);
                            for (int b = 0; b < fullBlocks; b++)
                                TransAKernel6x32(ap + (long)b * kCap * Mr, Mr, ps, Nr512,
                                    pC + (long)(rb0 + b) * Mr * nCap + stripStart, nCap, kCap);
                            for (int rbIdx = rb0; rbIdx < rb1; rbIdx++)
                            {
                                int i = rbIdx * Mr;
                                int mc = Math.Min(Mr, mCap - i);
                                if (mc == Mr) continue;
                                float* pARow = pA + (long)i * ldaCap;
                                float* pCRow = pC + (long)i * nCap;
                                for (int j = stripStart; j < stripEnd; j += Nr)
                                    DirectKernelMxNMaskedStore(pARow, ldaCap, pB + j, ldbCap, pCRow + j, nCap,
                                        kCap, mcActual: mc, ncActual: Nr);
                            }
                        }
                        continue;
                    }
                    if (stripBuffer is not null && stripEnd - stripStart == Nr512)
                    {
                        fixed (float* ps = stripBuffer)
                        {
                            // k in blocks of s_stripKc rows: each block's strip (kc x 32) stays in L1 across the row
                            // blocks, which park their partial sums in C and resume the same FMA chains on the next block.
                            int kc = s_stripKc > 0 ? Math.Min(s_stripKc, kCap) : kCap;
                            for (int k0 = 0; k0 < kCap; k0 += kc)
                            {
                                int kb = Math.Min(kc, kCap - k0);
                                PackStrip32(pB + (long)k0 * ldbCap + stripStart, ldbCap, ps, kb);
                                for (int rbIdx = rb0; rbIdx < rb1; rbIdx++)
                                {
                                    int i = rbIdx * Mr;
                                    if (mCap - i < Mr) continue;
                                    float* pARow = pA + (long)i * ldaCap + k0;
                                    float* pCRow = pC + (long)i * nCap + stripStart;
                                    if (k0 == 0) DirectKernel6x32Store(pARow, ldaCap, ps, Nr512, pCRow, nCap, kb);
                                    else DirectKernel6x32Resume(pARow, ldaCap, ps, Nr512, pCRow, nCap, kb);
                                }
                            }
                            for (int rbIdx = rb0; rbIdx < rb1; rbIdx++)
                            {
                                int i = rbIdx * Mr;
                                int mc = Math.Min(Mr, mCap - i);
                                if (mc == Mr) continue;
                                float* pARow = pA + (long)i * ldaCap;
                                float* pCRow = pC + (long)i * nCap;
                                for (int j = stripStart; j < stripEnd; j += Nr)
                                    DirectKernelMxNMaskedStore(pARow, ldaCap, pB + j, ldbCap, pCRow + j, nCap,
                                        kCap, mcActual: mc, ncActual: Nr);
                            }
                        }
                        continue;
                    }
                    // probe: AVX2 16-column strips packed contiguously (rows ldb apart thrash L1 sets at power-of-two n).
                    if (s_zzPack16 && stripCap == Nr && stripEnd - stripStart == Nr && kCap <= MaxPackedStripK
                        && Math.Min(rb1, mCap / Mr) - rb0 >= 2)
                    {
                        fixed (float* ps = PackedStripBuffer(kCap))
                        {
                            float* src = pB + stripStart; float* dst = ps;
                            for (int pp = 0; pp < kCap; pp++, src += ldbCap, dst += Nr)
                            {
                                System.Runtime.Intrinsics.Vector256.Store(System.Runtime.Intrinsics.Vector256.Load(src), dst);
                                System.Runtime.Intrinsics.Vector256.Store(System.Runtime.Intrinsics.Vector256.Load(src + 8), dst + 8);
                            }
                            for (int rbIdx = rb0; rbIdx < rb1; rbIdx++)
                            {
                                int i = rbIdx * Mr;
                                int mc = Math.Min(Mr, mCap - i);
                                float* pARow = pA + (long)i * ldaCap;
                                float* pCRow = pC + (long)i * nCap + stripStart;
                                if (mc == Mr) DirectKernel6x16Store(pARow, ldaCap, ps, Nr, pCRow, nCap, kCap);
                                else DirectKernelMxNMaskedStore(pARow, ldaCap, ps, Nr, pCRow, nCap, kCap, mcActual: mc, ncActual: Nr);
                            }
                        }
                        continue;
                    }
#endif
                    for (int rbIdx = rb0; rbIdx < rb1; rbIdx++)
                    {
                        int i = rbIdx * Mr;
                        int mc = Math.Min(Mr, mCap - i);
                        float* pARow = pA + (long)i * ldaCap;
                        float* pCRow = pC + (long)i * nCap;
#if NET9_0_OR_GREATER
                        if (mc == Mr && stripCap == Nr512 && stripEnd - stripStart == Nr512)
                        {
                            DirectKernel6x32Store(pARow, ldaCap, pB + stripStart, ldbCap, pCRow + stripStart, nCap, kCap);
                            continue;
                        }
#endif
                        for (int j = stripStart; j < stripEnd; j += Nr)
                        {
                            int nc = Math.Min(Nr, stripEnd - j);
                            if (mc == Mr && nc == Nr)
                                DirectKernel6x16Store(pARow, ldaCap, pB + j, ldbCap, pCRow + j, nCap, kCap);
                            else
                                DirectKernelMxNMaskedStore(pARow, ldaCap, pB + j, ldbCap, pCRow + j, nCap,
                                    kCap, mcActual: mc, ncActual: nc);
                        }
                    }
                }
            };
            if (items <= 1) body(0);
            else PersistentParallelExecutor.Instance.Execute(items, body);
        }
        t_directParallel2DRuns++;
        return true;
    }
}
#endif
