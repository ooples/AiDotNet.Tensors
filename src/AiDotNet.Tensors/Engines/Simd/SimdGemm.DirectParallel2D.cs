// Copyright (c) AiDotNet. All rights reserved.
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

        int rowBlocks = (m + Mr - 1) / Mr;    // the last block may be a partial (masked) one
        int colStrips = (n + Nr - 1) / Nr;    // likewise the last strip
        int colChunks = Math.Max(1, Math.Min(chunks, colStrips));
        int rowChunks = Math.Min(rowBlocks, Math.Max(1, chunks / colChunks));
        int items = colChunks * rowChunks;

        fixed (float* pAroot = a, pBroot = b, pCroot = c)
        {
            IntPtr ipA = (IntPtr)pAroot, ipB = (IntPtr)pBroot, ipC = (IntPtr)pCroot;
            int mCap = m, kCap = k, nCap = n, ldaCap = lda, ldbCap = ldb;
            int colChunksCap = colChunks, rowChunksCap = rowChunks, rowBlocksCap = rowBlocks, colStripsCap = colStrips;

            Action<int> body = item =>
            {
                int rc = item / colChunksCap, cc = item % colChunksCap;
                // Balanced split of whole row blocks / column strips: every chunk gets floor or ceil of the share.
                int rb0 = (int)((long)rc * rowBlocksCap / rowChunksCap), rb1 = (int)((long)(rc + 1) * rowBlocksCap / rowChunksCap);
                int cs0 = (int)((long)cc * colStripsCap / colChunksCap), cs1 = (int)((long)(cc + 1) * colStripsCap / colChunksCap);
                float* pA = (float*)ipA, pB = (float*)ipB, pC = (float*)ipC;

                for (int rbIdx = rb0; rbIdx < rb1; rbIdx++)
                {
                    int i = rbIdx * Mr;
                    int mc = Math.Min(Mr, mCap - i);
                    float* pARow = pA + (long)i * ldaCap;
                    float* pCRow = pC + (long)i * nCap;
                    for (int cs = cs0; cs < cs1; cs++)
                    {
                        int j = cs * Nr;
                        int nc = Math.Min(Nr, nCap - j);
                        if (mc == Mr && nc == Nr)
                            DirectKernel6x16Store(pARow, ldaCap, pB + j, ldbCap, pCRow + j, nCap, kCap);
                        else
                            DirectKernelMxNMaskedStore(pARow, ldaCap, pB + j, ldbCap, pCRow + j, nCap,
                                kCap, mcActual: mc, ncActual: nc);
                    }
                }
            };
            if (items <= 1) body(0);
            else PersistentParallelExecutor.Instance.Execute(items, body);
        }
        return true;
    }
}
#endif
