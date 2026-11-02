// Copyright (c) CrowdTrainer LLC. All rights reserved.
//
// A·Bᵀ (B stored [n, k]) on the 512-bit direct tiles, the transpose fused into the strip packing.
//
// A linear layer's input gradient dX = dY·Wᵀ (W stored [in, out]) is this product. Transposing all of B first and then
// running the NN product spends a whole serial pass over B before any FMA: on a Xeon 8573C at 784->512 the transpose
// alone took 0.2 ms of 0.56 (the NN product 0.36, PyTorch 0.28 for everything). Here each worker transposes only the
// 32 rows of B its column strip needs, straight into the contiguous strip buffer the NN path copies into anyway, so the
// transpose is split across the workers and its output is consumed from L2 while it is still there.

#if NET9_0_OR_GREATER
using System;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>GEMMs <see cref="TrySgemmDirectParallel2DTransB"/> computed on this thread; a test probe.</summary>
    [ThreadStatic] internal static int t_directParallel2DTransBRuns;

    /// <summary>
    /// <c>C[m,n] = A[m,k] · B[n,k]ᵀ</c> (row-major, lda = ldb = k, ldc = n, C overwritten) over disjoint output rectangles
    /// in parallel, on the 512-bit direct tiles. Every 32-column strip of C is formed from a packed copy of the strip's
    /// 32 rows of B, transposed to [k][32] by the worker that uses it; each output element is one FMA chain over k, in
    /// the same order as the NN direct path on an explicitly transposed B (bit-identical to it). Returns false, having
    /// touched nothing, where the 512-bit tiles or the strip buffer do not apply.
    /// </summary>
    internal static unsafe bool TrySgemmDirectParallel2DTransB(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n, bool allowParallel)
    {
        if (!UseVector512Direct || !Avx.IsSupported || !Fma.IsSupported
            || m < Mr || n < Nr || k <= 0 || k > MaxPackedStripK)
            return false;
        if (a.Length < (long)m * k || b.Length < (long)n * k || c.Length < (long)m * n)
            throw new ArgumentException("An operand is shorter than its extent.");

        int cores = allowParallel ? Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism) : 1;
        int chunks = CapDirectChunksByWork(cores, (long)m * k * n);
        int rowBlocks = (m + Mr - 1) / Mr;
        int colStrips = (n + Nr512 - 1) / Nr512;
        int colChunks = Math.Max(1, Math.Min(chunks, colStrips));
        int rowChunks = Math.Min(rowBlocks, Math.Max(1, chunks / colChunks));
        int items = colChunks * rowChunks;

        fixed (float* pAroot = a, pBroot = b, pCroot = c)
        {
            IntPtr ipA = (IntPtr)pAroot, ipB = (IntPtr)pBroot, ipC = (IntPtr)pCroot;
            int mCap = m, kCap = k, nCap = n;
            int colChunksCap = colChunks, rowChunksCap = rowChunks, rowBlocksCap = rowBlocks, colStripsCap = colStrips;

            Action<int> body = item =>
            {
                int rc = item / colChunksCap, cc = item % colChunksCap;
                int rb0 = (int)((long)rc * rowBlocksCap / rowChunksCap), rb1 = (int)((long)(rc + 1) * rowBlocksCap / rowChunksCap);
                int cs0 = (int)((long)cc * colStripsCap / colChunksCap), cs1 = (int)((long)(cc + 1) * colStripsCap / colChunksCap);
                float* pA = (float*)ipA, pB = (float*)ipB, pC = (float*)ipC;
                int kc = s_stripKc > 0 ? Math.Min(s_stripKc, kCap) : kCap;

                fixed (float* ps = PackedStripBuffer(kCap))
                {
                    for (int cs = cs0; cs < cs1; cs++)
                    {
                        int stripStart = cs * Nr512;
                        int width = Math.Min(Nr512, nCap - stripStart);
                        PackStrip32Transposed(pB + (long)stripStart * kCap, kCap, ps, kCap, width);

                        if (width == Nr512)
                        {
                            // Full row blocks: k in blocks of kc rows of the packed strip (L1-resident across the row
                            // blocks), partial sums parked in C between blocks, as in the NN path.
                            for (int k0 = 0; k0 < kCap; k0 += kc)
                            {
                                int kb = Math.Min(kc, kCap - k0);
                                float* strip = ps + (long)k0 * Nr512;
                                for (int rbIdx = rb0; rbIdx < rb1; rbIdx++)
                                {
                                    int i = rbIdx * Mr;
                                    if (mCap - i < Mr) continue;
                                    float* pARow = pA + (long)i * kCap + k0;
                                    float* pCRow = pC + (long)i * nCap + stripStart;
                                    if (k0 == 0) DirectKernel6x32Store(pARow, kCap, strip, Nr512, pCRow, nCap, kb);
                                    else DirectKernel6x32Resume(pARow, kCap, strip, Nr512, pCRow, nCap, kb);
                                }
                            }
                        }

                        // The rest (a partial row block; every row block of a partial strip): 16-column tiles over the
                        // full k of the packed strip, the tail masked.
                        for (int rbIdx = rb0; rbIdx < rb1; rbIdx++)
                        {
                            int i = rbIdx * Mr;
                            int mc = Math.Min(Mr, mCap - i);
                            if (mc == Mr && width == Nr512) continue;
                            float* pARow = pA + (long)i * kCap;
                            float* pCRow = pC + (long)i * nCap + stripStart;
                            for (int j = 0; j < width; j += Nr)
                            {
                                int nc = Math.Min(Nr, width - j);
                                if (mc == Mr && nc == Nr)
                                    DirectKernel6x16Store(pARow, kCap, ps + j, Nr512, pCRow + j, nCap, kCap);
                                else
                                    DirectKernelMxNMaskedStore(pARow, kCap, ps + j, Nr512, pCRow + j, nCap,
                                        kCap, mcActual: mc, ncActual: nc);
                            }
                        }
                    }
                }
            };
            if (items <= 1) body(0);
            else PersistentParallelExecutor.Instance.Execute(items, body);
        }
        t_directParallel2DTransBRuns++;
        return true;
    }

    /// <summary>
    /// dst[p * 32 + j] = src[j * ldb + p] for j &lt; <paramref name="width"/>, p &lt; <paramref name="k"/>: rows [0, width)
    /// of B (stored [n, k]) as a [k][32] strip, 8x8 tiles through AVX shuffles. Columns at and past width are zeroed, so
    /// a masked tile reading them never sees stale data.
    /// </summary>
    internal static unsafe void PackStrip32Transposed(float* src, int ldb, float* dst, int k, int width)
    {
        int w8 = width & ~7, k8 = k & ~7;
        for (int j = 0; j < w8; j += 8)
        {
            float* s = src + (long)j * ldb;
            for (int p = 0; p < k8; p += 8)
                Transpose8x8Avx(s + p, ldb, dst + (long)p * Nr512 + j, Nr512, 1f, scaled: false);
            for (int p = k8; p < k; p++)
                for (int jj = 0; jj < 8; jj++)
                    dst[(long)p * Nr512 + j + jj] = s[(long)jj * ldb + p];
        }
        for (int j = w8; j < width; j++)
        {
            float* s = src + (long)j * ldb;
            for (int p = 0; p < k; p++) dst[(long)p * Nr512 + j] = s[p];
        }
        if (width < Nr512)
            for (int p = 0; p < k; p++)
                new Span<float>(dst + (long)p * Nr512 + width, Nr512 - width).Clear();
    }
}
#endif
