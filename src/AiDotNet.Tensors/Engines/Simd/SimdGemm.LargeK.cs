using System;
using System.Buffers;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    // #1307: a deep reduction into a small output - the weight gradient of every linear layer, dW = Xᵀ·dY with the batch
    // rows as K - ran its K loop serially inside each output tile, and the transposed operand was first copied whole into
    // row-major order. On a 3950X, Cᵀ·dY [256x4096]ᵀ·[4096x256] took 3.6 ms and Xᵀ·dQKV [256x4096]ᵀ·[4096x768] 5.6 ms,
    // against 0.96 and 2.3 ms in PyTorch (oneDNN). This route splits K into fixed slices, each task transposing only its
    // own slice of A (cache-hot) and running the direct kernel over 256-column panels of B (a [256 x 256] panel stays in
    // L2; past that the kernel re-streams B from L3 for every 6-row block, 17 GFLOP/s at 256x512x768), then sums the
    // per-slice partials in slice order. The slice plan depends only on the shape, so the result is the same for any
    // thread count, and with or without parallelism.

    /// <summary>A/B and test toggle for <see cref="SgemmLargeK"/>.</summary>
    internal static bool UseLargeKSplit = true;

    private const int LargeKMinK = 2048;
    private const int LargeKSliceK = 256;
    private const int LargeKMaxSlices = 32;
    private const int LargeKPanelN = 256;
    private const long LargeKMaxOutput = 1L << 20;
    private const long LargeKMaxPartialFloats = 16L << 20;

    [ThreadStatic] private static float[]? t_largeKSliceA;
    [ThreadStatic] private static int t_largeKRuns, t_tallMRuns;

    /// <summary>Products this thread ran on <see cref="SgemmLargeK"/> / <see cref="SgemmTallM"/> (test positive controls).</summary>
    internal static int LargeKRunsOnThisThread => t_largeKRuns;
    internal static int TallMRunsOnThisThread => t_tallMRuns;

    /// <summary>Whether <c>[m x k]·[k x n]</c> (A optionally stored transposed as [k x m]) takes <see cref="SgemmLargeK"/>.</summary>
    internal static bool QualifiesForLargeK(bool transA, bool transB, int lda, int m, int k, int n)
    {
#if NET5_0_OR_GREATER
        return UseLargeKSplit && !transB && Avx2.IsSupported && Fma.IsSupported
            && k >= LargeKMinK && k >= 2L * Math.Max(m, n)
            && m >= Mr && n >= Nr && (long)m * n <= LargeKMaxOutput
            && (transA ? lda == m : lda >= k);
#else
        return false;
#endif
    }

    private static int LargeKSlices(int m, int k, int n)
    {
        int slices = Math.Min(LargeKMaxSlices, (k + LargeKSliceK - 1) / LargeKSliceK);
        return (int)Math.Max(1, Math.Min(slices, LargeKMaxPartialFloats / ((long)m * n)));
    }

#if NET5_0_OR_GREATER
    private static unsafe void SgemmLargeK(
        ReadOnlySpan<float> a, int lda, bool transA, ReadOnlySpan<float> b, int ldb, Span<float> c,
        int m, int k, int n, bool clearedOutput, bool allowParallel)
    {
        t_largeKRuns++;
        int slices = LargeKSlices(m, k, n);
        int outSize = m * n;
        var partials = ArrayPool<float>.Shared.Rent(slices * outSize);
        try
        {
            fixed (float* pA = a, pB = b)
            {
                IntPtr ipA = (IntPtr)pA, ipB = (IntPtr)pB;
                // One task per slice: its slice of A is transposed once and stays in L2 while the task walks the
                // 256-column panels of B (a task per panel transposed it once per panel: 3x at n = 768).
                void Task(int s)
                {
                    int k0 = (int)((long)s * k / slices), k1 = (int)((long)(s + 1) * k / slices), ks = k1 - k0;
                    float[]? sliceA = null;
                    if (transA)
                    {
                        // A is stored [k x m] (lda == m): its rows k0..k1 are one contiguous [ks x m] block.
                        sliceA = t_largeKSliceA is { } cached && cached.Length >= m * ks ? cached : (t_largeKSliceA = new float[m * ks]);
                        fixed (float* pT = sliceA)
                            SimdTranspose.TransposeSerial((float*)ipA + (long)k0 * m, pT, ks, m);
                    }
                    fixed (float* pPart = partials, pS = sliceA)
                    {
                        float* aS = transA ? pS : (float*)ipA + k0;
                        int ldaS = transA ? ks : lda;
                        for (int j0 = 0; j0 < n; j0 += LargeKPanelN)
                            DirectPanel(aS, ldaS, (float*)ipB + (long)k0 * ldb + j0, ldb, pPart + (long)s * outSize + j0, n,
                                m, ks, Math.Min(LargeKPanelN, n - j0), clearedOutput: true);
                    }
                }
                int tasks = slices;
                if (allowParallel && Helpers.CpuParallelSettings.MaxDegreeOfParallelism > 1)
                    Helpers.CpuParallelSettings.LightweightParallel(tasks, Task);
                else
                    for (int t = 0; t < tasks; t++) Task(t);
            }

            int cores = Math.Max(1, Helpers.CpuParallelSettings.MaxDegreeOfParallelism);
            int chunks = allowParallel ? Math.Min(cores, Math.Max(1, outSize / 4096)) : 1;
            int per = (outSize + chunks - 1) / chunks;
            fixed (float* pC = c)
            {
                IntPtr ipC = (IntPtr)pC;
                bool cleared = clearedOutput;
                void Reduce(int ch)
                {
                    int i0 = ch * per, i1 = Math.Min(outSize, i0 + per);
                    if (i0 >= i1) return;
                    var dst = new Span<float>((float*)ipC + i0, i1 - i0);
                    if (cleared) new ReadOnlySpan<float>(partials, i0, i1 - i0).CopyTo(dst);
                    else AddInto(dst, new ReadOnlySpan<float>(partials, i0, i1 - i0));
                    for (int s = 1; s < slices; s++)
                        AddInto(dst, new ReadOnlySpan<float>(partials, s * outSize + i0, i1 - i0));
                }
                if (chunks > 1) Helpers.CpuParallelSettings.LightweightParallel(chunks, Reduce);
                else Reduce(0);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(partials);
        }
    }
#endif

    // #1307: a tall product whose A is too large for the column-panel route (PrefersParallelN streams all of A once per
    // panel, so it caps A at 4 MB) - dX = dY·Wᵀ of a layer with a wide fan-out, e.g.
    // [4096x768]·[768x256] - fell to the packed 2-D route at 5.1-5.8 ms, while the same FLOPs as [4096x256]·[256x768]
    // take 1.4-2.6 ms on the column-panel route (PyTorch: 2.1 ms). Here each task owns a block of TallMRowBlock rows,
    // which stays in L2 while it walks the TallMPanelN-column panels of B (48 x 768 + 768 x 64 floats = 340 KB at K = 768).
    // Every C element is one direct-kernel dot product in K order, whatever the split, so the result does not depend
    // on the thread count; the route runs serially, in the same order, when parallelism is off.

    /// <summary>A/B and test toggle for <see cref="SgemmTallM"/>.</summary>
    internal static bool UseTallM = true;

    private const int TallMRowBlock = 8 * Mr;
    private const int TallMPanelN = 64;
    private const int TallMMinM = 1024;
    private const int TallMMaxK = 4096;
    // Below this K the small-K direct and JIT kernels win: dY·Woᵀ at K = 128 and 256 ran 0.26 -> 0.51 ms and 0.59 -> 0.80 ms here.
    private const int TallMMinK = 384;

    /// <summary>Whether the untransposed <c>[m x k]·[k x n]</c> takes <see cref="SgemmTallM"/>.</summary>
    internal static bool QualifiesForTallM(bool transA, bool transB, int m, int k, int n)
    {
#if NET5_0_OR_GREATER
        return UseTallM && !transA && !transB && Avx2.IsSupported && Fma.IsSupported
            && m >= TallMMinM && n >= Nr && k <= TallMMaxK
            && k >= TallMMinK
            && (long)m * k * n >= ParallelWorkThreshold;
#else
        return false;
#endif
    }

#if NET5_0_OR_GREATER
    private static unsafe void SgemmTallM(
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c,
        int m, int k, int n, bool clearedOutput, bool allowParallel)
    {
        t_tallMRuns++;
        int blocks = (m + TallMRowBlock - 1) / TallMRowBlock;
        fixed (float* pA = a, pB = b, pC = c)
        {
            IntPtr ipA = (IntPtr)pA, ipB = (IntPtr)pB, ipC = (IntPtr)pC;
            bool cleared = clearedOutput;
            void Block(int blk)
            {
                int i0 = blk * TallMRowBlock, rows = Math.Min(TallMRowBlock, m - i0);
                float* aRows = (float*)ipA + (long)i0 * lda;
                float* cRows = (float*)ipC + (long)i0 * n;
                for (int j0 = 0; j0 < n; j0 += TallMPanelN)
                    DirectPanel(aRows, lda, (float*)ipB + j0, ldb, cRows + j0, n, rows, k, Math.Min(TallMPanelN, n - j0), cleared);
            }
            if (allowParallel && Helpers.CpuParallelSettings.MaxDegreeOfParallelism > 1)
                Helpers.CpuParallelSettings.LightweightParallel(blocks, Block);
            else
                for (int blk = 0; blk < blocks; blk++) Block(blk);
        }
    }
#endif
}
