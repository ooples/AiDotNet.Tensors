// Copyright (c) AiDotNet. All rights reserved.
//
// fp32 batch=1 / small-M N-parallel GEMM + fused-projection helper (Phase A / #1622, L2).
//
// The tuned fp32 paths parallelize over M (rows): the small-matmul fast path needs m >= Mr and
// SgemmDirectParallelM needs m >= 64. So a tiny-M GEMM — the foundation-model batch=1 / decode
// forward and the per-projection QKV/FFN matmuls — runs single-threaded over a huge N, leaving
// every other core idle. This kernel partitions the OUTPUT columns (N) across cores instead, so
// even m=1 saturates the machine. Column-blocked accumulation: each thread owns a disjoint
// contiguous column range (so it is race-free for both overwrite and accumulate), iterates k
// once per row, and reads B row p + the C column block contiguously (FMA-vectorized).

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>Largest M routed to the N-parallel small-M kernel. Above this the existing
    /// M-parallel / small-matmul fast paths have enough rows to slice and are preferred.</summary>
    internal const int NParallelSmallMMaxM = 8;

    /// <summary>
    /// N-parallel GEMM for small M (no transpose): <c>C[m,n] (+)= A[m,k] · B[k,n]</c>, row-major
    /// (lda=k, ldb=n, ldc=n). Partitions N across cores. When <paramref name="clearedOutput"/> is
    /// true the output is overwritten; otherwise it is accumulated into (SgemmAdd semantics).
    /// </summary>
    internal static unsafe void SgemmNParallelSmallM(
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c,
        int m, int k, int n, bool clearedOutput)
    {
        if (m <= 0 || k <= 0 || n <= 0) return;

        int cores = Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism);
        int numChunks = Math.Max(1, Math.Min(n, cores * 4)); // oversubscribe 4x for load balance
        int colsPerChunk = (n + numChunks - 1) / numChunks; // numChunks==1 → whole range on one worker

        fixed (float* pAroot = a, pBroot = b, pCroot = c)
        {
            IntPtr ipA = (IntPtr)pAroot, ipB = (IntPtr)pBroot, ipC = (IntPtr)pCroot;
            int kCap = k, nCap = n, ldaCap = lda, ldbCap = ldb, mCap = m;
            bool cleared = clearedOutput;

            PersistentParallelExecutor.Instance.Execute(numChunks, chunk =>
            {
                int j0 = chunk * colsPerChunk;
                if (j0 >= nCap) return;
                int j1 = Math.Min(j0 + colsPerChunk, nCap);
                int bw = j1 - j0;

                float* pA = (float*)ipA, pB = (float*)ipB, pC = (float*)ipC;
                int vw = Vector<float>.Count;

                for (int i = 0; i < mCap; i++)
                {
                    float* cRow = pC + (long)i * nCap + j0;
                    if (cleared)
                        for (int j = 0; j < bw; j++) cRow[j] = 0f;

                    float* aRow = pA + (long)i * ldaCap;
                    for (int p = 0; p < kCap; p++)
                    {
                        float av = aRow[p];
                        float* bRow = pB + (long)p * ldbCap + j0;
                        var avv = new Vector<float>(av);
                        int j = 0;
                        for (; j + vw <= bw; j += vw)
                        {
                            ref byte cb = ref Unsafe.AsRef<byte>(cRow + j);
                            var cv = Unsafe.ReadUnaligned<Vector<float>>(ref cb);
                            var bv = Unsafe.ReadUnaligned<Vector<float>>(ref Unsafe.AsRef<byte>(bRow + j));
                            cv += avv * bv;
                            Unsafe.WriteUnaligned(ref cb, cv);
                        }
                        for (; j < bw; j++) cRow[j] += av * bRow[j];
                    }
                }
            });
        }
    }

    /// <summary>
    /// Fused multi-projection matmul: one activation <c>A[m,k]</c> against a CONCATENATED weight
    /// <c>B[k, nTotal]</c> (e.g. <c>[Wq|Wk|Wv]</c> for QKV, or <c>[Wgate|Wup]</c> for SwiGLU FFN),
    /// producing <c>C[m, nTotal]</c> in ONE dispatch instead of 2-3 separate small GEMMs — the
    /// caller slices C by the per-projection widths. At small M this routes through the N-parallel
    /// kernel (the combined nTotal gives more columns to partition than any single projection);
    /// otherwise it uses the standard M-parallel dispatcher. Output is overwritten.
    /// </summary>
    internal static void SgemmConcatenatedProjections(
        ReadOnlySpan<float> a, ReadOnlySpan<float> bConcat, Span<float> cConcat,
        int m, int k, int nTotal)
    {
        if (UseParallelGemm && m > 0 && m <= NParallelSmallMMaxM && nTotal >= Nr
            && (long)m * k * nTotal >= ParallelWorkThreshold)
        {
            SgemmNParallelSmallM(a, k, bConcat, nTotal, cConcat, m, k, nTotal, clearedOutput: true);
        }
        else
        {
            SgemmAddInternal(a, k, false, bConcat, nTotal, false, cConcat, m, k, nTotal,
                allowParallel: true, clearedOutput: true);
        }
    }
}
