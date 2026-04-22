// Copyright (c) AiDotNet. All rights reserved.
// Dgemm (double-precision) SIMD kernel — issue #243 companion to the
// existing Sgemm path. The public entry point is SimdGemm.Dgemm (see
// partial below); this file carries the AVX2 Vector256<double>-based
// tiled kernel + scalar fallback.
//
// Current perf envelope:
//   - CpuEngine.MatMul<double> on [64, 2048] × [2048, 512] went from
//     ~2.8 GFLOPS (MultiplyBlocked + per-row numOps.MultiplyAdd dispatch)
//     to ~12 GFLOPS here (inline Vector256<double> FMA, 2D parallel).
//   - Still well below MKL's ~75 GFLOPS — closing that gap would require
//     packed-B panels + K-outer tiling at AVX-512 width (tracked under
//     issue #131 follow-up). This kernel closes the majority of the gap
//     without new deps, matching the supply-chain-free build constraint.

using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>
    /// C = A · B where A is [m, k], B is [k, n], C is [m, n], all in
    /// row-major order. C is cleared before computation.
    /// Uses AVX2 Vector256&lt;double&gt; FMA when available, scalar
    /// fallback otherwise. Parallelizes over the 2D (M, N) block grid.
    /// </summary>
    internal static void Dgemm(
        ReadOnlySpan<double> a,
        ReadOnlySpan<double> b,
        Span<double> c,
        int m, int k, int n)
    {
        if (m <= 0 || n <= 0) return;
        c.Clear();
        if (k <= 0) return;

#if NET5_0_OR_GREATER
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DgemmAvx2(a, b, c, m, k, n);
            return;
        }
#endif
        DgemmScalar(a, b, c, m, k, n);
    }

    // ─────────────────────────────────────────────────────────────────────
    // AVX2 tiled kernel. Block size = 64 (fits L1 for M+N, double).
    // Inner-most micro-kernel: per (i, kk+kIndex), broadcast a[i,kk+kIndex]
    // across a Vector256<double>, then FMA against B panels 4 lanes at a
    // time. This avoids the Action<int> dispatch + numOps.MultiplyAdd
    // indirection in MatrixMultiplyHelper.MultiplyBlocked, which is most
    // of the ~4× speedup on small-M transformer shapes.
    // ─────────────────────────────────────────────────────────────────────
    private const int DgemmBlockSize = 64;

#if NET5_0_OR_GREATER
    private static unsafe void DgemmAvx2(
        ReadOnlySpan<double> a,
        ReadOnlySpan<double> b,
        Span<double> c,
        int m, int k, int n)
    {
        int numRowBlocks = (m + DgemmBlockSize - 1) / DgemmBlockSize;
        int numColBlocks = (n + DgemmBlockSize - 1) / DgemmBlockSize;
        bool doParallel = (long)m * n >= 16384 && Environment.ProcessorCount > 1;

        // Pin once at the top level so the inner kernels can do raw
        // pointer arithmetic — the Span's pointers are already pinned
        // for the call duration so we capture them via fixed statements.
        fixed (double* aPtr0 = a)
        fixed (double* bPtr0 = b)
        fixed (double* cPtr0 = c)
        {
            // We need to close over these raw pointers; marshal through
            // IntPtr so the lambda captures stable handles (C# can't
            // capture unsafe pointers directly into a lambda).
            IntPtr aHandle = (IntPtr)aPtr0;
            IntPtr bHandle = (IntPtr)bPtr0;
            IntPtr cHandle = (IntPtr)cPtr0;

            void Tile(int iiBlock, int jjBlock)
            {
                double* aP = (double*)aHandle;
                double* bP = (double*)bHandle;
                double* cP = (double*)cHandle;

                int iStart = iiBlock * DgemmBlockSize;
                int iEnd = Math.Min(iStart + DgemmBlockSize, m);
                int jStart = jjBlock * DgemmBlockSize;
                int jEnd = Math.Min(jStart + DgemmBlockSize, n);
                int nLen = jEnd - jStart;

                for (int kk = 0; kk < k; kk += DgemmBlockSize)
                {
                    int kLen = Math.Min(DgemmBlockSize, k - kk);
                    for (int i = iStart; i < iEnd; i++)
                    {
                        double* aRow = aP + (long)i * k + kk;
                        double* cRow = cP + (long)i * n + jStart;
                        for (int kIndex = 0; kIndex < kLen; kIndex++)
                        {
                            double aik = aRow[kIndex];
                            if (aik == 0.0) continue;  // Sparse skip.
                            Vector256<double> vAik = Vector256.Create(aik);
                            double* bRow = bP + (long)(kk + kIndex) * n + jStart;
                            int j = 0;
                            // 4-wide FMA loop.
                            for (; j <= nLen - 4; j += 4)
                            {
                                var vB = Avx.LoadVector256(bRow + j);
                                var vC = Avx.LoadVector256(cRow + j);
                                vC = Fma.MultiplyAdd(vB, vAik, vC);
                                Avx.Store(cRow + j, vC);
                            }
                            // Scalar tail for n % 4.
                            for (; j < nLen; j++)
                                cRow[j] += aik * bRow[j];
                        }
                    }
                }
            }

            if (doParallel)
            {
                int total = numRowBlocks * numColBlocks;
                // Mirror MatrixMultiplyHelper's heuristic: 2D grid only
                // when M-axis alone would under-subscribe cores.
                int procs = Environment.ProcessorCount;
                if (numRowBlocks * 2 < procs && numColBlocks > 1)
                {
                    Parallel.For(0, total, bi => Tile(bi / numColBlocks, bi % numColBlocks));
                }
                else
                {
                    Parallel.For(0, numRowBlocks, ii =>
                    {
                        for (int jj = 0; jj < numColBlocks; jj++) Tile(ii, jj);
                    });
                }
            }
            else
            {
                for (int ii = 0; ii < numRowBlocks; ii++)
                for (int jj = 0; jj < numColBlocks; jj++)
                    Tile(ii, jj);
            }
        }
    }

#endif

    /// <summary>Pre-AVX2 scalar fallback — row-wise FMA accumulation.</summary>
    private static void DgemmScalar(
        ReadOnlySpan<double> a,
        ReadOnlySpan<double> b,
        Span<double> c,
        int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            int aRow = i * k;
            int cRow = i * n;
            for (int kk = 0; kk < k; kk++)
            {
                double aik = a[aRow + kk];
                if (aik == 0.0) continue;
                int bRow = kk * n;
                for (int j = 0; j < n; j++)
                    c[cRow + j] += aik * b[bRow + j];
            }
        }
    }
}
