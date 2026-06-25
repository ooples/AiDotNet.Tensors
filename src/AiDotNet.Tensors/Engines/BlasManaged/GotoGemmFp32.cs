using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Clean-slate GotoBLAS-style FP32 GEMM (rewrite to match/beat MKL on the Threadripper 3990X).
/// C = A·B, all row-major. Built on the proven machine-code microkernel (EmitFp32TileWindows,
/// ~125 GFLOP/s single-tile = 91% of FMA peak). The win vs the existing BlasManaged parallel paths
/// is the parallel DECOMPOSITION: the existing M-axis shares one big B-panel that all threads stream
/// from L3 (bandwidth-bound, scales 9x); MKL keeps each thread's working set L2-resident and scales
/// 22x. This implements the canonical 3-loop macro-kernel (jc→pc→ic) with per-thread packed A/B
/// panels sized for L2, parallelized over the IC×JC tile grid so each worker owns disjoint C tiles
/// (no shared writes, no cross-thread B streaming during compute).
///
/// Stage 1 (this file): correct single-thread core + the packing + the tile macro-loop + scalar tails.
/// Stage 2: the low-overhead, CCD-pinned parallel scheme over the tile grid.
/// </summary>
internal static class GotoGemmFp32
{
    // Microkernel tile: 6 rows x 16 cols (nrYmm=2). Proven 125 GFLOP/s, 12 YMM accumulators.
    private const int Mr = 6;
    private const int NrYmm = 2;
    private const int Nr = NrYmm * 8; // 16

    // Emitted-once kernel: void(float* packedA, float* packedB, float* c, long ldcBytes, long kc).
    private static IntPtr s_kernel;
    private static ExecutableMemory? s_kernelMem;
    private static int s_kernelInit;

    private static unsafe delegate* unmanaged<float*, float*, float*, long, long, void> Kernel()
    {
        if (Volatile.Read(ref s_kernelInit) == 0)
        {
            lock (typeof(GotoGemmFp32))
            {
                if (s_kernelInit == 0)
                {
                    var code = MachineCodeFmaKernel.EmitFp32TileWindows(Mr, NrYmm);
                    s_kernelMem = ExecutableMemory.TryAllocate(code);
                    s_kernel = s_kernelMem?.Pointer ?? IntPtr.Zero;
                    Volatile.Write(ref s_kernelInit, 1);
                }
            }
        }
        return (delegate* unmanaged<float*, float*, float*, long, long, void>)(void*)s_kernel;
    }

    /// <summary>True when the machine-code microkernel is available on this CPU/OS (AVX2+FMA, Windows x64).</summary>
    internal static unsafe bool IsAvailable => MachineKernelGemm.IsFp32Available && Kernel() != null;

    // Default L2-resident blocking for Zen2 (512 KB L2/core): Kc=512 (the measured-best K-block),
    // Mc/Nc chosen so packA(Mc*Kc) + packB(Kc*Nc) + Ctile(Mc*Nc) fit ~half L2. With Kc=512:
    // 120*512 + 512*128 + 120*128 ≈ 0.5 MB → tune per measurement.
    internal const int DefaultMc = 120;
    internal const int DefaultNc = 256;
    internal const int DefaultKc = 512;

    /// <summary>
    /// Single-thread C = A·B (row-major). A[M,K] lda, B[K,N] ldb, C[M,N] ldc (element strides).
    /// C is OVERWRITTEN (zeroed for the first K-panel via the pack-once-then-accumulate structure).
    /// </summary>
    internal static unsafe void RunSingle(
        float* a, int lda, float* b, int ldb, float* c, int ldc,
        int m, int n, int k, int mc, int nc, int kc)
    {
        var kern = Kernel();
        // Per-call scratch (Stage 1: plain alloc; Stage 2 will pool these per worker).
        // packA holds one Mc×Kc A-panel as consecutive mr-row sub-panels: [mt][k][mr].
        // packB holds one Kc×Nc B-panel as consecutive nr-col sub-panels: [nt][k][nr].
        float[] packAArr = new float[mc * kc];
        float[] packBArr = new float[kc * nc];

        fixed (float* packA = packAArr)
        fixed (float* packB = packBArr)
        {
            for (int jc = 0; jc < n; jc += nc)
            {
                int effNc = Math.Min(nc, n - jc);
                int nFull = effNc - (effNc % Nr);          // full Nr-tiles span
                for (int pc = 0; pc < k; pc += kc)
                {
                    int effKc = Math.Min(kc, k - pc);
                    bool firstK = pc == 0;

                    // Pack B[pc:pc+effKc, jc:jc+effNc] → packB[nt][kk][col].
                    int nTiles = nFull / Nr;
                    for (int nt = 0; nt < nTiles; nt++)
                    {
                        float* dst = packB + (long)nt * effKc * Nr;
                        int col0 = jc + nt * Nr;
                        for (int kk = 0; kk < effKc; kk++)
                        {
                            float* brow = b + (long)(pc + kk) * ldb + col0;
                            float* d = dst + (long)kk * Nr;
                            for (int col = 0; col < Nr; col++) d[col] = brow[col];
                        }
                    }

                    for (int ic = 0; ic < m; ic += mc)
                    {
                        int effMc = Math.Min(mc, m - ic);
                        int mFull = effMc - (effMc % Mr);
                        int mTiles = mFull / Mr;

                        // Pack A[ic:ic+effMc, pc:pc+effKc] → packA[mt][kk][row].
                        for (int mt = 0; mt < mTiles; mt++)
                        {
                            float* dst = packA + (long)mt * effKc * Mr;
                            int row0 = ic + mt * Mr;
                            for (int kk = 0; kk < effKc; kk++)
                            {
                                float* d = dst + (long)kk * Mr;
                                for (int row = 0; row < Mr; row++)
                                    d[row] = a[(long)(row0 + row) * lda + (pc + kk)];
                            }
                        }

                        // On the first K-panel, zero this C block so the kernel's C+= accumulates from 0.
                        if (firstK)
                            ZeroCBlock(c, ldc, ic, jc, effMc, effNc);

                        // Full mr×nr tiles via the machine-code microkernel.
                        for (int nt = 0; nt < nTiles; nt++)
                        {
                            float* bp = packB + (long)nt * effKc * Nr;
                            int col0 = jc + nt * Nr;
                            for (int mt = 0; mt < mTiles; mt++)
                            {
                                float* ap = packA + (long)mt * effKc * Mr;
                                float* cp = c + (long)(ic + mt * Mr) * ldc + col0;
                                kern(ap, bp, cp, (long)ldc * 4, effKc);
                            }
                        }

                        // M-tail rows [ic+mFull, ic+effMc) × full N tiles + everything in the N-tail:
                        // handle all remaining (row,col) of this C block with a scalar K-accumulate.
                        ScalarTails(a, lda, b, ldb, c, ldc, ic, jc, effMc, effNc, mFull, nFull, pc, effKc);
                    }
                }
            }
        }
    }

    private static unsafe void ZeroCBlock(float* c, int ldc, int ic, int jc, int effMc, int effNc)
    {
        for (int r = 0; r < effMc; r++)
        {
            float* crow = c + (long)(ic + r) * ldc + jc;
            for (int col = 0; col < effNc; col++) crow[col] = 0f;
        }
    }

    // Scalar K-accumulate for the rows/cols not covered by the full mr×nr tiles (the M-tail rows and
    // the N-tail cols). Correct but slow — tails are a small fraction of a well-blocked GEMM.
    private static unsafe void ScalarTails(
        float* a, int lda, float* b, int ldb, float* c, int ldc,
        int ic, int jc, int effMc, int effNc, int mFull, int nFull, int pc, int effKc)
    {
        // N-tail columns [jc+nFull, jc+effNc) for ALL rows [ic, ic+effMc).
        for (int r = 0; r < effMc; r++)
        {
            float* crow = c + (long)(ic + r) * ldc;
            float* arow = a + (long)(ic + r) * lda + pc;
            for (int col = jc + nFull; col < jc + effNc; col++)
            {
                float s = 0f;
                for (int kk = 0; kk < effKc; kk++) s += arow[kk] * b[(long)(pc + kk) * ldb + col];
                crow[col] += s;
            }
        }
        // M-tail rows [ic+mFull, ic+effMc) for the full-N span [jc, jc+nFull) (the N-tail already done above).
        for (int r = ic + mFull; r < ic + effMc; r++)
        {
            float* crow = c + (long)r * ldc;
            float* arow = a + (long)r * lda + pc;
            for (int col = jc; col < jc + nFull; col++)
            {
                float s = 0f;
                for (int kk = 0; kk < effKc; kk++) s += arow[kk] * b[(long)(pc + kk) * ldb + col];
                crow[col] += s;
            }
        }
    }
}
