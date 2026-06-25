using System;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Helpers;

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

    // Default L2-resident blocking for Zen2 (512 KB L2/core): Kc=512 (the measured-best K-block),
    // Mc/Nc chosen so packA(Mc*Kc) + packB(Kc*Nc) + Ctile(Mc*Nc) fit ~half L2. Small skewed shapes
    // (M small) win with smaller Mc/Nc (more tiles); large square shapes want larger tiles.
    internal const int DefaultMc = 120;
    internal const int DefaultNc = 256;
    internal const int DefaultKc = 512;

    /// <summary>Minimum M·N·K work for the parallel GotoBLAS path to beat the existing machine-kernel
    /// dispatch (below this the per-tile pack/launch overhead dominates — stay on the existing path).</summary>
    internal const long ParallelMinWork = 8L * 1024 * 1024; // ~8.4e6

    /// <summary>Shape-adaptive (Mc, Nc, Kc) for RunParallel, tuned on the 3990X (measured --ab-goto-par).
    /// Memory-bound regime: larger square shapes want larger tiles (fewer redundant DRAM re-reads);
    /// skewed / smaller shapes want smaller Mc for enough IC-blocks. Kc=512 is universally best.</summary>
    internal static (int mc, int nc, int kc) ChooseParallelBlocks(int m, int n)
    {
        if (m >= 1536 && n >= 1536) return (192, 256, DefaultKc); // large square
        if (m >= 640 && n >= 640) return (96, 128, DefaultKc);    // medium square
        return (120, 128, DefaultKc);                              // skewed / smaller
    }

#if NET5_0_OR_GREATER
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

    /// <summary>
    /// Parallel C = A·B over the IC×JC tile grid. Each worker owns a disjoint (mc×nc) C-tile and packs
    /// its OWN A-panel + B-panel into L2 — so during the microkernel NO thread streams a shared B-panel
    /// from L3 (the existing M-axis bottleneck that caps scaling at ~9x). Redundant B-pack across the
    /// ic dimension is the trade for L2-resident, contention-free compute. Each tile runs its full
    /// K-loop independently (disjoint C writes ⇒ no races, no reduction).
    /// </summary>
    internal static unsafe void RunParallel(
        float* a, int lda, float* b, int ldb, float* c, int ldc,
        int m, int n, int k, int mc, int nc, int kc)
    {
        int numIc = (m + mc - 1) / mc;
        int numJc = (n + nc - 1) / nc;
        int totalTiles = numIc * numJc;
        if (totalTiles <= 1) { RunSingle(a, lda, b, ldb, c, ldc, m, n, k, mc, nc, kc); return; }

        // Per-tile parallelism: each worker owns a disjoint (mc×nc) C-tile and packs ITS OWN A-panel
        // + B-panel into per-thread buffers, so the microkernel's operands are cache-resident (A→L1,
        // B→L2 when mc/nc are sized small) with NO shared-L3 streaming during compute. Sized small,
        // this is the cache-resident-tile scheme MKL uses to scale past the existing M-axis's L3 cap.
        nint ai = (nint)a, bi = (nint)b, ci = (nint)c;
        int numIcL = numIc, mcL = mc, ncL = nc, kcL = kc, mL = m, nL = n, kL = k, ldaL = lda, ldbL = ldb, ldcL = ldc;
        CpuParallelSettings.ParallelForOrSerial(0, totalTiles, (long)m * n * k, tileIdx =>
        {
            int ic = (int)(tileIdx % numIcL) * mcL;
            int jc = (int)(tileIdx / numIcL) * ncL;
            int effMc = Math.Min(mcL, mL - ic);
            int effNc = Math.Min(ncL, nL - jc);
            if (effMc <= 0 || effNc <= 0) return;
            RunTile((float*)ai, ldaL, (float*)bi, ldbL, (float*)ci, ldcL, ic, jc, effMc, effNc, kL, mcL, ncL, kcL);
        }, deterministicSafe: true);
    }

    /// <summary>Compute one disjoint (effMc×effNc) C-tile at (ic,jc): C[tile] = A[ic:, :]·B[:, jc:] over
    /// the full K, packing this tile's A and B panels into per-call L2 buffers. C is zeroed on the first
    /// K-panel then accumulated, so the tile is self-contained and deterministic (fixed K order). Shared
    /// by RunParallel (TPL tile grid) and the CCX-aware driver (pinned per-CCX dispatch).</summary>
    internal static unsafe void RunTile(
        float* a, int lda, float* b, int ldb, float* c, int ldc,
        int ic, int jc, int effMc, int effNc, int k, int mc, int nc, int kc)
    {
        int nFull = effNc - (effNc % Nr); int nTiles = nFull / Nr;
        int mFull = effMc - (effMc % Mr); int mTiles = mFull / Mr;
        float[] paArr = ArrayPool<float>.Shared.Rent(mc * kc);
        float[] pbArr = ArrayPool<float>.Shared.Rent(kc * nc);
        try
        {
            fixed (float* pa = paArr, pb = pbArr)
            {
                var kern = Kernel();
                for (int pc = 0; pc < k; pc += kc)
                {
                    int effKc = Math.Min(kc, k - pc);
                    for (int nt = 0; nt < nTiles; nt++)
                    {
                        float* dst = pb + (long)nt * effKc * Nr; int col0 = jc + nt * Nr;
                        for (int kk = 0; kk < effKc; kk++)
                        {
                            float* brow = b + (long)(pc + kk) * ldb + col0; float* d = dst + (long)kk * Nr;
                            for (int col = 0; col < Nr; col++) d[col] = brow[col];
                        }
                    }
                    for (int mt = 0; mt < mTiles; mt++)
                    {
                        float* dst = pa + (long)mt * effKc * Mr; int row0 = ic + mt * Mr;
                        for (int kk = 0; kk < effKc; kk++)
                        {
                            float* d = dst + (long)kk * Mr;
                            for (int row = 0; row < Mr; row++) d[row] = a[(long)(row0 + row) * lda + (pc + kk)];
                        }
                    }
                    if (pc == 0) ZeroCBlock(c, ldc, ic, jc, effMc, effNc);
                    for (int nt = 0; nt < nTiles; nt++)
                    {
                        float* bp = pb + (long)nt * effKc * Nr; int col0 = jc + nt * Nr;
                        for (int mt = 0; mt < mTiles; mt++)
                        {
                            float* ap = pa + (long)mt * effKc * Mr;
                            float* cp = c + (long)(ic + mt * Mr) * ldc + col0;
                            kern(ap, bp, cp, (long)ldc * 4, effKc);
                        }
                    }
                    ScalarTails(a, lda, b, ldb, c, ldc, ic, jc, effMc, effNc, mFull, nFull, pc, effKc);
                }
            }
        }
        finally { ArrayPool<float>.Shared.Return(paArr); ArrayPool<float>.Shared.Return(pbArr); }
    }

    /// <summary>Floats needed by <see cref="PackBPanel"/> / <see cref="RunTilePackedB"/> to hold ONE
    /// jc-block's whole-K packed B (numKc panels × nTiles × kc × Nr; last K-panel padded to kc).</summary>
    internal static long PackedBLen(int effNc, int k, int kc)
    {
        int nTiles = (effNc - (effNc % Nr)) / Nr;
        int numKc = (k + kc - 1) / kc;
        return (long)numKc * nTiles * kc * Nr;
    }

    /// <summary>Pack a whole-K B column-block B[0:k, jc:jc+effNc] ONCE into <paramref name="dst"/> (the
    /// CCX-shared layout): K-panel p at p·nTiles·kc·Nr, Nr-tile nt at +nt·kc·Nr, rows kk·Nr (effKc used).
    /// Packed once per CCX so the CCX's threads reuse it from their shared L3 with no redundant packing.
    /// <paramref name="ntStart"/>/<paramref name="ntStride"/> let the CCX's lanes split the pack work
    /// (lane, threadsPerCcx) so the pack itself is parallel — no serial-pack stall on skewed shapes.</summary>
    internal static unsafe void PackBPanel(float* b, int ldb, int jc, int effNc, int k, int kc, float* dst,
        int ntStart = 0, int ntStride = 1)
    {
        int nFull = effNc - (effNc % Nr); int nTiles = nFull / Nr;
        for (int p = 0, pc = 0; pc < k; p++, pc += kc)
        {
            int effKc = Math.Min(kc, k - pc);
            for (int nt = ntStart; nt < nTiles; nt += ntStride)
            {
                float* d = dst + (long)p * nTiles * kc * Nr + (long)nt * kc * Nr; int col0 = jc + nt * Nr;
                for (int kk = 0; kk < effKc; kk++)
                {
                    float* brow = b + (long)(pc + kk) * ldb + col0; float* dd = d + (long)kk * Nr;
                    for (int col = 0; col < Nr; col++) dd[col] = brow[col];
                }
            }
        }
    }

    /// <summary>Like <see cref="RunTile"/> but B is already packed (by <see cref="PackBPanel"/>) in
    /// <paramref name="packedB"/> — this tile only packs its own A-panel and runs the kernel + tails
    /// (tails read the original b). Deterministic (fixed K order, disjoint C). Used by the CCX driver.</summary>
    internal static unsafe void RunTilePackedB(
        float* a, int lda, float* b, int ldb, float* c, int ldc,
        int ic, int jc, int effMc, int effNc, int k, int mc, int nc, int kc, float* packedB)
    {
        int nFull = effNc - (effNc % Nr); int nTiles = nFull / Nr;
        int mFull = effMc - (effMc % Mr); int mTiles = mFull / Mr;
        float[] paArr = ArrayPool<float>.Shared.Rent(mc * kc);
        try
        {
            fixed (float* pa = paArr)
            {
                var kern = Kernel();
                for (int p = 0, pc = 0; pc < k; p++, pc += kc)
                {
                    int effKc = Math.Min(kc, k - pc);
                    for (int mt = 0; mt < mTiles; mt++)
                    {
                        float* dst = pa + (long)mt * effKc * Mr; int row0 = ic + mt * Mr;
                        for (int kk = 0; kk < effKc; kk++)
                        {
                            float* d = dst + (long)kk * Mr;
                            for (int row = 0; row < Mr; row++) d[row] = a[(long)(row0 + row) * lda + (pc + kk)];
                        }
                    }
                    if (pc == 0) ZeroCBlock(c, ldc, ic, jc, effMc, effNc);
                    for (int nt = 0; nt < nTiles; nt++)
                    {
                        float* bp = packedB + (long)p * nTiles * kc * Nr + (long)nt * kc * Nr; int col0 = jc + nt * Nr;
                        for (int mt = 0; mt < mTiles; mt++)
                        {
                            float* ap = pa + (long)mt * effKc * Mr;
                            float* cp = c + (long)(ic + mt * Mr) * ldc + col0;
                            kern(ap, bp, cp, (long)ldc * 4, effKc);
                        }
                    }
                    ScalarTails(a, lda, b, ldb, c, ldc, ic, jc, effMc, effNc, mFull, nFull, pc, effKc);
                }
            }
        }
        finally { ArrayPool<float>.Shared.Return(paArr); }
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
#else
    // net471 / pre-net5: the machine-code microkernel uses `delegate* unmanaged`, which requires the
    // net5.0+ runtime calling-convention support. Unavailable here; callers gate on IsAvailable.
    internal static bool IsAvailable => false;

    internal static unsafe void RunSingle(
        float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n, int k, int mc, int nc, int kc)
        => throw new NotSupportedException("GotoGemmFp32 requires the net5.0+ machine-code JIT path; gate on IsAvailable.");

    internal static unsafe void RunParallel(
        float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n, int k, int mc, int nc, int kc)
        => throw new NotSupportedException("GotoGemmFp32 requires the net5.0+ machine-code JIT path; gate on IsAvailable.");
#endif
}
