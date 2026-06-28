using System;
using System.Buffers;
using System.Runtime.InteropServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
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

    /// <summary>Shape regime where the per-tile GotoBLAS path beats the PackBoth strategy (measured on the
    /// 3990X via --ab-prod): large/balanced (M≥512) OR wide-K (K≥2N, e.g. MLP-fc2). PackBoth's wide-N
    /// N-axis path wins the small-M wide-N shapes (DiT QKV M256×N3456, MLP-fc1 M256×N4608 — GotoGemm was
    /// 0.86-0.88× there), so those are excluded to avoid a production regression on the diffusion forward.</summary>
    // GotoGemm beats PackBoth for: large-M (m>=512), deep-K (k>=2n), AND moderate-M decent-N — EXCEPT
    // thin-M VERY-wide-N (m<512 && n>=2k), where PackBoth's N-axis path wins by sharing A. PerfView PMC +
    // OpenBLAS-source analysis (2026-06-26): RunParallel packs A/B into PRIVATE per-thread tile panels, so
    // for thin-M wide-N it re-packs A ~n/nc times (the per-tile work shows 86% of LLC misses; per-core
    // 19.6 GF/s vs N-axis's shared-A 32). Clean A/B (--ab-goto-vs-naxis, DOP32): N-axis beats RunParallel
    // 384x2048x1024 1.17x, 384x4096x1536 1.32x, 384x6144x1536 1.31x; while GotoGemm still wins balanced
    // 768x2048x2048 1.33x and 1536^3 1.49x. So claim large/deep/moderate, cede thin-M-very-wide-N.
    // Cede thin-M very-wide-N ONLY when PackBoth's N-axis can actually take it (m % Mr == 0, Mr=6 for the
    // FP32 6x16 kernel): for m%6!=0 (e.g. the m=256 DiT QKV/MLP shapes) N-axis is disabled and PackBoth
    // falls to M-axis/2D which LOSES to GotoGemm (the old --ab-shortm 2.06x), so GotoGemm must keep those.
    internal static bool BeatsPackBoth(int m, int n, int k)
        => m >= 512
        || (long)k >= 2L * n
        || (m >= 128 && n >= 512 && ((long)n < 2L * k || (m % Mr) != 0));

    /// <summary>bf16-B win regime (opt-in, gated by BlasManaged.s_gemmBf16). bf16-B halves the streamed
    /// B bytes, so it only pays off once B no longer fits cache and the GEMM is bandwidth-bound — measured
    /// crossover between sq1024 (compute-bound, bf16 0.80×) and sq2048 (bandwidth-bound, 1.13×). Gate on
    /// total work ≥ 4e9 (≈ sq1585³) so opt-in users still get exact fp32 on the smaller shapes where bf16
    /// would only add pack overhead and precision loss. Wide-N short-M (n≥2k) is excluded — it streams B
    /// less and stays compute/pack-bound (matches the bf16 measurement showing the win is on the squares).</summary>
    internal static bool ShouldUseBf16(int m, int n, int k)
        => (long)m * n * k >= 4_000_000_000L && (long)n < 2L * k && (long)m >= 512;

    /// <summary>Shape-adaptive (Mc, Nc, Kc) for RunParallel, tuned on the 3990X (measured --ab-goto-par /
    /// --profile-gemm). Memory-bound regime: larger square shapes want larger tiles (fewer redundant DRAM
    /// re-reads); skewed / smaller shapes want smaller Mc for enough IC-blocks. Kc=256 (see ParallelKc).</summary>
    internal static int s_kcOverride; // 0 = use ParallelKc; A/B knob for the kc sweep
    // kc=256 (not 512): once the pack is SIMD (B 2xVector256, A 8x8 transpose), the smaller K-block that
    // keeps the microkernel's per-C-tile working set (A 6×kc + B kc×16 ≈ 22·kc floats) in L1 (32KB) wins —
    // sq1024 +25%, sq2048 +9% vs kc=512 (measured --profile-gemm). With the old scalar pack, kc=512 won
    // because it amortized the (then-expensive) packing; SIMD packing flipped the tradeoff toward L1 fit.
    private const int ParallelKc = 256;
    internal static (int mc, int nc, int kc) ChooseParallelBlocks(int m, int n)
    {
        int kc = s_kcOverride > 0 ? s_kcOverride : ParallelKc;
        if (m >= 1536 && n >= 1536) return (192, 256, kc); // large square
        if (m >= 640 && n >= 640) return (96, 128, kc);    // medium square
        if (m <= 512)
        {
            // Short-M (the DiT/transformer batch regime). Two coupled levers (measured via --ab-shortm):
            //  (1) mc must make FEW, BALANCED M-blocks — a partial M-block runs a scalar M-tail that re-does
            //      the WHOLE K per tail-row, so for deep-K it is very expensive. mc=64 on m=256 = 4 blocks ×
            //      4-row tail craters deep-K to 32% MKL; mc=128 = 2 even blocks holds ~49%. So derive the
            //      block count from M (~128-row blocks), NOT from the core count.
            //  (2) nc then fills the cores (ic*ceil(n/nc) ~ cores), narrow for narrow-N, clamped [32,128]
            //      (a multiple of Nr=16) for microkernel efficiency + a small L2-resident B-panel.
            // Net: m=256,n=4608 374→455 GF (1.2x); m=256,n=1152 deep-K 251→~387 GF (1.5x) vs the old (120,128).
            int cores = CpuParallelSettings.MaxDegreeOfParallelism;
            if (cores <= 0) cores = System.Environment.ProcessorCount;
            int ic = System.Math.Max(1, (int)System.Math.Round(m / 128.0));
            int mc = (m + ic - 1) / ic;
            int targetJc = System.Math.Max(1, (cores + ic - 1) / ic);
            int nc = System.Math.Max(32, System.Math.Min(128, n / targetJc));
            nc = System.Math.Max(16, (nc / 16) * 16);
            return (mc, nc, kc);
        }
        return (120, 128, kc);                              // skewed / smaller
    }

#if NET5_0_OR_GREATER
    // Direct pack-vs-kernel timing (diagnostic; set s_timing during a profiling run only). Coarse
    // per-K-panel Stopwatch deltas summed across worker threads — answers "is the cost packing or
    // the microkernel?" without profiler pseudo-frame ambiguity.
    internal static bool s_timing;
    internal static long s_packTicks, s_kernTicks, s_packBTicks, s_packATicks;
    internal static void ResetTiming() { s_packTicks = 0; s_kernTicks = 0; s_packBTicks = 0; s_packATicks = 0; }
    /// <summary>Format the pack-vs-kernel timing as a string for the caller (bench) to log — src must not
    /// write to Console directly.</summary>
    internal static string ReportTiming()
    {
        double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        long tot = s_packTicks + s_kernTicks; if (tot == 0) return "(no timing)";
        return $"  pack   = {s_packTicks * f,9:F0} ms  ({100.0 * s_packTicks / tot:F1}%)  [B={100.0 * s_packBTicks / tot:F1}% A={100.0 * s_packATicks / tot:F1}%]\n"
             + $"  kernel = {s_kernTicks * f,9:F0} ms  ({100.0 * s_kernTicks / tot:F1}%)";
    }

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
                    var code = MachineCodeFmaKernel.EmitFp32TileAbi(Mr, NrYmm, false);
                    s_kernelMem = ExecutableMemory.TryAllocate(code);
                    s_kernel = s_kernelMem?.Pointer ?? IntPtr.Zero;
                    Volatile.Write(ref s_kernelInit, 1);
                }
            }
        }
        return (delegate* unmanaged<float*, float*, float*, long, long, void>)(void*)s_kernel;
    }

    // Overwrite variant: C = acc (no read+add). Used for the FIRST K-panel so callers skip the ZeroCBlock
    // zero-pass + the panel-0 C read-back.
    private static IntPtr s_kernelOw;
    private static ExecutableMemory? s_kernelOwMem;
    private static int s_kernelOwInit;

    private static unsafe delegate* unmanaged<float*, float*, float*, long, long, void> KernelOw()
    {
        if (Volatile.Read(ref s_kernelOwInit) == 0)
        {
            lock (typeof(GotoGemmFp32))
            {
                if (s_kernelOwInit == 0)
                {
                    var code = MachineCodeFmaKernel.EmitFp32TileAbi(Mr, NrYmm, true);
                    s_kernelOwMem = ExecutableMemory.TryAllocate(code);
                    s_kernelOw = s_kernelOwMem?.Pointer ?? IntPtr.Zero;
                    Volatile.Write(ref s_kernelOwInit, 1);
                }
            }
        }
        return (delegate* unmanaged<float*, float*, float*, long, long, void>)(void*)s_kernelOw;
    }

    // bf16-B kernels (B stored bf16 → half the bytes; A fp32; fp32 accumulate). #19 bandwidth lever.
    private static IntPtr s_kBf16, s_kBf16Ow;
    private static ExecutableMemory? s_kBf16Mem, s_kBf16OwMem;
    private static int s_kBf16Init, s_kBf16OwInit;
    private static unsafe delegate* unmanaged<float*, float*, float*, long, long, void> KernelBf16(bool ow)
    {
        ref int init = ref (ow ? ref s_kBf16OwInit : ref s_kBf16Init);
        if (Volatile.Read(ref init) == 0)
            lock (typeof(GotoGemmFp32))
                if (init == 0)
                {
                    var code = MachineCodeFmaKernel.EmitBf16BTileWindows(Mr, NrYmm, ow);
                    var mem = ExecutableMemory.TryAllocate(code);
                    if (ow) { s_kBf16OwMem = mem; s_kBf16Ow = mem?.Pointer ?? IntPtr.Zero; }
                    else { s_kBf16Mem = mem; s_kBf16 = mem?.Pointer ?? IntPtr.Zero; }
                    Volatile.Write(ref init, 1);
                }
        return (delegate* unmanaged<float*, float*, float*, long, long, void>)(void*)(ow ? s_kBf16Ow : s_kBf16);
    }

    /// <summary>Truncate fp32→bf16 (take the high 16 bits). Used to pack B as bf16 for the bf16-B GEMM.</summary>
    private static ushort ToBf16(float f) => (ushort)(System.BitConverter.SingleToUInt32Bits(f) >> 16);

    /// <summary>Zero only the M-tail rows + N-tail cols of a C block (the kernel overwrites the aligned
    /// Mr×Nr interior on the first K-panel; only the scalar-tail regions still need pre-zeroing).</summary>
    private static unsafe void ZeroTailStrips(float* c, int ldc, int ic, int jc, int effMc, int effNc, int mFull, int nFull)
    {
        for (int r = mFull; r < effMc; r++) { float* cr = c + (long)(ic + r) * ldc + jc; for (int col = 0; col < effNc; col++) cr[col] = 0f; }
        // N-tail cols [nFull, effNc) of the full rows. cr is based at jc (NOT jc+nFull) so cr[col] hits the
        // actual tail columns and never writes past the tile (the +nFull offset double-counted before).
        for (int r = 0; r < mFull; r++) { float* cr = c + (long)(ic + r) * ldc + jc; for (int col = nFull; col < effNc; col++) cr[col] = 0f; }
    }

    /// <summary>True when the machine-code microkernel is available on this CPU/OS (AVX2+FMA, x64). The tile
    /// kernel is emitted for the CURRENT OS's ABI via MachineCodeFmaKernel.EmitFp32TileAbi — Microsoft x64 on
    /// Windows, System V AMD64 on Linux/macOS — so it matches the platform-default `delegate* unmanaged`
    /// convention on both. (A Windows-ABI kernel on Linux read the kc arg from the [rsp+0x28] shadow space
    /// that SysV doesn't have → SIGSEGV; the SysV kernel reads it from r8.)</summary>
    internal static unsafe bool IsAvailable => MachineKernelGemm.IsFp32Available && Kernel() != null;

    /// <summary>
    /// Single-thread C = A·B (row-major). A[M,K] lda, B[K,N] ldb, C[M,N] ldc (element strides).
    /// C is OVERWRITTEN (zeroed for the first K-panel via the pack-once-then-accumulate structure).
    /// </summary>
    internal static unsafe void RunSingle(
        float* a, int lda, float* b, int ldb, float* c, int ldc,
        int m, int n, int k, int mc, int nc, int kc)
    {
        if (k <= 0) { ZeroCBlock(c, ldc, 0, 0, m, n); return; } // C = A·B with K=0 is the zero matrix
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
        if (k <= 0) { ZeroCBlock(c, ldc, 0, 0, m, n); return; } // C = A·B with K=0 is the zero matrix
        int numIc = (m + mc - 1) / mc;
        int numJc = (n + nc - 1) / nc;
        int totalTiles = numIc * numJc;
        if (totalTiles <= 1) { RunSingle(a, lda, b, ldb, c, ldc, m, n, k, mc, nc, kc); return; }

        // Per-tile parallelism: each worker owns a disjoint (mc×nc) C-tile and packs ITS OWN A-panel +
        // B-panel (per kc-panel) into per-thread L2 buffers, so the microkernel's operands stay L2-resident
        // with no shared-L3 streaming. This redundantly re-packs A across jc and B across ic, but that
        // redundancy is the PRICE of L2-residency: both the A-cache (ic-major) and B-cache (full-K B[jc] =
        // K·nc > L2 → L3 reads) variants were MEASURED to regress (the cached panel spills L2 / source
        // locality breaks). The genuine fix is the CCX/NUMA hierarchy (per-CCX shared panel), not a flat
        // cache. Each tile runs its full K-loop independently (disjoint C ⇒ no races, deterministic).
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

    /// <summary>Deep-K + short-M is UNDER-PARALLELIZED (few (m/mc)·(n/nc) tiles for many cores). Split-K
    /// helps there. The split count G is derived from the SHAPE ONLY (never the thread/core count) so the
    /// reduction order is fixed ⇒ the result is thread-count-INDEPENDENT (Deterministic-mode safe).</summary>
    internal static bool ShouldSplitK(int m, int n, int k)
        => m <= 512 && k >= 4 * Math.Max(m, n) && (long)m * n * k >= ParallelMinWork;

    internal static int SplitKGroups(int m, int n, int k)
        => Math.Min(4, Math.Max(2, (int)Math.Round((double)k / (2 * Math.Max(m, n)))));

    /// <summary>Split-K GEMM: G shape-based K-groups computed as G partial products in parallel
    /// ((tile×G) work units instead of tiles), then summed in fixed g-order. For the deep-K short-M shapes
    /// (e.g. the DiT MLP down-projection) this fills the cores the per-tile scheme leaves idle. Deterministic
    /// (G is shape-only; each partial is a fixed-K-order tile; the reduction order is fixed).</summary>
    internal static unsafe void RunParallelSplitK(
        float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n, int k, int mc, int nc, int kc)
    {
        int G = SplitKGroups(m, n, k);
        int kg = (k + G - 1) / G;
        int numIc = (m + mc - 1) / mc, numJc = (n + nc - 1) / nc, totalTiles = numIc * numJc;
        long mn = (long)m * n;
        float[] tmp = ArrayPool<float>.Shared.Rent(checked((int)(mn * G)));
        try
        {
            fixed (float* pt = tmp)
            {
                nint ai = (nint)a, bi = (nint)b, ti = (nint)pt;
                int numIcL = numIc, mcL = mc, ncL = nc, kcL = kc, mL = m, nL = n, kL = k, ldaL = lda, ldbL = ldb, GL = G, kgL = kg;
                // (tile, g) → partial C-tile for K-range [g*kg, …) into tmp[g] (disjoint per g, per tile).
                CpuParallelSettings.ParallelForOrSerial(0, totalTiles * G, (long)m * n * k, taskIdx =>
                {
                    int g = taskIdx % GL;
                    int tile = taskIdx / GL;
                    int ic = (tile % numIcL) * mcL;
                    int jc = (tile / numIcL) * ncL;
                    int effMc = Math.Min(mcL, mL - ic);
                    int effNc = Math.Min(ncL, nL - jc);
                    if (effMc <= 0 || effNc <= 0) return;
                    int k0 = g * kgL; int kThis = Math.Min(kgL, kL - k0); if (kThis <= 0) return;
                    float* gbuf = (float*)ti + (long)g * mL * nL;
                    RunTile((float*)ai + k0, ldaL, (float*)bi + (long)k0 * ldbL, ldbL, gbuf, nL, ic, jc, effMc, effNc, kThis, mcL, ncL, kcL);
                }, deterministicSafe: true);
                // C = Σ_g tmp[g] in fixed g order (deterministic), parallel over rows.
                nint ci = (nint)c; int mL2 = m, nL2 = n, GL2 = G, ldcL = ldc;
                CpuParallelSettings.ParallelForOrSerial(0, m, (long)m * n * G, i =>
                {
                    float* cp = (float*)ci + (long)i * ldcL;
                    float* t0 = (float*)ti + (long)i * nL2;
                    long stride = (long)mL2 * nL2;
                    int j = 0;
                    if (Avx.IsSupported)
                        for (; j + 8 <= nL2; j += 8)
                        {
                            var acc = Avx.LoadVector256(t0 + j);
                            for (int g = 1; g < GL2; g++) acc = Avx.Add(acc, Avx.LoadVector256(t0 + g * stride + j));
                            Avx.Store(cp + j, acc);
                        }
                    for (; j < nL2; j++)
                    {
                        float s = t0[j];
                        for (int g = 1; g < GL2; g++) s += t0[g * stride + j];
                        cp[j] = s;
                    }
                }, deterministicSafe: true);
            }
        }
        finally { ArrayPool<float>.Shared.Return(tmp); }
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
        float[] paArr = ArrayPool<float>.Shared.Rent(mc * kc + 8); // +8: SIMD A-pack transpose may store 2 past the last tile
        float[] pbArr = ArrayPool<float>.Shared.Rent(kc * nc);
        try
        {
            fixed (float* pa = paArr, pb = pbArr)
            {
                var kAcc = Kernel(); var kOw = KernelOw();
                bool timing = s_timing;
                for (int pc = 0; pc < k; pc += kc)
                {
                    int effKc = Math.Min(kc, k - pc);
                    long t0 = timing ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    for (int nt = 0; nt < nTiles; nt++)
                    {
                        float* dst = pb + (long)nt * effKc * Nr; int col0 = jc + nt * Nr;
                        for (int kk = 0; kk < effKc; kk++)
                        {
                            // SIMD copy of the Nr=16 contiguous B-cols (was scalar — packing was ~50% of GEMM).
                            float* brow = b + (long)(pc + kk) * ldb + col0; float* d = dst + (long)kk * Nr;
                            Vector256.Store(Vector256.Load(brow), d);
                            Vector256.Store(Vector256.Load(brow + 8), d + 8);
                        }
                    }
                    long tB = timing ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    PackA6(a, lda, pa, ic, pc, effKc, mTiles); // SIMD A-pack (8x8 AVX2 transpose; scalar fallback)
                    // First K-panel: overwrite kernel (C = acc) + zero only the scalar-tail strips → skip the
                    // full ZeroCBlock zero-pass + the panel-0 C read-back. Later panels accumulate.
                    var kern = pc == 0 ? kOw : kAcc;
                    if (pc == 0) ZeroTailStrips(c, ldc, ic, jc, effMc, effNc, mFull, nFull);
                    long t1 = timing ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
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
                    if (timing)
                    {
                        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
                        Interlocked.Add(ref s_packTicks, t1 - t0);
                        Interlocked.Add(ref s_packBTicks, tB - t0);
                        Interlocked.Add(ref s_packATicks, t1 - tB);
                        Interlocked.Add(ref s_kernTicks, t2 - t1);
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
                    Vector256.Store(Vector256.Load(brow), dd);          // SIMD copy of the Nr=16 contiguous B-cols
                    Vector256.Store(Vector256.Load(brow + 8), dd + 8);
                }
            }
        }
    }

    /// <summary>Floats for ONE kc×nc B-panel (nTiles·kc·Nr) — the 2D-NUMA GotoBLAS step packs B per (pc,jc),
    /// not whole-K, so the active panel stays ~L2-sized and the microkernel never reads B from a big L3 block.</summary>
    internal static long PackedBPanelLen(int effNc, int kc)
    {
        int nTiles = (effNc - (effNc % Nr)) / Nr;
        return (long)nTiles * kc * Nr;
    }

    /// <summary>Pack ONE kc-panel of B[pc:pc+effKc, jc:jc+effNc] into <paramref name="pkB"/> as [nt][kk][col]
    /// (nt-stride = kc·Nr). Lane-strided (ntStart/ntStride) for parallel packing within a CCX. Used by the
    /// 2D-NUMA driver: the panel lives in the CCX's L3 and is reused across that block's ic-blocks.</summary>
    internal static unsafe void PackBPanelPc(float* b, int ldb, int jc, int effNc, int pc, int effKc, int kc,
        float* pkB, int ntStart, int ntStride)
    {
        int nFull = effNc - (effNc % Nr); int nTiles = nFull / Nr;
        for (int nt = ntStart; nt < nTiles; nt += ntStride)
        {
            float* dst = pkB + (long)nt * kc * Nr; int col0 = jc + nt * Nr;
            for (int kk = 0; kk < effKc; kk++)
            {
                float* brow = b + (long)(pc + kk) * ldb + col0; float* d = dst + (long)kk * Nr;
                Vector256.Store(Vector256.Load(brow), d);
                Vector256.Store(Vector256.Load(brow + 8), d + 8);
            }
        }
    }

    /// <summary>One GotoBLAS macro step for the 2D-NUMA driver: C[ic:ic+effMc, jc:jc+effNc] += A[ic,pc]·B[pc],
    /// where B's pc-panel is already packed in <paramref name="pkB"/> (PackBPanelPc, in the CCX's L3) and A's
    /// pc-panel is packed here into a per-lane L2 buffer (PackA6). C is zeroed when <paramref name="firstPc"/>
    /// then accumulated across pc (RMW in L2). Tails read original a,b. Deterministic (fixed K order, disjoint C).</summary>
    internal static unsafe void RunMacroPanelStep(
        float* a, int lda, float* b, int ldb, float* c, int ldc,
        int ic, int jc, int effMc, int effNc, int pc, int effKc, int mc, int kc, float* pkB, bool firstPc)
    {
        int nFull = effNc - (effNc % Nr); int nTiles = nFull / Nr;
        int mFull = effMc - (effMc % Mr); int mTiles = mFull / Mr;
        float[] paArr = ArrayPool<float>.Shared.Rent(mc * kc + 8); // +8: PackA6 transpose may store 2 past the last tile
        try
        {
            fixed (float* pa = paArr)
            {
                PackA6(a, lda, pa, ic, pc, effKc, mTiles);
                var kern = firstPc ? KernelOw() : Kernel();    // first panel overwrites C; later accumulate
                if (firstPc) ZeroTailStrips(c, ldc, ic, jc, effMc, effNc, mFull, nFull);
                for (int nt = 0; nt < nTiles; nt++)
                {
                    float* bp = pkB + (long)nt * kc * Nr; int col0 = jc + nt * Nr;
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
        finally { ArrayPool<float>.Shared.Return(paArr); }
    }

    /// <summary>Floats needed to hold ONE ic-block's whole-K packed A (numKc · mTiles · kc · Mr).</summary>
    internal static long PackedALen(int effMc, int k, int kc)
    {
        int mTiles = (effMc - (effMc % Mr)) / Mr;
        int numKc = (k + kc - 1) / kc;
        return (long)numKc * mTiles * kc * Mr;
    }

    /// <summary>Pack a whole-K A row-block A[ic:ic+effMc, 0:k] ONCE into <paramref name="dst"/>: K-panel p
    /// at p·mTiles·kc·Mr, Mr-tile mt at +mt·kc·Mr, rows kk·Mr (effKc used). mtStart/mtStride let the CCX's
    /// lanes pack in parallel. Used by the 2D CCX grid so each CCX's A-block lives in its own L3.</summary>
    internal static unsafe void PackAPanel(float* a, int lda, int ic, int effMc, int k, int kc, float* dst,
        int mtStart = 0, int mtStride = 1)
    {
        int mFull = effMc - (effMc % Mr); int mTiles = mFull / Mr;
        for (int p = 0, pc = 0; pc < k; p++, pc += kc)
        {
            int effKc = Math.Min(kc, k - pc);
            for (int mt = mtStart; mt < mTiles; mt += mtStride)
            {
                float* d = dst + (long)p * mTiles * kc * Mr + (long)mt * kc * Mr; int row0 = ic + mt * Mr;
                for (int kk = 0; kk < effKc; kk++)
                {
                    float* dd = d + (long)kk * Mr;
                    for (int row = 0; row < Mr; row++) dd[row] = a[(long)(row0 + row) * lda + (pc + kk)];
                }
            }
        }
    }

    /// <summary>2D CCX-grid block: compute C[ic0:ic0+effMc, jc0:jc0+effNc] from a whole-K pre-packed
    /// A-block (<paramref name="packedA"/>, PackAPanel) and B-block (<paramref name="packedB"/>, PackBPanel)
    /// — BOTH resident in this CCX's L3, so the microkernel streams neither from DRAM. The CCX's lanes
    /// split the Mr×Nr micro-tiles (laneStart/laneStride); each micro-tile is zeroed then accumulated over
    /// the full K (fixed order ⇒ deterministic, disjoint C ⇒ no races). lane 0 does the block's M/N tails
    /// (reads original a,b). mTilesAll/nTilesAll are the block's full Mr/Nr tile counts (pack strides).</summary>
    internal static unsafe void RunBlockPackedAB(
        float* a, int lda, float* b, int ldb, float* c, int ldc,
        int ic0, int jc0, int effMc, int effNc, int k, int kc,
        float* packedA, float* packedB, int laneStart, int laneStride)
    {
        int nFull = effNc - (effNc % Nr); int nTiles = nFull / Nr;
        int mFull = effMc - (effMc % Mr); int mTiles = mFull / Mr;
        var kern = Kernel();
        int totalMt = mTiles * nTiles;
        for (int idx = laneStart; idx < totalMt; idx += laneStride)
        {
            int mt = idx % mTiles; int nt = idx / mTiles;
            float* cp = c + (long)(ic0 + mt * Mr) * ldc + (jc0 + nt * Nr);
            for (int r = 0; r < Mr; r++) { float* cr = cp + (long)r * ldc; for (int col = 0; col < Nr; col++) cr[col] = 0f; }
            for (int p = 0, pc = 0; pc < k; p++, pc += kc)
            {
                int effKc = Math.Min(kc, k - pc);
                float* ap = packedA + (long)p * mTiles * kc * Mr + (long)mt * kc * Mr;
                float* bp = packedB + (long)p * nTiles * kc * Nr + (long)nt * kc * Nr;
                kern(ap, bp, cp, (long)ldc * 4, effKc);
            }
        }
        // Block M/N tails (single lane): zero the tail strips then scalar-accumulate over full K.
        if (laneStart == 0 && (mFull < effMc || nFull < effNc))
        {
            for (int r = mFull; r < effMc; r++) { float* cr = c + (long)(ic0 + r) * ldc + jc0; for (int col = 0; col < effNc; col++) cr[col] = 0f; }
            for (int r = 0; r < mFull; r++) { float* cr = c + (long)(ic0 + r) * ldc + jc0 + nFull; for (int col = nFull; col < effNc; col++) cr[col - nFull] = 0f; }
            for (int p = 0, pc = 0; pc < k; p++, pc += kc)
            {
                int effKc = Math.Min(kc, k - pc);
                ScalarTails(a, lda, b, ldb, c, ldc, ic0, jc0, effMc, effNc, mFull, nFull, pc, effKc);
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
        float[] paArr = ArrayPool<float>.Shared.Rent(mc * kc + 8); // +8: SIMD A-pack transpose may store 2 past the last tile
        try
        {
            fixed (float* pa = paArr)
            {
                var kAcc = Kernel(); var kOw = KernelOw();
                for (int p = 0, pc = 0; pc < k; p++, pc += kc)
                {
                    int effKc = Math.Min(kc, k - pc);
                    PackA6(a, lda, pa, ic, pc, effKc, mTiles); // SIMD A-pack (8x8 AVX2 transpose; scalar fallback)
                    var kern = pc == 0 ? kOw : kAcc;           // first panel overwrites C; later panels accumulate
                    if (pc == 0) ZeroTailStrips(c, ldc, ic, jc, effMc, effNc, mFull, nFull);
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

    /// <summary>Pack the A-panel A[ic:ic+mTiles*Mr, pc:pc+effKc] into <paramref name="pa"/> as [mt][kk][row]
    /// (effKc*Mr stride per mt) via an 8x8 AVX2 transpose (A is row-major K-contiguous → microkernel's
    /// kk-major layout). Each store writes 8 floats (low Mr=6 valid); the +2 overlap is overwritten by the
    /// next column/chunk — the caller MUST pad the pa buffer by 8. Scalar K-tail + scalar fallback (non-AVX).
    /// Shared by RunTile and RunTilePackedB.</summary>
    private static unsafe void PackA6(float* a, int lda, float* pa, int ic, int pc, int effKc, int mTiles)
    {
        for (int mt = 0; mt < mTiles; mt++)
        {
            float* dst = pa + (long)mt * effKc * Mr; int row0 = ic + mt * Mr;
            if (Avx.IsSupported)
            {
                float* a0 = a + (long)(row0 + 0) * lda + pc, a1 = a + (long)(row0 + 1) * lda + pc;
                float* a2 = a + (long)(row0 + 2) * lda + pc, a3 = a + (long)(row0 + 3) * lda + pc;
                float* a4 = a + (long)(row0 + 4) * lda + pc, a5 = a + (long)(row0 + 5) * lda + pc;
                int kc8 = effKc & ~7;
                for (int kk = 0; kk < kc8; kk += 8)
                {
                    var r0 = Avx.LoadVector256(a0 + kk); var r1 = Avx.LoadVector256(a1 + kk);
                    var r2 = Avx.LoadVector256(a2 + kk); var r3 = Avx.LoadVector256(a3 + kk);
                    var r4 = Avx.LoadVector256(a4 + kk); var r5 = Avx.LoadVector256(a5 + kk);
                    var zz = Vector256<float>.Zero;
                    var u0 = Avx.UnpackLow(r0, r1); var u1 = Avx.UnpackHigh(r0, r1);
                    var u2 = Avx.UnpackLow(r2, r3); var u3 = Avx.UnpackHigh(r2, r3);
                    var u4 = Avx.UnpackLow(r4, r5); var u5 = Avx.UnpackHigh(r4, r5);
                    var u6 = Avx.UnpackLow(zz, zz); var u7 = Avx.UnpackHigh(zz, zz);
                    var s0 = Avx.Shuffle(u0, u2, 0x44); var s1 = Avx.Shuffle(u0, u2, 0xEE);
                    var s2 = Avx.Shuffle(u1, u3, 0x44); var s3 = Avx.Shuffle(u1, u3, 0xEE);
                    var s4 = Avx.Shuffle(u4, u6, 0x44); var s5 = Avx.Shuffle(u4, u6, 0xEE);
                    var s6 = Avx.Shuffle(u5, u7, 0x44); var s7 = Avx.Shuffle(u5, u7, 0xEE);
                    float* dd = dst + (long)kk * Mr;
                    Avx.Store(dd + 0 * Mr, Avx.Permute2x128(s0, s4, 0x20));
                    Avx.Store(dd + 1 * Mr, Avx.Permute2x128(s1, s5, 0x20));
                    Avx.Store(dd + 2 * Mr, Avx.Permute2x128(s2, s6, 0x20));
                    Avx.Store(dd + 3 * Mr, Avx.Permute2x128(s3, s7, 0x20));
                    Avx.Store(dd + 4 * Mr, Avx.Permute2x128(s0, s4, 0x31));
                    Avx.Store(dd + 5 * Mr, Avx.Permute2x128(s1, s5, 0x31));
                    Avx.Store(dd + 6 * Mr, Avx.Permute2x128(s2, s6, 0x31));
                    Avx.Store(dd + 7 * Mr, Avx.Permute2x128(s3, s7, 0x31));
                }
                for (int kk = kc8; kk < effKc; kk++)
                {
                    float* d = dst + (long)kk * Mr;
                    d[0] = a0[kk]; d[1] = a1[kk]; d[2] = a2[kk]; d[3] = a3[kk]; d[4] = a4[kk]; d[5] = a5[kk];
                }
            }
            else
            {
                for (int row = 0; row < Mr; row++)
                {
                    float* asrc = a + (long)(row0 + row) * lda + pc; float* d = dst + row;
                    for (int kk = 0; kk < effKc; kk++) d[(long)kk * Mr] = asrc[kk];
                }
            }
        }
    }

    /// <summary>bf16-B tile: like RunTile but B is packed as bf16 (half the bytes → 2× bandwidth headroom),
    /// A stays fp32, fp32 accumulate. Precision opt-in (bf16 ~3 sig digits). Tails are exact fp32 (small).</summary>
    internal static unsafe void RunTileBf16(float* a, int lda, float* b, int ldb, float* c, int ldc,
        int ic, int jc, int effMc, int effNc, int k, int mc, int nc, int kc)
    {
        int nFull = effNc - (effNc % Nr); int nTiles = nFull / Nr;
        int mFull = effMc - (effMc % Mr); int mTiles = mFull / Mr;
        float[] paArr = ArrayPool<float>.Shared.Rent(mc * kc + 8);
        ushort[] pbArr = ArrayPool<ushort>.Shared.Rent(kc * nc);
        try
        {
            fixed (float* pa = paArr) fixed (ushort* pbb = pbArr)
            {
                var kAcc = KernelBf16(false); var kOw = KernelBf16(true);
                for (int pc = 0; pc < k; pc += kc)
                {
                    int effKc = Math.Min(kc, k - pc);
                    for (int nt = 0; nt < nTiles; nt++)
                    {
                        ushort* dst = pbb + (long)nt * effKc * Nr; int col0 = jc + nt * Nr;
                        if (Avx2.IsSupported && Nr == 16)
                        {
                            // SIMD fp32→bf16 truncation: shift each fp32 right 16 (bf16 = high 16 bits), pack
                            // 8+8 u32→16 u16 (vpackusdw — exact since hi16=0 after the shift), then fix the
                            // AVX2 in-lane pack order with vpermq [0,2,1,3]=0xD8 → 16 bf16 in column order.
                            for (int kk = 0; kk < effKc; kk++)
                            {
                                float* brow = b + (long)(pc + kk) * ldb + col0; ushort* d = dst + (long)kk * Nr;
                                var s0 = Avx2.ShiftRightLogical(Avx.LoadVector256(brow).AsUInt32(), 16);
                                var s1 = Avx2.ShiftRightLogical(Avx.LoadVector256(brow + 8).AsUInt32(), 16);
                                var packed = Avx2.PackUnsignedSaturate(s0.AsInt32(), s1.AsInt32());
                                Avx.Store(d, Avx2.Permute4x64(packed.AsUInt64(), 0xD8).AsUInt16());
                            }
                        }
                        else
                        {
                            for (int kk = 0; kk < effKc; kk++)
                            {
                                float* brow = b + (long)(pc + kk) * ldb + col0; ushort* d = dst + (long)kk * Nr;
                                for (int col = 0; col < Nr; col++) d[col] = ToBf16(brow[col]);
                            }
                        }
                    }
                    PackA6(a, lda, pa, ic, pc, effKc, mTiles);
                    var kern = pc == 0 ? kOw : kAcc;
                    if (pc == 0) ZeroTailStrips(c, ldc, ic, jc, effMc, effNc, mFull, nFull);
                    for (int nt = 0; nt < nTiles; nt++)
                    {
                        float* bp = (float*)(pbb + (long)nt * effKc * Nr); int col0 = jc + nt * Nr;
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
        finally { ArrayPool<float>.Shared.Return(paArr); ArrayPool<ushort>.Shared.Return(pbArr); }
    }

    /// <summary>Parallel bf16-B GEMM (per-tile, B bf16) — the #19 bandwidth-doubler measurement path.</summary>
    internal static unsafe void RunParallelBf16(float* a, int lda, float* b, int ldb, float* c, int ldc,
        int m, int n, int k, int mc, int nc, int kc)
    {
        int numIc = (m + mc - 1) / mc, numJc = (n + nc - 1) / nc, totalTiles = numIc * numJc;
        if (totalTiles <= 1) { RunTileBf16(a, lda, b, ldb, c, ldc, 0, 0, m, n, k, mc, nc, kc); return; }
        nint ai = (nint)a, bi = (nint)b, ci = (nint)c;
        int numIcL = numIc, mcL = mc, ncL = nc, kcL = kc, mL = m, nL = n, kL = k, ldaL = lda, ldbL = ldb, ldcL = ldc;
        CpuParallelSettings.ParallelForOrSerial(0, totalTiles, (long)m * n * k, tileIdx =>
        {
            int ic = (int)(tileIdx % numIcL) * mcL; int jc = (int)(tileIdx / numIcL) * ncL;
            int effMc = Math.Min(mcL, mL - ic); int effNc = Math.Min(ncL, nL - jc);
            if (effMc <= 0 || effNc <= 0) return;
            RunTileBf16((float*)ai, ldaL, (float*)bi, ldbL, (float*)ci, ldcL, ic, jc, effMc, effNc, kL, mcL, ncL, kcL);
        }, deterministicSafe: true);
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
