using System;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Three-level Goto loop nest for the Pack-Both strategy: pack A panels into
/// vpanel layout, pack B panels into stripe layout, dispatch the microkernel
/// over the (M, N) tile grid. Used when both A and B benefit from cache-blocked
/// packing — typically large K shapes where pack cost amortizes across the
/// inner loops.
///
/// <para>
/// Microkernel dispatch follows an AVX-512 → AVX2 → Neon → scalar hierarchy:
/// FP64 prefers <see cref="Avx512Fp64_8x16"/> (Mr=8, Nr=16), falls back to
/// <see cref="Avx2Fp64_4x8"/> (Mr=4, Nr=8), then <see cref="NeonFp64_4x4"/>
/// (Mr=4, Nr=4) on ARM64, then <see cref="ScalarFp64_4x4"/>;
/// FP32 prefers <see cref="Avx512Fp32_16x16"/> (Mr=16, Nr=16), falls back to
/// <see cref="Avx2Fp32_8x8"/> (Mr=8, Nr=8), then <see cref="NeonFp32_8x4"/>
/// (Mr=8, Nr=4) on ARM64, then <see cref="ScalarFp32_4x4"/>.
/// </para>
///
/// <para>
/// Caller-supplied (mc, nc, kc) blocking parameters; (mr, nr) are fixed to
/// the microkernel's tile widths. The implementation handles partial outer
/// blocks (effective_mc/nc/kc = min(block, remaining)) but requires every
/// effective block to be exactly divisible by mr/nr — tail handling is added
/// in Phase G.
/// </para>
/// </summary>
internal static class PackBothStrategy
{
    // #653 diagnostic gate (env AIDOTNET_GEMM_TRACE=1): trace the chosen parallel path.
    private static readonly bool s_trace =
        System.Environment.GetEnvironmentVariable("AIDOTNET_GEMM_TRACE") == "1";

    // A/B toggle for the FP32 6×16 machine-code panel fast path (one asm call per M-block
    // vs per-tile managed dispatch). Test-only; default off = panel ON.
    internal static bool s_disablePanel;

    // Max Nr=16-wide tiles per machine-code panel call. Bounds each asm call's B/C working
    // set — a single call over a very wide panel (e.g. 384 tiles at n=6144) regresses vs
    // chunks of ~256, while chunking still amortizes per-tile dispatch. Settable for sweeps.
    internal static int MaxPanelTiles = 256;

    // #475: route the FP32 N-axis compute loop's Mr-sweep through the machine-code MACRO kernel
    // (RunMacroPanelFp32) — the whole numMr × njr × kc sweep in one asm call, taking RyuJIT off the
    // hot path (PerfView showed ~36% of single-thread time in the managed RunNAxisParallelUnsafe
    // loop body). A/B toggle; default off until bit-exactness is validated, then flipped on.
    internal static bool s_macroKernel;

    // #85 low-barrier jc-blocked GotoBLAS path (RunGotoBlasParallelUnsafe) — pack-B-per-Nc-block once
    // into shared L3, ONE parallel region per Nc-block, shared packed-A read by DISJOINT rows.
    // DEAD END — kept only as a measurement record (s_forceGotoBlas + the --ab-gotoblas* benches).
    //
    // WHY DEAD: the early "1.1-1.4x medium win" was a MEASUREMENT ARTIFACT. The production large-float
    // path is GotoGemmFp32 at BlasManaged.cs:542 (its own GotoBLAS macro-kernel + CCX pool), gated by
    // BeatsPackBoth = m≥512 || k≥2n || (m≥128 && n≥512). Every shape in this path's intended envelope
    // (k≤n≤2k with m≥128, k≥512 ⇒ n≥512) satisfies (m≥128 && n≥512) ⇒ routes to GotoGemmFp32 and NEVER
    // reaches PackBoth. So toggling s_forceGotoBlas changed nothing for those shapes — the "win" was
    // ±25% box noise. Forcing the path for real (AIDOTNET_DISABLE_GOTO=1 ⇒ falls to PackBoth) measured
    // it SLOWER than the N-axis path: ffn 0.40x, medium ~1.0x — the jc-outer-serial barrier-per-Nc-block
    // structure throttles wide-N (the single-region path exists precisely to avoid that barrier).
    // Conclusion: GotoGemmFp32 already owns this regime; this PackBoth variant adds nothing. Default OFF.
    internal static bool s_forceGotoBlas;
    internal static bool s_gotoBlasAuto; // default OFF — dead end (see above); GotoGemmFp32 owns this regime

    // #653 software-prefetch microkernel (env AIDOTNET_GEMM_PREFETCH=1, default off). Routes the
    // FP32 6×16 tile through Avx2Fp32_6x16.RunPrefetch (PREFETCHT0 of packed-B ahead) — same inlined
    // intrinsic kernel, so NO per-tile call/fixed overhead (unlike the machine-code-per-tile path
    // that regressed). Tests whether L2/L3 latency is the residual per-core gap at large N.
    // (Merge: independent of the machine-code panel path above — the 6×16 tile chooses prefetch
    // vs panel vs plain-intrinsic at dispatch; both knobs coexist.)
    private static readonly bool s_prefetch =
        System.Environment.GetEnvironmentVariable("AIDOTNET_GEMM_PREFETCH") == "1";

    // #653 single-persistent-region path (DEFAULT ON, opt-out AIDOTNET_GEMM_SINGLE_REGION=0).
    // The M-axis RunParallelUnsafe re-dispatches the parallel region (and barriers) once per
    // (jc, pc) macro-tile. For the wide-nc forward case (numNB==1) that's one barriered
    // pack-B + ic-compute pair per K-panel — e.g. the FFN's 4 K-panels = 8 parallel regions,
    // capping utilization (~8/16 cores -> 8x). This path packs ALL K-panels of B once (one
    // dispatch) then runs ONE parallel ic region where each thread walks the full K-loop for
    // its rows — the BLIS persistent-team shape, no per-K-panel barrier. Bit-identical: each
    // C element still reduces over k in the same panel order. Self-limiting (only fires on the
    // numNB==1 wide-nc case under the packed-B residency cap below); measured on top of
    // forward-packboth it lifts FFN-up busyCores ~12->15.7 (9.7->9.05ms) with no regression
    // across the 10-shape A/B, so flipped default-on alongside s_forwardPackBothBlocking.
    private static readonly bool s_singleRegion =
        System.Environment.GetEnvironmentVariable("AIDOTNET_GEMM_SINGLE_REGION") != "0";

    // Cap on the all-panels packed-B residency for the single-region path (bytes). Above this
    // the buffer blows the cache budget; fall back to the per-panel M-axis path.
    private const long SingleRegionPackBBudgetBytes = 64L * 1024 * 1024;

    /// <summary>
    /// Compute C += op(A) · op(B) using the Pack-Both 3-level loop nest.
    /// C is zeroed by the caller (BlasManaged.Gemm); the kernel accumulates.
    /// </summary>
    /// <typeparam name="T">Element type. Must be float or double.</typeparam>
    /// <param name="a">Source A buffer.</param>
    /// <param name="lda">Leading dimension of A.</param>
    /// <param name="transA">True if A is stored transposed: [K, M] layout for logical [M, K].</param>
    /// <param name="b">Source B buffer.</param>
    /// <param name="ldb">Leading dimension of B.</param>
    /// <param name="transB">True if B is stored transposed: [N, K] layout for logical [K, N].</param>
    /// <param name="c">Output matrix C, row-major [M, N] with leading dimension ldc.</param>
    /// <param name="ldc">Leading dimension of C (number of columns in the full C matrix).</param>
    /// <param name="m">Number of rows in op(A) and C.</param>
    /// <param name="n">Number of columns in op(B) and C.</param>
    /// <param name="k">Shared inner dimension: columns of op(A), rows of op(B).</param>
    /// <param name="mc">Row blocking factor (Mc); each A panel covers mc rows.</param>
    /// <param name="nc">Column blocking factor (Nc); each B stripe covers nc cols.</param>
    /// <param name="kc">K blocking factor (Kc); each packed panel covers kc K-steps.</param>
    /// <param name="mr">Microkernel row-tile width (Mr). Must divide mc exactly (Phase G adds tail).</param>
    /// <param name="nr">Microkernel column-tile width (Nr). Must divide nc exactly (Phase G adds tail).</param>
    /// <param name="options">Allocator options: workspace buffer, pre-pack handles, packing mode.</param>
    public static unsafe void Run<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        int mc, int nc, int kc,
        int mr, int nr,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        // Compute byte sizes for worst-case panel buffers.
        int elemSize = Unsafe.SizeOf<T>();
        int packABytes = mc * kc * elemSize;
        int packBBytes = kc * nc * elemSize;

        // When a caller-supplied workspace is present (Layer 5), we cannot split it
        // safely across threads. Run serial: workspace carved for both pack-A and pack-B.
        // When no workspace is supplied, the ic loop is parallelized across threads:
        // each thread rents its own pack-A from PerThreadPool.Current ([ThreadStatic]),
        // while pack-B is shared (read-only inside the parallel region).
        bool hasWorkspace = !options.Workspace.IsEmpty;

        if (hasWorkspace)
        {
            // ── Serial path (workspace-backed) ──────────────────────────────────────
            var carver = new WorkspaceCarver(options.Workspace);
            Span<byte> packABytesSpan = carver.TryCarve(packABytes);
            if (packABytesSpan.IsEmpty) packABytesSpan = ArenaIntegration.TryRentBytes(packABytes);
            if (packABytesSpan.IsEmpty) packABytesSpan = PerThreadPool.Current.RentPackA(packABytes);

            Span<byte> packBBytesSpan = carver.TryCarve(packBBytes);
            if (packBBytesSpan.IsEmpty) packBBytesSpan = ArenaIntegration.TryRentBytes(packBBytes);
            if (packBBytesSpan.IsEmpty) packBBytesSpan = PerThreadPool.Current.RentPackB(packBBytes);

            Span<T> packA = MemoryMarshal.Cast<byte, T>(packABytesSpan).Slice(0, mc * kc);
            Span<T> packB = MemoryMarshal.Cast<byte, T>(packBBytesSpan).Slice(0, kc * nc);

            RunSerial<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, mc, nc, kc, mr, nr,
                packA, packB, in options, elemSize);
        }
        else
        {
            // ── Parallel path (no workspace) ────────────────────────────────────────
            // Sub-B5 (#370): dispatch to 2D MN-grid when shape has too few M-blocks
            // to fill all cores via M-axis parallel alone. ShouldUse2DGrid returns
            // true when (numMBlocks * 2 < procs && numNBlocks > 1) — i.e., the
            // existing M-axis split would leave half or more cores idle.
            // #475: default fan-out caps at PHYSICAL cores (FMA-bound GEMM regresses on SMT threads —
            // measured ffn peak at 16, regress at 24/32). An explicit NumThreads request is honoured.
            int procs = options.NumThreads > 0 ? options.NumThreads : CpuParallelSettings.GemmThreadCount(Environment.ProcessorCount);
            if (options.NumThreads < 0) procs = 1;
            int numMBlocks = (m + mc - 1) / mc;
            int numNBlocks = (n + nc - 1) / nc;
            // ShouldUse2DGrid fires when M-axis underutilizes. ALSO require m < 256
            // (absolute) — when m is meaningful (≥ 256), M-axis can usually keep
            // enough cores busy AND the per-thread redundant pack-B in 2D mode hurts
            // memory bandwidth. Concrete cases:
            //   ViT 192×768×768 → m=192 < 256: 2D wins (~2× speedup)
            //   GPT-2 512×N×K   → m=512 ≥ 256: M-axis wins (2D would lose 30%)
            //   LargeSquare 2048 → m=2048 ≥ 256: M-axis wins
            // Threshold 256 = empirical from baseline-after-B5 sweep on
            // x64-amd-avx2-cpu16. Future autotune (#374 F.3 deferred) should
            // measure both paths per-shape and cache the winner.
            // The m < 256 guard kept moderate-M shapes on the M-axis path (where its shared
            // packed-B avoids 2D's redundant per-thread B-pack). But on high core counts a
            // wide-N shape with moderate M (e.g. FFN 512×512×2048: numMBlocks≈2-4 vs 32 cores)
            // leaves ≥75% of cores idle on the M-axis path — there the extra MN parallelism of
            // the 2D grid wins despite the redundant B-pack. So ALSO take 2D when the M-axis
            // would use ≤ a quarter of the cores (numMBlocks*4 < procs), regardless of absolute M.
            bool use2D = MN2DDriver.ShouldUse2DGrid(numMBlocks, numNBlocks, procs)
                         && (m < 256 || numMBlocks * 4 <= procs);

            // #653 diagnostic: AIDOTNET_GEMM_TRACE=1 prints the chosen parallel path +
            // blocking so we can see what a real forward GEMM actually runs. stderr only,
            // so the probe's stdout GEMM line stays parseable.
            if (s_trace)
                Console.Error.WriteLine(
                    $"[PackBoth] m={m} n={n} k={k} mc={mc} nc={nc} kc={kc} mr={mr} nr={nr} " +
                    $"numMB={numMBlocks} numNB={numNBlocks} procs={procs} path={(use2D ? "2D" : "Maxis")}");

            // Pin a, b, c for the duration of the parallel region. Both paths use
            // unsafe pointer access from worker threads.
            // N-axis private-B path: wide-N moderate-M FP32, m%mr==0, enough N-blocks to fan
            // out, no pre-pack/workspace/transpose. Each thread owns an N-block with a private
            // L2-resident B — beats the M-axis shared-B path's L3 contention on these shapes.
            // N-axis uses its OWN L2-sized Nc (the autotune may have widened nc to span N for
            // A-pack-once, which leaves 1 N-block — no fan-out). Target the Kc×Nc B-block at
            // ~half L2 (256 KB) so each thread's private B stays L2-resident.
            int effProcs = Math.Min(procs, Environment.ProcessorCount);
            int ncN = nc;
            {
                int l2Target = 256 * 1024 / (Math.Max(1, kc) * sizeof(float));
                ncN = Math.Max(nr, (l2Target / nr) * nr);
                int nRounded = ((n + nr - 1) / nr) * nr;
                // Only shrink toward n when n>0; a zero-column GEMM would clamp ncN to 0 and make
                // the numNBlocksN divide below throw. (Callers guard n>0, but keep route selection safe.)
                if (nRounded > 0) ncN = Math.Min(ncN, nRounded);
                // DOP-aware fan-out: the L2-sized ncN above can leave FEWER N-blocks than threads
                // (e.g. n=1024, ncN=256 → 4 blocks → only 4 of 32 threads do work). When parallel,
                // shrink ncN so there are at least effProcs N-blocks — each thread's private B is
                // just smaller, still L2-resident. Measured at DOP=32: +36% on ffn-big (n=4096),
                // +44% on a 1024³ shape, neutral on ffn-up. Single-thread (effProcs==1) skips this,
                // keeping the wide B-panel the serial per-block A re-read amortizes best.
                if (effProcs > 1)
                {
                    int ncFanout = Math.Max(nr, (n / effProcs / nr) * nr);
                    ncN = Math.Min(ncN, ncFanout);
                }
            }
            int numNBlocksN = (n + ncN - 1) / ncN;
            bool useNAxis = !s_disableNAxis
                && MachineKernelGemm.IsFp32PanelAvailable
                && typeof(T) == typeof(float)
                && !transA && !transB
                && options.PackedA is null && options.PackedB is null
                && (m % mr) == 0 && m >= mr
                // Wide-N gate (n >= k): the N-axis re-reads the shared A (m×k) once per N-block,
                // so it pays off only when N is wide enough to amortize that — measured across 6
                // interleaved reps: ffn-big n=4096≥k=3456 wins +27% (DOP32, 6/6) and is neutral
                // at DOP4, while narrow-N attn-out n=1536<k=4608 loses −19/−26% (0/6). The
                // earlier k<=2048 gate was the wrong proxy.
                && (long)n >= (long)k
                && numNBlocksN >= 2 && numNBlocksN >= Math.Min(4, effProcs)
                && options.Epilogue.Activation == AiDotNet.Tensors.Engines.FusedActivationType.None
                && options.Epilogue.BiasN.IsEmpty && options.Epilogue.SkipMxN.IsEmpty;
            // #85 low-barrier jc-blocked GotoBLAS path. Same FP32 / no-transpose / no-prepack /
            // no-epilogue / m%mr==0 envelope as the N-axis path. s_gotoBlasAuto is DEAD (default off):
            // every shape in the k≤n≤2k gate satisfies BeatsPackBoth's (m≥128 && n≥512) clause and is
            // claimed by GotoGemmFp32 at BlasManaged.cs:542 before PackBoth is ever reached; when forced
            // onto PackBoth (AIDOTNET_DISABLE_GOTO=1) this path measured SLOWER than N-axis (ffn 0.40x).
            // The gate stays only so s_forceGotoBlas can still drive the --ab-gotoblas* measurement record.
            bool gotoEnvelope =
                MachineKernelGemm.IsFp32PanelAvailable
                && typeof(T) == typeof(float)
                && !transA && !transB
                && options.PackedA is null && options.PackedB is null
                && (m % mr) == 0 && m >= mr
                && options.Epilogue.Activation == AiDotNet.Tensors.Engines.FusedActivationType.None
                && options.Epilogue.BiasN.IsEmpty && options.Epilogue.SkipMxN.IsEmpty;
            bool gotoAutoGate = s_gotoBlasAuto
                && m >= 128 && k >= 512
                && (long)n >= (long)k && (long)n <= 2L * k;
            if (gotoEnvelope && (s_forceGotoBlas || gotoAutoGate))
            {
                fixed (T* aPtrG = a)
                fixed (T* bPtrG = b)
                fixed (T* cPtrG = c)
                {
                    RunGotoBlasParallelUnsafe<T>(
                        aPtrG, a.Length, lda,
                        bPtrG, b.Length, ldb,
                        cPtrG, c.Length, ldc,
                        m, n, k, mc, nc, kc, mr, nr, elemSize);
                }
                return;
            }

            if (useNAxis)
            {
                System.Threading.Interlocked.Increment(ref s_nAxisRunCount); // test observable (see s_nAxisRunCount)
                fixed (T* aPtrN = a)
                fixed (T* bPtrN = b)
                fixed (T* cPtrN = c)
                {
                    RunNAxisParallelUnsafe(
                        (float*)aPtrN, a.Length, lda,
                        (float*)bPtrN, b.Length, ldb,
                        (float*)cPtrN, c.Length, ldc,
                        m, n, k, mc, ncN, kc, mr, nr);
                }
                return;
            }

            fixed (T* aPtr = a)
            fixed (T* bPtr = b)
            fixed (T* cPtr = c)
            {
                if (use2D)
                {
                    // 2D-grid: each thread packs its OWN B for a unique (jcIdx, icIdx).
                    // No shared packed-B; redundant pack-B work across threads but the
                    // extra parallelism wins when M-axis alone has few blocks.
                    Run2DParallelUnsafe<T>(
                        aPtr, a.Length, lda, transA,
                        bPtr, b.Length, ldb, transB,
                        cPtr, c.Length, ldc,
                        m, n, k, mc, nc, kc, mr, nr,
                        in options, elemSize);
                }
                else
                {
                    // #653 single-region: wide-nc (numNB==1), no pre-pack/workspace/transpose,
                    // packed-B-all within budget → pack all K-panels once + one ic region (no
                    // per-K-panel barrier). Otherwise the existing per-panel M-axis path.
                    int numKPanelsSr = (k + kc - 1) / kc;
                    int packedNcSr = ((n + nr - 1) / nr) * nr;
                    long packBAllBytes = (long)numKPanelsSr * kc * packedNcSr * elemSize;
                    bool useSingleRegion = s_singleRegion
                        && nc >= n                       // numNB == 1
                        && !transA && !transB
                        && options.PackedA is null && options.PackedB is null
                        && packBAllBytes <= SingleRegionPackBBudgetBytes;

                    if (useSingleRegion)
                    {
                        RunParallelSingleRegionUnsafe<T>(
                            aPtr, a.Length, lda,
                            bPtr, b.Length, ldb,
                            cPtr, c.Length, ldc,
                            m, n, k, mc, kc, mr, nr,
                            in options, elemSize);
                    }
                    else
                    {
                        // M-axis (existing): shared packed-B, parallel over ic.
                        byte[]? packBArray = null;
                        try
                        {
                            Span<byte> packBBytesSpan = ArenaIntegration.TryRentBytes(packBBytes);
                            if (packBBytesSpan.IsEmpty)
                            {
                                packBArray = ArrayPool<byte>.Shared.Rent(packBBytes);
                                packBBytesSpan = packBArray.AsSpan(0, packBBytes);
                            }
                            RunParallelUnsafe<T>(
                                aPtr, a.Length, lda, transA,
                                bPtr, b.Length, ldb, transB,
                                cPtr, c.Length, ldc,
                                m, n, k, mc, nc, kc, mr, nr,
                                packBBytesSpan, in options, elemSize);
                        }
                        finally
                        {
                            if (packBArray != null) ArrayPool<byte>.Shared.Return(packBArray);
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Serial execution of the 3-level Goto loop nest. Used when the caller
    /// provides a workspace buffer (Layer 5) that cannot be split across threads.
    /// Pack-A and pack-B buffers are already allocated by the caller.
    /// </summary>
    private static void RunSerial<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        int mc, int nc, int kc,
        int mr, int nr,
        Span<T> packA, Span<T> packB,
        in BlasOptions<T> options, int elemSize) where T : unmanaged
    {
        for (int jc = 0; jc < n; jc += nc)
        {
            int effectiveNc = Math.Min(nc, n - jc);

            for (int pc = 0; pc < k; pc += kc)
            {
                int effectiveKc = Math.Min(kc, k - pc);

                // Layer 3: short-circuit pack-B when pre-pack handle is current.
                // Use effective tile bytes, not nominal, because PrePackB clamps Kc/Nc to
                // the actual matrix dimensions (small matrices produce smaller handles).
                int effectivePackBBytes = effectiveKc * effectiveNc * elemSize;
                // Round up Nc to the next multiple of Nr so the last partial-N stripe
                // is zero-padded into a full Nr-wide row (Task G2). The packed buffer
                // (packBBytes = kc * nc * elemSize) always has room for this extra padding.
                int packedNc = ((effectiveNc + nr - 1) / nr) * nr;
                int packedBElemCount = effectiveKc * packedNc;
                int packedBByteCount = packedBElemCount * elemSize;
                bool packBFromPrePack = false;
                Span<T> activePackB = packB;
                // CodeRabbit #366: DispatchMicrokernelWithTail reads
                // `effectiveKc * nr` per stripe, so the cached handle must
                // hold the PADDED byte count (packedBByteCount), not the
                // dense effectivePackBBytes. Without this, partial-N tiles
                // would slice past activePackB or pick up stale padding from
                // a previous (jc, pc) iteration.
                // Sub-E (#373): pre-pack consumption supports both legacy single-panel
                // and the new multi-panel layout. Multi-panel requires the handle's
                // (TileMc, TileKc) to match this call's (mc, kc); else fall back to
                // live pack to avoid reading a misaligned tile.
                if (options.PackedB != null && WeightPackCache.IsCacheCurrent(options.PackedB))
                {
                    var pkdB = options.PackedB;
                    // PackNr gate: the packed panel is interleaved in nr-column
                    // stripes; consuming with a different active nr (e.g. the
                    // dispatcher's scalar (4,4) too-small-shape fallback on a
                    // machine whose prepack tile is 8/16-wide) reads the stripes
                    // at the wrong stride and yields garbage. Mismatch -> live
                    // pack below (correct, just unaccelerated).
                    if (pkdB.PackNr == nr && pkdB.MultiPanelStride > 0)
                    {
                        // Multi-panel: jc-blocks live in NumIcBlocks slot for PackedB.
                        if (pkdB.TileMc == nc && pkdB.TileKc == kc)
                        {
                            int jcIdx = jc / nc;
                            int pcIdx = pc / kc;
                            var tile = pkdB.GetTileSlice(jcIdx, pcIdx);
                            if (!tile.IsEmpty)
                            {
                                activePackB = MemoryMarshal.Cast<byte, T>(tile);
                                packBFromPrePack = true;
                            }
                        }
                    }
                    else if (pkdB.PackNr == nr && pkdB.PackedBuffer.Length >= packedBByteCount)
                    {
                        // Legacy single-panel.
                        activePackB = MemoryMarshal.Cast<byte, T>(pkdB.PackedBuffer.AsSpan(0, packedBByteCount));
                        packBFromPrePack = true;
                    }
                }

                if (packBFromPrePack)
                {
                    BlasManagedStatsTracker.IncrementPackCacheHit();
                }
                else
                {
                    if (options.PackedB != null) BlasManagedStatsTracker.IncrementPackCacheMiss();
                    // Pack B[pc..pc+effectiveKc, jc..jc+effectiveNc] into packB.
                    int bSliceOffset = transB ? jc * ldb + pc : pc * ldb + jc;
                    long _pbStart = PackBothProfiler.Enabled ? Stopwatch.GetTimestamp() : 0L;
                    Avx2Pack.PackB<T>(
                        b: b.Slice(bSliceOffset), ldb, transB,
                        packed: activePackB.Slice(0, packedBElemCount),
                        nc: effectiveNc, kc: effectiveKc, nr);
                    if (PackBothProfiler.Enabled) PackBothProfiler.AddPackB(Stopwatch.GetTimestamp() - _pbStart);
                }

                for (int ic = 0; ic < m; ic += mc)
                {
                    int effectiveMc = Math.Min(mc, m - ic);

                    int effectivePackABytes = effectiveMc * effectiveKc * elemSize;
                    bool packAFromPrePack = false;
                    Span<T> activePackA = packA;
                    if (options.PackedA != null && WeightPackCache.IsCacheCurrent(options.PackedA))
                    {
                        var pkdA = options.PackedA;
                        // PackMr gate: the packed panel is interleaved in mr-row
                        // stripes; a consumer running a different active mr (the
                        // dispatcher's scalar (4,4) too-small-shape fallback on a
                        // machine whose prepack tile is 8 rows wide) reads the
                        // stripes at the wrong stride and yields garbage —
                        // exactly the CI-only ScalarKernelTests failures on
                        // AVX-512 runners. Mismatch -> live pack (correct).
                        if (pkdA.PackMr == mr && pkdA.MultiPanelStride > 0)
                        {
                            if (pkdA.TileMc == mc && pkdA.TileKc == kc)
                            {
                                int icIdx = ic / mc;
                                int pcIdx = pc / kc;
                                var tile = pkdA.GetTileSlice(icIdx, pcIdx);
                                if (!tile.IsEmpty)
                                {
                                    activePackA = MemoryMarshal.Cast<byte, T>(tile);
                                    packAFromPrePack = true;
                                }
                            }
                        }
                        else if (pkdA.PackMr == mr && pkdA.PackedBuffer.Length >= effectivePackABytes)
                        {
                            activePackA = MemoryMarshal.Cast<byte, T>(pkdA.PackedBuffer.AsSpan(0, effectivePackABytes));
                            packAFromPrePack = true;
                        }
                    }

                    if (packAFromPrePack)
                    {
                        BlasManagedStatsTracker.IncrementPackCacheHit();
                    }
                    else
                    {
                        if (options.PackedA != null) BlasManagedStatsTracker.IncrementPackCacheMiss();
                        // Pack A[ic..ic+effectiveMc, pc..pc+effectiveKc] into packA.
                        // transA=false: A is [M, K] row-major, panel starts at a[ic * lda + pc].
                        // transA=true:  A is [K, M] row-major, panel starts at a[pc * lda + ic].
                        int aSliceOffset = transA ? pc * lda + ic : ic * lda + pc;
                        long _paStart = PackBothProfiler.Enabled ? Stopwatch.GetTimestamp() : 0L;
                        Avx2Pack.PackA<T>(
                            a: a.Slice(aSliceOffset), lda, transA,
                            packed: activePackA.Slice(0, effectiveMc * effectiveKc),
                            mc: effectiveMc, kc: effectiveKc, mr);
                        if (PackBothProfiler.Enabled) PackBothProfiler.AddPackA(Stopwatch.GetTimestamp() - _paStart);
                    }

                    // Iterate microkernel tiles within this Mc × Nc panel.
                    long _krStart = PackBothProfiler.Enabled ? Stopwatch.GetTimestamp() : 0L;
                    for (int jr = 0; jr < effectiveNc; jr += nr)
                    {
                        // Partial-N tile on the last jr iteration when effectiveNc % nr != 0.
                        int effectiveNr = Math.Min(nr, effectiveNc - jr);
                        for (int ir = 0; ir < effectiveMc; ir += mr)
                        {
                            // M-tail: skip partial rows — caller guarantees m % mr == 0
                            // (BlasManaged.cs falls back to scalar otherwise). Guard here
                            // in case effectiveMc is not a multiple of mr for any reason.
                            if (ir + mr > effectiveMc) break;

                            // Offset into packA for the current Mr-stripe (ir/mr th stripe).
                            // Stripe layout: [numStripes, Kc, Mr] → stripe * Kc * Mr.
                            int packedAStripeOff = (ir / mr) * effectiveKc * mr;
                            // Offset into packB for the current Nr-stripe (jr/nr th stripe).
                            // Stripe layout: [numStripes, Kc, Nr] → stripe * Kc * Nr.
                            int packedBStripeOff = (jr / nr) * effectiveKc * nr;

                            // C tile starts at row (ic + ir), col (jc + jr) in the full C matrix.
                            int cTileOff = (ic + ir) * ldc + (jc + jr);
                            DispatchMicrokernelWithTail<T>(
                                activePackA.Slice(packedAStripeOff, effectiveKc * mr),
                                activePackB.Slice(packedBStripeOff, effectiveKc * nr),
                                c.Slice(cTileOff),
                                ldc, effectiveKc,
                                mr, nr, effectiveNr);
                        }
                    }
                    if (PackBothProfiler.Enabled) PackBothProfiler.AddKernel(Stopwatch.GetTimestamp() - _krStart);
                }
            }
        }
    }

    /// <summary>
    /// Parallel execution of the 3-level Goto loop nest (M-axis split on the ic loop).
    /// a, b, c are passed as raw pointers (pinned by the caller via <c>fixed</c>).
    ///
    /// <para>
    /// Pack-B is packed once on the calling thread, then shared across all ic threads as
    /// a read-only <c>byte[]</c> captured by the parallel lambda. Pack-A is rented per-thread
    /// from <see cref="PerThreadPool.Current"/> (<c>[ThreadStatic]</c>), providing each
    /// worker thread its own dedicated buffer with zero cross-thread contention.
    /// C writes are disjoint — each icIdx block owns rows [ic, ic+effectiveMc) — so no
    /// synchronization is needed on the C matrix.
    /// </para>
    ///
    /// <para>
    /// Note on allocator bypass for pack-A under parallelism:
    /// - Workspace (Layer 5): a single buffer carved by <see cref="WorkspaceCarver"/> cannot
    ///   be split safely across threads. Bypassed; caller routes workspace callers to serial.
    /// - Arena (Layer 4): <c>[ThreadStatic]</c> — only the calling thread may have an arena
    ///   active; worker threads do not inherit it. Bypassed for pack-A.
    /// - PerThreadPool (Layer 1): always used. Each worker thread's <c>[ThreadStatic]</c>
    ///   pool instance is lazily created on first access, ensuring isolation.
    /// </para>
    /// </summary>
    private static unsafe void RunParallelUnsafe<T>(
        T* aPtr, int aLen, int lda, bool transA,
        T* bPtr, int bLen, int ldb, bool transB,
        T* cPtr, int cLen, int ldc,
        int m, int n, int k,
        int mc, int nc, int kc,
        int mr, int nr,
        Span<byte> packBBytesSpan,
        in BlasOptions<T> options, int elemSize) where T : unmanaged
    {
        // Snapshot options fields: BlasOptions<T> is a ref struct and cannot be captured.
        WeightPackHandle? packedA = options.PackedA;
        WeightPackHandle? packedB = options.PackedB;

        // Convert pinned pointers to nint (platform int) so they can be captured by the lambda.
        // Worker threads reconstruct spans from these captured pointers.
        nint aPtrInt = (nint)aPtr;
        nint bPtrInt = (nint)bPtr;
        nint cPtrInt = (nint)cPtr;

        // Pack-B backing array: capturable by the lambda as a managed byte[].
        // The bytes are shared across all ic-parallel iterations (read-only after packing).
        // CodeRabbit #366: was `new byte[packBBytesSpan.Length]`, defeating the
        // allocator hierarchy (Layer 5/4/1) and adding a full managed alloc on
        // every parallel GEMM call. Rent from the shared array pool instead and
        // return in finally; capacity is bounded by packBBytesSpan.Length which
        // itself caps at kc*nc*elemSize.
        //
        // Sub-P (#406): over-rent by 31 bytes so we can slice the pack buffer at
        // a 32-byte-aligned offset — required for Avx2Pack's non-temporal-store
        // fast path. Without alignment the NT path silently falls back to
        // regular cached stores (still correct, just no perf win).
        const int PackBAlignment = 32;
        byte[] packBArr = System.Buffers.ArrayPool<byte>.Shared.Rent(packBBytesSpan.Length + PackBAlignment - 1);
        int packBAlignedOffset;
        unsafe
        {
            fixed (byte* pbPtr = packBArr)
            {
                nuint addr = (nuint)pbPtr;
                nuint mask = PackBAlignment - 1;
                packBAlignedOffset = (int)((PackBAlignment - (uint)(addr & mask)) & mask);
            }
        }
        try
        {

        for (int jc = 0; jc < n; jc += nc)
        {
            int effectiveNc = Math.Min(nc, n - jc);

            for (int pc = 0; pc < k; pc += kc)
            {
                int effectiveKc = Math.Min(kc, k - pc);

                // ── Pack B (shared, single calling-thread, before the parallel ic region) ──
                int effectivePackBBytes = effectiveKc * effectiveNc * elemSize;
                // Round up Nc to the next multiple of Nr so the last partial-N stripe
                // is zero-padded into a full Nr-wide row (Task G2). The backing byte[]
                // (packBBytesSpan.Length = kc * nc * elemSize) always has room for this padding.
                int packedNc = ((effectiveNc + nr - 1) / nr) * nr;
                int packedBElemCount = effectiveKc * packedNc;
                int packedBByteCount = packedBElemCount * elemSize;
                bool packBFromPrePack = false;
                // CodeRabbit #366: gate on the PADDED byte count and copy the
                // full padded region — see the serial path for rationale.
                // Sub-E (#373): multi-panel consume — when the handle is
                // multi-panel and (TileMc, TileKc) match (nc, kc), copy from
                // the specific (jcIdx, pcIdx) tile rather than offset 0. The
                // pre-Sub-E single-panel path stays as the fallback else-branch.
                // PackNr gate (see WeightPackHandle.PackNr): the panel stripes are
                // nr-interleaved; an active-nr mismatch (scalar fallback on small
                // shapes vs a wide prepack tile) reads garbage. Mismatch -> live pack.
                if (packedB != null && WeightPackCache.IsCacheCurrent(packedB) && packedB.PackNr == nr)
                {
                    if (packedB.MultiPanelStride > 0
                        && packedB.TileMc == nc && packedB.TileKc == kc)
                    {
                        int jcIdx = jc / nc;
                        int pcIdx = pc / kc;
                        var tile = packedB.GetTileSlice(jcIdx, pcIdx);
                        if (!tile.IsEmpty && tile.Length >= packedBByteCount)
                        {
                            tile.Slice(0, packedBByteCount)
                                .CopyTo(packBArr.AsSpan(packBAlignedOffset, packedBByteCount));
                            packBFromPrePack = true;
                        }
                    }
                    else if (packedB.PackedBuffer.Length >= packedBByteCount)
                    {
                        // Legacy single-panel: pre-packed B is already in byte[] form — copy
                        // into packBArr so the parallel lambda reads from a stable, capturable byte[].
                        packedB.PackedBuffer.AsSpan(0, packedBByteCount)
                               .CopyTo(packBArr.AsSpan(packBAlignedOffset, packedBByteCount));
                        packBFromPrePack = true;
                    }
                }

                if (packBFromPrePack)
                {
                    BlasManagedStatsTracker.IncrementPackCacheHit();
                }
                else
                {
                    if (packedB != null) BlasManagedStatsTracker.IncrementPackCacheMiss();
                    // Pack B[pc..pc+effectiveKc, jc..jc+effectiveNc] into packBArr.
                    // transB=false: panel starts at b[pc * ldb + jc].
                    // transB=true:  panel starts at b[jc * ldb + pc].
                    int bSliceOffset = transB ? jc * ldb + pc : pc * ldb + jc;
                    int totalNumStripes = (effectiveNc + nr - 1) / nr;

                    // Sub-N (#404): parallelize pack-B across stripes when the
                    // buffer is large enough to amortize Parallel.For overhead.
                    // Each stripe writes a disjoint region of packBArr
                    // ([stripeIdx, Kc, Nr]) — no synchronization needed.
                    // Threshold: ≥4 stripes AND ≥procs threads available AND
                    // ≥256 KB of pack work — below that, serial wins.
                    int packBStripeSize = effectiveKc * nr * elemSize;
                    int procsLocal = options.NumThreads > 0 ? options.NumThreads : Environment.ProcessorCount;
                    if (options.NumThreads < 0) procsLocal = 1;
                    bool packBParallelWorthwhile =
                        totalNumStripes >= 4
                        && procsLocal >= 2
                        && (long)packBStripeSize * totalNumStripes >= 256 * 1024;

                    int bSliceOffset_cap = bSliceOffset;
                    int effectiveKc_packB_cap = effectiveKc;
                    int effectiveNc_packB_cap = effectiveNc;
                    int bLen_cap = bLen;

                    if (packBParallelWorthwhile)
                    {
                        // Partition stripes across procs threads. Each thread packs
                        // [chunkStart, chunkEnd) stripes into its own slice of packBArr.
                        int chunkSize = Math.Max(1, (totalNumStripes + procsLocal - 1) / procsLocal);
                        int numChunks = (totalNumStripes + chunkSize - 1) / chunkSize;
                        long packTotalWork = (long)packBStripeSize * totalNumStripes;

                        CpuParallelSettings.ParallelForOrSerial(0, numChunks, packTotalWork, chunkIdx =>
                        {
                            int stripeStart = chunkIdx * chunkSize;
                            int stripeEnd = Math.Min(stripeStart + chunkSize, totalNumStripes);
                            int numStripesInChunk = stripeEnd - stripeStart;
                            if (numStripesInChunk <= 0) return;

                            unsafe
                            {
                                int srcAdjustedOffset = bSliceOffset_cap;  // base for the (jc, pc) panel
                                ReadOnlySpan<T> bSliceLocal = new ReadOnlySpan<T>((T*)bPtrInt + srcAdjustedOffset, bLen_cap - srcAdjustedOffset);
                                int packedStripeOffElems = stripeStart * effectiveKc_packB_cap * nr;
                                Span<T> packBSliceLocal = MemoryMarshal.Cast<byte, T>(
                                    packBArr.AsSpan(packBAlignedOffset + packedStripeOffElems * elemSize,
                                                    numStripesInChunk * effectiveKc_packB_cap * nr * elemSize));
                                Avx2Pack.PackBStripeRange<T>(
                                    bSliceLocal, ldb, transB,
                                    packBSliceLocal,
                                    stripeStart, numStripesInChunk,
                                    effectiveNc_packB_cap, effectiveKc_packB_cap, nr);
                            }
                        }, deterministicSafe: true); // disjoint stripe packing — order-independent
                    }
                    else
                    {
                        // Serial pack — buffer too small or single-thread context.
                        // Pass a padded slice (packedBByteCount) so ScalarPack.PackB can
                        // zero-pad the partial tail stripe when effectiveNc % nr != 0.
                        ReadOnlySpan<T> bSlice = new ReadOnlySpan<T>((T*)bPtrInt + bSliceOffset, bLen - bSliceOffset);
                        Span<T> packBTemp = MemoryMarshal.Cast<byte, T>(packBArr.AsSpan(packBAlignedOffset, packedBByteCount));
                        long _pbStart = PackBothProfiler.Enabled ? Stopwatch.GetTimestamp() : 0L;
                        Avx2Pack.PackB<T>(
                            b: bSlice, ldb, transB,
                            packed: packBTemp,
                            nc: effectiveNc, kc: effectiveKc, nr);
                        if (PackBothProfiler.Enabled) PackBothProfiler.AddPackB(Stopwatch.GetTimestamp() - _pbStart);
                    }
                }

                // ── M-axis parallel split on the ic loop ──────────────────────────────────
                // Each icIdx body owns disjoint C rows [ic, ic+effectiveMc) → no sync on C.
                int numIcBlocks = (m + mc - 1) / mc;
                // Grain-size estimate: mc × effectiveNc × effectiveKc multiplied by
                // numIcBlocks gives total MACs across all parallel iterations.
                long totalWork = (long)mc * effectiveNc * effectiveKc * numIcBlocks;

                // Capture loop-iteration locals (int/nint are value-type, safe to capture).
                int jc_cap = jc, pc_cap = pc;
                int effectiveNc_cap = effectiveNc, effectiveKc_cap = effectiveKc;
                int packedBByteCount_cap = packedBByteCount;
                byte[] packBArr_cap = packBArr;
                int packBAlignedOffset_cap = packBAlignedOffset;

                CpuParallelSettings.ParallelForOrSerial(0, numIcBlocks, totalWork, icIdx =>
                {
                    int ic = icIdx * mc;
                    int effectiveMc = Math.Min(mc, m - ic);

                    // ── Pack A (per-thread, from PerThreadPool.Current) ────────────────
                    int effectivePackABytes = effectiveMc * effectiveKc_cap * elemSize;
                    Span<T> activePackA;
                    bool prePackHitParallel = false;
                    Span<byte> packAByteSlice = default;
                    // PackMr gate (see WeightPackHandle.PackMr): the panel stripes
                    // are mr-interleaved at PACK time; consuming with a different
                    // active mr reads the wrong stride and yields garbage. Both
                    // branches below require the recorded tile to match.
                    if (packedA != null && WeightPackCache.IsCacheCurrent(packedA) && packedA.PackMr == mr)
                    {
                        if (packedA.MultiPanelStride > 0)
                        {
                            // Sub-E: multi-panel. Tile sizes must match this call's mc/kc.
                            if (packedA.TileMc == mc && packedA.TileKc == effectiveKc_cap)
                            {
                                int icIdxLocal = ic / mc;
                                int pcIdxLocal = pc_cap / effectiveKc_cap;
                                var tile = packedA.GetTileSlice(icIdxLocal, pcIdxLocal);
                                if (!tile.IsEmpty)
                                {
                                    packAByteSlice = tile;
                                    prePackHitParallel = true;
                                }
                            }
                        }
                        else if (packedA.PackedBuffer.Length >= effectivePackABytes)
                        {
                            packAByteSlice = packedA.PackedBuffer.AsSpan(0, effectivePackABytes);
                            prePackHitParallel = true;
                        }
                    }
                    if (prePackHitParallel)
                    {
                        BlasManagedStatsTracker.IncrementPackCacheHit();
                        activePackA = MemoryMarshal.Cast<byte, T>(packAByteSlice);
                    }
                    else
                    {
                        if (packedA != null) BlasManagedStatsTracker.IncrementPackCacheMiss();
                        // Layer 1: per-thread pool — [ThreadStatic] ensures each worker
                        // thread rents from its own PerThreadPool instance.
                        // Workspace (Layer 5) and Arena (Layer 4) are intentionally bypassed:
                        //   Workspace: single caller buffer, not thread-partitionable.
                        //   Arena: [ThreadStatic] — worker threads may have no arena active.
                        Span<byte> packABytesSpan_inner = PerThreadPool.Current.RentPackA(effectivePackABytes);
                        activePackA = MemoryMarshal.Cast<byte, T>(packABytesSpan_inner)
                                                   .Slice(0, effectiveMc * effectiveKc_cap);

                        int aSliceOffset = transA ? pc_cap * lda + ic : ic * lda + pc_cap;
                        ReadOnlySpan<T> aSlice = new ReadOnlySpan<T>((T*)aPtrInt + aSliceOffset, aLen - aSliceOffset);
                        long _paStart = PackBothProfiler.Enabled ? Stopwatch.GetTimestamp() : 0L;
                        Avx2Pack.PackA<T>(
                            a: aSlice, lda, transA,
                            packed: activePackA,
                            mc: effectiveMc, kc: effectiveKc_cap, mr);
                        if (PackBothProfiler.Enabled) PackBothProfiler.AddPackA(Stopwatch.GetTimestamp() - _paStart);
                    }

                    // Shared pack-B: reconstruct span from captured byte[] (read-only).
                    // Use packedBByteCount_cap (Nr-padded size) so the tail stripe
                    // (packed by ScalarPack.PackB with zero-padding) is accessible.
                    Span<T> activePackB = MemoryMarshal.Cast<byte, T>(
                        packBArr_cap.AsSpan(packBAlignedOffset_cap, packedBByteCount_cap));

                    // ── Inner microkernel loop ─────────────────────────────────────────
                    long _krStart = PackBothProfiler.Enabled ? Stopwatch.GetTimestamp() : 0L;
                    int njrFull = effectiveNc_cap / nr;          // full Nr-wide tiles
                    int nrTail = effectiveNc_cap - njrFull * nr; // partial-N remainder
                    // FP32 6×16 MACHINE-CODE PANEL fast path: one hand-emitted asm call per 6-row
                    // block processes ALL its full-Nr tiles (A reused, B streamed, N-loop in asm),
                    // eliminating the per-tile dispatch + re-prologue that caps in-context
                    // throughput — the OpenBLAS macro-kernel granularity. Bit-identical to the
                    // per-tile loop (MachineKernelPanelTests). Partial-N tail keeps the generic
                    // tail dispatch. Gated to the live (non-pre-packed) Auto path.
                    bool mcPanel = !s_disablePanel && typeof(T) == typeof(float)
                        && mr == 6 && nr == 16 && njrFull > 0
                        && MachineKernelGemm.IsFp32PanelAvailable;
                    if (mcPanel)
                    {
                        int njrBytesA = effectiveKc_cap * mr;
                        for (int ir = 0; ir < effectiveMc; ir += mr)
                        {
                            if (ir + mr > effectiveMc) break;
                            int packedAStripeOff = (ir / mr) * effectiveKc_cap * mr;
                            var aPanel = MemoryMarshal.Cast<T, float>(activePackA.Slice(packedAStripeOff, njrBytesA));
                            // Chunk the N-tiles per asm call: one giant panel call (e.g. 384 tiles
                            // for n=6144) regresses vs ~256 (the single call's B/C working set
                            // spills cache); chunking keeps each call's footprint bounded while
                            // still amortizing dispatch over MaxPanelTiles tiles.
                            int maxPanelTiles = Math.Max(1, MaxPanelTiles); // guard sweep-set 0/negative
                            for (int jb = 0; jb < njrFull; jb += maxPanelTiles)
                            {
                                int chunk = Math.Min(maxPanelTiles, njrFull - jb);
                                int cPanelOff = (ic + ir) * ldc + (jc_cap + jb * nr);
                                MachineKernelGemm.RunPanelFp32(
                                    aPanel,
                                    MemoryMarshal.Cast<T, float>(activePackB.Slice(jb * effectiveKc_cap * nr, chunk * effectiveKc_cap * nr)),
                                    MemoryMarshal.Cast<T, float>(new Span<T>((T*)cPtrInt + cPanelOff, cLen - cPanelOff)),
                                    ldc, effectiveKc_cap, chunk);
                            }
                            if (nrTail > 0)
                            {
                                int jrTail = njrFull * nr;
                                int cTileOff = (ic + ir) * ldc + (jc_cap + jrTail);
                                DispatchMicrokernelWithTail<T>(
                                    activePackA.Slice(packedAStripeOff, effectiveKc_cap * mr),
                                    activePackB.Slice(njrFull * effectiveKc_cap * nr, effectiveKc_cap * nr),
                                    new Span<T>((T*)cPtrInt + cTileOff, cLen - cTileOff),
                                    ldc, effectiveKc_cap, mr, nr, nrTail);
                            }
                        }
                    }
                    else
                    for (int jr = 0; jr < effectiveNc_cap; jr += nr)
                    {
                        // Partial-N tile on the last jr iteration when effectiveNc % nr != 0.
                        int effectiveNr = Math.Min(nr, effectiveNc_cap - jr);
                        for (int ir = 0; ir < effectiveMc; ir += mr)
                        {
                            // M-tail: skip partial rows — caller guarantees m % mr == 0
                            // (BlasManaged.cs falls back to scalar otherwise). Guard here
                            // in case effectiveMc is not a multiple of mr for any reason.
                            if (ir + mr > effectiveMc) break;

                            // Stripe offsets into packed panels.
                            // Stripe layout: [numStripes, Kc, Mr/Nr] → stripe * Kc * Mr/Nr.
                            int packedAStripeOff = (ir / mr) * effectiveKc_cap * mr;
                            int packedBStripeOff = (jr / nr) * effectiveKc_cap * nr;

                            // C tile: rows [ic+ir, ic+ir+mr), cols [jc_cap+jr, jc_cap+jr+nr).
                            // Disjoint from all other icIdx values → no write synchronization.
                            int cTileOff = (ic + ir) * ldc + (jc_cap + jr);
                            Span<T> cTile = new Span<T>((T*)cPtrInt + cTileOff, cLen - cTileOff);
                            DispatchMicrokernelWithTail<T>(
                                activePackA.Slice(packedAStripeOff, effectiveKc_cap * mr),
                                activePackB.Slice(packedBStripeOff, effectiveKc_cap * nr),
                                cTile,
                                ldc, effectiveKc_cap,
                                mr, nr, effectiveNr);
                        }
                    }
                    if (PackBothProfiler.Enabled) PackBothProfiler.AddKernel(Stopwatch.GetTimestamp() - _krStart);
                }, deterministicSafe: true); // M-axis (ic) split: disjoint C rows, fixed-order K reduction per tile
            }
        }

        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(packBArr);
        }
    }

    // Route wide-N moderate-M float GEMMs to the N-axis private-B path. Test-only toggle.
    internal static bool s_disableNAxis;

    // Test-only observable: incremented each time the N-axis private-B path actually runs, so the
    // parity test can assert the N-axis path was exercised (not silently falling back to M-axis,
    // which would make the bit-identity comparison vacuous M-vs-M).
    internal static long s_nAxisRunCount;

    /// <summary>
    /// N-axis-parallel path (FP32 panel kernel): packs ALL of A once into a shared read-only
    /// buffer, then parallelises over N-blocks — each thread packs its OWN B-block into a
    /// PRIVATE per-thread buffer (its L2) and computes all M for that N-block with the panel
    /// kernel. This is the OpenBLAS macro-loop: no shared-L3 B contention (each thread's B is
    /// private) AND no redundant B-packing (each N-block's B packed exactly once) — the two
    /// failure modes of the M-axis shared-B path and the 2D grid respectively. Gated to
    /// m%mr==0 float (the M-tail-free case, e.g. FFN m=384); other shapes use the M-axis path.
    /// Bit-identical: each C[ir,jc] tile reduces over k in ascending K-panel order; jc blocks
    /// own disjoint C columns so no cross-thread write sync.
    /// </summary>
    private static unsafe void RunNAxisParallelUnsafe(
        float* aPtr, int aLen, int lda,
        float* bPtr, int bLen, int ldb,
        float* cPtr, int cLen, int ldc,
        int m, int n, int k, int mc, int nc, int kc, int mr, int nr)
    {
        int numKPanels = (k + kc - 1) / kc;
        int numNBlocks = (n + nc - 1) / nc;
        int numMr = m / mr; // m % mr == 0 guaranteed by caller

        // ── Pack ALL of A once (shared, read-only). Layout: [K-panel][Mr-stripe][Kc × Mr]. ──
        // Panel pc occupies numMr * effKc * mr floats; pc panels are laid end to end with a
        // fixed per-panel stride of numMr * kc * mr so a worker can index (pc, ir) directly.
        int aPanelStride = numMr * kc * mr;
        long packAElems = (long)numKPanels * aPanelStride;
        var packAArr = System.Buffers.ArrayPool<float>.Shared.Rent((int)packAElems);
        try
        {
            for (int pIdx = 0; pIdx < numKPanels; pIdx++)
            {
                int pc = pIdx * kc;
                int effKc = Math.Min(kc, k - pc);
                // PackA writes [Mr-stripe][effKc × mr] (mc=m → all stripes).
                Avx2Pack.PackA<float>(
                    new ReadOnlySpan<float>(aPtr + pc, aLen - pc), lda, false,
                    packAArr.AsSpan(pIdx * aPanelStride, numMr * effKc * mr),
                    mc: m, kc: effKc, mr);
            }

            nint aPackAddr = 0, bAddr = (nint)bPtr, cAddr = (nint)cPtr;
            fixed (float* paBase = packAArr) { aPackAddr = (nint)paBase;

            int kcL = kc, ncL = nc, nrL = nr, mrL = mr, ldbL = ldb, ldcL = ldc;
            int numKPanelsL = numKPanels, numMrL = numMr, aPanelStrideL = aPanelStride;
            int kL = k, nL = n, bLenL = bLen, cLenL = cLen;
            long totalWork = (long)m * n * k * 2;

            CpuParallelSettings.ParallelForOrSerial(0, numNBlocks, totalWork, jcIdx =>
            {
                int jc = jcIdx * ncL;
                int effNc = Math.Min(ncL, nL - jc);
                int njrFull = effNc / nrL;
                int nrTail = effNc - njrFull * nrL;
                int packedNc = ((effNc + nrL - 1) / nrL) * nrL;

                // Private per-thread packed-B for this N-block: [K-panel][Kc × packedNc].
                int bPanelStride = kcL * packedNc;
                byte[] packBArr = System.Buffers.ArrayPool<byte>.Shared.Rent(numKPanelsL * bPanelStride * sizeof(float));
                try
                {
                    float* pa = (float*)aPackAddr;
                    float* bb = (float*)bAddr;
                    float* cc = (float*)cAddr;
                    var packBSpan = MemoryMarshal.Cast<byte, float>(packBArr.AsSpan());
                    int numStripes = (effNc + nrL - 1) / nrL;

                    for (int pIdx = 0; pIdx < numKPanelsL; pIdx++)
                    {
                        int pc = pIdx * kcL;
                        int effKc = Math.Min(kcL, kL - pc);
                        Avx2Pack.PackBStripeRange<float>(
                            new ReadOnlySpan<float>(bb + pc * ldbL + jc, bLenL - (pc * ldbL + jc)), ldbL, false,
                            packBSpan.Slice(pIdx * bPanelStride, effKc * packedNc),
                            0, numStripes, effNc, effKc, nrL);
                    }

                    // Compute: for each K-panel (ascending → correct C accumulation), each
                    // Mr-block does one chunked panel call over the N-block's full tiles + tail.
                    // #475: when enabled, the whole Mr-sweep runs in the machine-code MACRO kernel
                    // (one asm call per K-panel/N-chunk), taking RyuJIT off the hot path.
                    bool useMacro = s_macroKernel && njrFull > 0 && MachineKernelGemm.IsFp32MacroAvailable;
                    fixed (float* pbPacked = packBSpan)
                    {
                        for (int pIdx = 0; pIdx < numKPanelsL; pIdx++)
                        {
                            int pc = pIdx * kcL;
                            int effKc = Math.Min(kcL, kL - pc);
                            var bPanel = packBSpan.Slice(pIdx * bPanelStride, effKc * packedNc);
                            int maxPanelTiles = Math.Max(1, MaxPanelTiles); // guard sweep-set 0/negative

                            if (useMacro)
                            {
                                float* aBase0 = pa + pIdx * aPanelStrideL;
                                float* bBase0 = pbPacked + pIdx * bPanelStride;
                                int aStrideBytes = effKc * mrL * sizeof(float);
                                for (int jb = 0; jb < njrFull; jb += maxPanelTiles)
                                {
                                    int chunk = Math.Min(maxPanelTiles, njrFull - jb);
                                    MachineKernelGemm.RunMacroPanelFp32(
                                        aBase0, bBase0 + jb * effKc * nrL, cc + jc + jb * nrL,
                                        ldcL, effKc, chunk, numMrL, aStrideBytes);
                                }
                            }
                            else if (njrFull > 0)
                            {
                                for (int ir = 0; ir < numMrL; ir++)
                                {
                                    var aPanel = new ReadOnlySpan<float>(pa + pIdx * aPanelStrideL + ir * effKc * mrL, effKc * mrL);
                                    int cOff = (ir * mrL) * ldcL + jc;
                                    for (int jb = 0; jb < njrFull; jb += maxPanelTiles)
                                    {
                                        int chunk = Math.Min(maxPanelTiles, njrFull - jb);
                                        MachineKernelGemm.RunPanelFp32(
                                            aPanel,
                                            bPanel.Slice(jb * effKc * nrL, chunk * effKc * nrL),
                                            new Span<float>(cc + cOff + jb * nrL, cLenL - (cOff + jb * nrL)),
                                            ldcL, effKc, chunk);
                                    }
                                }
                            }

                            // Nr tail (effNc % nr != 0): per-Mr-block managed dispatch (small).
                            if (nrTail > 0)
                            {
                                int jrTail = njrFull * nrL;
                                var bTail = bPanel.Slice(njrFull * effKc * nrL, effKc * nrL);
                                for (int ir = 0; ir < numMrL; ir++)
                                {
                                    var aPanel = new ReadOnlySpan<float>(pa + pIdx * aPanelStrideL + ir * effKc * mrL, effKc * mrL);
                                    int cOff = (ir * mrL) * ldcL + jc + jrTail;
                                    DispatchMicrokernelWithTail<float>(
                                        aPanel, bTail,
                                        new Span<float>(cc + cOff, cLenL - cOff),
                                        ldcL, effKc, mrL, nrL, nrTail);
                                }
                            }
                        }
                    }
                }
                finally { System.Buffers.ArrayPool<byte>.Shared.Return(packBArr); }
            }, deterministicSafe: true); // N-axis split: disjoint C columns, fixed-order K reduction
            }
        }
        finally { System.Buffers.ArrayPool<float>.Shared.Return(packAArr); }
    }

    /// <summary>
    /// #653 single-persistent-region M-axis path for the wide-nc case (numNB==1, no pre-pack,
    /// no workspace, no transpose). Packs all K-panels of B once into a shared buffer, then runs
    /// ONE parallel region over ic blocks where each thread walks the full K-loop for its rows —
    /// eliminating the per-K-panel pack/compute barriers of <see cref="RunParallelUnsafe"/>.
    /// Bit-identical: each C[ic,jr] tile still reduces over k in ascending panel order, and
    /// ic blocks own disjoint C rows so there is no cross-thread write contention.
    /// </summary>
    private static unsafe void RunParallelSingleRegionUnsafe<T>(
        T* aPtr, int aLen, int lda,
        T* bPtr, int bLen, int ldb,
        T* cPtr, int cLen, int ldc,
        int m, int n, int k,
        int mc, int kc, int mr, int nr,
        in BlasOptions<T> options, int elemSize) where T : unmanaged
    {
        int numKPanels = (k + kc - 1) / kc;
        int numStripes = (n + nr - 1) / nr;
        int packedNc = numStripes * nr;
        long packBAllElems = (long)numKPanels * kc * packedNc;

        const int PackBAlignment = 32;
        byte[] packBArr = System.Buffers.ArrayPool<byte>.Shared.Rent((int)(packBAllElems * elemSize) + PackBAlignment - 1);
        int packBAlignedOffset;
        fixed (byte* pbPtr = packBArr)
        {
            nuint addr = (nuint)pbPtr;
            nuint mask = PackBAlignment - 1;
            packBAlignedOffset = (int)((PackBAlignment - (uint)(addr & mask)) & mask);
        }

        nint aPtrInt = (nint)aPtr;
        nint bPtrInt = (nint)bPtr;
        nint cPtrInt = (nint)cPtr;
        byte[] packBArrCap = packBArr;
        int packBAlignedOffsetCap = packBAlignedOffset;
        int kcCap = kc, ncCol = n, packedNcCap = packedNc, numStripesCap = numStripes, nrCap = nr;
        int ldbCap = ldb, bLenCap = bLen, kCap = k, numKPanelsCap = numKPanels;

        try
        {
            // ── Pack ALL K-panels of B once: single dispatch over (panel × stripe) ──
            // Each work-item packs one Nr-stripe of one K-panel into a disjoint region of the
            // shared buffer — order-independent, so deterministicSafe.
            int totalPackItems = numKPanels * numStripes;
            long packWork = packBAllElems;
            CpuParallelSettings.ParallelForOrSerial(0, totalPackItems, packWork, item =>
            {
                int panel = item / numStripesCap;
                int stripe = item % numStripesCap;
                int pc = panel * kcCap;
                int effKc = Math.Min(kcCap, kCap - pc);
                int panelOffElems = panel * kcCap * packedNcCap;
                int stripeOffElems = stripe * effKc * nrCap;
                ReadOnlySpan<T> bSlice = new ReadOnlySpan<T>((T*)bPtrInt + pc * ldbCap, bLenCap - pc * ldbCap);
                Span<T> packBStripe = MemoryMarshal.Cast<byte, T>(
                    packBArrCap.AsSpan(packBAlignedOffsetCap + (panelOffElems + stripeOffElems) * elemSize,
                                       effKc * nrCap * elemSize));
                Avx2Pack.PackBStripeRange<T>(
                    bSlice, ldbCap, false,
                    packBStripe,
                    stripe, 1,
                    ncCol, effKc, nrCap);
            }, deterministicSafe: true);

            // ── ONE parallel region over ic blocks; each thread walks the full K-loop ──
            int numIcBlocks = (m + mc - 1) / mc;
            long totalWork = (long)m * n * k;
            int mcCap = mc, mCap = m, ldaCap = lda, aLenCap = aLen, ldcCap = ldc, cLenCap = cLen, mrCap = mr;
            CpuParallelSettings.ParallelForOrSerial(0, numIcBlocks, totalWork, icIdx =>
            {
                int ic = icIdx * mcCap;
                int effMc = Math.Min(mcCap, mCap - ic);
                for (int panel = 0; panel < numKPanelsCap; panel++)
                {
                    int pc = panel * kcCap;
                    int effKc = Math.Min(kcCap, kCap - pc);
                    int panelOffElems = panel * kcCap * packedNcCap;

                    // Pack A[ic, pc] into this thread's [ThreadStatic] pool (consumed below).
                    Span<byte> packAByteSpan = PerThreadPool.Current.RentPackA(effMc * effKc * elemSize);
                    Span<T> packA = MemoryMarshal.Cast<byte, T>(packAByteSpan).Slice(0, effMc * effKc);
                    int aSliceOffset = ic * ldaCap + pc;
                    ReadOnlySpan<T> aSlice = new ReadOnlySpan<T>((T*)aPtrInt + aSliceOffset, aLenCap - aSliceOffset);
                    Avx2Pack.PackA<T>(aSlice, ldaCap, false, packA, effMc, effKc, mrCap);

                    Span<T> packBPanel = MemoryMarshal.Cast<byte, T>(
                        packBArrCap.AsSpan(packBAlignedOffsetCap + panelOffElems * elemSize, effKc * packedNcCap * elemSize));

                    // Inner (jr, ir) microkernel — accumulates into C (cleared by caller); the
                    // K-panels accumulate in ascending order, matching RunParallelUnsafe bit-for-bit.
                    for (int jr = 0; jr < ncCol; jr += nrCap)
                    {
                        int effNr = Math.Min(nrCap, ncCol - jr);
                        for (int ir = 0; ir < effMc; ir += mrCap)
                        {
                            if (ir + mrCap > effMc) break;
                            int packedAStripeOff = (ir / mrCap) * effKc * mrCap;
                            int packedBStripeOff = (jr / nrCap) * effKc * nrCap;
                            int cTileOff = (ic + ir) * ldcCap + jr;
                            Span<T> cTile = new Span<T>((T*)cPtrInt + cTileOff, cLenCap - cTileOff);
                            DispatchMicrokernelWithTail<T>(
                                packA.Slice(packedAStripeOff, effKc * mrCap),
                                packBPanel.Slice(packedBStripeOff, effKc * nrCap),
                                cTile, ldcCap, effKc, mrCap, nrCap, effNr);
                        }
                    }
                }
            }, deterministicSafe: true);
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(packBArr);
        }
    }

    /// <summary>
    /// #85 low-barrier jc-blocked GotoBLAS path (FP32). Packs ALL of A once into a shared Mr-stripe
    /// buffer — each ic-block reads its DISJOINT rows, so there is no cross-thread A contention
    /// (unlike the N-axis path, where every thread re-reads the whole shared A for its N-block). Then
    /// for each Nc-block of N (serial outer, disjoint C columns) it packs that block's B once into a
    /// shared, L3-resident buffer and runs ONE parallel region over ic-blocks, each thread walking the
    /// full K-loop for its rows. This is the BLIS persistent-team structure that no existing path has:
    /// low barrier count (one region per Nc-block, not per (jc,pc) macro-tile like RunParallelUnsafe),
    /// B blocked to stay L3-resident (vs single-region's all-B), shared packed-B (vs 2D's redundant
    /// per-thread B-pack), and disjoint-row shared-A (vs N-axis's whole-A-per-thread re-read).
    /// Bit-identical to the other paths: each C[i,j] reduces over k in ascending panel order and
    /// ic-blocks own disjoint C rows.
    /// </summary>
    private static unsafe void RunGotoBlasParallelUnsafe<T>(
        T* aPtr, int aLen, int lda,
        T* bPtr, int bLen, int ldb,
        T* cPtr, int cLen, int ldc,
        int m, int n, int k,
        int mc, int nc, int kc, int mr, int nr, int elemSize) where T : unmanaged
    {
        int numKPanels = (k + kc - 1) / kc;
        int numStripesM = (m + mr - 1) / mr;
        int packedM = numStripesM * mr;
        long packAElems = (long)numKPanels * kc * packedM;

        const int Align = 32;
        byte[] packAArr = System.Buffers.ArrayPool<byte>.Shared.Rent((int)(packAElems * elemSize) + Align - 1);
        int packAOff;
        fixed (byte* pa0 = packAArr)
        {
            nuint addr = (nuint)pa0; nuint mask = Align - 1;
            packAOff = (int)((Align - (uint)(addr & mask)) & mask);
        }

        nint aI = (nint)aPtr, bI = (nint)bPtr, cI = (nint)cPtr;
        int ldaC = lda, ldbC = ldb, ldcC = ldc, kC = k, mC = m, nC = n, aLenC = aLen, bLenC = bLen, cLenC = cLen;
        int kcC = kc, mrC = mr, nrC = nr, numKPanelsC = numKPanels, packedMC = packedM, numStripesMC = numStripesM;
        byte[] packAArrC = packAArr; int packAOffC = packAOff;

        try
        {
            // ── Pack ALL of A once: (panel × m-stripe) shared, Mr-stripe layout ──
            int numPackAItems = numKPanelsC * numStripesMC;
            CpuParallelSettings.ParallelForOrSerial(0, numPackAItems, packAElems, item =>
            {
                int panel = item / numStripesMC;
                int mstripe = item % numStripesMC;
                int pc = panel * kcC;
                int effKc = Math.Min(kcC, kC - pc);
                int row = mstripe * mrC;
                int effMr = Math.Min(mrC, mC - row);
                long panelBase = (long)panel * packedMC * kcC;
                long stripeOff = (long)mstripe * effKc * mrC;
                int aOff = row * ldaC + pc;
                ReadOnlySpan<T> aSlice = new ReadOnlySpan<T>((T*)aI + aOff, aLenC - aOff);
                Span<T> dst = MemoryMarshal.Cast<byte, T>(
                    packAArrC.AsSpan(packAOffC + (int)((panelBase + stripeOff) * elemSize), effKc * mrC * elemSize));
                Avx2Pack.PackA<T>(aSlice, ldaC, false, dst, effMr, effKc, mrC);
            }, deterministicSafe: true);

            // ── Per Nc-block: pack that block's B once (shared, L3) + ONE ic parallel region ──
            int numIcBlocks = (mC + mc - 1) / mc;
            int mcC = mc;
            for (int jc = 0; jc < nC; jc += nc)
            {
                int effNc = Math.Min(nc, nC - jc);
                int numStripesN = (effNc + nrC - 1) / nrC;
                int packedNc = numStripesN * nrC;
                long packBElems = (long)numKPanelsC * kcC * packedNc;
                byte[] packBArr = System.Buffers.ArrayPool<byte>.Shared.Rent((int)(packBElems * elemSize) + Align - 1);
                int packBOff;
                fixed (byte* pb0 = packBArr)
                {
                    nuint addr = (nuint)pb0; nuint mask = Align - 1;
                    packBOff = (int)((Align - (uint)(addr & mask)) & mask);
                }
                byte[] packBArrC = packBArr; int packBOffC = packBOff;
                int jcC = jc, effNcC = effNc, numStripesNC = numStripesN, packedNcC = packedNc;
                try
                {
                    // Pack B[*, jc:jc+effNc] once: (panel × n-stripe) shared.
                    int numPackBItems = numKPanelsC * numStripesNC;
                    CpuParallelSettings.ParallelForOrSerial(0, numPackBItems, packBElems, item =>
                    {
                        int panel = item / numStripesNC;
                        int nstripe = item % numStripesNC;
                        int pc = panel * kcC;
                        int effKc = Math.Min(kcC, kC - pc);
                        long panelBase = (long)panel * packedNcC * kcC;
                        long stripeOff = (long)nstripe * effKc * nrC;
                        int bOff = pc * ldbC + jcC;
                        ReadOnlySpan<T> bSlice = new ReadOnlySpan<T>((T*)bI + bOff, bLenC - bOff);
                        Span<T> dst = MemoryMarshal.Cast<byte, T>(
                            packBArrC.AsSpan(packBOffC + (int)((panelBase + stripeOff) * elemSize), effKc * nrC * elemSize));
                        Avx2Pack.PackBStripeRange<T>(bSlice, ldbC, false, dst, nstripe, 1, effNcC, effKc, nrC);
                    }, deterministicSafe: true);

                    // ONE parallel region over ic-blocks; each thread walks the full K-loop for its rows.
                    long compWork = (long)mC * effNc * kC;
                    CpuParallelSettings.ParallelForOrSerial(0, numIcBlocks, compWork, icIdx =>
                    {
                        int ic = icIdx * mcC;
                        int effMc = Math.Min(mcC, mC - ic);
                        for (int panel = 0; panel < numKPanelsC; panel++)
                        {
                            int pc = panel * kcC;
                            int effKc = Math.Min(kcC, kC - pc);
                            long panelBaseA = (long)panel * packedMC * kcC;
                            long panelBaseB = (long)panel * packedNcC * kcC;
                            for (int jr = 0; jr < effNcC; jr += nrC)
                            {
                                int effNr = Math.Min(nrC, effNcC - jr);
                                long bStripeOff = panelBaseB + (long)(jr / nrC) * effKc * nrC;
                                Span<T> packBStripe = MemoryMarshal.Cast<byte, T>(
                                    packBArrC.AsSpan(packBOffC + (int)(bStripeOff * elemSize), effKc * nrC * elemSize));
                                for (int ir = 0; ir < effMc; ir += mrC)
                                {
                                    if (ir + mrC > effMc) break;
                                    int globalStripe = (ic + ir) / mrC;
                                    long aStripeOff = panelBaseA + (long)globalStripe * effKc * mrC;
                                    Span<T> packAStripe = MemoryMarshal.Cast<byte, T>(
                                        packAArrC.AsSpan(packAOffC + (int)(aStripeOff * elemSize), effKc * mrC * elemSize));
                                    int cOff = (ic + ir) * ldcC + jcC + jr;
                                    Span<T> cTile = new Span<T>((T*)cI + cOff, cLenC - cOff);
                                    DispatchMicrokernelWithTail<T>(
                                        packAStripe, packBStripe, cTile, ldcC, effKc, mrC, nrC, effNr);
                                }
                            }
                        }
                    }, deterministicSafe: true);
                }
                finally { System.Buffers.ArrayPool<byte>.Shared.Return(packBArr); }
            }
        }
        finally { System.Buffers.ArrayPool<byte>.Shared.Return(packAArr); }
    }

    /// <summary>
    /// Sub-B5 (#370) — 2D MN-grid parallel path. For each pc (K-block, serial),
    /// flatten the (jcIdx, icIdx) tile grid into a 1D work-item index and
    /// dispatch via <see cref="CpuParallelSettings.ParallelForOrSerial"/>. Each
    /// worker:
    /// <list type="number">
    ///   <item>Packs A for its (ic, pc) panel into thread-local PerThreadPool buffer.</item>
    ///   <item>Packs B for its (pc, jc) stripe into thread-local PerThreadPool buffer.</item>
    ///   <item>Iterates the inner (jr, ir) microkernel tiles writing C disjointly.</item>
    /// </list>
    ///
    /// <para>
    /// Memory model: per-thread packed-A AND packed-B (vs RunParallelUnsafe's
    /// shared packed-B). Redundant pack-B work across threads handling different
    /// icIdx within the same jcIdx, but the extra parallelism wins on shapes
    /// where M-axis alone underutilizes cores (e.g., m=192 with mc=128 gives
    /// 2 M-blocks; on 16 cores that's 87.5% idle without 2D).
    /// </para>
    ///
    /// <para>
    /// C-write disjointness: each thread owns a unique (ic, jc) tile, so writes
    /// to C[ic..ic+effectiveMc, jc..jc+effectiveNc] don't collide. Same property
    /// as M-axis parallel (where each thread owns rows [ic, ic+effectiveMc)).
    /// </para>
    /// </summary>
    private static unsafe void Run2DParallelUnsafe<T>(
        T* aPtrInt, int aLen, int lda, bool transA,
        T* bPtrInt, int bLen, int ldb, bool transB,
        T* cPtrInt, int cLen, int ldc,
        int m, int n, int k,
        int mc, int nc, int kc,
        int mr, int nr,
        in BlasOptions<T> options, int elemSize) where T : unmanaged
    {
        int numIcBlocks = (m + mc - 1) / mc;
        int numJcBlocks = (n + nc - 1) / nc;
        int totalTiles = MN2DDriver.TotalItems(numIcBlocks, numJcBlocks);

        // Capture pre-pack handle refs out of the `in` options param so the
        // ParallelForOrSerial lambda can read them. Both reads are safe across
        // threads because PackedBuffer is treated read-only inside the closure.
        var packedA = options.PackedA;
        var packedB = options.PackedB;

        // K-block outer loop: serial. Each iteration parallelizes across all
        // (jcIdx, icIdx) tiles.
        for (int pc = 0; pc < k; pc += kc)
        {
            int effectiveKc = Math.Min(kc, k - pc);
            int pc_cap = pc;
            int effectiveKc_cap = effectiveKc;

            long workPerTile = (long)mc * nc * effectiveKc;
            long totalWork = workPerTile * totalTiles;

            CpuParallelSettings.ParallelForOrSerial(0, totalTiles, totalWork, flatIdx =>
            {
                var (icIdx, jcIdx) = MN2DDriver.UnflattenIndex(flatIdx, numJcBlocks);
                int ic = icIdx * mc;
                int jc = jcIdx * nc;
                int effectiveMc = Math.Min(mc, m - ic);
                int effectiveNc = Math.Min(nc, n - jc);
                if (effectiveMc <= 0 || effectiveNc <= 0) return;

                int paddedNc = ((effectiveNc + nr - 1) / nr) * nr;
                int pcIdx = pc_cap / kc;

                // ── Pack A: prefer pre-packed tile, else thread-local live pack ──
                // Multi-panel matches when (TileMc, TileKc) equals the blocking
                // parameters (mc, kc) — the tile at (icIdx, pcIdx) holds the
                // effective-size-clamped pack for tail blocks within the full
                // allocation, and the inner microkernel stripe math uses
                // effectiveKc_cap so it reads only the live data.
                int effectivePackABytes = effectiveMc * effectiveKc_cap * elemSize;
                Span<T> activePackA;
                bool packAFromPrePack = false;
                if (packedA != null && WeightPackCache.IsCacheCurrent(packedA)
                    && packedA.PackMr == mr // stripe-interleave gate (WeightPackHandle.PackMr)
                    && packedA.MultiPanelStride > 0
                    && packedA.TileMc == mc && packedA.TileKc == kc)
                {
                    var tile = packedA.GetTileSlice(icIdx, pcIdx);
                    if (!tile.IsEmpty && tile.Length >= effectivePackABytes)
                    {
                        activePackA = MemoryMarshal.Cast<byte, T>(tile.Slice(0, effectivePackABytes));
                        packAFromPrePack = true;
                    }
                    else
                    {
                        activePackA = default;
                    }
                }
                else
                {
                    activePackA = default;
                }

                if (packAFromPrePack)
                {
                    BlasManagedStatsTracker.IncrementPackCacheHit();
                }
                else
                {
                    if (packedA != null) BlasManagedStatsTracker.IncrementPackCacheMiss();
                    Span<byte> packABytesSpan_inner = PerThreadPool.Current.RentPackA(effectivePackABytes);
                    activePackA = MemoryMarshal.Cast<byte, T>(packABytesSpan_inner)
                                              .Slice(0, effectiveMc * effectiveKc_cap);
                    int aSliceOffset = transA ? pc_cap * lda + ic : ic * lda + pc_cap;
                    ReadOnlySpan<T> aSlice = new ReadOnlySpan<T>((T*)aPtrInt + aSliceOffset, aLen - aSliceOffset);
                    Avx2Pack.PackA<T>(
                        a: aSlice, lda, transA,
                        packed: activePackA,
                        mc: effectiveMc, kc: effectiveKc_cap, mr);
                }

                // ── Pack B: prefer pre-packed tile, else thread-local live pack ──
                int effectivePackBBytes = effectiveKc_cap * paddedNc * elemSize;
                Span<T> activePackB;
                bool packBFromPrePack = false;
                if (packedB != null && WeightPackCache.IsCacheCurrent(packedB)
                    && packedB.PackNr == nr // stripe-interleave gate (WeightPackHandle.PackNr)
                    && packedB.MultiPanelStride > 0
                    && packedB.TileMc == nc && packedB.TileKc == kc)
                {
                    var tile = packedB.GetTileSlice(jcIdx, pcIdx);
                    if (!tile.IsEmpty && tile.Length >= effectivePackBBytes)
                    {
                        activePackB = MemoryMarshal.Cast<byte, T>(tile.Slice(0, effectivePackBBytes));
                        packBFromPrePack = true;
                    }
                    else
                    {
                        activePackB = default;
                    }
                }
                else
                {
                    activePackB = default;
                }

                if (packBFromPrePack)
                {
                    BlasManagedStatsTracker.IncrementPackCacheHit();
                }
                else
                {
                    if (packedB != null) BlasManagedStatsTracker.IncrementPackCacheMiss();
                    Span<byte> packBBytesSpan_inner = PerThreadPool.Current.RentPackB(effectivePackBBytes);
                    activePackB = MemoryMarshal.Cast<byte, T>(packBBytesSpan_inner)
                                              .Slice(0, effectiveKc_cap * paddedNc);
                    int bSliceOffset = transB ? jc * ldb + pc_cap : pc_cap * ldb + jc;
                    ReadOnlySpan<T> bSlice = new ReadOnlySpan<T>((T*)bPtrInt + bSliceOffset, bLen - bSliceOffset);
                    Avx2Pack.PackB<T>(
                        b: bSlice, ldb, transB,
                        packed: activePackB,
                        nc: effectiveNc, kc: effectiveKc_cap, nr);
                }

                // ── Inner microkernel loop (jr, ir) — disjoint C writes ──────────
                for (int jr = 0; jr < effectiveNc; jr += nr)
                {
                    int effectiveNr = Math.Min(nr, effectiveNc - jr);
                    for (int ir = 0; ir < effectiveMc; ir += mr)
                    {
                        if (ir + mr > effectiveMc) break;
                        int packedAStripeOff = (ir / mr) * effectiveKc_cap * mr;
                        int packedBStripeOff = (jr / nr) * effectiveKc_cap * nr;
                        int cTileOff = (ic + ir) * ldc + (jc + jr);
                        Span<T> cTile = new Span<T>((T*)cPtrInt + cTileOff, cLen - cTileOff);
                        DispatchMicrokernelWithTail<T>(
                            activePackA.Slice(packedAStripeOff, effectiveKc_cap * mr),
                            activePackB.Slice(packedBStripeOff, effectiveKc_cap * nr),
                            cTile,
                            ldc, effectiveKc_cap,
                            mr, nr, effectiveNr);
                    }
                }
            }, deterministicSafe: true); // 2D MN-grid split: each (ic,jc) tile disjoint, fixed-order K reduction
        }
    }

    /// <summary>
    /// Routes to the appropriate microkernel for T based on (mr, nr) and
    /// runtime SIMD availability. Dispatch order: AVX-512 → AVX2 → Neon → scalar,
    /// so the widest available ISA is always selected.
    /// </summary>
    private static void DispatchMicrokernel<T>(
        ReadOnlySpan<T> packedA, ReadOnlySpan<T> packedB,
        Span<T> c, int ldc, int kc,
        int mr, int nr) where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            if (mr == 8 && nr == 16 && Avx512Fp64_8x16.IsSupported)
            {
                Avx512Fp64_8x16.Run(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packedA),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packedB),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(c),
                    ldc, kc);
                return;
            }
            // #409 S.4: higher-intensity 6×8 FP64 tile (Fast-mode default via PickMicrokernelTile).
            if (mr == 6 && nr == 8 && Avx2Fp64_6x8.IsSupported)
            {
                Avx2Fp64_6x8.Run(
                    MemoryMarshal.Cast<T, double>(packedA),
                    MemoryMarshal.Cast<T, double>(packedB),
                    MemoryMarshal.Cast<T, double>(c),
                    ldc, kc);
                return;
            }
            if (mr == 4 && nr == 8 && Avx2Fp64_4x8.IsSupported)
            {
                Avx2Fp64_4x8.Run(
                    MemoryMarshal.Cast<T, double>(packedA),
                    MemoryMarshal.Cast<T, double>(packedB),
                    MemoryMarshal.Cast<T, double>(c),
                    ldc, kc);
                return;
            }
            if (mr == 4 && nr == 4)
            {
                if (NeonFp64_4x4.IsSupported)
                {
                    NeonFp64_4x4.Run(
                        MemoryMarshal.Cast<T, double>(packedA),
                        MemoryMarshal.Cast<T, double>(packedB),
                        MemoryMarshal.Cast<T, double>(c),
                        ldc, kc);
                    return;
                }
                ScalarFp64_4x4.Run(
                    MemoryMarshal.Cast<T, double>(packedA),
                    MemoryMarshal.Cast<T, double>(packedB),
                    MemoryMarshal.Cast<T, double>(c),
                    ldc, kc);
                return;
            }
            throw new NotSupportedException($"Unsupported FP64 microkernel shape Mr={mr} Nr={nr}");
        }
        if (typeof(T) == typeof(float))
        {
            if (mr == 16 && nr == 16 && Avx512Fp32_16x16.IsSupported)
            {
                Avx512Fp32_16x16.Run(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packedA),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packedB),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(c),
                    ldc, kc);
                return;
            }
            // #409 S.3: higher-intensity 6×16 FP32 tile (Fast-mode default via SelectMrNr).
            // NOTE: tried driving the AVX2 6×16 MACHINE-CODE kernel here — it REGRESSED (per-
            // micro-tile fixed + indirect-call overhead; 1024³ 393→369, FFN-up 266→210). The
            // C# kernel already saturates the harness, so the kernel is NOT the bottleneck in
            // PackBoth — the gap to native is the macro-harness (packing / blocking / bandwidth).
            if (mr == 6 && nr == 16 && Avx2Fp32_6x16.IsSupported)
            {
                if (s_prefetch)
                    Avx2Fp32_6x16.RunPrefetch(
                        MemoryMarshal.Cast<T, float>(packedA),
                        MemoryMarshal.Cast<T, float>(packedB),
                        MemoryMarshal.Cast<T, float>(c),
                        ldc, kc);
                else
                    Avx2Fp32_6x16.Run(
                        MemoryMarshal.Cast<T, float>(packedA),
                        MemoryMarshal.Cast<T, float>(packedB),
                        MemoryMarshal.Cast<T, float>(c),
                        ldc, kc);
                return;
            }
            if (mr == 8 && nr == 8 && Avx2Fp32_8x8.IsSupported)
            {
                Avx2Fp32_8x8.Run(
                    MemoryMarshal.Cast<T, float>(packedA),
                    MemoryMarshal.Cast<T, float>(packedB),
                    MemoryMarshal.Cast<T, float>(c),
                    ldc, kc);
                return;
            }
            // Neon FP32 uses Mr=8 Nr=4. PickMicrokernelTile returns (8, 4) on ARM64
            // hosts where Neon is available but AVX is not.
            if (mr == 8 && nr == 4 && NeonFp32_8x4.IsSupported)
            {
                NeonFp32_8x4.Run(
                    MemoryMarshal.Cast<T, float>(packedA),
                    MemoryMarshal.Cast<T, float>(packedB),
                    MemoryMarshal.Cast<T, float>(c),
                    ldc, kc);
                return;
            }
            // (4, 4) is only reached when no SIMD is available (no AVX, no Neon).
            if (mr == 4 && nr == 4)
            {
                ScalarFp32_4x4.Run(
                    MemoryMarshal.Cast<T, float>(packedA),
                    MemoryMarshal.Cast<T, float>(packedB),
                    MemoryMarshal.Cast<T, float>(c),
                    ldc, kc);
                return;
            }
            throw new NotSupportedException($"Unsupported FP32 microkernel shape Mr={mr} Nr={nr}");
        }
        throw new NotSupportedException($"PackBothStrategy does not support T={typeof(T).Name}.");
    }

    /// <summary>
    /// Routes to the appropriate microkernel for T, handling a partial-N tile
    /// (<paramref name="effectiveNr"/> &lt; <paramref name="nr"/>). When
    /// <paramref name="effectiveNr"/> equals <paramref name="nr"/> the full
    /// <see cref="DispatchMicrokernel{T}"/> path is taken. For partial tiles,
    /// the matching Avx512Tail or Avx2Tail kernel is preferred; if no matching
    /// SIMD tail kernel exists, a scalar column-by-column fallback is used.
    /// </summary>
    private static void DispatchMicrokernelWithTail<T>(
        ReadOnlySpan<T> packedA, ReadOnlySpan<T> packedB,
        Span<T> c, int ldc, int kc,
        int mr, int nr, int effectiveNr) where T : unmanaged
    {
        if (effectiveNr == nr)
        {
            // Full tile — use the regular (non-tail) microkernel.
            DispatchMicrokernel<T>(packedA, packedB, c, ldc, kc, mr, nr);
            return;
        }

        // Partial-N tile: dispatch to a tail kernel or scalar fallback.
        if (typeof(T) == typeof(double))
        {
            if (mr == 8 && nr == 16 && Avx512Tail.IsSupported)
            {
                Avx512Tail.RunFp64_8xN(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packedA),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packedB),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(c),
                    ldc, kc, effectiveNr);
                return;
            }
            if (mr == 4 && nr == 8 && Avx2Tail.IsSupported)
            {
                Avx2Tail.RunFp64_4xN(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packedA),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packedB),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(c),
                    ldc, kc, effectiveNr);
                return;
            }
            // Scalar fallback: process column-by-column.
            var packedAd = System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packedA);
            var packedBd = System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packedB);
            var cd = System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(c);
            for (int col = 0; col < effectiveNr; col++)
            {
                for (int row = 0; row < mr; row++)
                {
                    double sum = cd[row * ldc + col];
                    for (int kk = 0; kk < kc; kk++)
                    {
                        sum += packedAd[kk * mr + row] * packedBd[kk * nr + col];
                    }
                    cd[row * ldc + col] = sum;
                }
            }
            return;
        }
        if (typeof(T) == typeof(float))
        {
            if (mr == 16 && nr == 16 && Avx512Tail.IsSupported)
            {
                Avx512Tail.RunFp32_16xN(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packedA),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packedB),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(c),
                    ldc, kc, effectiveNr);
                return;
            }
            if (mr == 8 && nr == 8 && Avx2Tail.IsSupported)
            {
                Avx2Tail.RunFp32_8xN(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packedA),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packedB),
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(c),
                    ldc, kc, effectiveNr);
                return;
            }
            // Scalar fallback
            var packedAf = System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packedA);
            var packedBf = System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packedB);
            var cf = System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(c);
            for (int col = 0; col < effectiveNr; col++)
            {
                for (int row = 0; row < mr; row++)
                {
                    float sum = cf[row * ldc + col];
                    for (int kk = 0; kk < kc; kk++)
                    {
                        sum += packedAf[kk * mr + row] * packedBf[kk * nr + col];
                    }
                    cf[row * ldc + col] = sum;
                }
            }
            return;
        }
        throw new NotSupportedException($"DispatchMicrokernelWithTail does not support T={typeof(T).Name}.");
    }
}
