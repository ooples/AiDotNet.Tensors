using System;
using System.Buffers;
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
            // Pack-B is allocated once on the calling thread (shared, read-only during
            // the parallel region). We use ArrayPool<byte> so the backing byte[] can be
            // captured by the parallel lambda without violating the Span<T>-is-stack-only rule.
            // Pack-A is NOT pre-allocated here: each parallel icIdx body calls
            // PerThreadPool.Current.RentPackA ([ThreadStatic]) → every thread gets its own
            // buffer with zero cross-thread contention.
            //
            // a, b, c are pinned via `fixed` before entering the parallel region so the GC
            // cannot relocate them while worker threads hold raw pointers into them.
            byte[]? packBArray = null;
            try
            {
                Span<byte> packBBytesSpan = ArenaIntegration.TryRentBytes(packBBytes);
                if (packBBytesSpan.IsEmpty)
                {
                    packBArray = ArrayPool<byte>.Shared.Rent(packBBytes);
                    packBBytesSpan = packBArray.AsSpan(0, packBBytes);
                }

                // Pin a, b, c for the duration of RunParallelUnsafe.
                // `fixed` on a Span<T> obtained from a managed array is supported in C# 7.3+.
                fixed (T* aPtr = a)
                fixed (T* bPtr = b)
                fixed (T* cPtr = c)
                {
                    RunParallelUnsafe<T>(
                        aPtr, a.Length, lda, transA,
                        bPtr, b.Length, ldb, transB,
                        cPtr, c.Length, ldc,
                        m, n, k, mc, nc, kc, mr, nr,
                        packBBytesSpan, in options, elemSize);
                }
            }
            finally
            {
                if (packBArray != null)
                    ArrayPool<byte>.Shared.Return(packBArray);
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
                bool packBFromPrePack = false;
                Span<T> activePackB = packB;
                if (options.PackedB != null && WeightPackCache.IsCacheCurrent(options.PackedB)
                    && options.PackedB.PackedBuffer.Length >= effectivePackBBytes)
                {
                    activePackB = MemoryMarshal.Cast<byte, T>(options.PackedB.PackedBuffer.AsSpan(0, effectivePackBBytes));
                    packBFromPrePack = true;
                }

                if (!packBFromPrePack)
                {
                    // Pack B[pc..pc+effectiveKc, jc..jc+effectiveNc] into packB.
                    // transB=false: B is [K, N] row-major, panel starts at b[pc * ldb + jc].
                    // transB=true:  B is [N, K] row-major, panel starts at b[jc * ldb + pc].
                    // Pass a padded slice (packedBElemCount) so ScalarPack.PackB can
                    // zero-pad the partial tail stripe when effectiveNc % nr != 0.
                    int bSliceOffset = transB ? jc * ldb + pc : pc * ldb + jc;
                    ScalarPack.PackB<T>(
                        b: b.Slice(bSliceOffset), ldb, transB,
                        packed: activePackB.Slice(0, packedBElemCount),
                        nc: effectiveNc, kc: effectiveKc, nr);
                }

                for (int ic = 0; ic < m; ic += mc)
                {
                    int effectiveMc = Math.Min(mc, m - ic);

                    // Layer 3: short-circuit pack-A when pre-pack handle is current.
                    // Use effective tile bytes, not nominal, because PrePackA clamps Mc/Kc to
                    // the actual matrix dimensions (small matrices produce smaller handles).
                    int effectivePackABytes = effectiveMc * effectiveKc * elemSize;
                    bool packAFromPrePack = false;
                    Span<T> activePackA = packA;
                    if (options.PackedA != null && WeightPackCache.IsCacheCurrent(options.PackedA)
                        && options.PackedA.PackedBuffer.Length >= effectivePackABytes)
                    {
                        activePackA = MemoryMarshal.Cast<byte, T>(options.PackedA.PackedBuffer.AsSpan(0, effectivePackABytes));
                        packAFromPrePack = true;
                    }

                    if (!packAFromPrePack)
                    {
                        // Pack A[ic..ic+effectiveMc, pc..pc+effectiveKc] into packA.
                        // transA=false: A is [M, K] row-major, panel starts at a[ic * lda + pc].
                        // transA=true:  A is [K, M] row-major, panel starts at a[pc * lda + ic].
                        int aSliceOffset = transA ? pc * lda + ic : ic * lda + pc;
                        ScalarPack.PackA<T>(
                            a: a.Slice(aSliceOffset), lda, transA,
                            packed: activePackA.Slice(0, effectiveMc * effectiveKc),
                            mc: effectiveMc, kc: effectiveKc, mr);
                    }

                    // Iterate microkernel tiles within this Mc × Nc panel.
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
        byte[] packBArr = new byte[packBBytesSpan.Length];

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
                if (packedB != null && WeightPackCache.IsCacheCurrent(packedB)
                    && packedB.PackedBuffer.Length >= effectivePackBBytes)
                {
                    // Layer 3: pre-packed B is already in byte[] form — copy into packBArr
                    // so the parallel lambda reads from a stable, capturable byte[].
                    packedB.PackedBuffer.AsSpan(0, effectivePackBBytes)
                           .CopyTo(packBArr.AsSpan(0, effectivePackBBytes));
                    packBFromPrePack = true;
                }

                if (!packBFromPrePack)
                {
                    // Pack B[pc..pc+effectiveKc, jc..jc+effectiveNc] into packBArr.
                    // transB=false: panel starts at b[pc * ldb + jc].
                    // transB=true:  panel starts at b[jc * ldb + pc].
                    // Pass a padded slice (packedBByteCount) so ScalarPack.PackB can
                    // zero-pad the partial tail stripe when effectiveNc % nr != 0.
                    int bSliceOffset = transB ? jc * ldb + pc : pc * ldb + jc;
                    ReadOnlySpan<T> bSlice = new ReadOnlySpan<T>((T*)bPtrInt + bSliceOffset, bLen - bSliceOffset);
                    Span<T> packBTemp = MemoryMarshal.Cast<byte, T>(packBArr.AsSpan(0, packedBByteCount));
                    ScalarPack.PackB<T>(
                        b: bSlice, ldb, transB,
                        packed: packBTemp,
                        nc: effectiveNc, kc: effectiveKc, nr);
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

                CpuParallelSettings.ParallelForOrSerial(0, numIcBlocks, totalWork, icIdx =>
                {
                    int ic = icIdx * mc;
                    int effectiveMc = Math.Min(mc, m - ic);

                    // ── Pack A (per-thread, from PerThreadPool.Current) ────────────────
                    int effectivePackABytes = effectiveMc * effectiveKc_cap * elemSize;
                    Span<T> activePackA;
                    if (packedA != null && WeightPackCache.IsCacheCurrent(packedA)
                        && packedA.PackedBuffer.Length >= effectivePackABytes)
                    {
                        // Layer 3: pre-packed A is a read-only byte[]. Create span from it.
                        activePackA = MemoryMarshal.Cast<byte, T>(
                            packedA.PackedBuffer.AsSpan(0, effectivePackABytes));
                    }
                    else
                    {
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
                        ScalarPack.PackA<T>(
                            a: aSlice, lda, transA,
                            packed: activePackA,
                            mc: effectiveMc, kc: effectiveKc_cap, mr);
                    }

                    // Shared pack-B: reconstruct span from captured byte[] (read-only).
                    // Use packedBByteCount_cap (Nr-padded size) so the tail stripe
                    // (packed by ScalarPack.PackB with zero-padding) is accessible.
                    Span<T> activePackB = MemoryMarshal.Cast<byte, T>(
                        packBArr_cap.AsSpan(0, packedBByteCount_cap));

                    // ── Inner microkernel loop (jr, ir) ────────────────────────────────
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
                });
            }
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
