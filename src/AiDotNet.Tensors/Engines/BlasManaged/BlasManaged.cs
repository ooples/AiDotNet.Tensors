using System;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// BLIS-style managed GEMM kernel. Replaces Avx512Sgemm + SimdGemm as the
/// codebase's primary GEMM path. See docs/superpowers/specs/2026-05-16-blas-managed-design.md.
/// </summary>
public static class BlasManaged
{
    /// <summary>
    /// Process-wide default execution mode. Applied when a caller's
    /// <see cref="BlasOptions{T}.Mode"/> equals <see cref="BlasMode.Deterministic"/>
    /// (the field default) — i.e., when the caller didn't explicitly opt in to
    /// <see cref="BlasMode.Fast"/>. Set once at process start by the routing
    /// shim (Sub-issue F, #374); otherwise stays <see cref="BlasMode.Deterministic"/>.
    /// </summary>
    public static BlasMode DefaultMode { get; set; } = BlasMode.Deterministic;

    /// <summary>
    /// Sub-issue C (#371): work threshold (M·N·K) below which <see cref="Gemm{T}"/>
    /// bypasses strategy selection / autotune / microkernel-tile picking and
    /// routes directly to <see cref="StreamingStrategy"/>. Measured on tiny shapes
    /// where dispatcher overhead exceeded actual compute by 100×+.
    /// </summary>
    internal const long TinyShapeWorkThreshold = 100_000;

    /// <summary>
    /// Computes C = op(A) · op(B), where op(X) is X or X^T.
    ///
    /// <para>
    /// The strategy (packing mode) is selected by <see cref="Dispatcher.SelectStrategy{T}"/>
    /// based on the user's <see cref="BlasOptions{T}.PackingMode"/> override or, when it is
    /// <see cref="PackingMode.Auto"/>, a static heuristic on K and M·N work size:
    /// </para>
    ///
    /// <list type="bullet">
    ///   <item>K &lt; 32 or M·N &lt; 1024 → <see cref="StreamingStrategy"/> (no pack overhead).</item>
    ///   <item>K &lt; 128 → <see cref="PackAOnlyStrategy"/> (pack A; B reuse too low to justify packing).</item>
    ///   <item>Otherwise → <see cref="PackBothStrategy"/> (pack A and B for maximum cache locality).</item>
    /// </list>
    ///
    /// <para>
    /// <c>C</c> is always zeroed on entry so the result is C := op(A)·op(B),
    /// not C += op(A)·op(B). A future beta=1 option may skip this zero.
    /// </para>
    ///
    /// <para>
    /// <see cref="PackingMode.DisableAutotune"/> falls through to the default Auto heuristic
    /// in this phase; the autotune cache is added in Phase H.
    /// </para>
    ///
    /// <para>
    /// An early return is taken when any of M, N, or K is zero; caller-supplied C is preserved.
    /// </para>
    ///
    /// <para>
    /// Optional epilogue (bias, activation, skip, dropout, output scale) and other advanced
    /// settings are specified via <paramref name="options"/>.
    /// Defaults when omitted: no epilogue, single-threaded, no autotune key.
    /// </para>
    /// </summary>
    public static void Gemm<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // Strategies do read-modify-write on C; zero it here so the first call
        // produces C = A · B (not C += A · B). Future versions may expose a
        // beta=1 option that skips this zero, but Phase B's contract is C := A · B.
        c.Clear();

        // Sub-issue C (#371): tiny-shape fast path. For very small (M, N, K),
        // dispatcher + autotune overhead exceeds the GEMM compute itself
        // (Tiny_8x6x4 at 192 flops takes 70us on the regular path → 99% overhead).
        // When M*N*K is below the threshold AND the caller hasn't requested
        // a specific PackingMode, skip strategy selection / autotune /
        // microkernel-tile picking and route directly to StreamingStrategy.
        if ((long)m * n * k <= TinyShapeWorkThreshold && options.PackingMode == PackingMode.Auto)
        {
            StreamingStrategy.Run<T>(
                a, lda, transA,
                b, ldb, transB,
                c, ldc,
                m, n, k,
                in options);
            // Apply epilogue if any (cheap check; rare for tiny shapes).
            var fastEpilogue = options.Epilogue;
            EpilogueChain.Apply<T>(c, ldc, m, n, in fastEpilogue);
            return;
        }

        PackingMode strategy = Dispatcher.SelectStrategy(m, n, k, options);

        // PackAOnly does not support transB=true in Phase B — fall back to
        // PackBoth (which absorbs the transpose into its pack) or Streaming
        // (which handles transB via in-place index branching).
        if (strategy == PackingMode.ForcePackAOnly && transB)
            strategy = PackingMode.ForcePackBoth;

        // Pick SIMD-aware (mr, nr) for PackBoth using the AVX-512 → AVX2 → Neon → scalar
        // hierarchy. Fall back to (4, 4) scalar only when the shape is too small for
        // the SIMD tile to even start (m < mr or n < nr), or when m is not an exact
        // multiple of mr (M-tail kernels deferred to a future phase).
        // When m % mr == 0 but n % nr != 0, PackBothStrategy handles the partial-N
        // column-block via Avx2Tail/Avx512Tail (Task G2).
        // PackAOnly always uses scalar (4, 4) because its strided-B path has no
        // AVX2/AVX-512 RunStridedB variant yet (deferred to Phase Cx).
        var (mr, nr) = PickMicrokernelTile<T>();
        if (m < mr || n < nr || m % mr != 0)
        {
            mr = 4; nr = 4;
        }

        // Sub-D (#372) — correctness fix for the partial-M silent-drop bug.
        // PackBoth and PackAOnly inner loops have `if (ir + mr > effectiveMc) break;`
        // which silently exits before computing the partial-M tail. Combined with
        // c.Clear() at function entry, shapes where m % mr != 0 produced all-zero
        // trailing rows. Route those shapes to StreamingStrategy which has no
        // row-alignment constraint. The original existing-tests cohort still hits
        // the PackBoth/PackAOnly paths because they use shapes with m % 4 == 0;
        // this branch only triggers on previously-broken shapes.
        if (m % mr != 0 || n % nr != 0)
        {
            StreamingStrategy.Run<T>(
                a, lda, transA,
                b, ldb, transB,
                c, ldc,
                m, n, k,
                in options);
            var streamingEpilogue = options.Epilogue;
            EpilogueChain.Apply<T>(c, ldc, m, n, in streamingEpilogue);
            return;
        }

        // Consult the autotune dispatcher for blocking parameters. The axis is
        // informational for now — strategy integration is a future task.
        bool hasEpilogue = options.Epilogue.Activation != AiDotNet.Tensors.Engines.FusedActivationType.None
            || !options.Epilogue.BiasN.IsEmpty
            || !options.Epilogue.SkipMxN.IsEmpty;
        int procs = options.NumThreads > 0 ? options.NumThreads : Environment.ProcessorCount;
        var (_, autotuneMc, autotuneNc, autotuneKc, _) =
            AutotuneDispatcher.Decide<T>(
                m, n, k,
                transA, transB,
                mr, nr,
                procs,
                BlasProvider.IsDeterministicMode,
                hasEpilogue,
                options.PackingMode);

        int mcFromAutotune = autotuneMc;
        int ncFromAutotune = autotuneNc;
        int kcFromAutotune = autotuneKc;

        switch (strategy)
        {
            case PackingMode.ForcePackBoth:
                PackBothStrategy.Run<T>(
                    a, lda, transA,
                    b, ldb, transB,
                    c, ldc,
                    m, n, k,
                    mc: mcFromAutotune, nc: ncFromAutotune, kc: kcFromAutotune,
                    mr: mr, nr: nr,
                    options);
                break;
            case PackingMode.ForcePackAOnly:
                {
                    // Sub-D (#372) D.3: PackAOnly now has Avx2Fp32_8x8.RunStridedB.
                    // Use AVX2 8×8 for FP32 when shape aligns; else fall back to scalar 4×4.
                    int paoMr = 4, paoNr = 4;
                    if (typeof(T) == typeof(float)
                        && Avx2Fp32_8x8.IsSupported
                        && m % 8 == 0 && n % 8 == 0)
                    {
                        paoMr = 8;
                        paoNr = 8;
                    }
                    PackAOnlyStrategy.Run<T>(
                        a, lda, transA,
                        b, ldb,
                        c, ldc,
                        m, n, k,
                        mc: mcFromAutotune, kc: kcFromAutotune,
                        mr: paoMr, nr: paoNr,
                        options);
                }
                break;
            case PackingMode.ForceStreaming:
                // Streaming dispatches AVX2 internally — no Mr/Nr tiling parameter.
                StreamingStrategy.Run<T>(
                    a, lda, transA,
                    b, ldb, transB,
                    c, ldc,
                    m, n, k,
                    in options);
                break;
            case PackingMode.DisableAutotune:
                // DisableAutotune is a power-user override that bypasses the
                // autotune cache. The autotune dispatcher already returns the
                // heuristic decision when DisableAutotune is set, so mc/nc/kc
                // are already populated from the heuristic above.
                {
                    var defaultedOptions = new BlasOptions<T> { PackingMode = PackingMode.Auto };
                    PackingMode fallback = Dispatcher.SelectStrategy(m, n, k, defaultedOptions);
                    if (fallback == PackingMode.ForcePackAOnly && transB)
                        fallback = PackingMode.ForcePackBoth;
                    switch (fallback)
                    {
                        case PackingMode.ForcePackBoth:
                            PackBothStrategy.Run<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, mcFromAutotune, ncFromAutotune, kcFromAutotune, mr, nr, options);
                            break;
                        case PackingMode.ForcePackAOnly:
                            PackAOnlyStrategy.Run<T>(a, lda, transA, b, ldb, c, ldc, m, n, k, mcFromAutotune, kcFromAutotune, 4, 4, options);
                            break;
                        case PackingMode.ForceStreaming:
                            StreamingStrategy.Run<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
                            break;
                    }
                }
                break;
            case PackingMode.Auto:
                throw new InvalidOperationException("Dispatcher.SelectStrategy should never return Auto.");
            default:
                throw new NotSupportedException($"Unknown PackingMode: {strategy}");
        }

        // Apply fused epilogue chain (post-pass). Fast-path when no stages are active.
        // Copy epilogue to a local so we can pass it by-ref (ref structs cannot be passed
        // in from a property of another in ref struct directly).
        var epilogue = options.Epilogue;
        EpilogueChain.Apply<T>(c, ldc, m, n, in epilogue);
    }

    /// <summary>
    /// Pre-pack the A matrix into BLIS vpanel layout, returning a handle
    /// that subsequent <see cref="Gemm{T}"/> calls can reuse via
    /// <see cref="BlasOptions{T}.PackedA"/>. Eliminates the pack step on
    /// every iteration of a training loop (PyTorch CPU re-packs every call;
    /// this is the PyTorch-surpassing feature).
    ///
    /// <para>
    /// Call <see cref="WeightPackHandle.MarkDirty"/> after mutating the
    /// underlying weight tensor (e.g., from an optimizer step); the next
    /// Gemm call will re-pack before use.
    /// </para>
    /// </summary>
    /// <typeparam name="T">Element type. Must be float or double.</typeparam>
    /// <param name="a">Source A buffer, length depends on lda × M or lda × K (depending on transA).</param>
    /// <param name="lda">Leading dimension of A.</param>
    /// <param name="transA">True if A is stored as A^T.</param>
    /// <param name="m">Logical M dimension (rows of op(A)).</param>
    /// <param name="k">Logical K dimension (cols of op(A)).</param>
    /// <param name="options">Options — currently used for <see cref="BlasOptions{T}.PackingMode"/> selection.</param>
    public static WeightPackHandle PrePackA<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        int m, int k,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        var (mr, _) = PickMicrokernelTile<T>();
        const int PanelMc = 64;
        const int PanelKc = 64;
        // CodeRabbit #366: this routine currently packs a single Mc×Kc panel.
        // Silently truncating larger weights would produce wrong GEMM results
        // when the handle is reused, so reject m > PanelMc / k > PanelKc until
        // multi-panel packing lands. Callers below PanelMc / PanelKc are the
        // common training-loop case (linear projections, attention heads).
        if (m > PanelMc || k > PanelKc)
            throw new NotSupportedException(
                $"PrePackA currently supports only single-panel weights up to {PanelMc}×{PanelKc} " +
                $"(got m={m}, k={k}). Multi-panel pre-packing is tracked as follow-up work; " +
                $"call Gemm directly without a pre-pack handle for larger weights.");
        int mc = PanelMc;
        int kc = PanelKc;
        // Don't pack at a Kc larger than k.
        if (kc > k) kc = k;
        // Don't pack at an Mc larger than m, rounded down to mr.
        if (mc > m) mc = (m / mr) * mr;
        if (mc <= 0)
            throw new ArgumentException($"M={m} smaller than microkernel Mr={mr}; pre-pack not supported.", nameof(m));

        int elemSize = Marshal.SizeOf<T>();
        int packedBytes = mc * kc * elemSize;

        var key = (mc, kc, transA, options.PackingMode, typeof(T));
        var handle = WeightPackCache.Allocate(packedBytes, key, isForA: true);

        // Snapshot Version BEFORE packing so a concurrent MarkDirty doesn't
        // race the post-pack MarkCacheCurrent and mark stale data current.
        long packedVersion = Interlocked.Read(ref handle.Version);

        // Initial pack: write the source weight into handle.PackedBuffer.
        if (typeof(T) == typeof(double))
        {
            var packedSpan = MemoryMarshal.Cast<byte, double>(handle.PackedBuffer.AsSpan());
            ScalarPack.PackA<double>(
                MemoryMarshal.Cast<T, double>(a),
                lda, transA,
                packedSpan, mc, kc, mr);
        }
        else if (typeof(T) == typeof(float))
        {
            var packedSpan = MemoryMarshal.Cast<byte, float>(handle.PackedBuffer.AsSpan());
            ScalarPack.PackA<float>(
                MemoryMarshal.Cast<T, float>(a),
                lda, transA,
                packedSpan, mc, kc, mr);
        }
        else
        {
            throw new NotSupportedException($"PrePackA does not support T={typeof(T).Name}.");
        }

        WeightPackCache.MarkCacheCurrent(handle, packedVersion);
        return handle;
    }

    /// <summary>
    /// Pre-pack the B matrix into BLIS stripe layout. See <see cref="PrePackA{T}"/>
    /// for the broader rationale.
    /// </summary>
    /// <typeparam name="T">Element type. Must be float or double.</typeparam>
    /// <param name="b">Source B buffer, length depends on ldb × K or ldb × N (depending on transB).</param>
    /// <param name="ldb">Leading dimension of B.</param>
    /// <param name="transB">True if B is stored as B^T.</param>
    /// <param name="k">Logical K dimension (rows of op(B)).</param>
    /// <param name="n">Logical N dimension (cols of op(B)).</param>
    /// <param name="options">Options — currently used for <see cref="BlasOptions{T}.PackingMode"/> selection.</param>
    public static WeightPackHandle PrePackB<T>(
        ReadOnlySpan<T> b, int ldb, bool transB,
        int k, int n,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        var (_, nr) = PickMicrokernelTile<T>();
        const int PanelNc = 64;
        const int PanelKc = 64;
        // CodeRabbit #366: same single-panel limit as PrePackA — see that
        // method for rationale. Reject larger weights up front.
        if (n > PanelNc || k > PanelKc)
            throw new NotSupportedException(
                $"PrePackB currently supports only single-panel weights up to {PanelKc}×{PanelNc} " +
                $"(got k={k}, n={n}). Multi-panel pre-packing is tracked as follow-up work; " +
                $"call Gemm directly without a pre-pack handle for larger weights.");
        int nc = PanelNc;
        int kc = PanelKc;
        if (kc > k) kc = k;
        if (nc > n) nc = (n / nr) * nr;
        if (nc <= 0)
            throw new ArgumentException($"N={n} smaller than microkernel Nr={nr}; pre-pack not supported.", nameof(n));

        int elemSize = Marshal.SizeOf<T>();
        int packedBytes = nc * kc * elemSize;

        var key = (nc, kc, transB, options.PackingMode, typeof(T));
        var handle = WeightPackCache.Allocate(packedBytes, key, isForA: false);

        // Snapshot Version BEFORE packing — see PrePackA for rationale.
        long packedVersion = Interlocked.Read(ref handle.Version);

        if (typeof(T) == typeof(double))
        {
            var packedSpan = MemoryMarshal.Cast<byte, double>(handle.PackedBuffer.AsSpan());
            ScalarPack.PackB<double>(
                MemoryMarshal.Cast<T, double>(b),
                ldb, transB,
                packedSpan, nc, kc, nr);
        }
        else if (typeof(T) == typeof(float))
        {
            var packedSpan = MemoryMarshal.Cast<byte, float>(handle.PackedBuffer.AsSpan());
            ScalarPack.PackB<float>(
                MemoryMarshal.Cast<T, float>(b),
                ldb, transB,
                packedSpan, nc, kc, nr);
        }
        else
        {
            throw new NotSupportedException($"PrePackB does not support T={typeof(T).Name}.");
        }

        WeightPackCache.MarkCacheCurrent(handle, packedVersion);
        return handle;
    }

    /// <summary>
    /// Returns a snapshot of process-wide BlasManaged diagnostic counters
    /// (autotune hits/misses, JIT cache stats, weight pre-pack cache stats).
    /// </summary>
    public static BlasManagedStats GetStats() => BlasManagedStatsTracker.Snapshot();

    /// <summary>
    /// Reset diagnostic counters AND clear in-memory caches. The on-disk
    /// autotune cache persists across calls — this clears the counters only.
    /// </summary>
    public static void ClearCaches()
    {
        BlasManagedStatsTracker.Reset();
        JittedKernelCache.Clear();
        // Future: clear weight pre-pack cache entries that are still in memory.
    }

    /// <summary>
    /// Pick the microkernel (mr, nr) tile widths based on element type and
    /// runtime SIMD availability. The selection follows a four-tier hierarchy
    /// — AVX-512 → AVX2 → Neon → scalar — so the widest available vector ISA
    /// is used. On any host only one of {AVX-512, AVX2, Neon} can be active:
    /// AVX paths are x64-only; Neon is ARM64-only.
    /// <list type="bullet">
    ///   <item>FP64: (8, 16) AVX-512 → (4, 8) AVX2 → (4, 4) Neon or scalar.</item>
    ///   <item>FP32: (16, 16) AVX-512 → (8, 8) AVX2 → (8, 4) Neon → (4, 4) scalar.</item>
    /// </list>
    /// For FP64 the Neon tile (Mr=4, Nr=4) is identical to the scalar tile, so
    /// no new branch is needed — <see cref="PackBothStrategy"/> picks between
    /// <see cref="NeonFp64_4x4"/> and <see cref="ScalarFp64_4x4"/> at dispatch time.
    /// For FP32 the Neon tile (Mr=8, Nr=4) differs from scalar (4, 4) so it
    /// requires its own branch here.
    /// This selection drives the layout of packed-A and packed-B, so it must
    /// match the microkernel the strategy ultimately dispatches to.
    /// </summary>
    private static (int Mr, int Nr) PickMicrokernelTile<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            if (Avx512Fp64_8x16.IsSupported) return (8, 16);
            if (Avx2Fp64_4x8.IsSupported) return (4, 8);
            // Neon FP64 uses (4, 4) — same as scalar; DispatchMicrokernel picks the right kernel.
            return (4, 4);
        }
        if (typeof(T) == typeof(float))
        {
            if (Avx512Fp32_16x16.IsSupported) return (16, 16);
            if (Avx2Fp32_8x8.IsSupported) return (8, 8);
            if (NeonFp32_8x4.IsSupported) return (8, 4);
            return (4, 4);
        }
        return (4, 4);
    }
}
