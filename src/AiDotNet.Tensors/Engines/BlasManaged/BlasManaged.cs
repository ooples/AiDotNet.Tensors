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
    /// Sub-issue F (#374): when true, <see cref="Helpers.BlasProvider.TryGemm"/> and
    /// <see cref="Helpers.BlasProvider.TryGemmEx"/> route through <see cref="Gemm{T}"/>
    /// instead of the native cblas P/Invoke. Provides a single flag to opt all 144
    /// production call sites into managed dispatch — zero caller-side edits.
    ///
    /// <para>
    /// Defaults to <see langword="false"/> (native path). Supply-chain-conscious
    /// deployments set this to <see langword="true"/> at process startup to
    /// eliminate the libopenblas / MKL attack surface.
    /// </para>
    /// </summary>
    public static bool PreferManaged { get; set; } = false;

    /// <summary>
    /// Sub-F3 (#374 follow-up): when true, <see cref="Helpers.BlasProvider.TryGemmEx"/>
    /// consults <see cref="PrefersManagedCache"/> on each call. The cache measures
    /// both managed and native paths once per (shape, hardware) tuple and routes
    /// future calls to the winner.
    ///
    /// <para>
    /// Defaults to <see langword="false"/> (no autotune routing).
    /// <see cref="PreferManaged"/> takes precedence — when both are true,
    /// every call routes managed regardless of autotune. This lets supply-chain
    /// deployments override autotune (forcing managed dispatch unconditionally).
    /// </para>
    /// </summary>
    public static bool AutotuneRouting { get; set; } = false;

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

        // Sub-R (#408): GEMV fast path. M=1 (row × matrix), N=1 (matrix × col),
        // or K=1 (outer product) bypass the GEMM dispatcher entirely — per-call
        // dispatch overhead (~10-30 µs of strategy + autotune + tile-pick +
        // epilogue) dominates actual compute for these vector ops. Honours
        // PackingMode.Auto only; explicit pack-mode overrides take the regular
        // path so callers can force-test the strategy if needed.
        if (GemvKernel.QualifiesFor(m, n, k) && options.PackingMode == PackingMode.Auto)
        {
            GemvKernel.Run<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
            var gemvEpilogue = options.Epilogue;
            EpilogueChain.Apply<T>(c, ldc, m, n, in gemvEpilogue);
            return;
        }

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

        // Sub-D4 (#372 follow-up): partial-M/N tail splitting. When the shape is
        // big enough for the SIMD tile (m >= mr AND n >= nr) but not aligned
        // (m % mr != 0 OR n % nr != 0), split the work into:
        //   (a) aligned interior — fast path through the regular dispatcher
        //   (b) M-tail rows                 — small, Streaming
        //   (c) N-tail cols                 — small, Streaming
        //   (d) MN corner (only if both)    — tiny, Streaming
        // This avoids the entire shape falling to Streaming for a 1-7 row/col tail.
        // Sub-D D.1 had previously routed all of (a..d) through Streaming for
        // correctness, sacrificing the AVX2 perf on the aligned bulk. This restores it.
        //
        // Parallelism gate: PackBoth uses M-axis parallel; when m_aligned has too
        // few mr-blocks to keep all cores busy, the streaming-N path wins despite
        // the slower scalar microkernel because it parallelizes N-axis. Concrete
        // example: ResNet50_layer4 m=49 → m_aligned=48 → 6 mr-blocks; on 16 cores
        // most threads idle, so PackBoth ends up slower than Streaming with N
        // split. Gate: m_aligned >= procs * mr (every core gets at least one
        // mr-block).
        //
        // c.Clear() at the top of Gemm already zeroed all of c, so each sub-call
        // accumulates from zero on its disjoint region (Streaming reads-modifies-
        // writes; the recursive Gemm call re-clears its sub-slice, harmless).
        // Epilogue is applied once at the very end on the full (m, n) — sub-calls
        // get an inner options with Epilogue stripped to avoid double-application.
        int splitMAligned = m - (m % mr);
        int splitNAligned = n - (n % nr);
        // Honor NumThreads = -1 as explicit single-thread (deterministic mode).
        // Pre-fix: -1 fell into the "> 0 is false" branch and used all processors,
        // breaking determinism/perf expectations. PR #402 CodeRabbit fix.
        int splitProcs = options.NumThreads switch
        {
            -1 => 1,
            > 0 => options.NumThreads,
            _ => System.Environment.ProcessorCount,
        };
        if (m >= mr && n >= nr && (m % mr != 0 || n % nr != 0)
            && splitMAligned >= splitProcs * mr)
        {
            int m_aligned = splitMAligned;
            int n_aligned = splitNAligned;
            int m_tail = m - m_aligned;
            int n_tail = n - n_aligned;

            // Strip epilogue for sub-calls; apply once at the end on full (m, n).
            var innerOptions = new BlasOptions<T>
            {
                PackingMode = options.PackingMode,
                Workspace = options.Workspace,
                PackedA = options.PackedA,
                PackedB = options.PackedB,
                NumThreads = options.NumThreads,
                AutotuneKey = options.AutotuneKey,
                MaxJitCacheBytes = options.MaxJitCacheBytes,
                Mode = options.Mode,
                // Epilogue intentionally omitted.
            };

            // (a) Aligned interior — recursive Gemm on (m_aligned, n_aligned).
            // m_aligned % mr == 0 AND n_aligned % nr == 0 by construction, so the
            // recursion won't re-enter this branch.
            if (m_aligned > 0 && n_aligned > 0)
            {
                Gemm<T>(a, lda, transA, b, ldb, transB, c, ldc,
                        m_aligned, n_aligned, k, in innerOptions);
            }

            // (b) M-tail rows: [m_aligned, m) × [0, n_aligned)
            if (m_tail > 0 && n_aligned > 0)
            {
                var aTail = transA ? a.Slice(m_aligned) : a.Slice(m_aligned * lda);
                var cTail = c.Slice(m_aligned * ldc);
                StreamingStrategy.Run<T>(
                    aTail, lda, transA, b, ldb, transB,
                    cTail, ldc, m_tail, n_aligned, k, in innerOptions);
            }

            // (c) N-tail cols: [0, m_aligned) × [n_aligned, n)
            if (n_tail > 0 && m_aligned > 0)
            {
                var bTail = transB ? b.Slice(n_aligned * ldb) : b.Slice(n_aligned);
                var cTail = c.Slice(n_aligned);
                StreamingStrategy.Run<T>(
                    a, lda, transA, bTail, ldb, transB,
                    cTail, ldc, m_aligned, n_tail, k, in innerOptions);
            }

            // (d) Corner: [m_aligned, m) × [n_aligned, n)
            if (m_tail > 0 && n_tail > 0)
            {
                var aTail = transA ? a.Slice(m_aligned) : a.Slice(m_aligned * lda);
                var bTail = transB ? b.Slice(n_aligned * ldb) : b.Slice(n_aligned);
                var cTail = c.Slice(m_aligned * ldc + n_aligned);
                StreamingStrategy.Run<T>(
                    aTail, lda, transA, bTail, ldb, transB,
                    cTail, ldc, m_tail, n_tail, k, in innerOptions);
            }

            // Apply epilogue once on the full (m, n).
            var splitEpilogue = options.Epilogue;
            EpilogueChain.Apply<T>(c, ldc, m, n, in splitEpilogue);
            return;
        }

        // Shapes too small for the SIMD tile entirely: fall back to scalar 4×4.
        if (m < mr || n < nr || m % mr != 0)
        {
            mr = 4; nr = 4;
        }

        // If still misaligned after the scalar fallback (m < 4 or n < 4 or m%4 != 0):
        // route to Streaming. This is the D.1 correctness fix for genuinely tiny
        // misaligned shapes; the Sub-D4 split above handles m >= mr cases.
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
        // Honor NumThreads = -1 as explicit single-thread (matches splitProcs above).
        int procs = options.NumThreads switch
        {
            -1 => 1,
            > 0 => options.NumThreads,
            _ => Environment.ProcessorCount,
        };
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

        // Sub-E (#373): when a multi-panel pre-pack handle is supplied, override
        // the autotuner's tile choice to match the handle. Otherwise the strategy
        // would reject the handle (tile-size mismatch) and fall back to live pack,
        // making the pre-pack a no-op. The handle's tile sizes are authoritative
        // because they were baked into the packed-byte layout at PrePack time.
        if (options.PackedA != null && options.PackedA.MultiPanelStride > 0
            && options.PackedA.TileMc > 0 && options.PackedA.TileKc > 0)
        {
            mcFromAutotune = options.PackedA.TileMc;
            kcFromAutotune = options.PackedA.TileKc;
        }
        if (options.PackedB != null && options.PackedB.MultiPanelStride > 0
            && options.PackedB.TileMc > 0 && options.PackedB.TileKc > 0)
        {
            ncFromAutotune = options.PackedB.TileMc;  // PackedB stores Nc in TileMc slot
            kcFromAutotune = options.PackedB.TileKc;
        }

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
                    // Sub-D D.3 (FP32) + Sub-D2 (FP64): use AVX2 strided-B kernel
                    // when shape aligns to the tile width.
                    //   FP32 — Avx2Fp32_8x8: Mr=8, Nr=8 → requires m%8==0, n%8==0
                    //   FP64 — Avx2Fp64_4x8: Mr=4, Nr=8 → requires m%4==0, n%8==0
                    int paoMr = 4, paoNr = 4;
                    if (typeof(T) == typeof(float)
                        && Avx2Fp32_8x8.IsSupported
                        && m % 8 == 0 && n % 8 == 0)
                    {
                        paoMr = 8;
                        paoNr = 8;
                    }
                    else if (typeof(T) == typeof(double)
                        && Avx2Fp64_4x8.IsSupported
                        && m % 4 == 0 && n % 8 == 0)
                    {
                        paoMr = 4;
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
                            // PR #402 CodeRabbit fix: pass caller options through so
                            // NumThreads / Mode / Epilogue stays honored on this fallback
                            // path. Pre-fix this used the parameterless overload which
                            // silently dropped all caller-provided execution settings.
                            StreamingStrategy.Run<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, in options);
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

        // Sub-E (#373): multi-panel pre-pack. Pack the entire weight into
        // (numIcBlocks × numPcBlocks) tiles, each of size Mc × Kc. Strategies
        // index by (ic/Mc, pc/Kc) at runtime via the new MultiPanelStride
        // metadata on the handle.
        //
        // Tile dimensions match the autotuner's FallbackToHeuristic defaults
        // (Mc = Kc = 64) so the strategy naturally consumes the pre-pack on
        // shapes the autotuner hasn't trained yet. The Gemm dispatch also
        // honors handle dims via an override (search "TileMc" in BlasManaged.cs)
        // so larger pre-pack tiles still work — they just override autotune.
        int mc = 64;
        int kc = 64;
        // Round mc UP to a multiple of mr — same rationale as PrePackB below.
        // Pre-fix the second `if (mc > m) mc = m;` line clamped mc back DOWN
        // to m, defeating the alignment intent and causing OOB writes in
        // ScalarPack.PackA / Avx2Pack.PackA on shapes with m < mr (e.g., the
        // ExecuteIntoTests m=2 mr=4 matmul plan).
        if (mc > m) mc = ((m + mr - 1) / mr) * mr;
        if (kc > k) kc = k;
        if (mc <= 0)
            throw new ArgumentException($"M={m} smaller than microkernel Mr={mr}; pre-pack not supported.", nameof(m));

        int numIcBlocks = (m + mc - 1) / mc;
        int numPcBlocks = (k + kc - 1) / kc;
        int elemSize = Marshal.SizeOf<T>();
        // PR #402 CodeRabbit fix: guard against int overflow on large shapes.
        // mc * kc * elemSize alone can hit ~32 MB for double/64×64; multiplied
        // by numIcBlocks × numPcBlocks for a 100k×100k pre-pack the product
        // overflows int.MaxValue silently and produces an invalid alloc size.
        long tileBytes64 = (long)mc * kc * elemSize;
        long packedBytes64 = (long)numIcBlocks * numPcBlocks * tileBytes64;
        if (tileBytes64 <= 0 || tileBytes64 > int.MaxValue
            || packedBytes64 <= 0 || packedBytes64 > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                $"Packed buffer size ({packedBytes64} bytes) exceeds supported limits " +
                $"for m={m} k={k} mc={mc} kc={kc} elemSize={elemSize}.");
        }
        int tileBytes = (int)tileBytes64;
        int packedBytes = (int)packedBytes64;

        var key = (mc, kc, transA, options.PackingMode, typeof(T));
        var handle = WeightPackCache.Allocate(packedBytes, key, isForA: true);
        handle.FullM = m;
        handle.FullK = k;
        handle.TileMc = mc;
        handle.TileKc = kc;
        handle.NumIcBlocks = numIcBlocks;
        handle.NumPcBlocks = numPcBlocks;
        handle.MultiPanelStride = tileBytes;
        BlasManagedStatsTracker.AddPackCacheBytes(packedBytes);

        // Snapshot Version BEFORE packing so a concurrent MarkDirty doesn't
        // race the post-pack MarkCacheCurrent and mark stale data current.
        long packedVersion = Interlocked.Read(ref handle.Version);

        // Pack each (icIdx, pcIdx) tile at its offset in the multi-panel buffer.
        for (int icIdx = 0; icIdx < numIcBlocks; icIdx++)
        {
            int ic = icIdx * mc;
            int effectiveMc = Math.Min(mc, m - ic);
            for (int pcIdx = 0; pcIdx < numPcBlocks; pcIdx++)
            {
                int pc = pcIdx * kc;
                int effectiveKc = Math.Min(kc, k - pc);

                int tileOffsetBytes = (icIdx * numPcBlocks + pcIdx) * tileBytes;
                int aSliceOffset = transA ? pc * lda + ic : ic * lda + pc;

                if (typeof(T) == typeof(double))
                {
                    var packedSpan = MemoryMarshal.Cast<byte, double>(
                        handle.PackedBuffer.AsSpan(tileOffsetBytes, tileBytes));
                    ScalarPack.PackA<double>(
                        MemoryMarshal.Cast<T, double>(a).Slice(aSliceOffset),
                        lda, transA,
                        packedSpan, effectiveMc, effectiveKc, mr);
                }
                else if (typeof(T) == typeof(float))
                {
                    var packedSpan = MemoryMarshal.Cast<byte, float>(
                        handle.PackedBuffer.AsSpan(tileOffsetBytes, tileBytes));
                    ScalarPack.PackA<float>(
                        MemoryMarshal.Cast<T, float>(a).Slice(aSliceOffset),
                        lda, transA,
                        packedSpan, effectiveMc, effectiveKc, mr);
                }
                else
                {
                    throw new NotSupportedException($"PrePackA does not support T={typeof(T).Name}.");
                }
            }
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

        // Sub-E (#373): multi-panel pre-pack — see PrePackA above. Tile sizes
        // match the autotuner heuristic defaults (Nc = Kc = 64) for natural
        // consume on un-trained shapes.
        int nc = 64;
        int kc = 64;
        // Round nc UP to a multiple of nr (microkernel column tile width) — the
        // packed-buffer layout in ScalarPack.PackB / Avx2Pack.PackB always
        // writes ceil(nc / nr) * nr columns per Kc row, zero-padding the tail
        // stripe. Pre-fix the code below clamped nc back DOWN to n when n < nr,
        // so a [k×n] PrePackB with n=2 nr=4 would size the tile as kc*n*sizeof(T)
        // but ScalarPack.PackB would still write kc*nr*sizeof(T) (one full
        // zero-padded stripe), producing an IndexOutOfRangeException in the
        // packed-buffer span. Keep nc aligned UP and let effectiveNc bound the
        // logical read range in the per-tile inner loop.
        if (nc > n) nc = ((n + nr - 1) / nr) * nr;
        if (kc > k) kc = k;
        if (nc <= 0)
            throw new ArgumentException($"N={n} smaller than microkernel Nr={nr}; pre-pack not supported.", nameof(n));

        int numJcBlocks = (n + nc - 1) / nc;
        int numPcBlocks = (k + kc - 1) / kc;
        int elemSize = Marshal.SizeOf<T>();
        // PR #402 CodeRabbit fix: same int-overflow guard as PrePackA.
        long tileBytes64 = (long)kc * nc * elemSize;
        long packedBytes64 = (long)numJcBlocks * numPcBlocks * tileBytes64;
        if (tileBytes64 <= 0 || tileBytes64 > int.MaxValue
            || packedBytes64 <= 0 || packedBytes64 > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(n),
                $"Packed buffer size ({packedBytes64} bytes) exceeds supported limits " +
                $"for k={k} n={n} kc={kc} nc={nc} elemSize={elemSize}.");
        }
        int tileBytes = (int)tileBytes64;
        int packedBytes = (int)packedBytes64;

        var key = (nc, kc, transB, options.PackingMode, typeof(T));
        var handle = WeightPackCache.Allocate(packedBytes, key, isForA: false);
        handle.FullM = n;          // B-side: store n in FullM slot
        handle.FullK = k;
        handle.TileMc = nc;        // B-side: store nc in TileMc slot
        handle.TileKc = kc;
        handle.NumIcBlocks = numJcBlocks;  // B-side: jc-blocks live in NumIcBlocks slot
        handle.NumPcBlocks = numPcBlocks;
        handle.MultiPanelStride = tileBytes;
        BlasManagedStatsTracker.AddPackCacheBytes(packedBytes);

        long packedVersion = Interlocked.Read(ref handle.Version);

        for (int jcIdx = 0; jcIdx < numJcBlocks; jcIdx++)
        {
            int jc = jcIdx * nc;
            int effectiveNc = Math.Min(nc, n - jc);
            for (int pcIdx = 0; pcIdx < numPcBlocks; pcIdx++)
            {
                int pc = pcIdx * kc;
                int effectiveKc = Math.Min(kc, k - pc);

                int tileOffsetBytes = (jcIdx * numPcBlocks + pcIdx) * tileBytes;
                int bSliceOffset = transB ? jc * ldb + pc : pc * ldb + jc;

                if (typeof(T) == typeof(double))
                {
                    var packedSpan = MemoryMarshal.Cast<byte, double>(
                        handle.PackedBuffer.AsSpan(tileOffsetBytes, tileBytes));
                    ScalarPack.PackB<double>(
                        MemoryMarshal.Cast<T, double>(b).Slice(bSliceOffset),
                        ldb, transB,
                        packedSpan, effectiveNc, effectiveKc, nr);
                }
                else if (typeof(T) == typeof(float))
                {
                    var packedSpan = MemoryMarshal.Cast<byte, float>(
                        handle.PackedBuffer.AsSpan(tileOffsetBytes, tileBytes));
                    ScalarPack.PackB<float>(
                        MemoryMarshal.Cast<T, float>(b).Slice(bSliceOffset),
                        ldb, transB,
                        packedSpan, effectiveNc, effectiveKc, nr);
                }
                else
                {
                    throw new NotSupportedException($"PrePackB does not support T={typeof(T).Name}.");
                }
            }
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
