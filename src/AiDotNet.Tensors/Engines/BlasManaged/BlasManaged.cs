using System;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// BLIS-style managed GEMM kernel. Replaces Avx512Sgemm + SimdGemm as the
/// codebase's primary GEMM path. See docs/superpowers/specs/2026-05-16-blas-managed-design.md.
/// </summary>
public static partial class BlasManaged
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
    /// #403 Phase D: minimum M·N·K work for the Sub-S machine-code microkernel
    /// fast path to be worth taking. The machine kernel has higher fixed per-call
    /// overhead than the managed strategies (JIT-cache lookup + M-stripe parallel
    /// partition + unpacked-B streamed over K), so below this it loses to managed
    /// PackBoth/Streaming. Measured at the DCGAN conv-forward shape
    /// (M=64,N=49,K=1024 → 3.2M work): machine-code ~218-316µs vs managed ~60µs
    /// (~4-5× slower); machine-code becomes competitive by ~16M work. The
    /// conv-forward GEMM is the only NN (non-transposed) conv shape that reaches
    /// this gate — the backward GEMMs are transposed and route to Streaming. Set
    /// conservatively above the conv-forward shape but well below where the
    /// machine kernel's throughput dominates (large square GEMMs run 100+ GF/s
    /// here and stay on the fast path).
    /// </summary>
    internal const long MachineKernelMinWork = 4_000_000;

    // #653: extend the PackBoth parallelism-floor + wide-N blocking optimizations to the
    // DisableAutotune shim path the forward GEMM uses (CpuEngine.BatchMatMul -> 6-arg
    // SimdGemm.Sgemm -> Gemm{DisableAutotune}). Those optimizations were gated on
    // strategy == ForcePackBoth, so the forward FFN/QKV matmuls skipped them and capped
    // M-axis parallelism (512x1024x4096 -> mc=60/nc=512 -> numMB=9, ~9 of 16 cores).
    // With this on, the same shape gets nc widened to N + mc shrunk -> numMB=15, 4.9x -> 8.0x
    // at 16 cores (measured). Default OFF: it's a hot path the code's own notes flag as
    // regression-prone for multi-N shapes, and the dev box is too noisy to validate a flip —
    // gated for the gemm-bench CI A/B (set =1 to enable; flip the default once CI on a
    // many-core runner confirms no tuned-shape regression). Only affects the DisableAutotune
    // path; direct ForcePackBoth callers are unchanged (they already get these optimizations).
    private static readonly bool s_forwardPackBothBlocking =
        System.Environment.GetEnvironmentVariable("AIDOTNET_GEMM_FORWARD_PACKBOTH") == "1";

    // #653 cache-blocking sweep hook: env overrides for the PackBoth Mc/Nc/Kc so the
    // cache-residency-optimal blocking can be found empirically (the MKL/BLIS tuning art).
    // 0 = unset (use the computed value). Only applied on the effective-PackBoth path.
    private static readonly int s_envMc = EnvBlock("AIDOTNET_GEMM_MC");
    private static readonly int s_envNc = EnvBlock("AIDOTNET_GEMM_NC");
    private static readonly int s_envKc = EnvBlock("AIDOTNET_GEMM_KC");
    private static int EnvBlock(string name) =>
        int.TryParse(System.Environment.GetEnvironmentVariable(name), out var v) && v > 0 ? v : 0;

    /// <summary>
    /// #368 thin-M fast path bounds. The no-pack direct 6×16 parallel kernel
    /// (<see cref="Simd.SimdGemm.SgemmDirectParallelMInto"/>) beats both the
    /// machine-code path and native OpenBLAS for thin/moderate-M, N-aligned GEMMs
    /// (measured on a 32-thread AVX2 box, K=784, N=512: M128 464 vs OpenBLAS 335,
    /// M512 745 vs 523 GF/s; the general dispatch otherwise routes these to the
    /// #409 machine-code path at ~55 GF/s). It loses past these bounds, where the
    /// B re-stream cost dominates and the packed paths win (M2048 613 vs 692,
    /// N1024 510 vs 629). Bounds are deliberately conservative — only the proven
    /// winning box — to keep every other shape on its tuned path.
    /// </summary>
    private const int ThinMDirectMinM = 64;     // enough Mr=6 blocks to parallelize
    private const int ThinMDirectMaxM = 1024;   // above this the packed path wins
    private const int ThinMDirectMaxN = 512;    // above this B re-stream dominates
    private const int ThinMDirectMaxK = 1024;   // tested winning range
    // Tiny GEMMs (e.g. 72×72×48 ≈ 0.25M) gain nothing from the parallel direct kernel
    // and should stay on the strategy/autotune path (which learns + caches a winner for
    // repeated small shapes). The validated wins are at the MLP-layer scale (L1 ≈ 8.4M,
    // L0 ≈ 51M), so a 1M floor excludes only negligible shapes.
    private const long ThinMDirectMinWork = 1L << 20; // 1,048,576

#if NET5_0_OR_GREATER
    /// <summary>
    /// #368 thin-M fast path (see the call site in <see cref="Gemm{T}"/>). Routes the
    /// measured winning regime to the no-pack direct-parallel kernel
    /// (<see cref="Simd.SimdGemm.SgemmDirectParallelMInto"/> /
    /// <see cref="Simd.SimdGemm.DgemmDirectParallelMInto"/>) and returns true; returns
    /// false to fall through to the tuned strategy paths. Transposed operands are
    /// transposed into pooled scratch (cheap relative to the GEMM at thin-M, and far
    /// faster than the ~57 GF/s the packed/strategy path gives transposed thin-M); a
    /// fused bias/activation epilogue is applied after the GEMM. The kernels write
    /// disjoint output rows in fixed K order, so the result is deterministic across
    /// thread counts.
    /// </summary>
    private static bool TryThinMDirect<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc, int m, int n, int k,
        in BlasOptions<T> options) where T : unmanaged
    {
        if (!(typeof(T) == typeof(float) || typeof(T) == typeof(double))) return false;
        if (!System.Runtime.Intrinsics.X86.Fma.IsSupported) return false;
        if (options.PackingMode != PackingMode.Auto && options.PackingMode != PackingMode.DisableAutotune) return false;
        if (options.PackedA is not null || options.PackedB is not null) return false;
        // both-transposed is rare and would need strided A *and* strided B (no
        // contiguous vector dimension) — left on the strategy.
        if (transA && transB) return false;
        if (ldc != n || (n & 7) != 0) return false;
        if (m < ThinMDirectMinM || m > ThinMDirectMaxM || n > ThinMDirectMaxN || k > ThinMDirectMaxK) return false;
        if ((long)m * n * k < ThinMDirectMinWork) return false; // tiny GEMMs stay on the strategy/autotune path
        // Each operand must be contiguous in its stored (possibly transposed) layout:
        // !transA → A is [m,k] (lda=k); transA → Aᵀ is [k,m] (lda=m). Likewise B.
        if (lda != (transA ? m : k)) return false;
        if (ldb != (transB ? k : n)) return false;

        bool isFloat = typeof(T) == typeof(float);
        if (transA)
        {
            // Aᵀ·B via the strided-A kernel (no transpose materialised).
            if (isFloat)
                Simd.SimdGemm.SgemmDirectParallelMIntoTransA(
                    MemoryMarshal.Cast<T, float>(a), MemoryMarshal.Cast<T, float>(b),
                    MemoryMarshal.Cast<T, float>(c), m, k, n);
            else
                Simd.SimdGemm.DgemmDirectParallelMIntoTransA(
                    MemoryMarshal.Cast<T, double>(a), MemoryMarshal.Cast<T, double>(b),
                    MemoryMarshal.Cast<T, double>(c), m, k, n);
        }
        else if (transB)
        {
            // A·Bᵀ via the NT dot-product kernel (Bstored rows are contiguous).
            if (isFloat)
                Simd.SimdGemm.SgemmDirectParallelMIntoTransB(
                    MemoryMarshal.Cast<T, float>(a), MemoryMarshal.Cast<T, float>(b),
                    MemoryMarshal.Cast<T, float>(c), m, k, n);
            else
                Simd.SimdGemm.DgemmDirectParallelMIntoTransB(
                    MemoryMarshal.Cast<T, double>(a), MemoryMarshal.Cast<T, double>(b),
                    MemoryMarshal.Cast<T, double>(c), m, k, n);
        }
        else if (isFloat)
            Simd.SimdGemm.SgemmDirectParallelMInto(
                MemoryMarshal.Cast<T, float>(a), MemoryMarshal.Cast<T, float>(b),
                MemoryMarshal.Cast<T, float>(c), m, k, n);
        else
            Simd.SimdGemm.DgemmDirectParallelMInto(
                MemoryMarshal.Cast<T, double>(a), MemoryMarshal.Cast<T, double>(b),
                MemoryMarshal.Cast<T, double>(c), m, k, n);

        // Fused bias/activation epilogue applied after the GEMM (a cheap elementwise
        // pass) — lets thin-M FusedLinear hit the fast kernel instead of the strategy.
        var epi = options.Epilogue;
        EpilogueChain.Apply<T>(c, ldc, m, n, in epi);
        return true;
    }
#endif

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
    /// Sub-F3 (#374 follow-up): when true, the <see cref="Helpers.BlasProvider"/>
    /// <c>TryGemm</c> / <c>TryGemmEx</c> entry points consult
    /// <see cref="PrefersManagedCache"/> in NON-deterministic mode. The cache measures
    /// both managed and native paths once per (shape, hardware) tuple (disk-persisted)
    /// and routes future calls to the faster one — managed where it wins, native where
    /// it wins — so flipping toward managed never costs throughput.
    ///
    /// <para>
    /// Defaults to <see langword="true"/> (ManagedBlas deterministic-parallel +
    /// non-deterministic best-of is the intended CPU GEMM behavior; native OpenBLAS is
    /// kept as an optional per-shape accelerator the router picks where it still wins).
    /// Precedence is resolved in <see cref="Helpers.BlasProvider.ShouldRouteManaged"/>:
    /// <see cref="PreferManaged"/> and <see cref="Helpers.BlasProvider.IsDeterministicMode"/>
    /// both force managed BEFORE this timing-based routing is consulted — deterministic
    /// mode must not pick its kernel by measurement noise (it would break
    /// bit-reproducibility), so autotune only governs the non-deterministic path.
    /// </para>
    /// </summary>
    public static bool AutotuneRouting { get; set; } = true;

    // #375: the earlier opt-in Gemm-level autotune (EnableAutotuneV2 + AutotuneCacheV2 +
    // a synchronous warmup sweep) was superseded by the hybrid hardware-aware strategy
    // selection — a per-fingerprint, disk-persistent, non-blocking learned cache wired into
    // Dispatcher.SelectStrategy (see StrategyDefaultTable + BlasManagedAutotune.TryLookupStrategy
    // + BackgroundAutotuner). The hybrid does the same job strictly better (persistent,
    // default-on, no first-call sweep cost), so the redundant V2 path was removed.

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
    // Shapes where the cache-blocking packed strategy (Dispatcher → PackBoth) beats the
    // #409 machine-code microkernel in PRODUCTION (deterministic) mode, so the Sub-S
    // interception below should DEFER to the strategy path instead. Measured by
    // ManagedVsNativeGemmAudit (perf/managed-blas-vs-native-audit) on a 32-core Ryzen:
    //   • thin-N (n < 128, k ≥ 128): the 6×Nr microkernel can't fill its Nr-wide tile when
    //     n is tiny → idle SIMD lanes. Packed wins ~3.6× for BOTH dtypes (4096×512×16:
    //     float 22→80, double 14→50 GF/s).
    //   • very large work (≥ 4e9): aggressive cache blocking pays for both dtypes
    //     (2048³: float 290→462, double 285→319). 1024³ (1.07e9) stays on the microkernel
    //     (153>81), so the cutoff is above it.
    //   • FP32-only wide-N / wide-K skew (FFN): packed wins big (512×512×2048 106→174;
    //     512×2048×512 101→177). FP64 is EXCLUDED — its microkernel BEATS packed on FFN
    //     (512×512×2048 double: 187 microkernel vs 80 packed), so FP64 falls through to the
    //     microkernel for these shapes.
    // The microkernel still wins (and is kept) for small/medium near-square shapes
    // (512³, 256×784×128, 128×64×256), where its low per-call overhead dominates.
    /// <summary>
    /// #475 tiny-K routing. When true (default), FP32 GEMMs with a small contraction
    /// dimension (K ≤ 128) but large M·N — the 1×1-conv / im2col shapes that dominate
    /// diffusion (e.g. 13824×1536×64) — are routed to the machine-code kernel instead of
    /// the cache-blocking packed strategy, whose A/B packing cost does not amortize over a
    /// tiny K. Measured on Kandinsky train (A/B in one process, MaxDOP=4): the 13824×1536×64
    /// shape went from 1.23s @ ~31 GF/s on the packed path to fast enough to drop off the
    /// top-8 (machine kernel ~5×), cutting total GEMM time 18.4s → 16.4s/iter. Settable for
    /// A/B regression testing.
    /// </summary>
    public static bool TinyKPreferMachineKernel = true;

    private static bool PrefersStrategyOverMachineKernel(int m, int n, int k, bool isFloat)
    {
        long work = (long)m * n * k;
        // #475: tiny-K, large-M·N FP32 (1×1-conv / im2col) — let the machine kernel take it.
        // Checked before the work>=250M cutoff that would otherwise force the packed path.
        if (isFloat && TinyKPreferMachineKernel && k <= 128 && (long)m * n >= 1_000_000L)
            return false;
        // #475 DEAD END (measured neutral, do not re-add): routing mid-large FP32 (≥250M)
        // to the machine kernel instead of PackBoth left the diffusion FFN GEMMs (e.g.
        // 384×4096×3456) at the SAME ~122-150 GF/s at MaxDOP=4 — both kernels hit the same
        // ~4-thread ceiling, so the FFN gap is raw kernel efficiency vs MKL, not routing.
        if (n < 128 && k >= 128) return true;                 // thin-N (both dtypes)
        if (!isFloat)
            // FP64 microkernel is competitive on FFN and mid squares (1024³ double 211 > packed
            // 172); packed only pays at very large work (2048³: 316 -> 323). So FP64 defers only
            // for thin-N (above) and the very-large tail.
            return work >= 4_000_000_000L;
        // FP32: the cache-blocking packed strategy wins on all mid-large work — squares from
        // ~640³ (microkernel 110 -> packed ~192 GF/s; 1024³ 178 -> 315; stable across runs) and
        // FFN (106 -> 174) — and on wide-N/wide-K skew. The microkernel only leads on SMALL
        // near-square shapes (512³ 552, 256×784×128 674), where packing overhead would crater
        // the packed path (512³ packed is only 45 GF/s). The cutoff sits just below 640³
        // (2.6e8) and safely above 512³ (1.3e8) — conservative because mis-routing a small shape
        // to the packed path is a ~12× loss, while mis-keeping a mid shape on the microkernel is mild.
        if (work >= 250_000_000L) return true;                // FP32 mid-large (squares ≥ ~640³, FFN)
        if (n >= 1024 && n > 2L * m) return true;             // FP32 wide-N skew below the cutoff
        if (k >= 1024 && k > 2L * m) return true;             // FP32 wide-K skew below the cutoff
        return false;
    }

    // Diagnostic timing wrapper (#475 shape audit). When GemmShapeHistogram.Enabled it times
    // only the TOP-LEVEL Gemm call (a ThreadStatic guard skips internal re-dispatches so a
    // shape is attributed once, to the outer call). Disabled → a single bool read + straight
    // tail-call into GemmCore, no Stopwatch, no measurable overhead.
    [ThreadStatic] private static bool s_gemmTiming;

    public static void Gemm<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (!GemmShapeHistogram.Enabled || s_gemmTiming)
        {
            GemmCore(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, in options);
            return;
        }
        s_gemmTiming = true;
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        try
        {
            GemmCore(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, in options);
        }
        finally
        {
            long elapsed = System.Diagnostics.Stopwatch.GetTimestamp() - start;
            s_gemmTiming = false;
            if (m > 0 && n > 0 && k > 0)
                GemmShapeHistogram.Record(m, n, k, transA, transB, typeof(T) == typeof(float), elapsed);
        }
    }

    private static void GemmCore<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (m <= 0 || n <= 0 || k <= 0) return;

        // Thin-M pre-packed B re-dispatch: the fast thin-M / machine-code kernels below cannot
        // consume pre-packed tiles (they're gated on PackedB == null), so a pre-packed thin-M GEMM
        // falls to the slow tuned strategy. That strategy parallelizes over the M-axis, so with few
        // M-rows AND heavy per-row work (large N·K) it leaves cores idle — measured ~7× slower than
        // fresh-pack at M=8/N=1024/K=1024 (PrePackSpeedupTest). Re-dispatch without the handle so it
        // takes the fast fresh-pack path (machine-code kernel parallelizes the N-axis); the result is
        // bit-identical (packing is only a memory-layout optimization). The per-row-work guard keeps
        // the handle for small-N·K thin-M GEMMs (e.g. M=32 K=128 N=64) where the tuned strategy is
        // already fast and dropping it would skip the pre-pack consumption other callers rely on.
        const long ThinMPrePackDropMinRowWork = 1L << 16; // 64K = N·K above which thin-M pre-pack stalls
        if (options.PackedB is not null && options.PackedA is null && m < ThinMDirectMinM
            && (long)n * k >= ThinMPrePackDropMinRowWork
            && !transA && !transB && options.PackingMode == PackingMode.Auto)
        {
            Gemm<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k,
                new BlasOptions<T>
                {
                    PackingMode = options.PackingMode,
                    Epilogue = options.Epilogue,
                    Workspace = options.Workspace,
                    PackedA = options.PackedA,
                    PackedB = null,
                    NumThreads = options.NumThreads,
                    AutotuneKey = options.AutotuneKey,
                    MaxJitCacheBytes = options.MaxJitCacheBytes,
                    Mode = options.Mode,
                });
            return;
        }

        // Strategies do read-modify-write on C; zero it here so the first call
        // produces C = A · B (not C += A · B). Future versions may expose a
        // beta=1 option that skips this zero, but Phase B's contract is C := A · B.
        //
        // Clear only the logical [m, n] output tile, NOT the whole span: with an
        // ldc-padded or offset-based caller the backing span extends past the
        // tile, and c.Clear() would clobber data the caller owns outside it.
        if (ldc == n)
        {
            // Contiguous rows — the tile is a single [0, m*n) run.
            c.Slice(0, m * n).Clear();
        }
        else
        {
            for (int row = 0; row < m; row++)
                c.Slice(row * ldc, n).Clear();
        }

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

#if NET5_0_OR_GREATER
        // #368 thin-M fast path: BlasManaged's general dispatch parallelises thin-M
        // GEMM poorly — at the AIsEval MLP L0 128×784×512 float falls to the #409
        // machine-code path (~55 GF/s) and double stays ~60, both LOSING to a no-pack
        // direct parallel kernel that splits disjoint output-row blocks (float 6×16
        // ~400-464 GF/s, beating OpenBLAS ~335; double 4×8 ~172). Disjoint rows →
        // bit-deterministic across thread counts (no K-split → no Deterministic-mode
        // concern). Also applies a fused bias/activation epilogue after the GEMM, so
        // thin-M FusedLinear hits the fast kernel. Bounded to the measured winning
        // regime; the Force* pack modes (caller pinned a strategy), pre-packed operands,
        // transposed (its serial transpose regresses thin-M — see TryThinMDirect),
        // out-of-box and non-FMA shapes fall through to the tuned strategy paths.
        if (TryThinMDirect<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, in options))
            return;
#endif

        // Sub-S (#409): machine-code microkernel fast path. The tile-aligned interior
        // runs on the hand-emitted kernel (FP64 6×8 ~57 GFLOPS/core ~95% of OpenBLAS;
        // FP32 6×16 — multithreaded, first-party machine code, no dependency); the
        // m%Mr / n%Nr tails go to Streaming. Honours PackingMode.Auto only (explicit
        // pack-mode overrides take the managed path so callers can force-test a
        // strategy), no transpose, no epilogue, no pre-pack. C is already zeroed
        // above (the kernel accumulates), and the tails read-modify-write their own
        // disjoint sub-tiles — so this can only add speed, never change results.
        if (options.PackingMode == PackingMode.Auto && !transA && !transB
            && options.PackedA is null && options.PackedB is null)
        {
            int mkMr = 0, mkNr = 0;
            bool mkAvail = false;
            if (typeof(T) == typeof(double) && MachineKernelGemm.IsFp64Available)
            { mkMr = MachineKernelGemm.Fp64Mr; mkNr = MachineKernelGemm.ActiveFp64Nr; mkAvail = true; }
            else if (typeof(T) == typeof(float) && MachineKernelGemm.IsFp32Available)
            { mkMr = MachineKernelGemm.Fp32Mr; mkNr = MachineKernelGemm.ActiveFp32Nr; mkAvail = true; }

            var epi409 = options.Epilogue;
            if (mkAvail && m >= mkMr && n >= mkNr
                && (long)m * n * k >= MachineKernelMinWork
                && !PrefersStrategyOverMachineKernel(m, n, k, typeof(T) == typeof(float))
                && EpilogueFlagsCompute.Compute(in epi409) == EpilogueFlags.None)
            {
                int mAl = m - (m % mkMr); // interior rows (× Mr)
                int nAl = n - (n % mkNr); // interior cols (× Nr)
                bool ran = typeof(T) == typeof(double)
                    ? MachineKernelGemm.TryGemmFp64(
                        MemoryMarshal.Cast<T, double>(a), lda, false,
                        MemoryMarshal.Cast<T, double>(b), ldb, false,
                        MemoryMarshal.Cast<T, double>(c), ldc, mAl, nAl, k, false, false)
                    : MachineKernelGemm.TryGemmFp32(
                        MemoryMarshal.Cast<T, float>(a), lda, false,
                        MemoryMarshal.Cast<T, float>(b), ldb, false,
                        MemoryMarshal.Cast<T, float>(c), ldc, mAl, nAl, k, false, false);
                if (ran)
                {
                    int mTail = m - mAl, nTail = n - nAl;
                    // (b) M-tail rows [mAl, m) × [0, nAl)
                    if (mTail > 0 && nAl > 0)
                        StreamingStrategy.Run<T>(a.Slice(mAl * lda), lda, false, b, ldb, false,
                            c.Slice(mAl * ldc), ldc, mTail, nAl, k, in options);
                    // (c) N-tail cols [0, mAl) × [nAl, n)
                    if (nTail > 0 && mAl > 0)
                        StreamingStrategy.Run<T>(a, lda, false, b.Slice(nAl), ldb, false,
                            c.Slice(nAl), ldc, mAl, nTail, k, in options);
                    // (d) corner [mAl, m) × [nAl, n)
                    if (mTail > 0 && nTail > 0)
                        StreamingStrategy.Run<T>(a.Slice(mAl * lda), lda, false, b.Slice(nAl), ldb, false,
                            c.Slice(mAl * ldc + nAl), ldc, mTail, nTail, k, in options);
                    return;
                }
            }
        }

        // #375 hybrid: pass transA/transB so the learned-strategy cache key matches the
        // actual call (the shapes reaching strategy selection are typically transposed —
        // Sub-S already handled non-transposed aligned GEMM above).
        PackingMode strategy = Dispatcher.SelectStrategy(m, n, k, transA, transB, in options);

        // PackAOnly does not support transB=true in Phase B — fall back to
        // PackBoth (which absorbs the transpose into its pack) or Streaming
        // (which handles transB via in-place index branching).
        if (strategy == PackingMode.ForcePackAOnly && transB)
            strategy = PackingMode.ForcePackBoth;

        // Sub-G (#375): Streaming per-call-overhead fast path. Streaming streams A
        // and B in place — it uses no packing/blocking params (mc/nc/kc) and no
        // microkernel tile (mr/nr), and handles its own M/N tails internally. So
        // for any shape SelectStrategy routes to Streaming (small shapes above the
        // TinyShapeWorkThreshold, thin-K tall-skinny, etc.), the PickMicrokernelTile
        // + partial-M/N split + AutotuneDispatcher.Decide work below is pure
        // overhead — and Decide does a JSON disk write on a cache miss, which on a
        // ~60 µs streaming GEMM dwarfs the actual compute. Dispatch directly here.
        if (strategy == PackingMode.ForceStreaming)
        {
            StreamingStrategy.Run<T>(
                a, lda, transA,
                b, ldb, transB,
                c, ldc,
                m, n, k,
                in options);
            var streamingFastEpilogue = options.Epilogue;
            EpilogueChain.Apply<T>(c, ldc, m, n, in streamingFastEpilogue);
            return;
        }

        // Pick SIMD-aware (mr, nr) for PackBoth using the AVX-512 → AVX2 → Neon → scalar
        // hierarchy. Fall back to (4, 4) scalar only when the shape is too small for
        // the SIMD tile to even start (m < mr or n < nr), or when m is not an exact
        // multiple of mr (M-tail kernels deferred to a future phase).
        // When m % mr == 0 but n % nr != 0, PackBothStrategy handles the partial-N
        // column-block via Avx2Tail/Avx512Tail (Task G2).
        // PackAOnly always uses scalar (4, 4) because its strided-B path has no
        // AVX2/AVX-512 RunStridedB variant yet (deferred to Phase Cx).
        //
        // #409 S.3: the live (non-pre-packed) path takes the mode-gated tile (Fast → 6×16);
        // a supplied pre-packed handle must consume with the SAME 8×8 tile it was packed
        // against (PickMicrokernelTilePrePack), independent of mode.
        var (mr, nr) = (options.PackedB is null && options.PackedA is null)
            ? PickMicrokernelTile<T>()
            : PickMicrokernelTilePrePack<T>();

        // The FP32 6×16 kernel is PackBoth-ONLY: PackAOnly's strided-B path uses an 8×8
        // (Avx2Fp32_8x8) tile, so the tail-split alignment and packing layout must stay on 8×8
        // when the dispatcher picked PackAOnly. Without this, the interior is aligned to mr=6 and
        // then handed to PackAOnly, which needs a multiple of 8 and slices out of range
        // (ArgumentOutOfRangeException). PackBoth keeps the faster 6×16 tile.
        if (strategy == PackingMode.ForcePackAOnly && typeof(T) == typeof(float)
            && mr == 6 && Avx2Fp32_8x8.IsSupported)
        {
            mr = 8;
            nr = 8;
        }

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
        // CORRECTNESS (reproducibility): the tail-split decomposition — aligned interior via the
        // SIMD microkernel + Streaming for the M/N tails — picks which kernel computes each output
        // element. It MUST be a function of the shape and the MACHINE only, never the per-call
        // thread count, or the same shape takes a different path (and a different microkernel) at
        // different thread counts and the output stops being bit-identical across them, violating
        // the deterministic-mode contract. The old gate used options.NumThreads, so a tailed shape
        // tail-split at NumThreads=1 (interior on the fast kernel) but fell through to all-Streaming
        // at NumThreads=N — different bits. This was a LATENT reproducibility bug that the FP32 6×16
        // tile (Mr=6 leaves an M-tail on most shapes) exposes; 8×8 only escaped because the test
        // shapes are 8-aligned. Gate on the fixed machine core count instead — a thread-count-
        // INVARIANT measure of "is the aligned interior big enough for its fast kernel to beat
        // all-Streaming?" The actual parallel degree still honours options.NumThreads downstream.
        int gateProcs = System.Environment.ProcessorCount;
        if (m >= mr && n >= nr && (m % mr != 0 || n % nr != 0))
        {
            if (splitMAligned < gateProcs * mr && options.PackingMode == PackingMode.Auto)
            {
                // Auto + too few aligned mr-blocks for the packed kernel to beat all-Streaming,
                // and a tail IS present — route the WHOLE shape to Streaming, which handles the
                // m%mr / n%nr tails internally and parallelizes the N axis. Decided by the fixed
                // machine core count (NOT options.NumThreads), so the same shape takes this path
                // at every thread count → bit-identical across thread counts. Force* pack modes
                // skip this and tail-split below (the caller explicitly asked for that strategy);
                // the tail-split keeps PackBoth's "caller guarantees m%mr==0" contract intact (it
                // never receives an unaligned m, which it would silently skip).
                StreamingStrategy.Run<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k, in options);
                var streamWholeEpi = options.Epilogue;
                EpilogueChain.Apply<T>(c, ldc, m, n, in streamWholeEpi);
                return;
            }

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

        // #653: the parallelism-floor + A-repack-reduction blocking optimizations below were gated
        // on strategy == ForcePackBoth, so they were SKIPPED for the DisableAutotune shim path the
        // forward GEMM uses (CpuEngine.BatchMatMul -> 6-arg SimdGemm.Sgemm -> Gemm{DisableAutotune}).
        // That left the FFN/QKV forward matmuls at the autotuner's cache-sized mc (e.g. the FFN
        // 512×1024×4096 -> mc=60/nc=512 -> numMB=9, ~9 of 16 cores busy). Resolve the EFFECTIVE
        // PackBoth decision (the same fallback the DisableAutotune switch case computes at dispatch
        // time) so these blocking optimizations fire on that path too — the DisableAutotune case
        // reads the mcFromAutotune/ncFromAutotune we adjust here.
        bool effectivePackBoth = strategy == PackingMode.ForcePackBoth;
        if (s_forwardPackBothBlocking && strategy == PackingMode.DisableAutotune)
        {
            var packBothProbe = new BlasOptions<T> { PackingMode = PackingMode.Auto };
            PackingMode effStrategy = Dispatcher.SelectStrategy(m, n, k, packBothProbe);
            if (effStrategy == PackingMode.ForcePackAOnly && transB) effStrategy = PackingMode.ForcePackBoth;
            effectivePackBoth = effStrategy == PackingMode.ForcePackBoth;
        }

        // Parallelism floor (PackBoth, no pre-pack): the autotuner sizes mc/nc for cache, but
        // on high core counts a shape can yield fewer (m/mc)×(n/nc) blocks than cores, leaving
        // cores idle. When N fits a single nc-block the 2D MN-grid can't add N-parallelism
        // (PackBothStrategy handles the multi-N-block case), so the only lever is mc. Shrink it
        // toward ~procs M-blocks — bounded to a multiple of mr (panels stay tile-aligned) and
        // never below 2·mr — so e.g. FFN-down 512×2048×512 (mc=64 → 8 M-blocks on 32 cores)
        // parallelizes across all cores via the shared-B M-axis path (no redundant B-pack).
        //
        // Gated to LARGE K (≥1024): a smaller mc means more A-panels and thus more pack passes,
        // which only pays off when K is large enough to amortize the pack over the FMA work.
        // On small-K shapes (e.g. 640³, K=640) the extra pack overhead dominates and the shrink
        // REGRESSES — so leave those on the autotuner's larger mc.
        //
        // NOTE: this is deliberately limited to numNB==1 (genuinely under-parallelized, ≤8 blocks).
        // Extending it to multi-N-block shapes (e.g. 1024³, mc=64 → 16 M-blocks) REGRESSED them:
        // the box is 16 PHYSICAL cores (32 SMT threads) and compute-bound GEMM gets nothing from
        // SMT, so 16 M-blocks already saturates the cores — shrinking mc only shrinks the panels.
        if (effectivePackBoth && k >= 1024
            && options.PackedA is null && options.PackedB is null && procs > 1)
        {
            int numNB = (n + ncFromAutotune - 1) / ncFromAutotune;
            int numMB = (m + mcFromAutotune - 1) / mcFromAutotune;
            if (numNB == 1 && numMB < procs && mcFromAutotune > 2 * mr)
            {
                int targetMc = Math.Max(2 * mr, (m + procs - 1) / procs);
                targetMc = ((targetMc + mr - 1) / mr) * mr; // round up to a whole mr tile
                if (targetMc < mcFromAutotune) mcFromAutotune = targetMc;
            }
        }

        // A-repack reduction (run AFTER the floor so it doesn't perturb the floor's numNB check):
        // the M-axis path re-packs A[ic,pc] once per nc-panel because the jc loop is outermost, so
        // numNB nc-blocks means A is packed numNB times. The autotuner sizes nc for L2 B-residency
        // (~512), but the kernel only ever reads a Kc×Nr B micro-panel (L1-sized) at a time — the
        // full Kc×Nc B-panel merely has to fit L3. So widen nc to span all of N when the Kc×N panel
        // fits an L3 budget: numNB collapses to 1 and A is packed ONCE (this is why OpenBLAS uses a
        // large Nc). Pure win — fewer pack passes, identical reduction order (bit-exact).
        if (effectivePackBoth
            && options.PackedA is null && options.PackedB is null && m >= 256)
        {
            int elemSz = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
            const long L3PanelBudgetBytes = 8L * 1024 * 1024;
            long fullPanelBytes = (long)kcFromAutotune * n * elemSz;
            if (n > ncFromAutotune && fullPanelBytes <= L3PanelBudgetBytes)
            {
                ncFromAutotune = ((n + nr - 1) / nr) * nr; // span all of N → numNB == 1, A packed once

                // With numNB now == 1, A is packed ONCE regardless of mc (total A-pack = M×K either
                // way), so the small-mc A-repack penalty that the floor avoids no longer applies.
                // That lets a wide-N moderate-M shape (e.g. FFN-up 512×512×2048, numMB=8) fill the
                // cores via the shared-B M-axis with NO redundant B-pack — strictly better than the
                // 2D grid, which re-packs B per ic-block. Shrink mc to ~physical M-blocks when under.
                int physicalCores = Math.Max(1, procs / 2); // 2-way SMT assumption
                int numMBwide = (m + mcFromAutotune - 1) / mcFromAutotune;
                if (numMBwide < physicalCores && mcFromAutotune > 2 * mr)
                {
                    int targetMc = Math.Max(2 * mr, (m + physicalCores - 1) / physicalCores);
                    targetMc = ((targetMc + mr - 1) / mr) * mr; // round up to a whole mr tile
                    if (targetMc < mcFromAutotune) mcFromAutotune = targetMc;
                }
            }
        }

        // #653 cache-blocking sweep hook: env overrides applied last, so a sweep can probe the
        // cache-residency-optimal (Mc, Nc, Kc) directly. Rounded to the micro-tile so the packed
        // panels stay tile-aligned. Effective-PackBoth only (the path the forward GEMM runs).
        if (effectivePackBoth && (s_envMc > 0 || s_envNc > 0 || s_envKc > 0))
        {
            if (s_envMc > 0) mcFromAutotune = ((s_envMc + mr - 1) / mr) * mr;
            if (s_envNc > 0) ncFromAutotune = Math.Min(((s_envNc + nr - 1) / nr) * nr, ((n + nr - 1) / nr) * nr);
            if (s_envKc > 0) kcFromAutotune = Math.Min(s_envKc, k);
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
                    // when shape aligns to the tile width. Respect PickMicrokernelTile's
                    // mode-based selection (Fast → 6×8, Deterministic → 4×8 for FP64).
                    //   FP32 — Avx2Fp32_8x8: Mr=8, Nr=8 → requires m%8==0, n%8==0
                    //   FP64 — mode-gated: (6,8) Fast or (4,8) Deterministic → requires m%(6or4)==0, n%8==0
                    int paoMr = 4, paoNr = 4;
                    if (typeof(T) == typeof(float)
                        && Avx2Fp32_8x8.IsSupported
                        && m % 8 == 0 && n % 8 == 0)
                    {
                        paoMr = 8;
                        paoNr = 8;
                    }
                    else if (typeof(T) == typeof(double))
                    {
                        // Query PickMicrokernelTile to respect Fast/Deterministic mode.
                        var (tileMr, tileNr) = PickMicrokernelTile<T>();
                        if (tileMr == 6 && tileNr == 8 && Avx2Fp64_6x8.IsSupported
                            && m % 6 == 0 && n % 8 == 0)
                        {
                            paoMr = 6;
                            paoNr = 8;
                        }
                        else if (Avx2Fp64_4x8.IsSupported
                            && m % 4 == 0 && n % 8 == 0)
                        {
                            paoMr = 4;
                            paoNr = 8;
                        }
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
        var (mr, _) = PickMicrokernelTilePrePack<T>();

        // Sub-E (#373): multi-panel pre-pack. Pack the entire weight into
        // (numIcBlocks × numPcBlocks) tiles, each of size Mc × Kc. Strategies
        // index by (ic/Mc, pc/Kc) at runtime via the new MultiPanelStride
        // metadata on the handle.
        //
        // Tile dimensions MUST match the autotuner's FallbackToHeuristic
        // defaults so the override at BlasManaged.Gemm (search "TileMc") sees
        // the same (Mc, Kc) the live-pack path would have used — otherwise
        // pre-pack forces a smaller block size on the consume path, multiplying
        // the inner-loop iteration count and producing a NET SLOWDOWN. Sub-Q
        // (#407) upgraded the autotune defaults to BLIS-style (128, 512, 256);
        // PrePackA's contract is "give the consume path identical tiling to
        // what live-pack would have produced" so we mirror those defaults.
        int mc = Math.Min(128, m);
        int kc = Math.Min(256, k);
        // Round mc UP to a multiple of mr (microkernel row tile height) so the
        // packed-byte layout matches what ScalarPack.PackA / Avx2Pack.PackA
        // expects. CodeRabbit #402: the previous gated form (`if (mc < m ...`)
        // skipped tiny-panel cases where Math.Min(128, m) made mc start equal
        // to m, leaving mc unaligned to mr. Concrete repro: m=2, mr=4 →
        // mc=Min(128,2)=2, mc<m is false, mc>m is false, no round-up runs and
        // tileBytes is sized for 2 rows instead of the mr-aligned padded
        // stripe the packer reads. Unconditional round-up handles both
        // "smaller than full tile" (mc=m < 128) and "edge of full tile"
        // (mc=128 already mr-aligned) consistently.
        if ((mc % mr) != 0) mc = ((mc + mr - 1) / mr) * mr;
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
        // Record the microkernel row tile the panel stripes are interleaved with —
        // consumers must run the SAME mr or fall back to live packing (see
        // WeightPackHandle.PackMr).
        handle.PackMr = mr;
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
        var (_, nr) = PickMicrokernelTilePrePack<T>();

        // Sub-E (#373): multi-panel pre-pack — see PrePackA above. Tile sizes
        // MUST match the autotuner's FallbackToHeuristic defaults (Sub-Q #407
        // upgraded these to BLIS-style Nc=512, Kc=256). The consume-path
        // override at BlasManaged.Gemm (search "TileMc") replaces the
        // autotune-picked Nc/Kc with the handle's TileMc/TileKc — so any
        // mismatch forces a smaller block size and hurts the GEMM (regression
        // was 0.35× speedup on FFN_128×768×768: pre-pack forced 64×64 tiles
        // = 6× more inner-loop iterations + far worse cache reuse than the
        // 512×256 live-pack baseline).
        int nc = Math.Min(512, n);
        int kc = Math.Min(256, k);
        // Round nc UP to a multiple of nr (microkernel column tile width) — the
        // packed-buffer layout in ScalarPack.PackB / Avx2Pack.PackB always
        // writes ceil(nc / nr) * nr columns per Kc row, zero-padding the tail
        // stripe. Pre-fix the code below clamped nc back DOWN to n when n < nr,
        // so a [k×n] PrePackB with n=2 nr=4 would size the tile as kc*n*sizeof(T)
        // but ScalarPack.PackB would still write kc*nr*sizeof(T) (one full
        // zero-padded stripe), producing an IndexOutOfRangeException in the
        // packed-buffer span. Keep nc aligned UP and let effectiveNc bound the
        // logical read range in the per-tile inner loop.
        // CodeRabbit #402: unconditional round-up (same rationale as PrePackA
        // above). The previous gated form skipped tiny-panel cases where
        // Math.Min(512, n) made nc == n on a value already smaller than nr.
        if ((nc % nr) != 0) nc = ((nc + nr - 1) / nr) * nr;
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
        // Record the microkernel column tile the panel stripes are interleaved
        // with — consumers must run the SAME nr or fall back to live packing (see
        // WeightPackHandle.PackNr).
        handle.PackNr = nr;
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
        BlasManagedAutotune.ClearStrategyMemo(); // in-memory strategy memo (#375 G13)
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
            // #409 S.4: the higher-arithmetic-intensity 6×8 FP64 kernel (1.5 FMA/load,
            // ~82% of peak vs the 4×8's 1.33 / ~75%) is used in FAST (non-deterministic)
            // mode only — the FP64 analog of the S.3 FP32 6×16 win. Deterministic mode
            // keeps 4×8 because 6×8's different reduction order (plain K-loop vs the 4×8's
            // unroll-4) would break the bit-exact invariant the Streaming bypass /
            // serial-vs-parallel paths assert (acceptable only where Fast mode is set).
            if (!Helpers.BlasProvider.IsDeterministicMode && Avx2Fp64_6x8.IsSupported) return (6, 8);
            if (Avx2Fp64_4x8.IsSupported) return (4, 8);
            // Neon FP64 uses (4, 4) — same as scalar; DispatchMicrokernel picks the right kernel.
            return (4, 4);
        }
        if (typeof(T) == typeof(float))
        {
            if (Avx512Fp32_16x16.IsSupported) return (16, 16);
            // #409 S.3: the higher-arithmetic-intensity 6×16 kernel (1.5 FMA/load, ~66% of
            // peak vs the load-bound 8×8's ~0.89 / ~49-59%) is used in FAST (non-deterministic)
            // mode only. Deterministic mode keeps 8×8 because 6×16's Mr=6 leaves an M-tail on
            // most shapes that gets computed by Streaming, and 6×16's interior bits do NOT match
            // Streaming's (8×8's DO) — so when the parallel M-split shifts the interior/tail
            // boundary, a row flips between 6×16 and Streaming and the output stops being
            // bit-identical across thread counts (verified: 6×16 fails 4 bit-exact tests, incl.
            // DeterministicParallelGemmContractTests at m=16). Acceptable only in Fast mode.
            if (Avx2Fp32_6x16.IsSupported) return (6, 16);   // now deterministic too — see gate fix in Gemm
            if (Avx2Fp32_8x8.IsSupported) return (8, 8);
            if (NeonFp32_8x4.IsSupported) return (8, 4);
            return (4, 4);
        }
        return (4, 4);
    }

    /// <summary>
    /// Tile selection for the PRE-PACK path (<see cref="PrePackA{T}"/> / <see cref="PrePackB{T}"/>
    /// and the consume side when a pre-packed handle is supplied). FP32 AVX2 stays on the legacy
    /// 8×8 tile regardless of Fast/Deterministic mode: a pre-packed handle's pack-time and
    /// consume-time tiles MUST agree, and the multi-panel byte layout + offset math are validated
    /// for 8×8. #409's 6×16 applies only to the live (non-pre-packed) Fast-mode path.
    /// </summary>
    private static (int Mr, int Nr) PickMicrokernelTilePrePack<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(float)
            && !Avx512Fp32_16x16.IsSupported   // AVX-512 FP32 already uses 16×16 (unchanged)
            && Avx2Fp32_8x8.IsSupported)
            return (8, 8);
        // #409 S.4: FP64 pre-pack stays on the legacy 4×8 tile regardless of Fast/
        // Deterministic mode — a pre-packed handle's pack-time and consume-time tiles MUST
        // agree, and the byte layout + offset math are validated for 4×8. The 6×8 applies
        // only to the live (non-pre-packed) Fast-mode path.
        if (typeof(T) == typeof(double)
            && !Avx512Fp64_8x16.IsSupported   // AVX-512 FP64 already uses 8×16 (unchanged)
            && Avx2Fp64_4x8.IsSupported)
            return (4, 8);
        return PickMicrokernelTile<T>();
    }
}
