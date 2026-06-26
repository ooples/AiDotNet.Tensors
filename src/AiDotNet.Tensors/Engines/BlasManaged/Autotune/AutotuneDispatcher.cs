using System;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Per-shape autotune dispatcher: looks up a cached choice for the current
/// (M, N, K, trans, precision, mr, nr) tuple, or falls back to the
/// <see cref="AxisSelector"/> heuristic and stores its decision for future
/// calls.
///
/// <para>
/// Phase H3 ships a "heuristic-only" autotune: on cache miss, we use the
/// AxisSelector heuristic and store the result. This gives the persistence
/// benefit (same heuristic decision on every future call, cross-process)
/// without the cost of benchmarking 3-5 candidates per shape. A future
/// refinement can add true benchmark-driven candidate ranking; the cache
/// schema already supports it.
/// </para>
///
/// <para>
/// Honors <see cref="PackingMode.DisableAutotune"/>: when set, the cache
/// lookup is bypassed and the heuristic decision is used directly (no
/// cache write either).
/// </para>
/// </summary>
internal static class AutotuneDispatcher
{
    /// <summary>
    /// Decide the parallelism axis + blocking parameters for the current GEMM
    /// call. Consults the on-disk autotune cache first; falls back to the
    /// AxisSelector heuristic on miss (and stores the heuristic decision).
    /// </summary>
    /// <typeparam name="T">Element type. Must be float or double.</typeparam>
    public static (ParallelismAxis Axis, int Mc, int Nc, int Kc, int ThreadCount)
        Decide<T>(
            int m, int n, int k,
            bool transA, bool transB,
            int mr, int nr,
            int procs,
            bool isDeterministic,
            bool hasEpilogue,
            PackingMode packingMode) where T : unmanaged
    {
        // Sub-Q (#407): when a BlockSizeSweep is measuring candidates, every
        // nested GEMM on this thread must (a) use the candidate blocking under
        // test and (b) NOT re-enter the cache or kick off another sweep — that
        // would recurse forever. The per-thread override supplies the candidate
        // blocking AND acts as the recursion guard. (This is the role the issue
        // earmarked for PrefersManagedCache.BypassAutotune; a dedicated override
        // is cleaner because it also carries the (Mc, Nc, Kc) being probed.)
        var ovr = BlockOverride;
        if (ovr.HasValue)
        {
            var axisOvr = AxisSelector.Select(m, n, k, mr, nr, procs, isDeterministic);
            var (omc, onc, okc) = ClampBlocking(ovr.Value.Mc, ovr.Value.Nc, ovr.Value.Kc, m, n, k, mr, nr);
            return (axisOvr, omc, onc, okc, procs);
        }

        // DisableAutotune: skip cache entirely, use heuristic directly.
        if (packingMode == PackingMode.DisableAutotune)
        {
            return FallbackToHeuristic(m, n, k, mr, nr, procs, isDeterministic);
        }

        var shape = BlasManagedAutotune.EncodeShape<T>(m, n, k, transA, transB, mr, nr, hasEpilogue, isDeterministic);

        // Cache hit?
        var cached = BlasManagedAutotune.TryLookup(shape);
        if (cached.HasValue)
        {
            BlasManagedStatsTracker.IncrementAutotuneHit();
            return cached.Value;
        }

        // Cache miss.
        BlasManagedStatsTracker.IncrementAutotuneMiss();

        // Sub-Q (#407): measurement-based tuning. When enabled (and the shape is
        // large enough that the block choice actually matters), benchmark a set
        // of candidate (Mc, Nc, Kc) tuples and cache the measured winner. Off by
        // default — falls through to the heuristic below.
        if ((MeasurementEnabled || ForceMeasureOnMiss) && ShouldMeasure(m, n, k))
        {
            try
            {
                var measured = BlockSizeSweep.Measure<T>(m, n, k, transA, transB, mr, nr, procs, isDeterministic);
                try
                {
                    BlasManagedAutotune.Store(shape, measured.Axis, measured.Mc, measured.Nc, measured.Kc, measured.ThreadCount, measured.MeasuredMs);
                }
                catch { /* cache write failure is non-fatal */ }
                return (measured.Axis, measured.Mc, measured.Nc, measured.Kc, measured.ThreadCount);
            }
            catch
            {
                // Any measurement failure (allocation, unexpected shape, etc.) is
                // non-fatal — fall through to the heuristic so the GEMM still runs.
            }
        }

        // Heuristic fallback — store the decision so future calls of this shape
        // are a cache hit even without measurement.
        var result = FallbackToHeuristic(m, n, k, mr, nr, procs, isDeterministic);
        try
        {
            BlasManagedAutotune.Store(shape, result.Axis, result.Mc, result.Nc, result.Kc, result.ThreadCount, measuredTimeMs: 0);
        }
        catch
        {
            // Cache write failure (disk full, read-only filesystem, etc.) is
            // non-fatal — return the heuristic decision and continue.
        }
        return result;
    }

    /// <summary>
    /// Sub-Q (#407): per-thread block-size override consumed by <see cref="Decide{T}"/>.
    /// <see cref="BlockSizeSweep"/> sets this to the candidate under test before
    /// each measurement GEMM and clears it afterwards. Non-null also suppresses
    /// cache lookup/store and recursive measurement on this thread.
    /// </summary>
    [ThreadStatic] internal static (int Mc, int Nc, int Kc)? BlockOverride;

    /// <summary>
    /// Gate for measurement-based autotune: only spend the (one-time, cached-
    /// forever) measurement cost on shapes big enough that (a) the timing is
    /// stable at ms scale and (b) a block choice beyond the heuristic clamp
    /// actually exists. Tiny/skinny shapes clamp to their dimensions anyway, so
    /// there is nothing to tune.
    /// </summary>
    private static bool ShouldMeasure(int m, int n, int k)
        => (long)m * n * k >= 1_000_000
           && (m > 128 || n > 512 || k > 256);

    /// <summary>
    /// Clamp a candidate (Mc, Nc, Kc) to the shape and microkernel alignment, so
    /// the strategy pipeline never sees a block bigger than the matrix or a
    /// non-positive / mis-aligned value. Mirrors the clamping in
    /// <see cref="FallbackToHeuristic"/>.
    /// </summary>
    internal static (int Mc, int Nc, int Kc) ClampBlocking(
        int mc, int nc, int kc, int m, int n, int k, int mr, int nr)
    {
        mc = Math.Min(mc, m);
        nc = Math.Min(nc, n);
        kc = Math.Min(kc, k);
        if (mr > 0) mc = (mc / mr) * mr;
        if (nr > 0) nc = (nc / nr) * nr;
        if (mc <= 0) mc = Math.Min(Math.Max(1, mr), m);
        if (nc <= 0) nc = Math.Min(Math.Max(1, nr), n);
        if (kc <= 0) kc = 1; // k > 0 is guaranteed by Gemm's early-out
        mc = Math.Min(mc, m);
        nc = Math.Min(nc, n);
        kc = Math.Min(kc, k);
        return (mc, nc, kc);
    }

    /// <summary>
    /// Sub-Q (#407): when set, on autotune cache miss benchmark a small set of
    /// candidate (Mc, Nc, Kc) tuples and store the winner instead of using the
    /// heuristic. Off by default — first-call overhead is several ms per shape.
    /// Set via env var <c>AIDOTNET_BLAS_AUTOTUNE_MEASURE=1</c> at process start
    /// (recommended: enable in production warmup, leave off in CI/tests).
    /// </summary>
    internal static bool MeasurementEnabled { get; } =
        Environment.GetEnvironmentVariable("AIDOTNET_BLAS_AUTOTUNE_MEASURE") == "1";

    /// <summary>
    /// Sub-Q (#407): per-thread force-measure flag set by
    /// <see cref="BlockSizeSweep.PrepopulateCommonShapes"/> so offline warmup
    /// measures-and-caches the common shapes even when runtime measurement
    /// (<see cref="MeasurementEnabled"/>) is off. Scoped to the warmup thread.
    /// </summary>
    [ThreadStatic] internal static bool ForceMeasureOnMiss;

    /// <summary>
    /// Use the AxisSelector heuristic to choose an axis, with default blocking
    /// parameters. The heuristic itself is shape-aware (m, n, k, mr, nr, procs).
    ///
    /// <para>
    /// Sub-Q (#407): block defaults upgraded from (64, 64, 64) to BLIS-style
    /// (128, 512, 256) for shapes large enough to benefit. Small shapes still
    /// clamp to dimension. The Pre-pack handle override in BlasManaged.Gemm
    /// (search "TileMc") keeps the (Sub-E) FrozenWeightRegistry consumption
    /// correct under the new defaults — handle dims override autotune choice.
    /// </para>
    /// </summary>
    private static (ParallelismAxis Axis, int Mc, int Nc, int Kc, int ThreadCount)
        FallbackToHeuristic(int m, int n, int k, int mr, int nr, int procs, bool isDeterministic)
    {
        var axis = AxisSelector.Select(m, n, k, mr, nr, procs, isDeterministic);

        // Sub-Q: BLIS-style defaults. Standard AVX2 + Zen-class cache geometry.
        // Mc=128 fits L1 with M-stripe + Kc K-stripe = 128 × 256 × 4 = 128 KB (FP32)
        //   — slightly over L1 but reads stream + microkernel reuses the stripe Nc/Nr
        //   times per pack-A, so the working set is dominated by the C tile.
        // Nc=512 lets B-pack fit L2 (256 KB FP32) and amortizes pack-B across many
        //   ic-blocks.
        // Kc=256 keeps the pack-A and pack-B regions cache-resident together.
        // For tiny shapes (m < 128, etc.) the Math.Min clamp falls back to dim-fits.
        int mc = Math.Min(128, m);
        int nc = Math.Min(512, n);
        // Kc=512 (was 256): a blocking sweep on Zen2 (Threadripper 3990X) measured Kc=512 as the
        // best K-block for EVERY tested shape (attn-proj/mlp-fc/square-1024/square-2048), +20–54%
        // all-core GFLOP/s vs Kc=256 — the larger K-stripe amortizes the pack + microkernel
        // ramp over more FMAs before the C-tile is re-touched. Matches OpenBLAS's Zen Kc≈248–512 band.
        int kc = Math.Min(512, k);

        // Sub-G (#375): M-axis occupancy floor. PackBoth parallelises over
        // numIcBlocks = ceil(m / mc). With mc=128 a shape like 512×512×512 has
        // only 4 ic-blocks, so it plateaus at 4 threads (measured: 165 GFLOPS @
        // 4 threads, no gain past that, vs torch's 331). Shrink mc to add blocks
        // when underutilised — but floor mc at 64 so the packed A panel stays
        // cache-friendly (mc=32 was measured to regress per-block efficiency
        // more than the extra parallelism gained). Target ≈ 2×procs blocks for
        // load-balance headroom without over-fragmenting.
        if (procs > 1 && mr > 0)
        {
            int mBlocks = (m + mc - 1) / mc;
            if (mBlocks < procs && m >= 2 * 64)
            {
                int floorMc = Math.Max(mr, 64);
                int targetBlocks = 2 * procs;
                int targetMc = ((m / targetBlocks) / mr) * mr;
                targetMc = Math.Max(floorMc, targetMc);
                if (targetMc < mc) mc = targetMc;
            }
        }

        // Round mc down to mr alignment, nc down to nr alignment.
        if (mr > 0) mc = (mc / mr) * mr;
        if (nr > 0) nc = (nc / nr) * nr;
        // CodeRabbit #366: alignment rounding can drive mc/nc to 0 on tiny
        // shapes (e.g., m < mr); clamping back to mr/nr alone can then make
        // mc > m or nc > n. Clamp to the actual dimension as well so drivers
        // never see a block bigger than the matrix.
        if (mc <= 0) mc = Math.Min(Math.Max(1, mr), m);
        if (nc <= 0) nc = Math.Min(Math.Max(1, nr), n);
        mc = Math.Min(mc, m);
        nc = Math.Min(nc, n);

        int threadCount = procs;  // Default to all available cores.
        return (axis, mc, nc, kc, threadCount);
    }
}
