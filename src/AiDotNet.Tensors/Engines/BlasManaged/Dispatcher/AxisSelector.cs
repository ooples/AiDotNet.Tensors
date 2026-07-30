using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Parallelism axis enumeration. Used by <see cref="AxisSelector"/> to
/// communicate which axis the strategy should split work along.
/// </summary>
internal enum ParallelismAxis : byte
{
    /// <summary>Sequential — no parallel split (below work threshold).</summary>
    None = 0,
    /// <summary>M-axis split (default for most shapes; each thread owns a row-block of C).</summary>
    M = 1,
    /// <summary>N-axis split (M small but N large — e.g., attention Q·Kᵀ at small batch).</summary>
    N = 2,
    /// <summary>K-axis split with deterministic reduction tree (tall-thin shapes, non-deterministic mode only).</summary>
    K = 3,
    /// <summary>2D MN-grid split (big square shapes; M alone underutilizes cores).</summary>
    MN_2D = 4,
}

/// <summary>
/// Heuristic that picks the optimal parallelism axis for a given GEMM shape.
/// Phase G5 ships the heuristic as a standalone helper; Phase H replaces it
/// with an autotune cache that learns the winning axis per shape over many
/// calls.
///
/// <para>
/// Decision tree (highest priority first):
/// </para>
/// <list type="number">
///   <item>Below work threshold → <see cref="ParallelismAxis.None"/>.</item>
///   <item>M has enough blocks AND (K small OR non-deterministic) → <see cref="ParallelismAxis.M"/>.</item>
///   <item>N has enough blocks → <see cref="ParallelismAxis.N"/>.</item>
///   <item>K large AND non-deterministic → <see cref="ParallelismAxis.K"/>.</item>
///   <item>Combined M·N has enough work → <see cref="ParallelismAxis.MN_2D"/>.</item>
///   <item>Fallback → <see cref="ParallelismAxis.M"/>.</item>
/// </list>
///
/// <para>
/// The K-axis is gated on non-deterministic mode because the reduction tree
/// adds work that doesn't pay off unless the throughput gain exceeds the
/// reduction cost. Deterministic mode also forbids K-axis to keep the path
/// predictable (Phase G6).
/// </para>
/// </summary>
internal static class AxisSelector
{
    /// <summary>
    /// Work threshold below which sequential execution beats parallel
    /// dispatch overhead. Same value used by
    /// <see cref="Helpers.CpuParallelSettings.ParallelForOrSerial"/>.
    /// </summary>
    public const long ParallelWorkThreshold = 32_768;

    /// <summary>
    /// Pick the parallelism axis for the given shape, microkernel tile, and
    /// runtime config.
    /// </summary>
    /// <param name="m">Rows of C.</param>
    /// <param name="n">Cols of C.</param>
    /// <param name="k">Inner reduction dim.</param>
    /// <param name="mr">Microkernel row-tile width.</param>
    /// <param name="nr">Microkernel column-tile width.</param>
    /// <param name="procs">Available processor count (≥ 1).</param>
    /// <param name="isDeterministic">True if BlasProvider.IsDeterministicMode is set.</param>
    public static ParallelismAxis Select(
        int m, int n, int k,
        int mr, int nr,
        int procs,
        bool isDeterministic)
    {
        // Universal guards run FIRST, before any forced-axis override, so a test/bench hook can
        // never route a shape that production would refuse to parallelize (procs<=1 or below the
        // work threshold) or pick an axis that violates the active mode (K-axis under determinism).
        if (procs <= 1) return ParallelismAxis.None;
        if ((long)m * n * k < ParallelWorkThreshold) return ParallelismAxis.None;

        // A/B test hook (#475 medium-axis routing): honor a pinned axis only AFTER the universal
        // guards above, and reject a forced K-axis in deterministic mode (the same gate the
        // heuristic enforces below). Null in production, so this is a no-op on the hot path.
        var forced = ForceAxisForTest;
        if (forced.HasValue && !(forced.Value == ParallelismAxis.K && isDeterministic))
            return forced.Value;

        // M-axis: enough M-blocks to spread across cores. Also gated on K size:
        // when K is large, pack overhead amortizes better with finer-grained
        // splits (K or 2D); switch off M if K > 256 AND mode is deterministic.
        if (m >= procs * mr * 2 && (k <= 256 || !isDeterministic))
            return ParallelismAxis.M;

        // N-axis: M small but N large.
        if (n >= procs * nr * 2)
            return ParallelismAxis.N;

        // K-axis: tall-thin shapes, only in non-deterministic mode (reduction-tree cost).
        if (k >= 512 && !isDeterministic)
            return ParallelismAxis.K;

        // 2D grid: combined M*N has enough work for partial-M underutilization to matter.
        if ((long)m * n >= (long)procs * mr * nr * 4)
            return ParallelismAxis.MN_2D;

        // Fallback.
        return ParallelismAxis.M;
    }

    /// <summary>
    /// A/B test hook (#475 medium-axis routing). When non-null on the calling thread,
    /// <see cref="Select"/> returns this axis — but only AFTER the universal procs/work-threshold
    /// guards and only when the axis is legal for the active mode (a forced K-axis is ignored in
    /// deterministic mode). Used by the in-process axis-routing bench to measure each axis on the
    /// same shape. Null in production (the default); thread-static so a bench thread can pin an axis
    /// without perturbing the dispatcher on other threads.
    /// </summary>
    [ThreadStatic] internal static ParallelismAxis? ForceAxisForTest;
}
