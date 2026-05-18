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
        // DisableAutotune: skip cache entirely, use heuristic directly.
        if (packingMode == PackingMode.DisableAutotune)
        {
            return FallbackToHeuristic(m, n, k, mr, nr, procs, isDeterministic);
        }

        var shape = BlasManagedAutotune.EncodeShape<T>(m, n, k, transA, transB, mr, nr, hasEpilogue);

        // Cache hit?
        var cached = BlasManagedAutotune.TryLookup(shape);
        if (cached.HasValue)
        {
            BlasManagedStatsTracker.IncrementAutotuneHit();
            return cached.Value;
        }

        // Cache miss — use heuristic, store the decision.
        BlasManagedStatsTracker.IncrementAutotuneMiss();
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
    /// Use the AxisSelector heuristic to choose an axis, with default blocking
    /// parameters. The heuristic itself is shape-aware (m, n, k, mr, nr, procs).
    /// </summary>
    private static (ParallelismAxis Axis, int Mc, int Nc, int Kc, int ThreadCount)
        FallbackToHeuristic(int m, int n, int k, int mr, int nr, int procs, bool isDeterministic)
    {
        var axis = AxisSelector.Select(m, n, k, mr, nr, procs, isDeterministic);
        // Default blocking: 64 across the board (matches BlasManaged.Gemm defaults).
        // Cap each block at the corresponding shape dimension.
        int mc = Math.Min(64, m);
        int nc = Math.Min(64, n);
        int kc = Math.Min(64, k);
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
