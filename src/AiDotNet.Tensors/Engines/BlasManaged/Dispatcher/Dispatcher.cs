using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Strategy selection for <see cref="BlasManaged"/>.
///
/// <para>
/// Phase B uses a simple static heuristic based on K size and total work
/// (M·N). Phase H replaces this with an autotune cache that learns the
/// winning strategy per shape across calls. The heuristic boundaries chosen
/// here match the spec's defaults:
/// </para>
///
/// <list type="bullet">
///   <item>K &lt; 32 OR M·N &lt; 1024 → <see cref="PackingMode.ForceStreaming"/>: no pack overhead amortizes at this scale.</item>
///   <item>K &lt; 128 → <see cref="PackingMode.ForcePackAOnly"/>: pack A only; B reuse is too low to justify packing.</item>
///   <item>Otherwise → <see cref="PackingMode.ForcePackBoth"/>: pack A and B for maximum cache locality.</item>
/// </list>
/// </summary>
internal static class Dispatcher
{
    /// <summary>
    /// Select the packing strategy for a given (M, N, K) shape.
    /// Honours the user's <see cref="BlasOptions{T}.PackingMode"/> if it's
    /// not <see cref="PackingMode.Auto"/>; otherwise applies the default
    /// heuristic.
    /// </summary>
    public static PackingMode SelectStrategy<T>(int m, int n, int k, in BlasOptions<T> options)
        where T : unmanaged
    {
        if (options.PackingMode != PackingMode.Auto)
            return options.PackingMode;

        if (k < 32 || (long)m * n < 1024) return PackingMode.ForceStreaming;

        // Sub-G readiness: small cubes (the 64×64×64 family) hit a perf cliff
        // on PackAOnly because pack-A + autotune dispatch overhead exceeds
        // the actual GEMM compute at these sizes. The A/B diagnostic
        // (Conv2DAbBench `--ab-blas-small-square-fp64`) showed:
        //   - 64³ FP64: Streaming 4.3 vs PackAOnly 2.4 GFLOPS (1.8× win)
        //   - 64³ FP32: Streaming 5.8 vs PackAOnly 3.3 GFLOPS (1.75× win)
        //   - 96×128×64 FP64 (786K): PackAOnly 9.0 GFLOPS still wins
        //     (above the cutoff)
        // 300K total-work is the empirical cutoff: 64³ (262K) routes to
        // Streaming; 96×128×64 (786K) stays on PackAOnly. Applies to FP32
        // and FP64 — both showed the same shape-dependent crossover.
        if ((long)m * n * k <= 300_000L)
            return PackingMode.ForceStreaming;

        if (k < 128) return PackingMode.ForcePackAOnly;
        return PackingMode.ForcePackBoth;
    }
}
