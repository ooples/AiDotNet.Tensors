using System;
using AiDotNet.Tensors.Helpers.Autotune;

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

        // #371 guard (merged from main): a supplied pre-pack handle (FrozenWeightRegistry)
        // MUST be consumed via the packing path — the Streaming kernel ignores packed
        // handles, which would silently make the pre-pack a no-op. So the work-based
        // Streaming routes below are gated on !hasPrePack; PackBoth/PackAOnly consume it.
        bool hasPrePack = options.PackedA != null || options.PackedB != null;

        // #375 hybrid: a supplied pre-pack handle MUST be consumed via the packing
        // path — Streaming/PackAOnly ignore a packed B, which would silently drop it.
        if (hasPrePack) return PackingMode.ForcePackBoth;

        // #375 hybrid: route via the per-hardware seed table (the cold-start tier of the
        // unified (hardwareKey, shapeBucket) → strategy map). Replaces the static
        // work<1M/k<128 heuristic, which baked one machine's optimum in for all hardware
        // (the amd-avx2-cpu16 vs cpu32 collision). The table reproduces the prior routing
        // for unmeasured shapes and encodes the measured per-hardware wins (e.g. cpu16
        // routes 512×512×64 / 128³ to Streaming, cpu32 to blocking). The learned cache
        // (Phase 2) and background autotuner (Phase 3) refine this per fingerprint.
        return StrategyDefaultTable.Route(HardwareFingerprint.Key, m, n, k);
    }

    /// <summary>
    /// Sub-C (#371): total-work ceiling below which packing is skipped (route to
    /// pack-free <see cref="PackingMode.ForceStreaming"/>). Chosen to capture
    /// small low-K shapes whose pack overhead dominates compute, while leaving
    /// larger shapes — where pack + cache blocking pays off — on the packing
    /// path. Above <see cref="BlasManaged.TinyShapeWorkThreshold"/> (100k), which
    /// Gemm already routes to Streaming directly.
    /// </summary>
    internal const long SmallShapeWorkThreshold = 1_000_000;
}
