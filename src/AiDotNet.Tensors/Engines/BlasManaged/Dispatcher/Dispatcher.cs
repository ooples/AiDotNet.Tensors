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

        // Sub-C (#371): skip packing entirely for small total work. PackAOnly
        // rents + packs the A panel (PackAOnlyStrategy) and PackBoth packs both;
        // at small M·N·K that rent+pack round-trip costs more than the GEMM
        // compute itself. Measured (FP64, OpenBLAS baseline): Tiny_64sq
        // (64×64×64 = 262k) drops from ~21× to ~13× when routed pack-free,
        // because the pack-A allocation is eliminated. The threshold stays well
        // below larger low-K shapes (e.g. WideFat 512×512×64 = 16.7M, where
        // PackAOnly's cache blocking still wins ~5× vs streaming's ~7×), so they
        // keep the packing path. (Shapes ≤ TinyShapeWorkThreshold never reach
        // here — BlasManaged.Gemm already fast-paths them to Streaming.)
        //
        // Guard: a supplied pre-pack handle (Sub-E FrozenWeightRegistry) MUST be
        // consumed via the packing path — the Streaming kernel ignores packed
        // handles, which would silently make the pre-pack a no-op. So never
        // route pack-free when PackedA/PackedB is present, regardless of size.
        bool hasPrePack = options.PackedA != null || options.PackedB != null;
        if (!hasPrePack && (long)m * n * k < SmallShapeWorkThreshold)
            return PackingMode.ForceStreaming;

        if (k < 128) return PackingMode.ForcePackAOnly;
        return PackingMode.ForcePackBoth;
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
