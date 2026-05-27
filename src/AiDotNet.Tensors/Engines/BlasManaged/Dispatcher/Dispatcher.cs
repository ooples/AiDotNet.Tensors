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

        // #371 guard (merged from main): a supplied pre-pack handle (FrozenWeightRegistry)
        // MUST be consumed via the packing path — the Streaming kernel ignores packed
        // handles, which would silently make the pre-pack a no-op. So the work-based
        // Streaming routes below are gated on !hasPrePack; PackBoth/PackAOnly consume it.
        bool hasPrePack = options.PackedA != null || options.PackedB != null;

        // Sub-C (#371, PR #464): small total work → pack-free Streaming. Below
        // SmallShapeWorkThreshold (1M) the pack-A rent+pack round-trip costs more
        // than the GEMM compute, so packing is skipped. This is the routing PR
        // #464 introduced and that SmallShapeStreamingDispatchTests pins; a prior
        // main→branch merge silently reverted it to a 300K cutoff (leaving the
        // 1M constant unused), which regressed 96³/80×80×100-class shapes back to
        // PackAOnly and turned those tests red. Restored here. Also routes
        // 96×128×64 (786K) pack-free, which the 16-core AVX2 A/B confirms
        // (Streaming ≫ PackAOnly).
        long work = (long)m * n * k;
        if (!hasPrePack && work < SmallShapeWorkThreshold) return PackingMode.ForceStreaming;

        // work ≥ 1M. Preserve the two special-case wins the old heuristic carried
        // (PR #464's clean version dropped both, which would regress them):
        //   (a) tall + thin-N, low-K → Streaming (e.g. ResNet50_layer1 3136×64×64:
        //       Streaming ≫ PackAOnly; the very tall M streams cheaply).
        //   (b) medium-M wide-shape, k≥256 → PackAOnly (e.g. 128×768×768: PackAOnly
        //       ≫ PackBoth — pack-B amortises poorly at small M/mr reuse).
        // Everything else low-K goes PackAOnly; high-K balanced goes PackBoth.
        // (The strategy choice on 128³ / 512×512×64 is hardware-dependent — see
        // the per-fingerprint autotune; the static default here matches #464's
        // tests.)
        if (!hasPrePack && k < 128 && n <= 64 && m >= 8 * n)
            return PackingMode.ForceStreaming;                              // (a) tall + thin

        if (k < 128) return PackingMode.ForcePackAOnly;                     // 512×512×64, 3136×32×32

        // Sub-G follow-on: medium-M wide-shape pattern. At k≥256 with
        // moderate M (128-256) and substantially larger N and K, PackBoth
        // amortises its pack-B cost poorly — M/mr reuse is small while
        // N×K pack-B work is huge. PackAOnly skips the pack-B hit and
        // parallelises over N effectively. A/B measured wins:
        //   - 128×768×768 FP64: PackAOnly@8thr 97.8 vs PackBoth-default 22 (4.4×)
        //   - 197×768×768 FP32: PackAOnly 118 vs PackBoth 87 (1.4×)
        // Tight gate (m≥128 AND k≥256) keeps the rule narrow:
        //   - Doesn't trigger for BERT_FFN_up 1024×3072×768 (m=1024 ≥ k=768
        //     so m<k is false anyway), preserving Auto's 217 GFLOPS.
        //   - Doesn't trigger for ResNet50_bwd_dW 64×147×3136 (m=64 <128),
        //     letting PackBoth's 22.9 GFLOPS stand vs PackAOnly's 21.2.
        //   - Doesn't trigger for 32×2048×256 FP64 (m=32 <128), preserving
        //     the existing PackBoth route.
        if (m >= 128 && m <= 256 && k >= 256 && m < k && m < n)
            return PackingMode.ForcePackAOnly;

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
