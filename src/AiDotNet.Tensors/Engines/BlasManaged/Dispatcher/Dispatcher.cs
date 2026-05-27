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

        // Sub-G readiness — A/B diagnostic (Conv2DAbBench
        // `--ab-blas-small-square-fp64`) measured every PackingMode for the
        // 6 worst-loss shapes in the refreshed baseline and revealed three
        // distinct K<128 regimes that PackAOnly handled badly. PackAOnly is
        // rarely the optimal choice — it wins only on a narrow band of
        // medium-work shapes (~96×128×64-ish), and the prior heuristic
        // routed every K<128 shape to it regardless of M, N, total work.
        //
        // Empirical optima (16-core AVX2 Windows host, 2026-05-26):
        //
        //   tiny cubes (M·N·K ≤ 300K):              Streaming
        //     - 64³ FP64: Streaming 4.3 vs PackAOnly 2.4 (1.8× win)
        //     - 64³ FP32: Streaming 5.8 vs PackAOnly 3.3 (1.75× win)
        //
        //   thin-K with substantial work, very thin N (≤32):  PackBoth
        //     - 3136×32×32 FP32: PackBoth 35.3 vs PackAOnly 18.2 (1.94×)
        //
        //   thin-K with substantial work, tall + thin N (n≤64, m≥8·n): Streaming
        //     - 3136×64×64 FP32: Streaming 55.6 vs PackAOnly 26.3 (2.11×)
        //
        //   thin-K with substantial work, balanced M·N:    PackBoth
        //     - 512×512×64 FP64: PackBoth 53.2 vs PackAOnly 12.0 (4.43×)
        //
        //   medium-K, medium work (PackAOnly's narrow band): PackAOnly
        //     - 96×128×64 FP64: PackAOnly 9.2 vs Streaming 8.6 vs PackBoth 3.3
        //
        // The branch ordering below codifies this map. The 300K Streaming
        // cutoff is intentionally BELOW 96×128×64's work (786K) so
        // PackAOnly's narrow band is preserved.

        long work = (long)m * n * k;
        if (!hasPrePack && work <= 300_000L) return PackingMode.ForceStreaming;

        if (k < 128 && work >= 1_000_000L)
        {
            // Substantial thin-K shape. Three sub-regimes:
            if (n <= 32) return PackingMode.ForcePackBoth;                  // micro-N
            if (!hasPrePack && n <= 64 && m >= 8 * n) return PackingMode.ForceStreaming;   // tall + thin
            return PackingMode.ForcePackBoth;                               // balanced
        }

        if (k < 128) return PackingMode.ForcePackAOnly;                     // 96×128×64-style

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
