using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Seed tier of the unified (hardwareKey, shapeBucket) → strategy map (#375). Hardcoded
/// per-{simd,vendor,cpuBucket} routing; the cold-start default before the learned disk
/// cache populates. Pure, no I/O. The learned cache (Phase 2) overrides this per shape;
/// the background autotuner (Phase 3) populates that cache.
///
/// <para>
/// IMPORTANT — this table is consulted ONLY for shapes the Sub-S machine-code fast path
/// declines (transposed / epilogue-fused / unaligned); Sub-S handles all non-transposed
/// aligned GEMM before strategy selection. So the seed values are calibrated for the
/// TRANSPOSED case (measured via `--prewarm-autotune`), NOT the non-transposed A/B — e.g.
/// transposed 512×512×64 wants PackBoth (≈43 ms) even though non-transposed 512×512×64 wants
/// Streaming (which never reaches here). Routing transposed ThinK to Streaming was a 2.5×
/// regression caught by the G12 anti-regression test.
/// </para>
/// </summary>
internal static class StrategyDefaultTable
{
    /// <summary>Coarse shape regime — the bucket axis of the seed map.</summary>
    internal enum ShapeBucket { TinyCube, SmallLowK, TallThin, MediumSquare, ThinK, MediumMWide, WideCompute, Large }

    internal static ShapeBucket Bucket(int m, int n, int k)
    {
        long work = (long)m * n * k;
        if (work <= 300_000L) return ShapeBucket.TinyCube;
        if (k <= 128 && work < 1_000_000L) return ShapeBucket.SmallLowK;
        // Tall + thin-N, low-K (e.g. ResNet50_layer1 3136×64×64): the very tall M
        // streams cheaply; Streaming ≫ packing on all hardware measured.
        if (k <= 128 && n <= 64 && m >= 8 * n) return ShapeBucket.TallThin;
        // True cube (m==n==k, e.g. 128³) — distinct from a thin-K m==n shape like
        // 512×512×64, which is ThinK (k≪m). On cpu32 the cube wants PackBoth but the
        // thin-K wants PackAOnly, so they must NOT share a bucket.
        if (k <= 128 && m == n && n == k) return ShapeBucket.MediumSquare;
        if (k <= 128) return ShapeBucket.ThinK;
        // Medium-M wide-shape, k≥256 (e.g. 128×768×768, 197×768×768): pack-B amortises
        // poorly at small M/mr reuse → PackAOnly skips the pack-B hit and wins.
        if (m >= 128 && m <= 256 && k >= 256 && m < k && m < n) return ShapeBucket.MediumMWide;
        if (work >= 50_000_000L) return ShapeBucket.WideCompute;
        return ShapeBucket.Large;
    }

    /// <summary>
    /// Route a shape to a packing strategy via the seed table for the given hardware key.
    /// Falls back to a conservative default for unmapped keys; never throws.
    /// </summary>
    internal static PackingMode Route(HardwareFingerprint.HwKey key, int m, int n, int k)
    {
        var bucket = Bucket(m, n, k);

        // #653: the MediumMWide bucket (m∈[128,256], wide-N, k≥256) spans two regimes for the
        // NON-transposed shapes that reach here (via PrefersStrategyOverMachineKernel's k>2m
        // decline): at SHALLOW K (≤512) packing overhead isn't amortized and Streaming wins big
        // (e.g. 128×256×1536: Streaming 1.5ms vs PackBoth 8.0ms — 5.3×); at DEEP K, PackBoth wins
        // (the per-hardware arms below). The shallow-K→Streaming win is packing-overhead-driven
        // (hardware-independent), so gate it here for all hardware. Measured on-box (avx2 cpu≤1).
        if (bucket == ShapeBucket.MediumMWide && k <= 512)
            return PackingMode.ForceStreaming;
        // Deep-K (>512) MediumMWide, resolved by INTERLEAVED min-of-N (two runs, <2% apart —
        // a reproducible signal, not noise): at VERY-wide-N (≥3072) with m>128, packing B
        // doesn't pay (huge B, low per-stripe reuse) and PackAOnly beats PackBoth by ~1.2-1.3×.
        // m=128 is special (PackBoth wins even at N=3072), and N=1536 favors PackBoth — both
        // fall through to the PackBoth arms below. (Under transB this PackAOnly redirects to
        // PackBoth, so it only affects the non-transposed large shapes the 8M autotuner ceiling
        // never measures live.)
        if (bucket == ShapeBucket.MediumMWide && n >= 3072 && m > 128)
            return PackingMode.ForcePackAOnly;

        // amd-avx2 mid-core (≤16T, this box). Calibrated for the TRANSPOSED shapes that
        // actually reach this table (--prewarm-autotune measurements): small/cube → Streaming
        // wins even transposed; ThinK (e.g. 512×512×64 transB) → PackBoth (Streaming is 2.5×
        // slower there — the G12-caught regression); MediumMWide → PackAOnly (redirects to
        // PackBoth under transB, which wins).
        if (key.Simd == "avx2" && key.CpuBucket <= 1)
        {
            return bucket switch
            {
                ShapeBucket.TinyCube => PackingMode.ForceStreaming,
                ShapeBucket.SmallLowK => PackingMode.ForceStreaming,     // 96×128×64 transB: Streaming
                ShapeBucket.TallThin => PackingMode.ForceStreaming,
                ShapeBucket.MediumSquare => PackingMode.ForceStreaming,  // 128³ transB: Streaming
                ShapeBucket.ThinK => PackingMode.ForcePackBoth,          // 512×512×64 transB: PackBoth ≫ Streaming
                // #653: was ForcePackAOnly, calibrated for TRANSPOSED shapes (which redirect
                // PackAOnly→PackBoth under transB and win). But PrefersStrategyOverMachineKernel
                // declines the machine kernel for deep-K small-M (k>2m), so NON-transposed
                // FFN-shaped GEMMs (m∈[132,256], wide-N, deep-K) also reach here — and for them
                // PackAOnly is pathologically 4-6× slower than PackBoth (132×384×1536: 26ms vs
                // ~6ms). PackBoth is also where the transB redirect lands, so route there directly.
                ShapeBucket.MediumMWide => PackingMode.ForcePackBoth,
                _ => PackingMode.ForcePackBoth,
            };
        }

        // avx2 large-core (>16T, Ryzen 3950X #464): blocking wins on medium squares + thin-K.
        if (key.Simd == "avx2") // CpuBucket == 2
        {
            return bucket switch
            {
                ShapeBucket.TinyCube => PackingMode.ForceStreaming,
                ShapeBucket.SmallLowK => PackingMode.ForceStreaming,
                ShapeBucket.TallThin => PackingMode.ForceStreaming,
                ShapeBucket.MediumSquare => PackingMode.ForcePackBoth,
                ShapeBucket.ThinK => PackingMode.ForcePackAOnly,
                ShapeBucket.MediumMWide => PackingMode.ForcePackBoth, // #653: see cpu≤1 note — PackAOnly is pathological for non-transposed MediumMWide
                _ => PackingMode.ForcePackBoth,
            };
        }

        // Conservative default for all other hardware (avx512/sse2/neon/scalar):
        // tiny/small-low-K/tall-thin→Streaming, low-K→PackAOnly, medium-M-wide→PackAOnly,
        // else PackBoth. Refined by measurement (Phase 3) per fingerprint.
        return bucket switch
        {
            ShapeBucket.TinyCube => PackingMode.ForceStreaming,
            ShapeBucket.SmallLowK => PackingMode.ForceStreaming,
            ShapeBucket.TallThin => PackingMode.ForceStreaming,
            ShapeBucket.ThinK => PackingMode.ForcePackAOnly,
            ShapeBucket.MediumSquare => PackingMode.ForcePackAOnly,
            ShapeBucket.MediumMWide => PackingMode.ForcePackBoth, // #653: see cpu≤1 note — PackAOnly is pathological for non-transposed MediumMWide
            _ => PackingMode.ForcePackBoth,
        };
    }
}
