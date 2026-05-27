using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Seed tier of the unified (hardwareKey, shapeBucket) → strategy map (#375). Hardcoded
/// per-{simd,vendor,cpuBucket} routing measured on real hardware; the cold-start default
/// before the learned disk cache populates. Pure, no I/O. The learned cache (Phase 2)
/// overrides this per shape; the background autotuner (Phase 3) populates that cache.
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

        // amd-avx2 mid-core (≤16T, this box): k≤128 wins on Streaming (measured A/B 2026-05).
        if (key.Simd == "avx2" && key.CpuBucket <= 1)
        {
            return bucket switch
            {
                ShapeBucket.TinyCube => PackingMode.ForceStreaming,
                ShapeBucket.SmallLowK => PackingMode.ForceStreaming,
                ShapeBucket.TallThin => PackingMode.ForceStreaming,
                ShapeBucket.MediumSquare => PackingMode.ForceStreaming,   // 128³: Streaming 80 vs PackBoth 18
                ShapeBucket.ThinK => PackingMode.ForceStreaming,          // 512×512×64: Streaming 80 vs PackBoth 59
                ShapeBucket.MediumMWide => PackingMode.ForcePackAOnly,    // 128×768×768: PackAOnly 110 vs PackBoth 34
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
                ShapeBucket.MediumMWide => PackingMode.ForcePackAOnly,
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
            ShapeBucket.MediumMWide => PackingMode.ForcePackAOnly,
            _ => PackingMode.ForcePackBoth,
        };
    }
}
