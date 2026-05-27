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
    internal enum ShapeBucket { TinyCube, SmallLowK, MediumSquare, ThinK, WideCompute, Large }

    internal static ShapeBucket Bucket(int m, int n, int k)
    {
        long work = (long)m * n * k;
        if (work <= 300_000L) return ShapeBucket.TinyCube;
        if (k <= 128 && work < 1_000_000L) return ShapeBucket.SmallLowK;
        if (k <= 128 && m == n) return ShapeBucket.MediumSquare;
        if (k <= 128) return ShapeBucket.ThinK;
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
                ShapeBucket.MediumSquare => PackingMode.ForceStreaming,
                ShapeBucket.ThinK => PackingMode.ForceStreaming,
                _ => PackingMode.ForcePackBoth,
            };
        }

        // avx2 large-core (>16T, Ryzen 3950X #464): blocking wins on medium squares.
        if (key.Simd == "avx2") // CpuBucket == 2
        {
            return bucket switch
            {
                ShapeBucket.TinyCube => PackingMode.ForceStreaming,
                ShapeBucket.SmallLowK => PackingMode.ForceStreaming,
                ShapeBucket.MediumSquare => PackingMode.ForcePackBoth,
                ShapeBucket.ThinK => PackingMode.ForcePackAOnly,
                _ => PackingMode.ForcePackBoth,
            };
        }

        // Conservative default for all other hardware (avx512/sse2/neon/scalar):
        // tiny→Streaming, small-low-K→Streaming, low-K→PackAOnly, else PackBoth.
        // Refined by measurement (Phase 3) per fingerprint.
        return bucket switch
        {
            ShapeBucket.TinyCube => PackingMode.ForceStreaming,
            ShapeBucket.SmallLowK => PackingMode.ForceStreaming,
            ShapeBucket.ThinK => PackingMode.ForcePackAOnly,
            _ => PackingMode.ForcePackBoth,
        };
    }
}
