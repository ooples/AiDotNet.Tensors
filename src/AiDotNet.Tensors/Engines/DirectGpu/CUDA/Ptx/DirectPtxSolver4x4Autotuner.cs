using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Bounded cold-path block-geometry tuner shared by the 4x4 solver family.</summary>
internal static class DirectPtxSolver4x4Autotuner
{
    internal const int DefaultBlockThreads = 128;
    internal const int TuneWarmups = 30;
    internal const int TuneSamples = 101;
    internal const int TuneLaunchesPerSample = 10;
    // At B=1024, competing geometries differ by fractions of a microsecond and a 10-launch
    // graph is shorter than one WDDM scheduling quantum. A 1,000-launch captured window makes
    // that one shape reproducible without multiplying the cost of the larger solver buckets.
    internal const int SmallBatchTuneLaunchesPerSample = 1000;
    // Prefer the first (smaller) geometry when candidates are effectively tied. This prevents
    // sub-percent device jitter from making the same hardware choose a different launch plan on
    // every cold start while still admitting any reproducible improvement above the noise band.
    internal const float MinimumRelativeImprovement = 0.01f;
    // Tiny batches are launch-wave limited. A 16-thread CTA still consumes one hardware warp,
    // but B=1024 exposes 64 independent CTAs instead of the 32 available at one full warp per
    // CTA. That can occupy every SM on 64-SM-class devices while preserving one-thread-per-matrix
    // semantics; the measured tuner rejects it on devices where the half-warp waste costs more.
    internal static ReadOnlySpan<int> Candidates => [16, 32, 64, 128, 256];

    internal static int Select(Func<int, float[]> measure)
    {
        PtxCompat.ThrowIfNull(measure, nameof(measure));
        int best = DefaultBlockThreads;
        float bestMedian = float.PositiveInfinity;
        foreach (int candidate in Candidates)
        {
            float[] samples = measure(candidate);
            if (samples.Length == 0) throw new InvalidOperationException("Autotune returned no samples.");
            Array.Sort(samples);
            float median = samples[samples.Length / 2];
            if (median < bestMedian * (1f - MinimumRelativeImprovement))
            {
                bestMedian = median;
                best = candidate;
            }
        }
        return best;
    }

    internal static int LaunchesPerSample(int batchCount) =>
        batchCount == 1024 ? SmallBatchTuneLaunchesPerSample : TuneLaunchesPerSample;

    internal static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (16 or 32 or 64 or 128 or 256))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Solver autotune candidates are exactly 16, 32, 64, 128, and 256 threads.");
    }
}
