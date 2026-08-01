using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Bounded cold-path block-geometry tuner shared by the 4x4 solver family.</summary>
internal static class DirectPtxSolver4x4Autotuner
{
    internal const int DefaultBlockThreads = 128;
    internal const int TuneWarmups = 30;
    internal const int TuneSamples = 101;
    internal const int TuneLaunchesPerSample = 10;
    // B=1024 needs the one-warp geometry to expose at least one CTA per SM on
    // common 20-32 SM devices. Starting at 64 threads caps that shape at 16
    // CTAs and can leave half the GPU idle regardless of kernel quality.
    internal static ReadOnlySpan<int> Candidates => [32, 64, 128, 256];

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
            if (median < bestMedian) { bestMedian = median; best = candidate; }
        }
        return best;
    }

    internal static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (32 or 64 or 128 or 256))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Solver autotune candidates are exactly 32, 64, 128, and 256 threads.");
    }
}
