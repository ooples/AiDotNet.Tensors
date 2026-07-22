#if NET5_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Bounded cold-path block-geometry tuner shared by the 4x4 solver family.</summary>
internal static class DirectPtxSolver4x4Autotuner
{
    internal const int DefaultBlockThreads = 128;
    internal static ReadOnlySpan<int> Candidates => [64, 128, 256];

    internal static int Select(Func<int, float[]> measure)
    {
        ArgumentNullException.ThrowIfNull(measure);
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
        if (blockThreads is not (64 or 128 or 256))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Solver autotune candidates are exactly 64, 128, and 256 threads.");
    }
}
#endif
