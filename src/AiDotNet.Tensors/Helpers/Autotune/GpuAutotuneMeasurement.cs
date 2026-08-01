using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Shared stability gate for production GPU autotuning. A candidate is eligible
/// for promotion only when its CUDA-event sample distribution is finite,
/// positive, and sufficiently tight. This prevents a transiently fast sample
/// from becoming a persistent per-device winner.
/// </summary>
public static class GpuAutotuneMeasurement
{
    private sealed class UnstableTimingException : InvalidOperationException
    {
        internal UnstableTimingException(string message) : base(message) { }
    }

    /// <summary>The default maximum accepted p95/median timing ratio.</summary>
    public const double DefaultMaxP95ToMedian = 1.05;

    /// <summary>
    /// Returns the median milliseconds for a stable sample distribution.
    /// Throws when the evidence is incomplete, invalid, or noisier than
    /// <paramref name="maxP95ToMedian"/> so the caller can reject the candidate.
    /// </summary>
    public static double StableMedianMilliseconds(
        IReadOnlyList<float> milliseconds,
        double maxP95ToMedian = DefaultMaxP95ToMedian)
        => StableMedianMillisecondsCore(milliseconds, maxP95ToMedian, retrySignal: false);

    private static double StableMedianMillisecondsCore(
        IReadOnlyList<float> milliseconds,
        double maxP95ToMedian,
        bool retrySignal)
    {
        if (milliseconds is null) throw new ArgumentNullException(nameof(milliseconds));
        if (milliseconds.Count < 3)
            throw new ArgumentException("At least three timing samples are required.", nameof(milliseconds));
        if (double.IsNaN(maxP95ToMedian) || double.IsInfinity(maxP95ToMedian) ||
            maxP95ToMedian < 1.0)
            throw new ArgumentOutOfRangeException(nameof(maxP95ToMedian));

        var sorted = new float[milliseconds.Count];
        for (int i = 0; i < milliseconds.Count; i++)
        {
            float sample = milliseconds[i];
            if (float.IsNaN(sample) || float.IsInfinity(sample) || sample <= 0f)
                throw new InvalidOperationException("GPU timing samples must be finite and positive.");
            sorted[i] = sample;
        }
        Array.Sort(sorted);

        (double median, double p95) = SortedDistributionStatistics(sorted);
        double p95ToMedian = p95 / median;
        if (p95ToMedian > maxP95ToMedian)
        {
            string message =
                $"GPU timing is unstable: p95/median={p95ToMedian:F4}, limit={maxP95ToMedian:F4}.";
            if (retrySignal) throw new UnstableTimingException(message);
            throw new InvalidOperationException(message);
        }

        return median;
    }

    /// <summary>
    /// Returns the conventional median and nearest-rank p95 for an ascending
    /// timing distribution. Kept in one place so diagnostics and the stability
    /// gate cannot report different percentiles for the same samples.
    /// </summary>
    internal static (double Median, double P95) SortedDistributionStatistics(
        IReadOnlyList<float> sorted)
    {
        if (sorted is null) throw new ArgumentNullException(nameof(sorted));
        if (sorted.Count == 0)
            throw new ArgumentException("At least one timing sample is required.", nameof(sorted));

        int middle = sorted.Count / 2;
        double median = sorted.Count % 2 == 0
            ? ((double)sorted[middle - 1] + sorted[middle]) / 2.0
            : sorted[middle];
        int p95Index = Math.Max(0, (int)Math.Ceiling(sorted.Count * 0.95) - 1);
        return (median, sorted[p95Index]);
    }

    /// <summary>Converts a stable median time and operation count to GFLOP/s.</summary>
    public static double StableGflops(
        IReadOnlyList<float> milliseconds,
        long operations,
        double maxP95ToMedian = DefaultMaxP95ToMedian)
    {
        if (operations <= 0) throw new ArgumentOutOfRangeException(nameof(operations));
        double medianMilliseconds = StableMedianMilliseconds(milliseconds, maxP95ToMedian);
        return operations / (medianMilliseconds * 1_000_000.0);
    }

    /// <summary>
    /// Measures with progressively larger launch groups until the averaged
    /// CUDA-event samples satisfy the stability gate. Very short kernels can
    /// otherwise appear noisy because event quantization and record overhead
    /// are large relative to one launch. The measurement callback itself is
    /// outside the retry catch, so launch failures are never mistaken for
    /// timing noise.
    /// </summary>
    public static double AdaptiveStableMedianMilliseconds(
        Func<int, IReadOnlyList<float>> measureSamples,
        int initialLaunchesPerSample = 8,
        int maxLaunchesPerSample = 512,
        double maxP95ToMedian = DefaultMaxP95ToMedian)
    {
        if (measureSamples is null) throw new ArgumentNullException(nameof(measureSamples));
        if (initialLaunchesPerSample <= 0)
            throw new ArgumentOutOfRangeException(nameof(initialLaunchesPerSample));
        if (maxLaunchesPerSample < initialLaunchesPerSample)
            throw new ArgumentOutOfRangeException(nameof(maxLaunchesPerSample));

        int launches = initialLaunchesPerSample;
        while (true)
        {
            // Keep launch/module failures outside the stability retry. Only a
            // completed timing distribution is eligible for a larger group.
            IReadOnlyList<float> samples = measureSamples(launches);
            try
            {
                return StableMedianMillisecondsCore(samples, maxP95ToMedian, retrySignal: true);
            }
            catch (UnstableTimingException) when (launches < maxLaunchesPerSample)
            {
                launches = launches > maxLaunchesPerSample / 4
                    ? maxLaunchesPerSample
                    : launches * 4;
            }
        }
    }

    /// <summary>
    /// Adaptive launch-group measurement converted to GFLOP/s.
    /// </summary>
    public static double AdaptiveStableGflops(
        Func<int, IReadOnlyList<float>> measureSamples,
        long operations,
        int initialLaunchesPerSample = 8,
        int maxLaunchesPerSample = 512,
        double maxP95ToMedian = DefaultMaxP95ToMedian)
    {
        if (operations <= 0) throw new ArgumentOutOfRangeException(nameof(operations));
        double medianMilliseconds = AdaptiveStableMedianMilliseconds(
            measureSamples, initialLaunchesPerSample, maxLaunchesPerSample, maxP95ToMedian);
        return operations / (medianMilliseconds * 1_000_000.0);
    }
}
