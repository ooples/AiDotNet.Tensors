// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Times a kernel with CUDA events and reports whether the result is stable enough to act on.
/// </summary>
/// <remarks>
/// <para>
/// THIS EXISTS BECAUSE THE NUMBERS MOVED MORE THAN THE EFFECTS BEING MEASURED. On one
/// elementwise kernel, three consecutive runs on an idle GPU at locked clocks reported 69.2%,
/// 46.5% and 73.0% of roofline -- a 1.57x spread with no code change between them. Findings
/// were ranked, work was scheduled, and one lever was built and reverted on numbers of that
/// quality.
/// </para>
/// <para>
/// Two changes fix it, and both matter. Timing moves to CUDA EVENTS on the stream, which
/// excludes host-side launch and synchronisation cost -- a host Stopwatch around a batch of
/// short kernels measures the driver as much as the kernel, which is exactly the regime the
/// small shapes on this campaign sat in. And every measurement is REPEATED until its spread
/// converges, so the caller learns how much to trust it.
/// </para>
/// <para>
/// The rule the whole class exists to enforce: <b>an unstable measurement is reported as
/// unstable, never as a number.</b> A single confident-looking figure derived from samples
/// that disagree by half is worse than no figure, because it gets acted on.
/// </para>
/// </remarks>
internal static class StableTimer
{
    /// <summary>A timing result, with the evidence for how much it can be trusted.</summary>
    /// <param name="Microseconds">Median of the accepted samples.</param>
    /// <param name="RelativeSpread">
    /// (max - min) / median across samples. The headline uncertainty.
    /// </param>
    /// <param name="Samples">How many samples were taken.</param>
    /// <param name="Stable">Whether the spread came within tolerance.</param>
    internal readonly record struct Result(
        double Microseconds, double RelativeSpread, int Samples, bool Stable)
    {
        /// <summary>Formats as a value with its uncertainty, or as a refusal.</summary>
        /// <remarks>
        /// An unstable result prints the spread rather than the value on purpose. Printing
        /// "44.4 us" from samples ranging 30 to 70 invites exactly the false conclusion this
        /// class was written to prevent.
        /// </remarks>
        public string Describe() => Stable
            ? Microseconds.ToString("0.0", CultureInfo.InvariantCulture) + " us"
            : "UNSTABLE +-" + (RelativeSpread * 100).ToString("0", CultureInfo.InvariantCulture) + "%";
    }

    /// <summary>Spread at or below which a measurement is considered stable.</summary>
    /// <remarks>
    /// Five percent is loose enough that a well-behaved kernel passes on the first attempt and
    /// tight enough to reject the 57% spread that prompted this. It is not a claim that 5% is
    /// negligible -- a 1.05x result should still not be called a win.
    /// </remarks>
    internal const double StableSpread = 0.05;

    /// <summary>
    /// Times <paramref name="launch"/>, repeating until the spread converges or attempts run out.
    /// </summary>
    /// <param name="runtime">Device runtime; supplies the event-timed measurement.</param>
    /// <param name="launch">One kernel launch.</param>
    /// <param name="workUnits">
    /// A size proxy -- MACs, or bytes moved -- used to pick an iteration count. A fixed count
    /// is wrong at both ends: too few for a 20 us kernel to escape launch noise, and minutes
    /// of wall clock for a 100 ms one.
    /// </param>
    /// <param name="maxAttempts">Samples to take before giving up on convergence.</param>
    internal static Result Measure(
        DirectPtxRuntime runtime, Action launch, long workUnits, int maxAttempts = 7)
    {
        if (runtime is null) throw new ArgumentNullException(nameof(runtime));
        if (launch is null) throw new ArgumentNullException(nameof(launch));

        int iterations = IterationsFor(workUnits);
        int warmup = Math.Max(3, iterations / 10);

        var samples = new List<double>(maxAttempts);

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            // Warm up on the FIRST attempt only. Later attempts follow immediately, so the
            // clocks and caches are already where the measurement wants them; re-warming would
            // just spend time re-reaching the same state.
            // MeasureKernelMilliseconds already divides by the iteration count -- it returns
            // milliseconds PER LAUNCH, not for the batch. Dividing again here made a 155 us
            // kernel read as 0.31 us and put rows at 6447% of their ceiling, which is the kind
            // of impossible number that at least announces itself.
            float msPerLaunch = runtime.MeasureKernelMilliseconds(
                launch, attempt == 0 ? warmup : 0, iterations);
            samples.Add(msPerLaunch * 1000.0);

            // Three samples is the fewest from which a spread means anything.
            if (samples.Count >= 3 && SpreadOf(samples) <= StableSpread) break;
        }

        double spread = SpreadOf(samples);
        return new Result(Median(samples), spread, samples.Count, spread <= StableSpread);
    }

    /// <summary>
    /// Iterations for a kernel of a given size: enough to swamp launch overhead, capped so a
    /// large kernel does not run for minutes.
    /// </summary>
    private static int IterationsFor(long workUnits) =>
        (int)Math.Max(5, Math.Min(500, 20_000_000_000L / Math.Max(1, workUnits)));

    /// <summary>(max - min) / median. Zero for a single sample, which is reported unstable.</summary>
    private static double SpreadOf(List<double> samples)
    {
        if (samples.Count < 2) return double.PositiveInfinity;

        double min = double.MaxValue, max = double.MinValue;
        foreach (double v in samples)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        double median = Median(samples);
        return median > 0 ? (max - min) / median : double.PositiveInfinity;
    }

    /// <summary>
    /// The MEDIAN, not the minimum.
    /// </summary>
    /// <remarks>
    /// A best-of-N minimum is the right summary when the noise is one-sided -- interference
    /// only ever makes a kernel slower -- and the wrong one when the question is stability,
    /// because the minimum hides exactly the variance being measured. The spread is reported
    /// alongside so nothing is hidden either way.
    /// </remarks>
    private static double Median(List<double> samples)
    {
        var ordered = new List<double>(samples);
        ordered.Sort();
        int mid = ordered.Count / 2;
        return ordered.Count % 2 == 1
            ? ordered[mid]
            : (ordered[mid - 1] + ordered[mid]) / 2.0;
    }
}
