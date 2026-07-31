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
/// Three changes fix it, and all matter. Timing moves to CUDA EVENTS on the stream, which
/// excludes host-side launch and synchronisation cost -- a host Stopwatch around a batch of
/// short kernels measures the driver as much as the kernel, which is exactly the regime the
/// small shapes on this campaign sat in. A/B comparisons use self-consistency-gated A/B/B/A
/// brackets, so an interrupted half-bracket is discarded from both the timing and ratio instead
/// of poisoning either side. And every measurement is REPEATED until its spread converges, so
/// the caller learns how much to trust it.
/// </para>
/// <para>
/// The rule the whole class exists to enforce: <b>an unstable measurement is reported as
/// unstable, never as a number.</b> A single confident-looking figure derived from samples
/// that disagree by half is worse than no figure, because it gets acted on.
/// </para>
/// </remarks>
internal static class StableTimer
{
    private const string TraceEnvironmentVariable = "AIDOTNET_STABLE_TIMER_TRACE";
    private static readonly bool TraceEnabled = string.Equals(
        Environment.GetEnvironmentVariable(TraceEnvironmentVariable), "1", StringComparison.Ordinal);

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
        public string Describe() => Samples == 0
            ? "NO CLEAN SAMPLE"
            : Stable
                ? Microseconds.ToString("0.0", CultureInfo.InvariantCulture) + " us"
                : "UNSTABLE +-" + (RelativeSpread * 100).ToString("0", CultureInfo.InvariantCulture) + "%";
    }

    /// <summary>
    /// Two measurements sampled next to each other, plus the distribution of their
    /// within-sample ratios.
    /// </summary>
    /// <param name="A">First timed operation.</param>
    /// <param name="B">Second timed operation.</param>
    /// <param name="Ratio">Median of A/B for each paired sample.</param>
    /// <param name="RelativeSpread">Spread of the paired ratios.</param>
    /// <param name="Samples">Number of paired samples.</param>
    /// <param name="RequiredSpread">Convergence threshold requested by the caller.</param>
    internal readonly record struct PairResult(
        Result A, Result B, double Ratio, double RelativeSpread, int Samples,
        double RequiredSpread)
    {
        /// <summary>
        /// A comparison is actionable only when both timings and their paired ratio
        /// independently converged.
        /// </summary>
        public bool Stable => A.Stable && B.Stable && RelativeSpread <= RequiredSpread;

        /// <summary>Formats the paired ratio, or refuses to print an unstable number.</summary>
        public string DescribeRatio() => Stable
            ? Ratio.ToString("0.00", CultureInfo.InvariantCulture) + "x"
            : "-";
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
    /// A size proxy -- MACs, or bytes moved -- used only if measured-duration calibration
    /// cannot produce a valid sample. Tensor extent alone cannot predict duration when two
    /// algorithms for the same operation differ by orders of magnitude.
    /// </param>
    /// <param name="maxAttempts">Samples to take before giving up on convergence.</param>
    internal static Result Measure(
        DirectPtxRuntime runtime, Action launch, long workUnits, int maxAttempts = 15)
    {
        if (runtime is null) throw new ArgumentNullException(nameof(runtime));
        if (launch is null) throw new ArgumentNullException(nameof(launch));

        int iterations = CalibrateDeviceIterations(runtime, launch, workUnits);

        var samples = new List<double>(3);

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            // Calibration already warmed the launch. Later attempts follow immediately, so the
            // clocks and caches are already where the measurement wants them.
            // MeasureKernelMilliseconds already divides by the iteration count -- it returns
            // milliseconds PER LAUNCH, not for the batch. Dividing again here made a 155 us
            // kernel read as 0.31 us and put rows at 6447% of their ceiling, which is the kind
            // of impossible number that at least announces itself.
            float msPerLaunch = runtime.MeasureKernelMilliseconds(launch, 0, iterations);
            AddToConsecutiveWindow(samples, msPerLaunch * 1000.0);

            // Three samples is the fewest from which a spread means anything.
            if (samples.Count >= 3 && SpreadOf(samples) <= StableSpread) break;
        }

        double spread = SpreadOf(samples);
        return new Result(
            Median(samples), spread, samples.Count,
            samples.Count >= 3 && spread <= StableSpread);
    }

    /// <summary>
    /// CUDA-event-times two launches as adjacent A/B/B/A batches and forms the ratio inside
    /// each sample. Each side is independently calibrated to a bounded-duration batch. A sample
    /// is the median of five internally consistent brackets; interrupted brackets are retried.
    /// </summary>
    internal static PairResult MeasurePair(
        DirectPtxRuntime runtime,
        Action launchA, Action launchB,
        long workUnitsA, long workUnitsB,
        int maxAttempts = 15,
        double targetSpread = StableSpread)
    {
        if (runtime is null) throw new ArgumentNullException(nameof(runtime));
        if (launchA is null) throw new ArgumentNullException(nameof(launchA));
        if (launchB is null) throw new ArgumentNullException(nameof(launchB));
        if (targetSpread <= 0 || targetSpread > StableSpread ||
            double.IsNaN(targetSpread))
            throw new ArgumentOutOfRangeException(nameof(targetSpread));

        int iterationsA = CalibrateDeviceIterations(runtime, launchA, workUnitsA);
        int iterationsB = CalibrateDeviceIterations(runtime, launchB, workUnitsB);
        Trace($"device pair iterations A={iterationsA} B={iterationsB}");

        var samplesA = new List<double>(3);
        var samplesB = new List<double>(3);
        var ratios = new List<double>(3);

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            var bracketA = new List<double>(BracketReplicates);
            var bracketB = new List<double>(BracketReplicates);
            var bracketRatios = new List<double>(BracketReplicates);
            for (int candidate = 0;
                 bracketA.Count < BracketReplicates && candidate < MaximumBracketCandidates;
                 candidate++)
            {
                double aFirst = runtime.MeasureKernelMilliseconds(launchA, 0, iterationsA) * 1000.0;
                double bFirst = runtime.MeasureKernelMilliseconds(launchB, 0, iterationsB) * 1000.0;
                double bSecond = runtime.MeasureKernelMilliseconds(launchB, 0, iterationsB) * 1000.0;
                double aSecond = runtime.MeasureKernelMilliseconds(launchA, 0, iterationsA) * 1000.0;
                bool accepted = BracketIsSelfConsistent(aFirst, aSecond) &&
                    BracketIsSelfConsistent(bFirst, bSecond);
                Trace($"device attempt {attempt + 1} bracket candidate {candidate + 1}: " +
                    $"A=({aFirst:0.000},{aSecond:0.000})us " +
                    $"B=({bFirst:0.000},{bSecond:0.000})us " +
                    (accepted ? "accepted" : "rejected"));
                if (!accepted) continue;

                double bracketAverageA = (aFirst + aSecond) * 0.5;
                double bracketAverageB = (bFirst + bSecond) * 0.5;
                bracketA.Add(bracketAverageA);
                bracketB.Add(bracketAverageB);
                bracketRatios.Add(bracketAverageA / bracketAverageB);
            }
            if (bracketA.Count < BracketReplicates)
            {
                Trace($"device pair sample {attempt + 1}: only {bracketA.Count} clean brackets; skipped");
                continue;
            }
            double a = Median(bracketA);
            double b = Median(bracketB);
            double ratio = Median(bracketRatios);
            Trace($"device pair sample {attempt + 1}: A={a:0.000}us " +
                $"B={b:0.000}us ratio={ratio:0.0000}");
            AddToConsecutiveWindow(samplesA, a);
            AddToConsecutiveWindow(samplesB, b);
            AddToConsecutiveWindow(ratios, ratio);

            if (samplesA.Count >= 3 &&
                SpreadOf(samplesA) <= targetSpread &&
                SpreadOf(samplesB) <= targetSpread &&
                SpreadOf(ratios) <= targetSpread)
            {
                break;
            }
        }

        return Pair(samplesA, samplesB, ratios, targetSpread);
    }

    /// <summary>
    /// Times a launch that does NOT go through the direct-PTX runtime, using the same
    /// convergence protocol.
    /// </summary>
    /// <remarks>
    /// The backend owns its own context and stream, so its kernels cannot be bracketed by
    /// events recorded on the direct-PTX stream. Host timing over a batch is the honest
    /// alternative, and the stability gate matters MORE here rather than less: host timing
    /// includes launch cost, so an unstable result is the expected outcome for a short kernel
    /// and must be reported as one rather than averaged into a confident number.
    ///
    /// Both sides of a head-to-head must use the same method, or the comparison measures the
    /// methods.
    /// </remarks>
    internal static Result MeasureHost(
        Action launch, Action synchronize, long workUnits, int maxAttempts = 15)
    {
        if (launch is null) throw new ArgumentNullException(nameof(launch));
        if (synchronize is null) throw new ArgumentNullException(nameof(synchronize));

        int iterations = CalibrateHostIterations(launch, synchronize, workUnits);

        var samples = new List<double>(3);
        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            AddToConsecutiveWindow(
                samples, TimeHostBatch(launch, synchronize, iterations));

            if (samples.Count >= 3 && SpreadOf(samples) <= StableSpread) break;
        }

        double spread = SpreadOf(samples);
        return new Result(
            Median(samples), spread, samples.Count,
            samples.Count >= 3 && spread <= StableSpread);
    }

    /// <summary>
    /// Host-times two operations as adjacent A/B/B/A batches and summarizes the ratios formed
    /// inside each sample. A sample is the median of five internally consistent brackets;
    /// interrupted brackets are retried.
    /// </summary>
    /// <remarks>
    /// Measuring every A sample and then every B sample lets clock and thermal drift become
    /// part of the apparent speedup. The current protocol brackets both sides: A, B, B, A,
    /// then averages each pair before forming A/B. The two runtimes are independently
    /// calibrated to the same target batch duration (for example an O(VDN) deterministic
    /// embedding backward against an O(ND) atomic form), so neither side creates a multi-second
    /// batch merely because its tensor extent matches a much faster algorithm.
    /// </remarks>
    internal static PairResult MeasureHostPair(
        Action launchA, Action synchronizeA, long workUnitsA,
        Action launchB, Action synchronizeB, long workUnitsB,
        int maxAttempts = 15)
    {
        if (launchA is null) throw new ArgumentNullException(nameof(launchA));
        if (synchronizeA is null) throw new ArgumentNullException(nameof(synchronizeA));
        if (launchB is null) throw new ArgumentNullException(nameof(launchB));
        if (synchronizeB is null) throw new ArgumentNullException(nameof(synchronizeB));

        int iterationsA = CalibrateHostIterations(launchA, synchronizeA, workUnitsA);
        int iterationsB = CalibrateHostIterations(launchB, synchronizeB, workUnitsB);
        Trace($"host pair iterations A={iterationsA} B={iterationsB}");

        var samplesA = new List<double>(3);
        var samplesB = new List<double>(3);
        var ratios = new List<double>(3);

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            var bracketA = new List<double>(BracketReplicates);
            var bracketB = new List<double>(BracketReplicates);
            var bracketRatios = new List<double>(BracketReplicates);
            for (int candidate = 0;
                 bracketA.Count < BracketReplicates && candidate < MaximumBracketCandidates;
                 candidate++)
            {
                double aFirst = TimeHostBatch(launchA, synchronizeA, iterationsA);
                double bFirst = TimeHostBatch(launchB, synchronizeB, iterationsB);
                double bSecond = TimeHostBatch(launchB, synchronizeB, iterationsB);
                double aSecond = TimeHostBatch(launchA, synchronizeA, iterationsA);
                bool accepted = BracketIsSelfConsistent(aFirst, aSecond) &&
                    BracketIsSelfConsistent(bFirst, bSecond);
                Trace($"host attempt {attempt + 1} bracket candidate {candidate + 1}: " +
                    $"A=({aFirst:0.000},{aSecond:0.000})us " +
                    $"B=({bFirst:0.000},{bSecond:0.000})us " +
                    (accepted ? "accepted" : "rejected"));
                if (!accepted) continue;

                double bracketAverageA = (aFirst + aSecond) * 0.5;
                double bracketAverageB = (bFirst + bSecond) * 0.5;
                bracketA.Add(bracketAverageA);
                bracketB.Add(bracketAverageB);
                bracketRatios.Add(bracketAverageA / bracketAverageB);
            }
            if (bracketA.Count < BracketReplicates)
            {
                Trace($"host pair sample {attempt + 1}: only {bracketA.Count} clean brackets; skipped");
                continue;
            }
            double a = Median(bracketA);
            double b = Median(bracketB);
            double ratio = Median(bracketRatios);
            Trace($"host pair sample {attempt + 1}: A={a:0.000}us " +
                $"B={b:0.000}us ratio={ratio:0.0000}");
            AddToConsecutiveWindow(samplesA, a);
            AddToConsecutiveWindow(samplesB, b);
            AddToConsecutiveWindow(ratios, ratio);

            if (samplesA.Count >= 3 &&
                SpreadOf(samplesA) <= StableSpread &&
                SpreadOf(samplesB) <= StableSpread &&
                SpreadOf(ratios) <= StableSpread)
            {
                break;
            }
        }

        return Pair(samplesA, samplesB, ratios, StableSpread);
    }

    private static PairResult Pair(
        List<double> samplesA, List<double> samplesB, List<double> ratios,
        double requiredSpread)
    {
        if (samplesA.Count == 0 || samplesB.Count == 0 || ratios.Count == 0)
        {
            var missing = new Result(0, double.PositiveInfinity, 0, Stable: false);
            return new PairResult(
                missing, missing, 0, double.PositiveInfinity, 0, requiredSpread);
        }

        double spreadA = SpreadOf(samplesA);
        double spreadB = SpreadOf(samplesB);
        double ratioSpread = SpreadOf(ratios);
        return new PairResult(
            new Result(Median(samplesA), spreadA, samplesA.Count,
                samplesA.Count >= 3 && spreadA <= requiredSpread),
            new Result(Median(samplesB), spreadB, samplesB.Count,
                samplesB.Count >= 3 && spreadB <= requiredSpread),
            Median(ratios), ratioSpread, ratios.Count, requiredSpread);
    }

    /// <summary>
    /// Keeps the latest three consecutive samples. A max-min gate over every attempt can
    /// never recover after one WDDM preemption because adding clean samples cannot lower
    /// the historical maximum. A consecutive window still requires three agreeing
    /// measurements, while letting later clean evidence replace a contaminated batch.
    /// </summary>
    private static void AddToConsecutiveWindow(List<double> samples, double sample)
    {
        samples.Add(sample);
        if (samples.Count > 3) samples.RemoveAt(0);
    }

    private static int CalibrateDeviceIterations(
        DirectPtxRuntime runtime, Action launch, long workUnits)
    {
        float milliseconds = runtime.MeasureKernelMilliseconds(launch, warmup: 3, iterations: 3);
        int iterations = IterationsForMeasuredDuration(milliseconds, workUnits);
        Trace($"device calibration initial={milliseconds * 1000.0:0.000}us count={iterations}");
        for (int refinement = 0; refinement < 2; refinement++)
        {
            milliseconds = runtime.MeasureKernelMilliseconds(launch, 0, iterations);
            int revised = IterationsForMeasuredDuration(milliseconds, workUnits);
            Trace($"device calibration refine {refinement + 1}={milliseconds * 1000.0:0.000}us " +
                $"count={iterations}->{revised}");
            if (CountsAreClose(iterations, revised)) break;
            iterations = revised;
        }
        return iterations;
    }

    private static int CalibrateHostIterations(
        Action launch, Action synchronize, long workUnits)
    {
        for (int i = 0; i < 3; i++) launch();
        synchronize();
        double microseconds = TimeHostBatch(launch, synchronize, iterations: 1);
        int iterations = IterationsForMeasuredDuration(microseconds / 1000.0, workUnits);
        Trace($"host calibration initial={microseconds:0.000}us count={iterations}");
        for (int refinement = 0; refinement < 2; refinement++)
        {
            microseconds = TimeHostBatch(launch, synchronize, iterations);
            int revised = IterationsForMeasuredDuration(microseconds / 1000.0, workUnits);
            Trace($"host calibration refine {refinement + 1}={microseconds:0.000}us " +
                $"count={iterations}->{revised}");
            if (CountsAreClose(iterations, revised)) break;
            iterations = revised;
        }
        return iterations;
    }

    private static bool CountsAreClose(int left, int right) =>
        Math.Abs((long)left - right) <= Math.Max(1L, left / 10L);

    private static bool BracketIsSelfConsistent(double first, double second)
    {
        double median = (first + second) * 0.5;
        return median > 0 && Math.Abs(first - second) / median <= BracketInternalSpread;
    }

    private static void Trace(string message)
    {
        if (TraceEnabled) Console.Error.WriteLine("[stable-timer] " + message);
    }

    private static double TimeHostBatch(Action launch, Action synchronize, int iterations)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++) launch();
        synchronize();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / iterations;
    }

    private const int BracketReplicates = 5;
    private const int MaximumBracketCandidates = 15;
    private const double BracketInternalSpread = 0.10;
    private const double TargetBatchMilliseconds = 10.0;
    private const int MaximumIterations = 10_000;

    /// <summary>
    /// Keeps each side of a bracket near 10 ms. An A/B/B/A bracket therefore spans about 40 ms,
    /// short enough to remain inside one desktop-GPU scheduling phase. Five internally consistent
    /// brackets form one robust sample, so the sample still contains about 200 ms of accepted
    /// device work while its median rejects a minority of WDDM-interrupted brackets. Duration,
    /// rather than tensor size, is the meaningful bound: a 2M-element sparsemax and masked fill
    /// move similar bytes but can differ by hundreds of times in runtime.
    /// </summary>
    private static int IterationsForMeasuredDuration(double milliseconds, long workUnits)
    {
        if (milliseconds > 0 && double.IsFinite(milliseconds))
        {
            double target = Math.Round(TargetBatchMilliseconds / milliseconds);
            return (int)Math.Max(1, Math.Min(MaximumIterations, target));
        }

        return IterationsFor(workUnits);
    }

    /// <summary>Conservative fallback when duration calibration is unavailable.</summary>
    private static int IterationsFor(long workUnits) =>
        // The old 20G/500 ceiling left a 35 us optimized convolution with only about
        // 6 ms of device work per sample. One ordinary WDDM preemption then dominated
        // that sample and permanently poisoned the strict max-min gate. Keep the gate
        // unchanged, but give short kernels a long enough event-timed batch that host
        // scheduling is amortized; calibrated large kernels can use a single launch.
        (int)Math.Max(1, Math.Min(MaximumIterations, 80_000_000_000L / Math.Max(1, workUnits)));

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
