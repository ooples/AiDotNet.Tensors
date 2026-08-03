// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
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
    /// <param name="Samples">
    /// Number of timing attempts performed, not the size of the retained
    /// three-sample convergence window.
    /// </param>
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

    /// <summary>
    /// Two measurements sampled next to each other, plus the distribution of their
    /// within-sample ratios.
    /// </summary>
    /// <param name="A">First timed operation.</param>
    /// <param name="B">Second timed operation.</param>
    /// <param name="Ratio">Median of A/B for each paired sample.</param>
    /// <param name="RelativeSpread">Spread of the paired ratios.</param>
    /// <param name="Samples">
    /// Number of paired timing attempts performed, not the size of the retained
    /// three-sample convergence window.
    /// </param>
    internal readonly record struct PairResult(
        Result A, Result B, double Ratio, double RelativeSpread, int Samples)
    {
        /// <summary>
        /// A comparison is actionable only when both timings and their paired ratio
        /// independently converged.
        /// </summary>
        public bool Stable => A.Stable && B.Stable && RelativeSpread <= StableSpread;

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
    /// A size proxy -- MACs, or bytes moved -- used to pick an iteration count. A fixed count
    /// is wrong at both ends: too few for a 20 us kernel to escape launch noise, and minutes
    /// of wall clock for a 100 ms one.
    /// </param>
    /// <param name="maxAttempts">Timing attempts to make before giving up on convergence.</param>
    internal static Result Measure(
        DirectPtxRuntime runtime, Action launch, long workUnits, int maxAttempts = 15)
    {
        if (runtime is null) throw new ArgumentNullException(nameof(runtime));
        if (launch is null) throw new ArgumentNullException(nameof(launch));

        int iterations = IterationsFor(workUnits);
        int warmup = Math.Max(3, iterations / 10);

        var samples = new List<double>(3);
        int attempts = 0;

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            attempts++;
            // Warm up on the FIRST attempt only. Later attempts follow immediately, so the
            // clocks and caches are already where the measurement wants them; re-warming would
            // just spend time re-reaching the same state.
            // MeasureKernelMilliseconds already divides by the iteration count -- it returns
            // milliseconds PER LAUNCH, not for the batch. Dividing again here made a 155 us
            // kernel read as 0.31 us and put rows at 6447% of their ceiling, which is the kind
            // of impossible number that at least announces itself.
            float msPerLaunch = runtime.MeasureKernelMilliseconds(
                launch, attempt == 0 ? warmup : 0, iterations);
            AddToConsecutiveWindow(samples, msPerLaunch * 1000.0);

            // Three samples is the fewest from which a spread means anything.
            if (samples.Count >= 3 && SpreadOf(samples) <= StableSpread) break;
        }

        double spread = SpreadOf(samples);
        return new Result(
            Median(samples), spread, attempts,
            samples.Count >= 3 && spread <= StableSpread);
    }

    /// <summary>
    /// CUDA-event-times two launches as adjacent A/B batches and forms the ratio inside
    /// each sample.
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

        int iterationsA = IterationsFor(workUnitsA);
        int iterationsB = IterationsFor(workUnitsB);

        // Calibrate with the ordinary work-size batch, then lengthen whichever lane receives
        // less device-time. Equal launch counts under-expose a fast specialization: in the
        // transposed-convolution search the 98 us baseline received about 196 ms per sample,
        // while its 23 us candidate received only 46 ms and repeatedly failed the same 5%
        // gate. The calibration batch is deliberately discarded rather than becoming a
        // favourable extra sample.
        double calibrationA = runtime.MeasureKernelMilliseconds(
            launchA, Math.Max(3, iterationsA / 10), iterationsA) * 1000.0;
        double calibrationB = runtime.MeasureKernelMilliseconds(
            launchB, Math.Max(3, iterationsB / 10), iterationsB) * 1000.0;
        (iterationsA, iterationsB) = CodegenMeasurementProtocol.CalibratePairIterations(
            iterationsA, calibrationA, iterationsB, calibrationB);

        var samplesA = new List<double>(3);
        var samplesB = new List<double>(3);
        var ratios = new List<double>(3);
        int attempts = 0;

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            attempts++;
            double a = runtime.MeasureKernelMilliseconds(
                launchA, 0, iterationsA) * 1000.0;
            double b = runtime.MeasureKernelMilliseconds(
                launchB, 0, iterationsB) * 1000.0;
            AddToConsecutiveWindow(samplesA, a);
            AddToConsecutiveWindow(samplesB, b);
            AddToConsecutiveWindow(ratios, a / b);

            if (samplesA.Count >= 3 &&
                SpreadOf(samplesA) <= targetSpread &&
                SpreadOf(samplesB) <= targetSpread &&
                SpreadOf(ratios) <= targetSpread)
            {
                break;
            }
        }

        return Pair(samplesA, samplesB, ratios, attempts);
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

        int iterations = IterationsFor(workUnits);

        Warm(launch, synchronize, iterations);

        var samples = new List<double>(3);
        int attempts = 0;
        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            attempts++;
            AddToConsecutiveWindow(
                samples, TimeHostBatch(launch, synchronize, iterations));

            if (samples.Count >= 3 && SpreadOf(samples) <= StableSpread) break;
        }

        double spread = SpreadOf(samples);
        return new Result(
            Median(samples), spread, attempts,
            samples.Count >= 3 && spread <= StableSpread);
    }

    /// <summary>
    /// Host-times two operations as adjacent A/B batches and summarizes the ratios formed
    /// inside each sample.
    /// </summary>
    /// <remarks>
    /// Measuring every A sample and then every B sample lets clock and thermal drift become
    /// part of the apparent speedup. The current protocol therefore pairs the operations:
    /// A batch, synchronize A, B batch, synchronize B, then A/B for that sample. The two
    /// runtimes may need different iteration counts (for example an O(VDN) deterministic
    /// embedding backward against an O(ND) atomic form), so elapsed time is normalized per
    /// launch before the ratio is formed.
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

        int iterationsA = IterationsFor(workUnitsA);
        int iterationsB = IterationsFor(workUnitsB);

        Warm(launchA, synchronizeA, iterationsA);
        Warm(launchB, synchronizeB, iterationsB);

        // Host-timed comparisons have the same exposure problem as CUDA-event pairs. Keep
        // the timing methods identical within a comparison, discard one calibration batch
        // per lane, and normalize the accepted samples per launch as before.
        double calibrationA = TimeHostBatch(launchA, synchronizeA, iterationsA);
        double calibrationB = TimeHostBatch(launchB, synchronizeB, iterationsB);
        (iterationsA, iterationsB) = CodegenMeasurementProtocol.CalibratePairIterations(
            iterationsA, calibrationA, iterationsB, calibrationB);

        var samplesA = new List<double>(3);
        var samplesB = new List<double>(3);
        var ratios = new List<double>(3);
        int attempts = 0;

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            attempts++;
            double a = TimeHostBatch(launchA, synchronizeA, iterationsA);
            double b = TimeHostBatch(launchB, synchronizeB, iterationsB);
            AddToConsecutiveWindow(samplesA, a);
            AddToConsecutiveWindow(samplesB, b);
            AddToConsecutiveWindow(ratios, a / b);

            if (samplesA.Count >= 3 &&
                SpreadOf(samplesA) <= StableSpread &&
                SpreadOf(samplesB) <= StableSpread &&
                SpreadOf(ratios) <= StableSpread)
            {
                break;
            }
        }

        return Pair(samplesA, samplesB, ratios, attempts);
    }

    /// <summary>
    /// Host-pairs two launches from separate GPU backends after calibrating a
    /// common batch length. AB/BA brackets remove launch-order bias; an odd
    /// median-of-brackets rejects an isolated desktop preemption while the
    /// unchanged outer three-sample spread gate decides actionability.
    /// </summary>
    internal static PairResult MeasureCalibratedHostPair(
        Action launchA,
        Action synchronizeA,
        Action launchB,
        Action synchronizeB,
        int operationsPerLaunchA = 1,
        int operationsPerLaunchB = 1,
        int warmups = 3,
        int maxAttempts = 15,
        double targetBatchMilliseconds = 5.0,
        int bracketsPerAttempt = 5)
    {
        if (launchA is null) throw new ArgumentNullException(nameof(launchA));
        if (synchronizeA is null) throw new ArgumentNullException(nameof(synchronizeA));
        if (launchB is null) throw new ArgumentNullException(nameof(launchB));
        if (synchronizeB is null) throw new ArgumentNullException(nameof(synchronizeB));
        if (operationsPerLaunchA <= 0)
            throw new ArgumentOutOfRangeException(nameof(operationsPerLaunchA));
        if (operationsPerLaunchB <= 0)
            throw new ArgumentOutOfRangeException(nameof(operationsPerLaunchB));
        if (warmups < 0) throw new ArgumentOutOfRangeException(nameof(warmups));
        if (maxAttempts < 3) throw new ArgumentOutOfRangeException(nameof(maxAttempts));
        if (!(targetBatchMilliseconds > 0) ||
            double.IsInfinity(targetBatchMilliseconds))
            throw new ArgumentOutOfRangeException(nameof(targetBatchMilliseconds));
        if (bracketsPerAttempt <= 0 || (bracketsPerAttempt & 1) == 0)
            throw new ArgumentOutOfRangeException(nameof(bracketsPerAttempt),
                "Use a positive odd bracket count so its median is an observed sample.");

        for (int i = 0; i < warmups; i++)
        {
            launchA();
            synchronizeA();
            launchB();
            synchronizeB();
        }

        const int calibrationLaunches = 3;
        double calibrationA = double.PositiveInfinity;
        double calibrationB = double.PositiveInfinity;
        for (int i = 0; i < 3; i++)
        {
            calibrationA = Math.Min(calibrationA,
                TimeHostBatch(launchA, synchronizeA, calibrationLaunches));
            calibrationB = Math.Min(calibrationB,
                TimeHostBatch(launchB, synchronizeB, calibrationLaunches));
        }
        int iterations = CalibratedIterationsFromMicroseconds(
            Math.Max(calibrationA, calibrationB), targetBatchMilliseconds);

        var samplesA = new List<double>(3);
        var samplesB = new List<double>(3);
        var ratios = new List<double>(3);
        int attempts = 0;
        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            attempts++;
            var bracketA = new List<double>(bracketsPerAttempt);
            var bracketB = new List<double>(bracketsPerAttempt);
            var bracketRatios = new List<double>(bracketsPerAttempt);
            for (int bracket = 0; bracket < bracketsPerAttempt; bracket++)
            {
                double aFirst = TimeHostBatch(launchA, synchronizeA, iterations) /
                    operationsPerLaunchA;
                double bSecond = TimeHostBatch(launchB, synchronizeB, iterations) /
                    operationsPerLaunchB;
                double bFirst = TimeHostBatch(launchB, synchronizeB, iterations) /
                    operationsPerLaunchB;
                double aSecond = TimeHostBatch(launchA, synchronizeA, iterations) /
                    operationsPerLaunchA;
                double a = (aFirst + aSecond) * 0.5;
                double b = (bFirst + bSecond) * 0.5;
                bracketA.Add(a);
                bracketB.Add(b);
                bracketRatios.Add(a / b);
            }
            AddToConsecutiveWindow(samplesA, Median(bracketA));
            AddToConsecutiveWindow(samplesB, Median(bracketB));
            AddToConsecutiveWindow(ratios, Median(bracketRatios));
            if (samplesA.Count >= 3 &&
                SpreadOf(samplesA) <= StableSpread &&
                SpreadOf(samplesB) <= StableSpread &&
                SpreadOf(ratios) <= StableSpread)
            {
                break;
            }
        }

        return Pair(samplesA, samplesB, ratios, attempts);
    }

    private static PairResult Pair(
        List<double> samplesA, List<double> samplesB, List<double> ratios,
        int attempts)
    {
        double spreadA = SpreadOf(samplesA);
        double spreadB = SpreadOf(samplesB);
        double ratioSpread = SpreadOf(ratios);
        return new PairResult(
            new Result(Median(samplesA), spreadA, attempts,
                samplesA.Count >= 3 && spreadA <= StableSpread),
            new Result(Median(samplesB), spreadB, attempts,
                samplesB.Count >= 3 && spreadB <= StableSpread),
            Median(ratios), ratioSpread, attempts);
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

    private static void Warm(Action launch, Action synchronize, int iterations)
    {
        for (int i = 0; i < Math.Max(3, iterations / 10); i++) launch();
        synchronize();
    }

    private static double TimeHostBatch(Action launch, Action synchronize, int iterations)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++) launch();
        synchronize();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / iterations;
    }

    private static int CalibratedIterationsFromMicroseconds(
        double microsecondsPerLaunch,
        double targetBatchMilliseconds)
    {
        if (!(microsecondsPerLaunch > 0) || double.IsInfinity(microsecondsPerLaunch))
            return 4_096;
        return (int)Math.Clamp(
            Math.Ceiling(targetBatchMilliseconds * 1_000.0 / microsecondsPerLaunch),
            1, 4_096);
    }

    /// <summary>
    /// Iterations for a kernel of a given size: enough to swamp launch overhead, capped so a
    /// large kernel does not run for minutes.
    /// </summary>
    private static int IterationsFor(long workUnits) =>
        // The old 20G/500 ceiling left a 35 us optimized convolution with only about
        // 6 ms of device work per sample. One ordinary WDDM preemption then dominated
        // that sample and permanently poisoned the strict max-min gate. Keep the gate
        // unchanged, but give short kernels a long enough event-timed batch that host
        // scheduling is amortized; large kernels still bottom out at five launches.
        (int)Math.Max(5, Math.Min(2_000, 80_000_000_000L / Math.Max(1, workUnits)));

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
