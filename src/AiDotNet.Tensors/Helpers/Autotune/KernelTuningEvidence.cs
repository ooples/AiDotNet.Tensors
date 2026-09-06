using System.Collections.ObjectModel;
using System.Diagnostics;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>Boundary included in a kernel timing measurement.</summary>
public enum KernelTuningTimingScope
{
    /// <summary>Steady-state device or CPU execution after preparation.</summary>
    SteadyStateExecution = 0,
    /// <summary>Steady-state end-to-end execution including transfers and synchronization.</summary>
    SteadyStateEndToEnd = 1,
    /// <summary>First-use execution including compilation and other one-time preparation.</summary>
    FirstUse = 2
}

/// <summary>Typed unit used to turn latency into a comparable work rate.</summary>
public enum KernelTuningWorkUnit
{
    /// <summary>One completed operation.</summary>
    Operations = 0,
    /// <summary>Floating-point operations.</summary>
    FloatingPointOperations = 1,
    /// <summary>Elements processed.</summary>
    Elements = 2,
    /// <summary>Bytes processed.</summary>
    Bytes = 3,
    /// <summary>Tokens processed.</summary>
    Tokens = 4
}

/// <summary>Whether a resource metric has a meaningful value for a backend.</summary>
public enum KernelTuningResourceMetricStatus
{
    /// <summary>The metric applies but the backend cannot currently observe it.</summary>
    Unavailable = 0,
    /// <summary>The backend measured the value.</summary>
    Measured = 1,
    /// <summary>The metric does not apply to this backend.</summary>
    NotApplicable = 2,
    /// <summary>The backend derived an estimate from declared configuration and device limits.</summary>
    Estimated = 3
}

/// <summary>Which side of paired replay evidence a deployment measurement describes.</summary>
public enum KernelTuningEvidenceRole
{
    /// <summary>The measurement describes the nominated candidate.</summary>
    Candidate = 0,
    /// <summary>The measurement describes the incumbent retained by the gate.</summary>
    Incumbent = 1
}

/// <summary>A typed resource value that cannot confuse missing data with a measured zero.</summary>
public readonly record struct KernelTuningResourceMetric<T>
    where T : struct
{
    private readonly T _value;

    private KernelTuningResourceMetric(KernelTuningResourceMetricStatus status, T value)
    {
        if (!Enum.IsDefined(typeof(KernelTuningResourceMetricStatus), status))
            throw new ArgumentOutOfRangeException(nameof(status));
        Status = status;
        _value = value;
    }

    /// <summary>Gets whether the value was measured, estimated, does not apply, or is unavailable.</summary>
    public KernelTuningResourceMetricStatus Status { get; }

    /// <summary>Gets the available measured or estimated value.</summary>
    /// <exception cref="InvalidOperationException">The metric has no numeric evidence.</exception>
    public T Value => HasValue
        ? _value
        : throw new InvalidOperationException("A resource metric without numeric evidence has no value.");

    /// <summary>Gets whether this evidence carries a numeric value.</summary>
    public bool HasValue => Status == KernelTuningResourceMetricStatus.Measured ||
                            Status == KernelTuningResourceMetricStatus.Estimated;

    /// <summary>Creates a measured resource value.</summary>
    public static KernelTuningResourceMetric<T> Measured(T value) =>
        new(KernelTuningResourceMetricStatus.Measured, value);

    /// <summary>Creates a value estimated from configuration and known device limits.</summary>
    public static KernelTuningResourceMetric<T> Estimated(T value) =>
        new(KernelTuningResourceMetricStatus.Estimated, value);

    /// <summary>Creates a marker for a resource that has no meaning on this backend.</summary>
    public static KernelTuningResourceMetric<T> NotApplicable() =>
        new(KernelTuningResourceMetricStatus.NotApplicable, default);

    /// <summary>Creates a marker for an applicable resource that cannot be observed.</summary>
    public static KernelTuningResourceMetric<T> Unavailable() =>
        new(KernelTuningResourceMetricStatus.Unavailable, default);

    internal bool TryGetValue(out T value)
    {
        value = _value;
        return HasValue;
    }
}

/// <summary>Fixed work performed by each timed operation.</summary>
public readonly record struct KernelTuningWorkload
{
    /// <summary>Creates a positive, typed workload.</summary>
    public KernelTuningWorkload(double unitsPerOperation, KernelTuningWorkUnit unit)
    {
        if (!KernelTuningMeasurement.IsFinite(unitsPerOperation) || unitsPerOperation <= 0)
            throw new ArgumentOutOfRangeException(nameof(unitsPerOperation));
        if (!Enum.IsDefined(typeof(KernelTuningWorkUnit), unit))
            throw new ArgumentOutOfRangeException(nameof(unit));
        UnitsPerOperation = unitsPerOperation;
        Unit = unit;
    }

    /// <summary>Gets work units completed by each timing sample.</summary>
    public double UnitsPerOperation { get; }
    /// <summary>Gets the kind of work represented by <see cref="UnitsPerOperation"/>.</summary>
    public KernelTuningWorkUnit Unit { get; }
}

/// <summary>One interleaved candidate/incumbent timing pair.</summary>
public readonly record struct KernelTuningPairedSample
{
    /// <summary>Creates a positive timing pair.</summary>
    public KernelTuningPairedSample(TimeSpan candidate, TimeSpan incumbent)
    {
        if (candidate <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(candidate));
        if (incumbent <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(incumbent));
        Candidate = candidate;
        Incumbent = incumbent;
    }

    /// <summary>Gets candidate latency in this pair.</summary>
    public TimeSpan Candidate { get; }
    /// <summary>Gets incumbent latency in this pair.</summary>
    public TimeSpan Incumbent { get; }
    /// <summary>Gets incumbent latency divided by candidate latency; values above one favor the candidate.</summary>
    public double Speedup => Incumbent.TotalSeconds / Candidate.TotalSeconds;
}

/// <summary>Raw, paired finalist evidence from a sealed holdout replay.</summary>
public sealed class KernelTuningPairedEvidence
{
    /// <summary>Minimum paired sample count accepted by the generic promotion gate.</summary>
    public const int MinimumSampleCount = 7;

    private readonly ReadOnlyCollection<KernelTuningPairedSample> _samples;

    /// <summary>Creates finalist evidence from interleaved holdout samples and a separately calibrated noise floor.</summary>
    public KernelTuningPairedEvidence(
        IEnumerable<KernelTuningPairedSample> samples,
        double calibratedNoiseRatio)
    {
        if (samples is null) throw new ArgumentNullException(nameof(samples));
        KernelTuningPairedSample[] copy = samples.ToArray();
        if (copy.Length < MinimumSampleCount)
            throw new ArgumentException(
                $"At least {MinimumSampleCount} paired holdout samples are required.", nameof(samples));
        if (!KernelTuningMeasurement.IsFinite(calibratedNoiseRatio) || calibratedNoiseRatio < 1d)
            throw new ArgumentOutOfRangeException(nameof(calibratedNoiseRatio));

        _samples = Array.AsReadOnly(copy);
        CandidateTiming = KernelTimingStatistics.FromSamples(copy.Select(item => item.Candidate));
        IncumbentTiming = KernelTimingStatistics.FromSamples(copy.Select(item => item.Incumbent));
        double[] speedups = copy.Select(item => item.Speedup).OrderBy(value => value).ToArray();
        MedianSpeedup = Median(speedups);
        int lowerIndex = Math.Max(0, (int)Math.Floor(speedups.Length * 0.05d));
        LowerSpeedupBound = speedups[lowerIndex];
        CalibratedNoiseRatio = calibratedNoiseRatio;
    }

    /// <summary>Gets an immutable copy of every raw holdout pair.</summary>
    public IReadOnlyList<KernelTuningPairedSample> Samples => _samples;
    /// <summary>Gets candidate timing statistics computed only from the holdout pairs.</summary>
    public KernelTimingStatistics CandidateTiming { get; }
    /// <summary>Gets incumbent timing statistics computed only from the holdout pairs.</summary>
    public KernelTimingStatistics IncumbentTiming { get; }
    /// <summary>Gets the median within-pair incumbent/candidate speedup.</summary>
    public double MedianSpeedup { get; }
    /// <summary>Gets the empirical lower five-percent speedup bound.</summary>
    public double LowerSpeedupBound { get; }
    /// <summary>Gets the symmetric noise ratio measured by a separate incumbent/incumbent control replay.</summary>
    public double CalibratedNoiseRatio { get; }
    /// <summary>Gets candidate P95 divided by incumbent P95; values no greater than one avoid tail regression.</summary>
    public double P95LatencyRatio => CandidateTiming.P95.TotalSeconds / IncumbentTiming.P95.TotalSeconds;

    internal static double SymmetricNoiseRatio(IEnumerable<KernelTuningPairedSample> controlSamples)
    {
        if (controlSamples is null) throw new ArgumentNullException(nameof(controlSamples));
        double[] deviations = controlSamples
            .Select(item => Math.Max(item.Speedup, 1d / item.Speedup))
            .OrderBy(value => value)
            .ToArray();
        if (deviations.Length < MinimumSampleCount)
            throw new ArgumentException(
                $"At least {MinimumSampleCount} paired control samples are required.", nameof(controlSamples));
        int p95Index = Math.Max(0, (int)Math.Ceiling(deviations.Length * 0.95d) - 1);
        return deviations[p95Index];
    }

    private static double Median(IReadOnlyList<double> sorted) => sorted.Count % 2 == 0
        ? (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2d
        : sorted[sorted.Count / 2];
}

/// <summary>Search finalist and incumbent results measured in one sealed replay.</summary>
public sealed class KernelTuningFinalistReplay<TConfiguration>
    where TConfiguration : notnull
{
    /// <summary>Creates a direct finalist replay.</summary>
    public KernelTuningFinalistReplay(
        TConfiguration incumbentConfiguration,
        KernelTuningMeasurement candidateMeasurement,
        KernelTuningMeasurement incumbentMeasurement,
        KernelTuningPairedEvidence evidence)
    {
        IncumbentConfiguration = incumbentConfiguration is null
            ? throw new ArgumentNullException(nameof(incumbentConfiguration))
            : incumbentConfiguration;
        CandidateMeasurement = candidateMeasurement ?? throw new ArgumentNullException(nameof(candidateMeasurement));
        IncumbentMeasurement = incumbentMeasurement ?? throw new ArgumentNullException(nameof(incumbentMeasurement));
        Evidence = evidence ?? throw new ArgumentNullException(nameof(evidence));
        ValidateMeasurement(candidateMeasurement, evidence.CandidateTiming, nameof(candidateMeasurement));
        ValidateMeasurement(incumbentMeasurement, evidence.IncumbentTiming, nameof(incumbentMeasurement));
        if (candidateMeasurement.Workload != incumbentMeasurement.Workload ||
            candidateMeasurement.TimingScope != incumbentMeasurement.TimingScope)
        {
            throw new ArgumentException("Candidate and incumbent replay measurements must cover identical work and timing scope.");
        }
    }

    /// <summary>Gets the exact incumbent configuration used by the replay.</summary>
    public TConfiguration IncumbentConfiguration { get; }
    /// <summary>Gets candidate evidence reconstructed from the holdout samples.</summary>
    public KernelTuningMeasurement CandidateMeasurement { get; }
    /// <summary>Gets incumbent evidence reconstructed from the holdout samples.</summary>
    public KernelTuningMeasurement IncumbentMeasurement { get; }
    /// <summary>Gets raw paired timing evidence and calibrated noise.</summary>
    public KernelTuningPairedEvidence Evidence { get; }

    private static void ValidateMeasurement(
        KernelTuningMeasurement measurement,
        KernelTimingStatistics expected,
        string parameterName)
    {
        if (measurement.Timing.SampleCount != expected.SampleCount ||
            measurement.Timing.Median != expected.Median ||
            measurement.Timing.P95 != expected.P95)
        {
            throw new ArgumentException(
                "Replay measurements must be computed from the paired holdout samples.", parameterName);
        }
    }
}

/// <summary>Replays a nominated search finalist directly against the current production incumbent.</summary>
public interface IKernelTuningFinalistEvaluator<TConfiguration>
    where TConfiguration : notnull
{
    /// <summary>Runs a sealed, paired holdout replay. Search measurements must not be reused.</summary>
    ValueTask<KernelTuningFinalistReplay<TConfiguration>> ReplayAsync(
        KernelTuningIdentity identity,
        TConfiguration candidate,
        KernelTuningDeploymentSnapshot<TConfiguration>? activeDeployment,
        CancellationToken cancellationToken = default);
}

/// <summary>Backend operations required by the first-party correctness and timing scaffold.</summary>
public interface IKernelTuningExperimentBackend<TConfiguration>
    where TConfiguration : notnull
{
    /// <summary>Prepares or compiles a configuration outside its steady-state timing samples.</summary>
    ValueTask PrepareAsync(TConfiguration configuration, CancellationToken cancellationToken = default);
    /// <summary>Executes the configured operation once against the scaffold-owned input.</summary>
    ValueTask ExecuteAsync(TConfiguration configuration, CancellationToken cancellationToken = default);
    /// <summary>Waits until backend execution is complete; CPU backends may return immediately.</summary>
    ValueTask SynchronizeAsync(CancellationToken cancellationToken = default);
    /// <summary>Runs the independent reference-oracle comparison before timing.</summary>
    ValueTask<KernelTuningCorrectnessEvidence> ValidateAsync(
        TConfiguration configuration,
        CancellationToken cancellationToken = default);
    /// <summary>Gets typed resource evidence for a prepared configuration.</summary>
    KernelTuningResourceUsage GetResourceUsage(TConfiguration configuration);
}

/// <summary>A correctness-oracle rejection carrying a typed trial status.</summary>
public sealed class KernelTuningValidationException : Exception
{
    /// <summary>Creates an output- or gradient-mismatch failure.</summary>
    public KernelTuningValidationException(KernelTuningTrialStatus status, string message)
        : base(message)
    {
        if (status is not KernelTuningTrialStatus.OutputMismatch and
            not KernelTuningTrialStatus.GradientMismatch)
        {
            throw new ArgumentOutOfRangeException(nameof(status));
        }
        Status = status;
    }

    /// <summary>Gets the typed correctness failure.</summary>
    public KernelTuningTrialStatus Status { get; }
}

/// <summary>Backend-specific clock used by the measurement scaffold.</summary>
public interface IKernelTuningTimer
{
    /// <summary>Measures one already-prepared execution and returns a positive elapsed duration.</summary>
    ValueTask<TimeSpan> MeasureAsync(
        Func<CancellationToken, ValueTask> operation,
        CancellationToken cancellationToken = default);
}

/// <summary>An opaque device timestamp that can be awaited and deterministically released.</summary>
public interface IKernelTuningDeviceTimestamp : IDisposable
{
    /// <summary>Waits until the device has reached this timestamp.</summary>
    ValueTask WaitAsync(CancellationToken cancellationToken = default);
}

/// <summary>Backend adapter that records and compares timestamps in a device clock domain.</summary>
public interface IKernelTuningDeviceClock
{
    /// <summary>Records a timestamp after all earlier work in the tuned execution stream.</summary>
    IKernelTuningDeviceTimestamp Record();

    /// <summary>Gets device time between two completed timestamps from this clock.</summary>
    TimeSpan GetElapsed(
        IKernelTuningDeviceTimestamp start,
        IKernelTuningDeviceTimestamp end);
}

/// <summary>
/// Measures asynchronous accelerator work with backend events, excluding host launch and synchronization latency.
/// </summary>
public sealed class DeviceEventKernelTuningTimer : IKernelTuningTimer
{
    private readonly IKernelTuningDeviceClock _clock;

    /// <summary>Creates a timer over a backend-specific device event clock.</summary>
    public DeviceEventKernelTuningTimer(IKernelTuningDeviceClock clock)
    {
        _clock = clock ?? throw new ArgumentNullException(nameof(clock));
    }

    /// <inheritdoc />
    public async ValueTask<TimeSpan> MeasureAsync(
        Func<CancellationToken, ValueTask> operation,
        CancellationToken cancellationToken = default)
    {
        if (operation is null) throw new ArgumentNullException(nameof(operation));
        cancellationToken.ThrowIfCancellationRequested();
        using IKernelTuningDeviceTimestamp start = _clock.Record();
        await operation(cancellationToken).ConfigureAwait(false);
        using IKernelTuningDeviceTimestamp end = _clock.Record();
        await end.WaitAsync(cancellationToken).ConfigureAwait(false);
        TimeSpan elapsed = _clock.GetElapsed(start, end);
        if (elapsed <= TimeSpan.Zero)
        {
            throw new InvalidOperationException(
                "The device event clock returned a non-positive elapsed duration.");
        }
        return elapsed;
    }
}

/// <summary>
/// Monotonic host timer for operations whose backend synchronization is included by the
/// experiment scaffold. This can time CPU work or a complete accelerator dispatch pipeline.
/// </summary>
public sealed class StopwatchKernelTuningTimer : IKernelTuningTimer
{
    /// <inheritdoc />
    public async ValueTask<TimeSpan> MeasureAsync(
        Func<CancellationToken, ValueTask> operation,
        CancellationToken cancellationToken = default)
    {
        if (operation is null) throw new ArgumentNullException(nameof(operation));
        var stopwatch = Stopwatch.StartNew();
        await operation(cancellationToken).ConfigureAwait(false);
        stopwatch.Stop();
        return stopwatch.Elapsed > TimeSpan.Zero ? stopwatch.Elapsed : TimeSpan.FromTicks(1);
    }
}

/// <summary>
/// First-party experiment orchestrator that owns warmup, oracle validation, raw timing, control calibration, and
/// sealed paired finalist replay. Backends supply operations and a typed clock, not precomputed scores.
/// </summary>
public sealed class KernelTuningExperiment<TConfiguration> : IKernelTuningFinalistEvaluator<TConfiguration>
    where TConfiguration : notnull
{
    private readonly IKernelTuningExperimentBackend<TConfiguration> _backend;
    private readonly IKernelTuningTimer _timer;
    private readonly TConfiguration _defaultIncumbent;
    private readonly KernelTuningWorkload _workload;
    private readonly KernelTuningTimingScope _timingScope;
    private readonly int _warmupCount;
    private readonly int _searchSampleCount;
    private readonly int _holdoutSampleCount;
    private readonly SemaphoreSlim _measurementGate = new(1, 1);

    /// <summary>Creates a deterministic measurement plan with predeclared search and holdout budgets.</summary>
    public KernelTuningExperiment(
        IKernelTuningExperimentBackend<TConfiguration> backend,
        IKernelTuningTimer timer,
        TConfiguration defaultIncumbent,
        KernelTuningWorkload workload,
        KernelTuningTimingScope timingScope = KernelTuningTimingScope.SteadyStateExecution,
        int warmupCount = 2,
        int searchSampleCount = 5,
        int holdoutSampleCount = 9)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _timer = timer ?? throw new ArgumentNullException(nameof(timer));
        _defaultIncumbent = defaultIncumbent is null
            ? throw new ArgumentNullException(nameof(defaultIncumbent))
            : defaultIncumbent;
        if (!Enum.IsDefined(typeof(KernelTuningTimingScope), timingScope))
            throw new ArgumentOutOfRangeException(nameof(timingScope));
        if (warmupCount < 0) throw new ArgumentOutOfRangeException(nameof(warmupCount));
        if (searchSampleCount < KernelTimingStatistics.MinimumSampleCount)
            throw new ArgumentOutOfRangeException(nameof(searchSampleCount));
        if (holdoutSampleCount < KernelTuningPairedEvidence.MinimumSampleCount)
            throw new ArgumentOutOfRangeException(nameof(holdoutSampleCount));
        _workload = workload;
        _timingScope = timingScope;
        _warmupCount = warmupCount;
        _searchSampleCount = searchSampleCount;
        _holdoutSampleCount = holdoutSampleCount;
    }

    /// <summary>Evaluates one search candidate through the owned oracle and measurement protocol.</summary>
    public async ValueTask<KernelTuningTrialResult> EvaluateAsync(
        TConfiguration configuration,
        AiDotNet.Evolution.EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        await _measurementGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            KernelTuningCorrectnessEvidence correctness =
                await PrepareAndValidateAsync(configuration, cancellationToken).ConfigureAwait(false);
            await WarmUpAsync(configuration, cancellationToken).ConfigureAwait(false);
            TimeSpan[] samples = await MeasureAsync(configuration, _searchSampleCount, cancellationToken)
                .ConfigureAwait(false);
            return KernelTuningTrialResult.Passed(CreateMeasurement(
                configuration, correctness, KernelTimingStatistics.FromSamples(samples)));
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (KernelTuningValidationException exception)
        {
            return KernelTuningTrialResult.Rejected(exception.Status, exception.Message);
        }
        catch (Exception exception)
        {
            return KernelTuningTrialResult.Rejected(
                KernelTuningTrialStatus.BenchmarkFailed,
                exception.Message);
        }
        finally
        {
            _measurementGate.Release();
        }
    }

    /// <inheritdoc />
    public async ValueTask<KernelTuningFinalistReplay<TConfiguration>> ReplayAsync(
        KernelTuningIdentity identity,
        TConfiguration candidate,
        KernelTuningDeploymentSnapshot<TConfiguration>? activeDeployment,
        CancellationToken cancellationToken = default)
    {
        if (identity is null) throw new ArgumentNullException(nameof(identity));
        TConfiguration incumbent = activeDeployment is null
            ? _defaultIncumbent
            : activeDeployment.Configuration;
        await _measurementGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            KernelTuningCorrectnessEvidence candidateCorrectness =
                await PrepareAndValidateAsync(candidate, cancellationToken).ConfigureAwait(false);
            KernelTuningCorrectnessEvidence incumbentCorrectness =
                await PrepareAndValidateAsync(incumbent, cancellationToken).ConfigureAwait(false);
            await WarmUpAsync(candidate, cancellationToken).ConfigureAwait(false);
            await WarmUpAsync(incumbent, cancellationToken).ConfigureAwait(false);

            KernelTuningPairedSample[] controls = await MeasurePairsAsync(
                incumbent, incumbent, _holdoutSampleCount, cancellationToken).ConfigureAwait(false);
            double calibratedNoise = KernelTuningPairedEvidence.SymmetricNoiseRatio(controls);
            KernelTuningPairedSample[] holdout = await MeasurePairsAsync(
                candidate, incumbent, _holdoutSampleCount, cancellationToken).ConfigureAwait(false);
            var evidence = new KernelTuningPairedEvidence(holdout, calibratedNoise);
            KernelTuningMeasurement candidateMeasurement = CreateMeasurement(
                candidate, candidateCorrectness, evidence.CandidateTiming);
            KernelTuningMeasurement incumbentMeasurement = CreateMeasurement(
                incumbent, incumbentCorrectness, evidence.IncumbentTiming);
            return new KernelTuningFinalistReplay<TConfiguration>(
                incumbent, candidateMeasurement, incumbentMeasurement, evidence);
        }
        finally
        {
            _measurementGate.Release();
        }
    }

    private async ValueTask<KernelTuningCorrectnessEvidence> PrepareAndValidateAsync(
        TConfiguration configuration,
        CancellationToken cancellationToken)
    {
        await _backend.PrepareAsync(configuration, cancellationToken).ConfigureAwait(false);
        return await _backend.ValidateAsync(configuration, cancellationToken).ConfigureAwait(false)
            ?? throw new InvalidOperationException("The kernel backend returned no correctness evidence.");
    }

    private async ValueTask WarmUpAsync(TConfiguration configuration, CancellationToken cancellationToken)
    {
        for (int i = 0; i < _warmupCount; i++)
            await ExecuteAndSynchronizeAsync(configuration, cancellationToken).ConfigureAwait(false);
    }

    private async ValueTask<TimeSpan[]> MeasureAsync(
        TConfiguration configuration,
        int count,
        CancellationToken cancellationToken)
    {
        var samples = new TimeSpan[count];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = await _timer.MeasureAsync(
                token => ExecuteAndSynchronizeAsync(configuration, token), cancellationToken).ConfigureAwait(false);
        }
        return samples;
    }

    private async ValueTask<KernelTuningPairedSample[]> MeasurePairsAsync(
        TConfiguration candidate,
        TConfiguration incumbent,
        int count,
        CancellationToken cancellationToken)
    {
        var samples = new KernelTuningPairedSample[count];
        for (int i = 0; i < samples.Length; i++)
        {
            TimeSpan candidateTime;
            TimeSpan incumbentTime;
            if ((i & 1) == 0)
            {
                candidateTime = await MeasureOneAsync(candidate, cancellationToken).ConfigureAwait(false);
                incumbentTime = await MeasureOneAsync(incumbent, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                incumbentTime = await MeasureOneAsync(incumbent, cancellationToken).ConfigureAwait(false);
                candidateTime = await MeasureOneAsync(candidate, cancellationToken).ConfigureAwait(false);
            }
            samples[i] = new KernelTuningPairedSample(candidateTime, incumbentTime);
        }
        return samples;
    }

    private ValueTask<TimeSpan> MeasureOneAsync(
        TConfiguration configuration,
        CancellationToken cancellationToken) =>
        _timer.MeasureAsync(token => ExecuteAndSynchronizeAsync(configuration, token), cancellationToken);

    private async ValueTask ExecuteAndSynchronizeAsync(
        TConfiguration configuration,
        CancellationToken cancellationToken)
    {
        await _backend.ExecuteAsync(configuration, cancellationToken).ConfigureAwait(false);
        await _backend.SynchronizeAsync(cancellationToken).ConfigureAwait(false);
    }

    private KernelTuningMeasurement CreateMeasurement(
        TConfiguration configuration,
        KernelTuningCorrectnessEvidence correctness,
        KernelTimingStatistics timing) =>
        new(
            _workload,
            _timingScope,
            timing,
            _backend.GetResourceUsage(configuration),
            correctness);
}
