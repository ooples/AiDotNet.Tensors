using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>Deterministic protocol double for adapter tests; real-kernel proofs use <see cref="KernelTuningExperiment{TConfiguration}"/>.</summary>
internal sealed class DeterministicFinalistEvaluator<TConfiguration> :
    IKernelTuningFinalistEvaluator<TConfiguration>
    where TConfiguration : notnull
{
    private static readonly double[] SampleMultipliers =
        { 1.00, 1.01, 0.99, 1.02, 0.98, 1.005, 0.995, 1.015, 0.985 };

    private readonly TConfiguration _defaultIncumbent;
    private readonly Func<TConfiguration, double> _throughput;
    private readonly Func<TConfiguration, KernelTuningResourceUsage> _resources;

    internal DeterministicFinalistEvaluator(
        TConfiguration defaultIncumbent,
        Func<TConfiguration, double> throughput,
        Func<TConfiguration, KernelTuningResourceUsage>? resources = null)
    {
        _defaultIncumbent = defaultIncumbent;
        _throughput = throughput ?? throw new ArgumentNullException(nameof(throughput));
        _resources = resources ?? (_ => KernelTuningResourceUsage.ForCpu(0, TimeSpan.Zero));
    }

    public ValueTask<KernelTuningFinalistReplay<TConfiguration>> ReplayAsync(
        KernelTuningIdentity identity,
        TConfiguration candidate,
        KernelTuningDeploymentSnapshot<TConfiguration>? activeDeployment,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        TConfiguration incumbent = activeDeployment is null
            ? _defaultIncumbent
            : activeDeployment.Configuration;
        double candidateThroughput = RequiredThroughput(candidate);
        double incumbentThroughput = RequiredThroughput(incumbent);
        var pairs = new KernelTuningPairedSample[SampleMultipliers.Length];
        for (int i = 0; i < pairs.Length; i++)
        {
            pairs[i] = new KernelTuningPairedSample(
                Latency(candidateThroughput, SampleMultipliers[i]),
                Latency(incumbentThroughput, SampleMultipliers[i]));
        }

        var evidence = new KernelTuningPairedEvidence(pairs, calibratedNoiseRatio: 1d);
        var workload = new KernelTuningWorkload(1e9d, KernelTuningWorkUnit.FloatingPointOperations);
        KernelTuningCorrectnessEvidence correctness = Correctness();
        var candidateMeasurement = new KernelTuningMeasurement(
            workload,
            KernelTuningTimingScope.SteadyStateExecution,
            evidence.CandidateTiming,
            _resources(candidate),
            correctness);
        var incumbentMeasurement = new KernelTuningMeasurement(
            workload,
            KernelTuningTimingScope.SteadyStateExecution,
            evidence.IncumbentTiming,
            _resources(incumbent),
            correctness);
        return new ValueTask<KernelTuningFinalistReplay<TConfiguration>>(
            new KernelTuningFinalistReplay<TConfiguration>(
                incumbent,
                candidateMeasurement,
                incumbentMeasurement,
                evidence));
    }

    internal static KernelTuningMeasurement SearchMeasurement(
        double throughputGflops,
        KernelTuningResourceUsage resources)
    {
        KernelTimingStatistics timing = KernelTimingStatistics.FromSamples(new[]
        {
            KernelTuningDuration.FromMilliseconds(1.1),
            KernelTuningDuration.FromMilliseconds(1.0),
            KernelTuningDuration.FromMilliseconds(0.9),
            KernelTuningDuration.FromMilliseconds(1.05),
            KernelTuningDuration.FromMilliseconds(0.95)
        });
        var workload = new KernelTuningWorkload(
            throughputGflops * 1e9d * timing.Median.TotalSeconds,
            KernelTuningWorkUnit.FloatingPointOperations);
        return new KernelTuningMeasurement(
            workload,
            KernelTuningTimingScope.SteadyStateExecution,
            timing,
            resources,
            Correctness());
    }

    private double RequiredThroughput(TConfiguration configuration)
    {
        double value = _throughput(configuration);
        if (!KernelTuningMeasurement.IsFinite(value) || value <= 0)
            throw new InvalidOperationException("The deterministic finalist throughput must be finite and positive.");
        return value;
    }

    private static TimeSpan Latency(double throughputGflops, double multiplier) =>
        KernelTuningDuration.FromSeconds(multiplier / throughputGflops);

    private static KernelTuningCorrectnessEvidence Correctness() => new(
        KernelTuningValidationScope.Output,
        outputAbsoluteError: 1e-7,
        outputRelativeError: 2e-7,
        outputAbsoluteTolerance: 1e-5,
        outputRelativeTolerance: 1e-5);
}
