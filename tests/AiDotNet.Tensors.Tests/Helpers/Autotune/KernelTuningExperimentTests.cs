using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

public sealed class KernelTuningExperimentTests
{
    [Fact]
    public async Task Replay_OwnsOracleWarmupControlAndInterleavedRawHoldout()
    {
        var backend = new RecordingBackend();
        var experiment = new KernelTuningExperiment<Configuration>(
            backend,
            new ConfigurationTimer(backend),
            Configuration.Incumbent,
            new KernelTuningWorkload(2048, KernelTuningWorkUnit.Elements),
            warmupCount: 1,
            searchSampleCount: 3,
            holdoutSampleCount: 7);

        KernelTuningFinalistReplay<Configuration> replay = await experiment.ReplayAsync(
            Identity(), Configuration.Candidate, activeDeployment: null);

        Assert.Equal(7, replay.Evidence.Samples.Count);
        Assert.Equal(1.25d, replay.Evidence.MedianSpeedup, 10);
        Assert.Equal(1d, replay.Evidence.CalibratedNoiseRatio, 10);
        Assert.True(replay.CandidateMeasurement.Timing.HasRawSamples);
        Assert.True(replay.IncumbentMeasurement.Timing.HasRawSamples);
        Assert.Equal(KernelTuningWorkUnit.Elements, replay.CandidateMeasurement.Workload.Unit);
        Assert.Equal(KernelTuningResourceMetricStatus.NotApplicable,
            replay.CandidateMeasurement.Resources.OccupancyRatioMetric.Status);

        Assert.Equal(1, backend.Events.Count(item =>
            item == new BackendEvent(BackendEventKind.Prepare, Configuration.Candidate)));
        Assert.Equal(1, backend.Events.Count(item =>
            item == new BackendEvent(BackendEventKind.Validate, Configuration.Candidate)));
        Assert.Equal(8, backend.Events.Count(item =>
            item == new BackendEvent(BackendEventKind.Execute, Configuration.Candidate)));
        Assert.Equal(22, backend.Events.Count(item =>
            item == new BackendEvent(BackendEventKind.Execute, Configuration.Incumbent)));

        Configuration[] holdoutExecutionOrder = backend.Events
            .Where(item => item.Kind == BackendEventKind.Execute)
            .Select(item => item.Configuration)
            .Skip(16)
            .ToArray();
        Assert.Equal(new[]
        {
            Configuration.Candidate, Configuration.Incumbent,
            Configuration.Incumbent, Configuration.Candidate,
            Configuration.Candidate, Configuration.Incumbent,
            Configuration.Incumbent, Configuration.Candidate,
            Configuration.Candidate, Configuration.Incumbent,
            Configuration.Incumbent, Configuration.Candidate,
            Configuration.Candidate, Configuration.Incumbent,
        }, holdoutExecutionOrder);
    }

    [Fact]
    public async Task SearchValidationFailure_PreservesTypedMismatchStatusAndSkipsTiming()
    {
        var backend = new RecordingBackend();
        var timer = new ConfigurationTimer(backend);
        var experiment = new KernelTuningExperiment<Configuration>(
            backend,
            timer,
            Configuration.Incumbent,
            new KernelTuningWorkload(1, KernelTuningWorkUnit.Operations),
            warmupCount: 0,
            searchSampleCount: 3,
            holdoutSampleCount: 7);

        KernelTuningTrialResult result = await experiment.EvaluateAsync(
            Configuration.Bad,
            new EvolutionEvaluationContext(0, 1, 2, 1));

        Assert.Equal(KernelTuningTrialStatus.OutputMismatch, result.Status);
        Assert.Equal(0, timer.MeasurementCount);
        Assert.DoesNotContain(backend.Events, item =>
            item == new BackendEvent(BackendEventKind.Execute, Configuration.Bad));
    }

    [Fact]
    public void PromotionGate_RejectsARegressionHiddenByTheMedian()
    {
        KernelTuningPairedSample[] stable = Enumerable.Range(0, 9)
            .Select(_ => new KernelTuningPairedSample(
                TimeSpan.FromMilliseconds(8), TimeSpan.FromMilliseconds(10)))
            .ToArray();
        KernelTuningPairedSample[] withRegression = stable.ToArray();
        withRegression[0] = new KernelTuningPairedSample(
            TimeSpan.FromMilliseconds(12), TimeSpan.FromMilliseconds(10));
        var options = new KernelTuningOptions
        {
            MinimumPromotionRatio = 1.01,
            MaximumP95LatencyRatio = 2,
        };

        Assert.True(options.QualifiesForPromotion(new KernelTuningPairedEvidence(stable, 1)));
        Assert.False(options.QualifiesForPromotion(new KernelTuningPairedEvidence(withRegression, 1)));
    }

    [Fact]
    public void ResourceState_DoesNotTurnUnavailableIntoMeasuredZero()
    {
        KernelTuningResourceMetric<int> metric = KernelTuningResourceMetric<int>.Unavailable();
        KernelTuningResourceMetric<int> uninitialized = default;

        Assert.Equal(KernelTuningResourceMetricStatus.Unavailable, metric.Status);
        Assert.Throws<InvalidOperationException>(() => metric.Value);
        Assert.Equal(KernelTuningResourceMetricStatus.Unavailable, uninitialized.Status);
        Assert.Throws<InvalidOperationException>(() => uninitialized.Value);
    }

    [Fact]
    public void ResourceState_PreservesEstimatedEvidenceWithoutCallingItMeasured()
    {
        KernelTuningResourceMetric<double> metric =
            KernelTuningResourceMetric<double>.Estimated(0.75);

        Assert.Equal(KernelTuningResourceMetricStatus.Estimated, metric.Status);
        Assert.True(metric.HasValue);
        Assert.Equal(0.75, metric.Value);
        Assert.Throws<ArgumentOutOfRangeException>(() => new KernelTuningResourceUsage(
            KernelTuningResourceMetric<long>.Measured(0),
            KernelTuningResourceMetric<double>.Estimated(1.01),
            KernelTuningResourceMetric<int>.Estimated(16),
            KernelTuningResourceMetric<TimeSpan>.Measured(TimeSpan.Zero),
            KernelTuningResourceMetric<int>.Measured(1)));
    }

    [Fact]
    public void ArchiveProfile_DeviceDefaultResolvesFromTypedDeviceKind()
    {
        var options = new KernelTuningOptions
        {
            MinimumPromotionRatio = 1.01,
            MaximumP95LatencyRatio = 1.1,
        };

        KernelTuningOptions cpu = options.SnapshotAndValidate(KernelTuningDeviceKind.Cpu);
        KernelTuningOptions gpu = options.SnapshotAndValidate(KernelTuningDeviceKind.NvidiaGpu);

        Assert.Equal(KernelTuningArchiveProfile.Cpu, cpu.ArchiveProfile);
        Assert.Contains(cpu.ArchiveDescriptors,
            descriptor => descriptor.Metric == KernelTuningMetric.KernelLaunchCount);
        Assert.DoesNotContain(cpu.ArchiveDescriptors,
            descriptor => descriptor.Metric == KernelTuningMetric.OccupancyRatio);
        Assert.Equal(KernelTuningArchiveProfile.PortableGpu, gpu.ArchiveProfile);
        Assert.Contains(gpu.ArchiveDescriptors,
            descriptor => descriptor.Metric == KernelTuningMetric.Log10NumericalError);
        Assert.DoesNotContain(gpu.ArchiveDescriptors,
            descriptor => descriptor.Metric == KernelTuningMetric.OccupancyRatio);
    }

    [Fact]
    public void ArchiveProfile_RejectsDescriptorsUnlessCustomIsExplicit()
    {
        var options = new KernelTuningOptions
        {
            ArchiveDescriptors = new[]
            {
                new KernelTuningDescriptorDefinition(KernelTuningMetric.KernelLaunchCount, 1, 8, 4)
            }
        };

        Assert.Throws<ArgumentException>(() =>
            options.SnapshotAndValidate(KernelTuningDeviceKind.Cpu));

        options.ArchiveProfile = KernelTuningArchiveProfile.Custom;
        KernelTuningOptions custom = options.SnapshotAndValidate(KernelTuningDeviceKind.Cpu);
        Assert.Equal(KernelTuningArchiveProfile.Custom, custom.ArchiveProfile);
        Assert.Single(custom.ArchiveDescriptors);
    }

    [Fact]
    public async Task DeviceEventTimer_RecordsAroundOperationAndWaitsBeforeReadingElapsedTime()
    {
        var clock = new RecordingDeviceClock(TimeSpan.FromMilliseconds(2));
        var timer = new DeviceEventKernelTuningTimer(clock);

        TimeSpan elapsed = await timer.MeasureAsync(cancellationToken =>
        {
            clock.Events.Add(DeviceClockEventKind.Operation);
            return default;
        });

        Assert.Equal(TimeSpan.FromMilliseconds(2), elapsed);
        Assert.Equal(new[]
        {
            DeviceClockEventKind.Record,
            DeviceClockEventKind.Operation,
            DeviceClockEventKind.Record,
            DeviceClockEventKind.Wait,
            DeviceClockEventKind.ReadElapsed,
            DeviceClockEventKind.Dispose,
            DeviceClockEventKind.Dispose,
        }, clock.Events);
    }

    [Fact]
    public async Task DeviceEventTimer_RejectsAnUnusableDeviceDuration()
    {
        var timer = new DeviceEventKernelTuningTimer(
            new RecordingDeviceClock(TimeSpan.Zero));

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            await timer.MeasureAsync(cancellationToken => default));
    }

    private static KernelTuningIdentity Identity() => new(
        new KernelId("test", "experiment"),
        new ShapeProfile(8, 8),
        KernelTuningDeviceFingerprint.CurrentCpu(),
        KernelTuningBackend.ManagedCpu,
        new KernelSearchSpaceVersion(1),
        new KernelBenchmarkProtocolVersion(1));

    private enum Configuration
    {
        Incumbent,
        Candidate,
        Bad,
    }

    private enum BackendEventKind
    {
        Prepare,
        Validate,
        Execute,
        Synchronize,
    }

    private enum DeviceClockEventKind
    {
        Record,
        Operation,
        Wait,
        ReadElapsed,
        Dispose,
    }

    private readonly record struct BackendEvent(
        BackendEventKind Kind,
        Configuration Configuration);

    private sealed class RecordingBackend : IKernelTuningExperimentBackend<Configuration>
    {
        private readonly List<BackendEvent> _events = new();

        internal IReadOnlyList<BackendEvent> Events => _events;
        internal Configuration LastExecuted { get; private set; }

        public ValueTask PrepareAsync(
            Configuration configuration,
            CancellationToken cancellationToken = default)
        {
            _events.Add(new BackendEvent(BackendEventKind.Prepare, configuration));
            return default;
        }

        public ValueTask ExecuteAsync(
            Configuration configuration,
            CancellationToken cancellationToken = default)
        {
            LastExecuted = configuration;
            _events.Add(new BackendEvent(BackendEventKind.Execute, configuration));
            return default;
        }

        public ValueTask SynchronizeAsync(CancellationToken cancellationToken = default)
        {
            _events.Add(new BackendEvent(BackendEventKind.Synchronize, LastExecuted));
            return default;
        }

        public ValueTask<KernelTuningCorrectnessEvidence> ValidateAsync(
            Configuration configuration,
            CancellationToken cancellationToken = default)
        {
            _events.Add(new BackendEvent(BackendEventKind.Validate, configuration));
            if (configuration == Configuration.Bad)
            {
                throw new KernelTuningValidationException(
                    KernelTuningTrialStatus.OutputMismatch,
                    "Independent oracle mismatch.");
            }
            return new ValueTask<KernelTuningCorrectnessEvidence>(new KernelTuningCorrectnessEvidence(
                KernelTuningValidationScope.Output, 0, 0, 0, 0));
        }

        public KernelTuningResourceUsage GetResourceUsage(Configuration configuration) =>
            KernelTuningResourceUsage.ForCpu(0, TimeSpan.Zero);
    }

    private sealed class ConfigurationTimer : IKernelTuningTimer
    {
        private readonly RecordingBackend _backend;

        internal ConfigurationTimer(RecordingBackend backend) => _backend = backend;

        internal int MeasurementCount { get; private set; }

        public async ValueTask<TimeSpan> MeasureAsync(
            Func<CancellationToken, ValueTask> operation,
            CancellationToken cancellationToken = default)
        {
            await operation(cancellationToken);
            MeasurementCount++;
            return _backend.LastExecuted == Configuration.Candidate
                ? TimeSpan.FromMilliseconds(8)
                : TimeSpan.FromMilliseconds(10);
        }
    }

    private sealed class RecordingDeviceClock : IKernelTuningDeviceClock
    {
        private readonly TimeSpan _elapsed;

        internal RecordingDeviceClock(TimeSpan elapsed) => _elapsed = elapsed;

        internal List<DeviceClockEventKind> Events { get; } = new();

        public IKernelTuningDeviceTimestamp Record()
        {
            Events.Add(DeviceClockEventKind.Record);
            return new RecordingTimestamp(Events);
        }

        public TimeSpan GetElapsed(
            IKernelTuningDeviceTimestamp start,
            IKernelTuningDeviceTimestamp end)
        {
            Events.Add(DeviceClockEventKind.ReadElapsed);
            return _elapsed;
        }
    }

    private sealed class RecordingTimestamp : IKernelTuningDeviceTimestamp
    {
        private readonly ICollection<DeviceClockEventKind> _events;

        internal RecordingTimestamp(ICollection<DeviceClockEventKind> events) => _events = events;

        public ValueTask WaitAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _events.Add(DeviceClockEventKind.Wait);
            return default;
        }

        public void Dispose() => _events.Add(DeviceClockEventKind.Dispose);
    }
}
