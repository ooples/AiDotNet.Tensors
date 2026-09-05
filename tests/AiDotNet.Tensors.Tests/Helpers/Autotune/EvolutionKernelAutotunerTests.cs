using System.Globalization;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>GPU-free contract tests for correctness-first typed evolutionary tuning.</summary>
[Collection("AutotuneCacheTests")]
public sealed class EvolutionKernelAutotunerTests : IDisposable
{
    private const string CacheEnvironmentVariable = "AIDOTNET_AUTOTUNE_CACHE_PATH";
    private readonly string? _originalCachePath;
    private readonly string _temporaryCachePath;

    public EvolutionKernelAutotunerTests()
    {
        _originalCachePath = Environment.GetEnvironmentVariable(CacheEnvironmentVariable);
        _temporaryCachePath = Path.Combine(
            Path.GetTempPath(), "aidotnet-evolution-tuning-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable(CacheEnvironmentVariable, _temporaryCachePath);
    }

    [Fact]
    public async Task TuneAsync_PublishesTypedWinnerWithRealArchiveDescriptors()
    {
        var registry = new KernelTuningDeploymentRegistry<FakeKernelConfiguration>();
        var store = new MemoryStore();
        EvolutionKernelAutotuner<FakeKernelConfiguration> tuner = CreateTuner(registry, store, MeasurePassed);

        EvolutionKernelTuningResult<FakeKernelConfiguration> result = await tuner.TuneAsync(Seeds());

        Assert.True(result.WasPromoted);
        Assert.True(result.WasPersisted);
        Assert.Equal(FakeKernelVariant.Wide, result.ProposedWinner.Configuration.Variant);
        Assert.Equal(240d, result.ActiveDeployment.Measurement.ThroughputGflops);
        Assert.Equal(3, result.Run.Counters.EvaluationAttempts);
        EvolutionArchiveEntry<FakeKernelConfiguration> best = result.Run.Best ??
            throw new InvalidOperationException("A successful tuning run must have a best entry.");
        Assert.Contains("log2-workspace-bytes", best.Evaluation.Descriptors.Keys);
        Assert.Contains("occupancy-ratio", best.Evaluation.Descriptors.Keys);
        Assert.Contains("registers-per-thread", best.Evaluation.Descriptors.Keys);
        Assert.DoesNotContain("throughput-gflops", best.Evaluation.Descriptors.Keys);
        Assert.True(tuner.Deployment.TryGet(out FakeKernelConfiguration active));
        Assert.Equal(result.ActiveDeployment.Configuration, active);
    }

    [Fact]
    public async Task TuneAsync_KeepsPublishedWinnerWhenPersistenceThrows()
    {
        EvolutionKernelAutotuner<FakeKernelConfiguration> tuner = CreateTuner(
            new KernelTuningDeploymentRegistry<FakeKernelConfiguration>(),
            new ThrowingStore(),
            MeasurePassed);

        EvolutionKernelTuningResult<FakeKernelConfiguration> result =
            await tuner.TuneAsync(Seeds());

        Assert.True(result.WasPromoted);
        Assert.False(result.WasPersisted);
        Assert.Same(result.ActiveDeployment, tuner.Deployment.Current);
        Assert.Equal(FakeKernelVariant.Wide, result.ActiveDeployment.Configuration.Variant);
    }

    [Fact]
    public async Task TuneAsync_DoesNotPromoteWinnerInsideNoiseFloor()
    {
        var registry = new KernelTuningDeploymentRegistry<FakeKernelConfiguration>();
        var store = new MemoryStore();
        EvolutionKernelTuningResult<FakeKernelConfiguration> initial = await
            CreateTuner(registry, store, MeasurePassed).TuneAsync(Seeds());
        EvolutionKernelAutotuner<FakeKernelConfiguration> replacement = CreateTuner(
            registry,
            store,
            (configuration, _, _) => new ValueTask<KernelTuningTrialResult>(Passed(configuration, bonus: 5)));

        EvolutionKernelTuningResult<FakeKernelConfiguration> result = await replacement.TuneAsync(Seeds());

        Assert.False(result.WasPromoted);
        Assert.False(result.WasPersisted);
        Assert.Equal(245d, result.ProposedWinner.Measurement.ThroughputGflops);
        Assert.Same(initial.ActiveDeployment, result.ActiveDeployment);
        Assert.Equal(240d, result.ActiveDeployment.Measurement.ThroughputGflops);
    }

    [Fact]
    public async Task ConcurrentTuners_CompareAgainstTheWinnerPublishedWhileWaitingForTheDevice()
    {
        var registry = new KernelTuningDeploymentRegistry<FakeKernelConfiguration>();
        var firstStarted = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseFirst = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        EvolutionKernelAutotuner<FakeKernelConfiguration> first = CreateTuner(
            registry,
            new MemoryStore(),
            async (configuration, _, cancellationToken) =>
            {
                firstStarted.TrySetResult(true);
                using (cancellationToken.Register(() => releaseFirst.TrySetCanceled()))
                    await releaseFirst.Task;
                return Passed(configuration, bonus: 200);
            });
        EvolutionKernelAutotuner<FakeKernelConfiguration> second = CreateTuner(
            registry,
            new MemoryStore(),
            MeasurePassed);

        Task<EvolutionKernelTuningResult<FakeKernelConfiguration>> firstRun =
            first.TuneAsync(Seeds().Take(1));
        await firstStarted.Task;
        Task<EvolutionKernelTuningResult<FakeKernelConfiguration>> secondRun =
            second.TuneAsync(Seeds().Take(1));
        releaseFirst.TrySetResult(true);

        EvolutionKernelTuningResult<FakeKernelConfiguration> firstResult = await firstRun;
        EvolutionKernelTuningResult<FakeKernelConfiguration> secondResult = await secondRun;
        Assert.True(firstResult.WasPromoted);
        Assert.False(secondResult.WasPromoted);
        Assert.Same(firstResult.ActiveDeployment, secondResult.ActiveDeployment);
        Assert.Equal(280d, secondResult.ActiveDeployment.Measurement.ThroughputGflops);
    }

    [Fact]
    public async Task TuneAsync_RejectsIncorrectFastCandidateBeforeDeployment()
    {
        EvolutionKernelAutotuner<FakeKernelConfiguration> tuner = CreateTuner(
            new KernelTuningDeploymentRegistry<FakeKernelConfiguration>(),
            new MemoryStore(),
            (configuration, _, _) => new ValueTask<KernelTuningTrialResult>(
                configuration.Variant == FakeKernelVariant.Wide
                    ? KernelTuningTrialResult.Rejected(
                        KernelTuningTrialStatus.OutputMismatch, "Reference output mismatch.")
                    : Passed(configuration)));

        EvolutionKernelTuningResult<FakeKernelConfiguration> result = await tuner.TuneAsync(Seeds());

        Assert.Equal(FakeKernelVariant.Fast, result.ActiveDeployment.Configuration.Variant);
        Assert.Equal(1, result.Run.Counters.StatusCounts[EvolutionEvaluationStatus.Rejected]);
    }

    [Fact]
    public async Task PersistedWinner_RoundTripsThroughLegacyCacheBoundaryWithoutBenchmarking()
    {
        var first = new EvolutionKernelAutotuner<FakeKernelConfiguration>(
            Identity(), new FakeKernelCodec(), new FakeKernelVariation(), MeasurePassed,
            EngineOptions(), deploymentRegistry: new KernelTuningDeploymentRegistry<FakeKernelConfiguration>());
        EvolutionKernelTuningResult<FakeKernelConfiguration> tuned = await first.TuneAsync(Seeds());
        Assert.True(tuned.WasPersisted);

        bool evaluatorCalled = false;
        var second = new EvolutionKernelAutotuner<FakeKernelConfiguration>(
            Identity(),
            new FakeKernelCodec(),
            new FakeKernelVariation(),
            (configuration, context, token) =>
            {
                evaluatorCalled = true;
                throw new InvalidOperationException("Hydration must not benchmark.");
            },
            EngineOptions(),
            deploymentRegistry: new KernelTuningDeploymentRegistry<FakeKernelConfiguration>());

        Assert.True(second.TryHydrate());
        Assert.False(evaluatorCalled);
        Assert.True(second.Deployment.TryGet(out FakeKernelConfiguration hydrated));
        Assert.Equal(tuned.ActiveDeployment.Configuration, hydrated);
        KernelTuningDeploymentSnapshot<FakeKernelConfiguration> hydratedSnapshot =
            second.Deployment.Current ??
            throw new InvalidOperationException("A successful hydration must publish a deployment snapshot.");
        Assert.Equal(
            tuned.ActiveDeployment.Measurement.Timing.P95,
            hydratedSnapshot.Measurement.Timing.P95);
        Assert.Equal(
            tuned.ActiveDeployment.Measurement.Correctness.OutputAbsoluteError,
            hydratedSnapshot.Measurement.Correctness.OutputAbsoluteError);
    }

    [Fact]
    public async Task TryHydrate_RejectsAConfigurationThatFailsTheDeploymentValidator()
    {
        var store = new MemoryStore();
        EvolutionKernelTuningResult<FakeKernelConfiguration> tuned = await CreateTuner(
            new KernelTuningDeploymentRegistry<FakeKernelConfiguration>(),
            store,
            MeasurePassed).TuneAsync(Seeds());
        Assert.True(tuned.WasPersisted);

        var replacement = new EvolutionKernelAutotuner<FakeKernelConfiguration>(
            Identity(),
            new FakeKernelCodec(),
            new FakeKernelVariation(),
            MeasurePassed,
            EngineOptions(),
            deploymentRegistry: new KernelTuningDeploymentRegistry<FakeKernelConfiguration>(),
            store: store,
            deploymentValidator: _ => false);

        Assert.False(replacement.TryHydrate());
        Assert.Null(replacement.Deployment.Current);
    }

    [Fact]
    public void TryHydrate_RejectsMismatchedIdentityAndGenomeEvidence()
    {
        KernelTuningIdentity identity = Identity();
        FakeKernelConfiguration configuration = Seeds()[0];
        var otherIdentity = new KernelTuningIdentity(
            new KernelId("test", "different-kernel"),
            identity.Shape,
            identity.Device,
            identity.SearchSpaceVersion,
            identity.BenchmarkProtocolVersion);
        KernelTuningDeploymentSnapshot<FakeKernelConfiguration> wrongIdentity =
            Snapshot(otherIdentity, configuration, bonus: 0);
        KernelTuningMeasurement measurement = Passed(configuration).Measurement ??
            throw new InvalidOperationException("A passed trial must carry a measurement.");
        var wrongGenome = new KernelTuningDeploymentSnapshot<FakeKernelConfiguration>(
            identity,
            configuration,
            "not-the-canonical-genome-id",
            measurement,
            "test-run-state");

        EvolutionKernelAutotuner<FakeKernelConfiguration> identityTuner = CreateTuner(
            new KernelTuningDeploymentRegistry<FakeKernelConfiguration>(),
            new FixedLoadStore(wrongIdentity),
            MeasurePassed);
        EvolutionKernelAutotuner<FakeKernelConfiguration> genomeTuner = CreateTuner(
            new KernelTuningDeploymentRegistry<FakeKernelConfiguration>(),
            new FixedLoadStore(wrongGenome),
            MeasurePassed);

        Assert.False(identityTuner.TryHydrate());
        Assert.Null(identityTuner.Deployment.Current);
        Assert.False(genomeTuner.TryHydrate());
        Assert.Null(genomeTuner.Deployment.Current);
    }

    [Fact]
    public void TryHydrate_TreatsStoreExceptionsAsCacheMisses()
    {
        EvolutionKernelAutotuner<FakeKernelConfiguration> tuner = CreateTuner(
            new KernelTuningDeploymentRegistry<FakeKernelConfiguration>(),
            new ThrowingStore(),
            MeasurePassed);

        Assert.False(tuner.TryHydrate());
        Assert.Null(tuner.Deployment.Current);
    }

    [Fact]
    public async Task DeploymentValidator_RejectsNewCandidateBeforeEvaluatorAndArchive()
    {
        int evaluatorCalls = 0;
        var tuner = new EvolutionKernelAutotuner<FakeKernelConfiguration>(
            Identity(),
            new FakeKernelCodec(),
            new FakeKernelVariation(),
            (configuration, context, cancellationToken) =>
            {
                evaluatorCalls++;
                return new ValueTask<KernelTuningTrialResult>(Passed(configuration));
            },
            EngineOptions(),
            deploymentRegistry: new KernelTuningDeploymentRegistry<FakeKernelConfiguration>(),
            store: new MemoryStore(),
            deploymentValidator: configuration => configuration.Variant != FakeKernelVariant.Wide);

        EvolutionKernelTuningResult<FakeKernelConfiguration> result = await tuner.TuneAsync(Seeds());

        Assert.Equal(2, evaluatorCalls);
        Assert.Equal(1, result.Run.Counters.StatusCounts[EvolutionEvaluationStatus.Rejected]);
        Assert.Equal(FakeKernelVariant.Fast, result.ActiveDeployment.Configuration.Variant);
    }

    [Fact]
    public void TryHydrate_DoesNotOverwriteDeploymentPublishedDuringStoreRead()
    {
        KernelTuningIdentity identity = Identity();
        var registry = new KernelTuningDeploymentRegistry<FakeKernelConfiguration>();
        KernelTuningDeployment<FakeKernelConfiguration> deployment = registry.GetOrCreate(identity);
        KernelTuningDeploymentSnapshot<FakeKernelConfiguration> persisted = Snapshot(
            identity, new FakeKernelConfiguration(FakeKernelVariant.Safe, 8), bonus: 0);
        KernelTuningDeploymentSnapshot<FakeKernelConfiguration> concurrent = Snapshot(
            identity, new FakeKernelConfiguration(FakeKernelVariant.Wide, 32), bonus: 200);
        var store = new CallbackLoadStore(persisted, () => deployment.Publish(concurrent));
        var tuner = new EvolutionKernelAutotuner<FakeKernelConfiguration>(
            identity,
            new FakeKernelCodec(),
            new FakeKernelVariation(),
            MeasurePassed,
            EngineOptions(),
            deploymentRegistry: registry,
            store: store);

        Assert.True(tuner.TryHydrate());
        Assert.Same(concurrent, deployment.Current);
    }

    [Fact]
    public async Task TuneAsync_RejectsSeedsBeyondProposalBudget()
    {
        EvolutionKernelAutotuner<FakeKernelConfiguration> tuner = CreateTuner(
            new KernelTuningDeploymentRegistry<FakeKernelConfiguration>(),
            new MemoryStore(),
            MeasurePassed);
        FakeKernelConfiguration[] tooMany = Seeds()
            .Append(new FakeKernelConfiguration(FakeKernelVariant.Safe, 16))
            .ToArray();

        await Assert.ThrowsAsync<ArgumentException>(() => tuner.TuneAsync(tooMany));
    }

    [Fact]
    public async Task BackgroundTune_WaitsForExplicitIdleAdmission()
    {
        var gate = new ManualIdleGate();
        bool evaluated = false;
        EvolutionKernelAutotuner<FakeKernelConfiguration> tuner = CreateTuner(
            new KernelTuningDeploymentRegistry<FakeKernelConfiguration>(),
            new MemoryStore(),
            (configuration, _, _) =>
            {
                evaluated = true;
                return new ValueTask<KernelTuningTrialResult>(Passed(configuration));
            });

        Task<EvolutionKernelTuningResult<FakeKernelConfiguration>> pending =
            tuner.TuneInBackgroundAsync(Seeds(), gate);
        await gate.Waiting.Task;
        Assert.False(evaluated);

        gate.Release.TrySetResult(true);
        EvolutionKernelTuningResult<FakeKernelConfiguration> result = await pending;
        Assert.True(evaluated);
        Assert.True(result.WasPromoted);
    }

    [Fact]
    public void TimingAndCorrectnessContracts_RejectWeakOrFailedEvidence()
    {
        Assert.Throws<ArgumentException>(() => KernelTimingStatistics.FromSamples(new[]
        {
            TimeSpan.FromMilliseconds(1), TimeSpan.FromMilliseconds(2)
        }));
        Assert.Throws<ArgumentException>(() => new KernelTuningCorrectnessEvidence(
            KernelTuningValidationScope.Output,
            outputAbsoluteError: 0.2,
            outputRelativeError: 0.3,
            outputAbsoluteTolerance: 0.1,
            outputRelativeTolerance: 0.1));
    }

    [Fact]
    public void Constructor_RejectsBudgetsThatCannotProduceADeployableWinner()
    {
        EvolutionEngineOptions noProposals = EngineOptions();
        noProposals.MaxProposals = 0;
        Assert.Throws<ArgumentException>(() => new EvolutionKernelAutotuner<FakeKernelConfiguration>(
            Identity(),
            new FakeKernelCodec(),
            new FakeKernelVariation(),
            MeasurePassed,
            noProposals));

        EvolutionEngineOptions noEvaluations = EngineOptions();
        noEvaluations.MaxEvaluationAttempts = 0;
        Assert.Throws<ArgumentException>(() => new EvolutionKernelAutotuner<FakeKernelConfiguration>(
            Identity(),
            new FakeKernelCodec(),
            new FakeKernelVariation(),
            MeasurePassed,
            noEvaluations));
    }

    [Fact]
    public void Identity_CoversDeviceShapeAndTypedProtocolVersions()
    {
        KernelTuningIdentity baseline = Identity();
        var otherShape = new KernelTuningIdentity(
            baseline.Kernel, new ShapeProfile(64, 64, 1024), baseline.Device,
            baseline.SearchSpaceVersion, baseline.BenchmarkProtocolVersion);
        var otherDevice = new KernelTuningIdentity(
            baseline.Kernel, baseline.Shape,
            new GpuDeviceFingerprint(GpuVendorKind.Nvidia, "Fake GPU", 8, 6, 550, "fake-1"),
            baseline.SearchSpaceVersion, baseline.BenchmarkProtocolVersion);
        var otherSearchSpace = new KernelTuningIdentity(
            baseline.Kernel, baseline.Shape, baseline.Device,
            new KernelSearchSpaceVersion(2), baseline.BenchmarkProtocolVersion);
        var otherProtocol = new KernelTuningIdentity(
            baseline.Kernel, baseline.Shape, baseline.Device,
            baseline.SearchSpaceVersion, new KernelBenchmarkProtocolVersion(2));

        Assert.NotEqual(baseline.StableKey, otherShape.StableKey);
        Assert.NotEqual(baseline.StableKey, otherDevice.StableKey);
        Assert.NotEqual(baseline.StableKey, otherSearchSpace.StableKey);
        Assert.NotEqual(baseline.StableKey, otherProtocol.StableKey);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(CacheEnvironmentVariable, _originalCachePath);
        if (Directory.Exists(_temporaryCachePath)) Directory.Delete(_temporaryCachePath, recursive: true);
    }

    private static EvolutionKernelAutotuner<FakeKernelConfiguration> CreateTuner(
        KernelTuningDeploymentRegistry<FakeKernelConfiguration> registry,
        IKernelTuningStore<FakeKernelConfiguration> store,
        Func<FakeKernelConfiguration, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningTrialResult>> evaluator) =>
        new(
            Identity(),
            new FakeKernelCodec(),
            new FakeKernelVariation(),
            evaluator,
            EngineOptions(),
            deploymentRegistry: registry,
            store: store);

    private static EvolutionEngineOptions EngineOptions() => new()
    {
        RunId = "typed-kernel-test",
        Seed = 42,
        MaxEvaluationAttempts = 3,
        MaxProposals = 3,
        MaxGenerations = 0,
        ProposalBatchSize = 3,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1
    };

    private static KernelTuningIdentity Identity() => new(
        new KernelId("test", "fake-gpu-kernel"),
        new ShapeProfile(64, 64, 4096),
        new GpuDeviceFingerprint(GpuVendorKind.Nvidia, "Fake GPU", 8, 6, 550, "fake-0"),
        new KernelSearchSpaceVersion(1),
        new KernelBenchmarkProtocolVersion(1));

    private static IReadOnlyList<FakeKernelConfiguration> Seeds() => new[]
    {
        new FakeKernelConfiguration(FakeKernelVariant.Safe, 8),
        new FakeKernelConfiguration(FakeKernelVariant.Fast, 16),
        new FakeKernelConfiguration(FakeKernelVariant.Wide, 32)
    };

    private static ValueTask<KernelTuningTrialResult> MeasurePassed(
        FakeKernelConfiguration configuration,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken) =>
        new(Passed(configuration));

    private static KernelTuningTrialResult Passed(FakeKernelConfiguration configuration, double bonus = 0)
    {
        double throughput = configuration.Variant switch
        {
            FakeKernelVariant.Safe => 80,
            FakeKernelVariant.Fast => 160,
            FakeKernelVariant.Wide => 240,
            _ => throw new ArgumentOutOfRangeException(nameof(configuration))
        };
        var timing = KernelTimingStatistics.FromSamples(new[]
        {
            TimeSpan.FromMilliseconds(1.1),
            TimeSpan.FromMilliseconds(1.0),
            TimeSpan.FromMilliseconds(0.9),
            TimeSpan.FromMilliseconds(1.05),
            TimeSpan.FromMilliseconds(0.95)
        });
        var resources = new KernelTuningResourceUsage(
            configuration.TileEdge * 1024L,
            0.5 + configuration.TileEdge / 100d,
            configuration.TileEdge,
            TimeSpan.FromMilliseconds(3));
        var correctness = new KernelTuningCorrectnessEvidence(
            KernelTuningValidationScope.Output,
            outputAbsoluteError: 1e-7,
            outputRelativeError: 2e-7,
            outputAbsoluteTolerance: 1e-5,
            outputRelativeTolerance: 1e-5);
        return KernelTuningTrialResult.Passed(
            new KernelTuningMeasurement(throughput + bonus, timing, resources, correctness));
    }

    private static KernelTuningDeploymentSnapshot<FakeKernelConfiguration> Snapshot(
        KernelTuningIdentity identity,
        FakeKernelConfiguration configuration,
        double bonus)
    {
        var codec = new FakeKernelCodec();
        KernelTuningMeasurement measurement = Passed(configuration, bonus).Measurement ??
            throw new InvalidOperationException("A passed trial must carry a measurement.");
        string payload = codec.Serialize(configuration);
        return new KernelTuningDeploymentSnapshot<FakeKernelConfiguration>(
            identity,
            configuration,
            EvolutionHash.Compute(payload),
            measurement,
            "test-run-state");
    }

    private enum FakeKernelVariant
    {
        Safe = 0,
        Fast = 1,
        Wide = 2
    }

    private readonly record struct FakeKernelConfiguration(FakeKernelVariant Variant, int TileEdge);

    private sealed class FakeKernelCodec : IEvolutionGenomeCodec<FakeKernelConfiguration>
    {
        public string Id => "fake-kernel-codec";
        public string VersionHash => "v2";

        public string Serialize(FakeKernelConfiguration genome) => string.Concat(
            ((int)genome.Variant).ToString(CultureInfo.InvariantCulture),
            ":",
            genome.TileEdge.ToString(CultureInfo.InvariantCulture));

        public FakeKernelConfiguration Deserialize(string payload)
        {
            string[] parts = payload.Split(':');
            if (parts.Length != 2 ||
                !int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out int variant) ||
                !Enum.IsDefined(typeof(FakeKernelVariant), variant) ||
                !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int tile) ||
                tile <= 0)
            {
                throw new InvalidDataException("Invalid fake kernel configuration.");
            }
            return new FakeKernelConfiguration((FakeKernelVariant)variant, tile);
        }
    }

    private sealed class FakeKernelVariation : IVariationOperator<FakeKernelConfiguration>
    {
        public string Id => "fake-kernel-variation";
        public string VersionHash => "v2";

        public ValueTask<FakeKernelConfiguration> ProposeAsync(
            EvolutionVariationContext<FakeKernelConfiguration> context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            IReadOnlyList<FakeKernelConfiguration> seeds = Seeds();
            return new ValueTask<FakeKernelConfiguration>(seeds[context.Random.NextInt(seeds.Count)]);
        }
    }

    private sealed class MemoryStore : IKernelTuningStore<FakeKernelConfiguration>
    {
        private KernelTuningDeploymentSnapshot<FakeKernelConfiguration>? _snapshot;

        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<FakeKernelConfiguration> codec,
            out KernelTuningDeploymentSnapshot<FakeKernelConfiguration>? snapshot)
        {
            snapshot = _snapshot;
            return snapshot is not null;
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<FakeKernelConfiguration> snapshot,
            IEvolutionGenomeCodec<FakeKernelConfiguration> codec)
        {
            _snapshot = snapshot;
            return true;
        }
    }

    private sealed class CallbackLoadStore : IKernelTuningStore<FakeKernelConfiguration>
    {
        private readonly KernelTuningDeploymentSnapshot<FakeKernelConfiguration> _snapshot;
        private readonly Action _duringLoad;

        internal CallbackLoadStore(
            KernelTuningDeploymentSnapshot<FakeKernelConfiguration> snapshot,
            Action duringLoad)
        {
            _snapshot = snapshot;
            _duringLoad = duringLoad;
        }

        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<FakeKernelConfiguration> codec,
            out KernelTuningDeploymentSnapshot<FakeKernelConfiguration>? snapshot)
        {
            _duringLoad();
            snapshot = _snapshot;
            return true;
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<FakeKernelConfiguration> snapshot,
            IEvolutionGenomeCodec<FakeKernelConfiguration> codec) => true;
    }

    private sealed class FixedLoadStore : IKernelTuningStore<FakeKernelConfiguration>
    {
        private readonly KernelTuningDeploymentSnapshot<FakeKernelConfiguration> _snapshot;

        internal FixedLoadStore(
            KernelTuningDeploymentSnapshot<FakeKernelConfiguration> snapshot)
        {
            _snapshot = snapshot;
        }

        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<FakeKernelConfiguration> codec,
            out KernelTuningDeploymentSnapshot<FakeKernelConfiguration>? snapshot)
        {
            snapshot = _snapshot;
            return true;
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<FakeKernelConfiguration> snapshot,
            IEvolutionGenomeCodec<FakeKernelConfiguration> codec) => true;
    }

    private sealed class ThrowingStore : IKernelTuningStore<FakeKernelConfiguration>
    {
        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<FakeKernelConfiguration> codec,
            out KernelTuningDeploymentSnapshot<FakeKernelConfiguration>? snapshot)
        {
            snapshot = null;
            throw new IOException("Simulated cache read failure.");
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<FakeKernelConfiguration> snapshot,
            IEvolutionGenomeCodec<FakeKernelConfiguration> codec) =>
            throw new IOException("Simulated cache write failure.");
    }

    private sealed class ManualIdleGate : IKernelTuningIdleGate
    {
        internal TaskCompletionSource<bool> Waiting { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal TaskCompletionSource<bool> Release { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public async ValueTask WaitUntilIdleAsync(
            KernelTuningIdentity identity,
            CancellationToken cancellationToken = default)
        {
            Waiting.TrySetResult(true);
            using (cancellationToken.Register(() => Release.TrySetCanceled()))
                await Release.Task;
        }
    }
}
