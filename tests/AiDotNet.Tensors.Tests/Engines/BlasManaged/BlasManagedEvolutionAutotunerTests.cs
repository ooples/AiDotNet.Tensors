using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.Tests.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("AutotuneCacheTests")]
public sealed class BlasManagedEvolutionAutotunerTests : IDisposable
{
    private const string CacheEnvironmentVariable = "AIDOTNET_AUTOTUNE_CACHE_PATH";
    private readonly string? _originalCachePath;
    private readonly string _temporaryCachePath;

    public BlasManagedEvolutionAutotunerTests()
    {
        _originalCachePath = Environment.GetEnvironmentVariable(CacheEnvironmentVariable);
        _temporaryCachePath = Path.Combine(
            Path.GetTempPath(), "aidotnet-blas-evolution-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable(CacheEnvironmentVariable, _temporaryCachePath);
        BlasManagedAutotune.ClearStrategyMemo();
    }

    [Fact]
    public void Seeds_RejectTransposedBAliasAndDeterministicKAxis()
    {
        var packAAlias = new BlasManagedGemmConfiguration(
            PackingMode.ForcePackAOnly, ParallelismAxis.M, 64, 64, 64, 1);
        var nondeterministicReduction = new BlasManagedGemmConfiguration(
            PackingMode.ForcePackBoth, ParallelismAxis.K, 64, 64, 64, 1);

        IReadOnlyList<BlasManagedGemmConfiguration> seeds =
            BlasManagedEvolutionAutotuner.GetSeeds<float>(
                256, 256, 256,
                transA: false,
                transB: true,
                deterministic: true,
                new[] { packAAlias, nondeterministicReduction });

        Assert.DoesNotContain(seeds, seed => seed.PackingMode == PackingMode.ForcePackAOnly);
        Assert.DoesNotContain(seeds, seed => seed.ParallelismAxis == ParallelismAxis.K);
    }

    [Fact]
    public void ReplayEligibility_RequiresTheExactFirstPartyBenchmarkContext()
    {
        var defaultOptions = new BlasOptions<float>();
        Assert.True(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged
            .IsEvolutionReplayEligible(in defaultOptions));

        var explicitThreads = new BlasOptions<float> { NumThreads = 2 };
        var explicitBlocks = new BlasOptions<float> { Mc = 8 };
        var betaZero = new BlasOptions<float> { BetaZero = true };
        byte[] workspaceBytes = new byte[64];
        var workspace = new BlasOptions<float> { Workspace = workspaceBytes };
        float[] bias = new float[8];
        var epilogue = new BlasOptions<float>
        {
            Epilogue = new Epilogue<float> { BiasN = bias }
        };

        Assert.False(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged
            .IsEvolutionReplayEligible(in explicitThreads));
        Assert.False(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged
            .IsEvolutionReplayEligible(in explicitBlocks));
        Assert.False(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged
            .IsEvolutionReplayEligible(in betaZero));
        Assert.False(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged
            .IsEvolutionReplayEligible(in workspace));
        Assert.False(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged
            .IsEvolutionReplayEligible(in epilogue));
    }

    [Fact]
    public async Task DeterminismGate_RejectsKAxisBeforeCpuEvaluator()
    {
        BlasManagedGemmConfiguration valid = BlasManagedEvolutionAutotuner.GetSeeds<float>(
            256, 256, 256, false, false, deterministic: true)[0];
        BlasManagedGemmConfiguration invalid = valid with
        {
            ParallelismAxis = ParallelismAxis.K,
            ThreadCount = 1
        };
        int evaluatorCalls = 0;
        EvolutionKernelAutotuner<BlasManagedGemmConfiguration> tuner =
            BlasManagedEvolutionAutotuner.Create<float>(
                256, 256, 256,
                transA: false,
                transB: false,
                deterministic: true,
                (configuration, context, cancellationToken) =>
                {
                    evaluatorCalls++;
                    return new ValueTask<KernelTuningTrialResult>(Passed(configuration));
                },
                Finalist(valid),
                new KernelSearchSpaceVersion(1),
                new KernelBenchmarkProtocolVersion(1),
                EngineOptions(2),
                deploymentRegistry: new KernelTuningDeploymentRegistry<BlasManagedGemmConfiguration>(),
                store: new MemoryStore());

        EvolutionKernelTuningResult<BlasManagedGemmConfiguration> result =
            await tuner.TuneAsync(new[] { invalid, valid });

        Assert.Equal(1, evaluatorCalls);
        Assert.Equal(1, result.Run.Counters.StatusCounts[EvolutionEvaluationStatus.Rejected]);
        Assert.Equal(valid, result.ActiveDeployment.Configuration);
    }

    [Fact]
    public async Task PromotedWinner_UpdatesExistingDispatchMemoWithoutNewHotPathLookup()
    {
        const int m = 288;
        const int n = 256;
        const int k = 128;
        IReadOnlyList<BlasManagedGemmConfiguration> seeds =
            BlasManagedEvolutionAutotuner.GetSeeds<float>(
                m, n, k, transA: true, transB: false, deterministic: false);
        BlasManagedGemmConfiguration incumbent = seeds.First(
            configuration => configuration.PackingMode == PackingMode.ForceStreaming);

        EvolutionKernelTuningResult<BlasManagedGemmConfiguration> result = await
            BlasManagedEvolutionAutotuner.TuneAsync<float>(
                m, n, k,
                transA: true,
                transB: false,
                deterministic: false,
                (configuration, context, cancellationToken) =>
                {
                    return new ValueTask<KernelTuningTrialResult>(Passed(
                        configuration,
                        Throughput(configuration)));
                },
                Finalist(incumbent),
                new KernelSearchSpaceVersion(1),
                new KernelBenchmarkProtocolVersion(1),
                engineOptions: EngineOptions(seeds.Count),
                deploymentRegistry: new KernelTuningDeploymentRegistry<BlasManagedGemmConfiguration>(),
                store: new MemoryStore());

        ShapeProfile shape = BlasManagedAutotune.EncodeShape<float>(
            m, n, k, transA: true, transB: false,
            mr: 0, nr: 0, hasEpilogue: false, isDeterministic: false);
        var active = BlasManagedAutotune.TryLookupStrategy(shape);
        bool hasEvolutionDeployment = BlasManagedAutotune.TryLookupEvolutionStrategy<float>(
            m, n, k, transA: true, transB: false, deterministic: false,
            out BlasManagedGemmConfiguration evolved);

        Assert.True(result.WasPromoted);
        Assert.True(active.HasValue);
        var activeValue = active.GetValueOrDefault();
        Assert.Equal(result.ActiveDeployment.Configuration.PackingMode, activeValue.Mode);
        Assert.Equal(result.ActiveDeployment.Configuration.ParallelismAxis, activeValue.Axis);
        Assert.Equal(result.ActiveDeployment.Configuration.Mc, activeValue.Mc);
        Assert.Equal(result.ActiveDeployment.Configuration.ThreadCount, activeValue.ThreadCount);
        Assert.True(hasEvolutionDeployment);
        Assert.Equal(result.ActiveDeployment.Configuration, evolved);
    }

    [Fact]
    public async Task FirstPartyExperiment_RunsRealGemmOracleTimingAndProductionReplay()
    {
        const int m = 96;
        const int n = 64;
        const int k = 64;
        IReadOnlyList<BlasManagedGemmConfiguration> seeds =
            BlasManagedEvolutionAutotuner.GetSeeds<float>(
                m, n, k, transA: true, transB: false, deterministic: true);
        var tuningOptions = new KernelTuningOptions
        {
            MinimumPromotionRatio = 1,
            MaximumP95LatencyRatio = 5,
        };

        EvolutionKernelTuningResult<BlasManagedGemmConfiguration> result = await
            BlasManagedEvolutionAutotuner.TuneAsync<float>(
                m, n, k,
                transA: true,
                transB: false,
                deterministic: true,
                new KernelSearchSpaceVersion(2),
                new KernelBenchmarkProtocolVersion(2),
                engineOptions: EngineOptions(seeds.Count),
                tuningOptions: tuningOptions,
                deploymentRegistry: new KernelTuningDeploymentRegistry<BlasManagedGemmConfiguration>(),
                store: new MemoryStore(),
                inputSeed: 9173,
                warmupCount: 1,
                searchSampleCount: 3,
                holdoutSampleCount: 7);

        Assert.Equal(seeds.Count, result.Run.Counters.EvaluationAttempts);
        Assert.Equal(7, result.ProposedWinner.PromotionEvidence.Samples.Count);
        Assert.True(result.ProposedWinner.Measurement.Timing.HasRawSamples);
        Assert.Equal(KernelTuningWorkUnit.FloatingPointOperations,
            result.ProposedWinner.Measurement.Workload.Unit);
        Assert.Equal(2d * m * n * k,
            result.ProposedWinner.Measurement.Workload.UnitsPerOperation);
        Assert.True(
            result.ProposedWinner.Measurement.Correctness.OutputAbsoluteError <=
            result.ProposedWinner.Measurement.Correctness.OutputAbsoluteTolerance ||
            result.ProposedWinner.Measurement.Correctness.OutputRelativeError <=
            result.ProposedWinner.Measurement.Correctness.OutputRelativeTolerance);

        bool hasEvolutionDeployment = BlasManagedAutotune.TryLookupEvolutionStrategy<float>(
            m, n, k, transA: true, transB: false, deterministic: true,
            out BlasManagedGemmConfiguration deployed);
        Assert.Equal(result.ActiveDeployment.Configuration.PackingMode != PackingMode.Auto,
            hasEvolutionDeployment);
        if (hasEvolutionDeployment)
            Assert.Equal(result.ActiveDeployment.Configuration, deployed);

        float[] a = Enumerable.Range(0, k * m)
            .Select(index => (index % 17 - 8) * 0.01f)
            .ToArray();
        float[] b = Enumerable.Range(0, k * n)
            .Select(index => (index % 13 - 6) * 0.02f)
            .ToArray();
        float[] actual = new float[m * n];
        var options = new BlasOptions<float> { Mode = BlasMode.Deterministic };
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
            a, m, transA: true,
            b, n, transB: false,
            actual, n,
            m, n, k,
            in options);
        for (int row = 0; row < m; row++)
        {
            for (int column = 0; column < n; column++)
            {
                float expected = 0;
                for (int inner = 0; inner < k; inner++)
                    expected += a[inner * m + row] * b[inner * n + column];
                Assert.InRange(Math.Abs(actual[row * n + column] - expected), 0, 2e-5f);
            }
        }
    }

    [Fact]
    public async Task PersistedActivation_RequiresExactIdentityAndCanonicalGenomeEvidence()
    {
        const int m = 288;
        const int n = 256;
        const int k = 128;
        var searchVersion = new KernelSearchSpaceVersion(2);
        var protocolVersion = new KernelBenchmarkProtocolVersion(2);
        IReadOnlyList<BlasManagedGemmConfiguration> seeds =
            BlasManagedEvolutionAutotuner.GetSeeds<float>(
                m, n, k, transA: false, transB: false, deterministic: true);
        BlasManagedGemmConfiguration incumbent = seeds.First(
            configuration => configuration.PackingMode == PackingMode.ForceStreaming);
        var store = new RecordingStore();
        EvolutionKernelTuningResult<BlasManagedGemmConfiguration> tuned = await
            BlasManagedEvolutionAutotuner.TuneAsync<float>(
                m, n, k,
                transA: false,
                transB: false,
                deterministic: true,
                (configuration, context, cancellationToken) =>
                    new ValueTask<KernelTuningTrialResult>(Passed(
                        configuration, Throughput(configuration))),
                Finalist(incumbent),
                searchVersion,
                protocolVersion,
                engineOptions: EngineOptions(seeds.Count),
                deploymentRegistry: new KernelTuningDeploymentRegistry<BlasManagedGemmConfiguration>(),
                store: store);
        Assert.True(tuned.WasPersisted);

        BlasManagedAutotune.ClearStrategyMemo();
        Assert.False(BlasManagedEvolutionAutotuner.TryActivatePersisted<float>(
            m, n, k,
            transA: false,
            transB: false,
            deterministic: true,
            searchVersion,
            new KernelBenchmarkProtocolVersion(3),
            store));
        Assert.False(BlasManagedAutotune.TryLookupEvolutionStrategy<float>(
            m, n, k, transA: false, transB: false, deterministic: true, out _));

        Assert.True(BlasManagedEvolutionAutotuner.TryActivatePersisted<float>(
            m, n, k,
            transA: false,
            transB: false,
            deterministic: true,
            searchVersion,
            protocolVersion,
            store));
        Assert.True(BlasManagedAutotune.TryLookupEvolutionStrategy<float>(
            m, n, k, transA: false, transB: false, deterministic: true,
            out BlasManagedGemmConfiguration activated));
        Assert.Equal(tuned.ActiveDeployment.Configuration, activated);
    }

    [Fact]
    public async Task ExhaustiveMode_ProvesEveryConfigurationInASmallCanonicalSpace()
    {
        var (m, n) = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged
            .GetEvolutionTuningTile<float>(PackingMode.ForcePackBoth, deterministic: true);
        const int k = 1;
        Assert.True(BlasManagedEvolutionAutotuner.TryGetExhaustiveConfigurations<float>(
            m, n, k,
            transA: false,
            transB: false,
            deterministic: true,
            maximumConfigurationCount: 32,
            out IReadOnlyList<BlasManagedGemmConfiguration> completeSpace));
        Assert.Contains(completeSpace,
            configuration => configuration.PackingMode == PackingMode.Auto);
        Assert.Contains(completeSpace,
            configuration => configuration.PackingMode == PackingMode.ForceStreaming);
        Assert.False(BlasManagedEvolutionAutotuner.TryGetExhaustiveConfigurations<float>(
            m, n, k,
            transA: false,
            transB: false,
            deterministic: true,
            maximumConfigurationCount: completeSpace.Count - 1,
            out IReadOnlyList<BlasManagedGemmConfiguration> partialSpace));
        Assert.Empty(partialSpace);

        var options = EngineOptions(completeSpace.Count);
        options.MaxGenerations = 100;
        EvolutionKernelTuningResult<BlasManagedGemmConfiguration> result = await
            BlasManagedEvolutionAutotuner.TuneAsync<float>(
                m, n, k,
                transA: false,
                transB: false,
                deterministic: true,
                new KernelSearchSpaceVersion(2),
                new KernelBenchmarkProtocolVersion(2),
                engineOptions: options,
                tuningOptions: new KernelTuningOptions
                {
                    MinimumPromotionRatio = 1,
                    MaximumP95LatencyRatio = 10,
                },
                deploymentRegistry: new KernelTuningDeploymentRegistry<BlasManagedGemmConfiguration>(),
                store: new MemoryStore(),
                inputSeed: 487,
                warmupCount: 0,
                searchSampleCount: 3,
                holdoutSampleCount: 7,
                searchMode: BlasManagedGemmSearchMode.Exhaustive);

        Assert.Equal(completeSpace.Count, result.Run.Counters.Proposals);
        Assert.Equal(completeSpace.Count, result.Run.Counters.EvaluationAttempts);
        Assert.Equal(1, result.Run.Counters.StatusCounts[EvolutionEvaluationStatus.Rejected]);
        Assert.False(result.Run.Counters.StatusCounts.ContainsKey(EvolutionEvaluationStatus.Failed));
    }

    public void Dispose()
    {
        BlasManagedAutotune.ClearStrategyMemo();
        Environment.SetEnvironmentVariable(CacheEnvironmentVariable, _originalCachePath);
        if (Directory.Exists(_temporaryCachePath)) Directory.Delete(_temporaryCachePath, recursive: true);
    }

    private static EvolutionEngineOptions EngineOptions(int count) => new()
    {
        RunId = "blas-managed-evolution-test",
        Seed = 71,
        MaxEvaluationAttempts = count,
        MaxProposals = count,
        MaxGenerations = 0,
        ProposalBatchSize = Math.Max(1, count),
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1
    };

    private static KernelTuningTrialResult Passed(
        BlasManagedGemmConfiguration configuration,
        double throughput = 100)
    {
        long workspace = configuration.PackingMode == PackingMode.ForceStreaming
            ? 0
            : (long)(configuration.Mc * configuration.Kc + configuration.Kc * configuration.Nc) * sizeof(float);
        double occupancy = configuration.ParallelismAxis == ParallelismAxis.None
            ? 1d / Math.Max(1, Environment.ProcessorCount)
            : Math.Min(1d, (double)configuration.ThreadCount / Math.Max(1, Environment.ProcessorCount));
        return KernelTuningTrialResult.Passed(
            DeterministicFinalistEvaluator<BlasManagedGemmConfiguration>.SearchMeasurement(
                throughput,
                new KernelTuningResourceUsage(
                    workspace, occupancy, registersPerThread: 0, compileTime: TimeSpan.Zero)));
    }

    private static DeterministicFinalistEvaluator<BlasManagedGemmConfiguration> Finalist(
        BlasManagedGemmConfiguration incumbent) => new(
        incumbent,
        Throughput,
        configuration =>
        {
            long workspace = configuration.PackingMode == PackingMode.ForceStreaming
                ? 0
                : (long)(configuration.Mc * configuration.Kc + configuration.Kc * configuration.Nc) * sizeof(float);
            return KernelTuningResourceUsage.ForCpu(workspace, TimeSpan.Zero);
        });

    private static double Throughput(BlasManagedGemmConfiguration configuration) =>
        configuration.PackingMode switch
        {
            PackingMode.ForceStreaming => 100,
            PackingMode.ForcePackAOnly => 200,
            PackingMode.ForcePackBoth => 300,
            _ => throw new ArgumentOutOfRangeException(nameof(configuration))
        };

    private sealed class MemoryStore : IKernelTuningStore<BlasManagedGemmConfiguration>
    {
        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<BlasManagedGemmConfiguration> codec,
            out KernelTuningDeploymentSnapshot<BlasManagedGemmConfiguration>? snapshot)
        {
            snapshot = null;
            return false;
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<BlasManagedGemmConfiguration> snapshot,
            IEvolutionGenomeCodec<BlasManagedGemmConfiguration> codec) => true;
    }

    private sealed class RecordingStore : IKernelTuningStore<BlasManagedGemmConfiguration>
    {
        private KernelTuningDeploymentSnapshot<BlasManagedGemmConfiguration>? _snapshot;

        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<BlasManagedGemmConfiguration> codec,
            out KernelTuningDeploymentSnapshot<BlasManagedGemmConfiguration>? snapshot)
        {
            snapshot = _snapshot;
            return snapshot is not null;
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<BlasManagedGemmConfiguration> snapshot,
            IEvolutionGenomeCodec<BlasManagedGemmConfiguration> codec)
        {
            _snapshot = snapshot;
            return true;
        }
    }
}
