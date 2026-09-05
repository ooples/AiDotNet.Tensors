using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers.Autotune;
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
        const int m = 257;
        const int n = 193;
        const int k = 129;
        IReadOnlyList<BlasManagedGemmConfiguration> seeds =
            BlasManagedEvolutionAutotuner.GetSeeds<float>(
                m, n, k, transA: true, transB: false, deterministic: false);

        EvolutionKernelTuningResult<BlasManagedGemmConfiguration> result = await
            BlasManagedEvolutionAutotuner.TuneAsync<float>(
                m, n, k,
                transA: true,
                transB: false,
                deterministic: false,
                (configuration, context, cancellationToken) =>
                {
                    double throughput = configuration.PackingMode switch
                    {
                        PackingMode.ForceStreaming => 100,
                        PackingMode.ForcePackAOnly => 200,
                        PackingMode.ForcePackBoth => 300,
                        _ => throw new ArgumentOutOfRangeException(nameof(configuration))
                    };
                    return new ValueTask<KernelTuningTrialResult>(Passed(configuration, throughput));
                },
                new KernelSearchSpaceVersion(1),
                new KernelBenchmarkProtocolVersion(1),
                engineOptions: EngineOptions(seeds.Count),
                deploymentRegistry: new KernelTuningDeploymentRegistry<BlasManagedGemmConfiguration>(),
                store: new MemoryStore());

        ShapeProfile shape = BlasManagedAutotune.EncodeShape<float>(
            m, n, k, transA: true, transB: false,
            mr: 0, nr: 0, hasEpilogue: false, isDeterministic: false);
        var active = BlasManagedAutotune.TryLookupStrategy(shape);

        Assert.True(result.WasPromoted);
        Assert.True(active.HasValue);
        var activeValue = active.GetValueOrDefault();
        Assert.Equal(result.ActiveDeployment.Configuration.PackingMode, activeValue.Mode);
        Assert.Equal(result.ActiveDeployment.Configuration.ParallelismAxis, activeValue.Axis);
        Assert.Equal(result.ActiveDeployment.Configuration.Mc, activeValue.Mc);
        Assert.Equal(result.ActiveDeployment.Configuration.ThreadCount, activeValue.ThreadCount);
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
        var timing = KernelTimingStatistics.FromSamples(new[]
        {
            TimeSpan.FromMilliseconds(1.1),
            TimeSpan.FromMilliseconds(1.0),
            TimeSpan.FromMilliseconds(0.9),
            TimeSpan.FromMilliseconds(1.05),
            TimeSpan.FromMilliseconds(0.95)
        });
        long workspace = configuration.PackingMode == PackingMode.ForceStreaming
            ? 0
            : (long)(configuration.Mc * configuration.Kc + configuration.Kc * configuration.Nc) * sizeof(float);
        double occupancy = configuration.ParallelismAxis == ParallelismAxis.None
            ? 1d / Math.Max(1, Environment.ProcessorCount)
            : Math.Min(1d, (double)configuration.ThreadCount / Math.Max(1, Environment.ProcessorCount));
        return KernelTuningTrialResult.Passed(new KernelTuningMeasurement(
            throughput,
            timing,
            new KernelTuningResourceUsage(
                workspace, occupancy, registersPerThread: 0, compileTime: TimeSpan.Zero),
            new KernelTuningCorrectnessEvidence(
                KernelTuningValidationScope.Output,
                1e-7, 2e-7, 1e-5, 1e-5)));
    }

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
}
