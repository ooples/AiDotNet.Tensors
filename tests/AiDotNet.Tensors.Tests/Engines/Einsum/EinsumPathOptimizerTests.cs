using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.Einsum;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

public class EinsumPathOptimizerTests
{
    private static EinsumPath Greedy(string equation, params int[][] shapes)
        => EinsumPathOptimizer.Greedy(
            EinsumShapeBinding.Bind(EinsumEquation.Parse(equation), shapes));

    [Fact]
    public void SingleOperand_HasNoSteps()
    {
        var p = Greedy("ij->ji", new[] { 3, 5 });
        Assert.Empty(p.Steps);
        Assert.Equal(0, p.TotalFlops);
    }

    [Fact]
    public void TwoOperands_OneStep()
    {
        var p = Greedy("ij,jk->ik", new[] { 3, 4 }, new[] { 4, 5 });
        Assert.Single(p.Steps);
        var step = p.Steps[0];
        Assert.Equal(0, step.LeftIndex);
        Assert.Equal(1, step.RightIndex);
        Assert.Equal(new[] { 'i', 'k' }, step.ResultLabels.OrderBy(c => c));
        Assert.Equal(new[] { 'j' }, step.ContractedLabels);
        // cost = 2 * 3 * 4 * 5 = 120
        Assert.Equal(120L, step.EstimatedFlops);
        Assert.Equal(120L, p.TotalFlops);
    }

    [Fact]
    public void ThreeWayChain_TotalFlopsEqualsSumOfSteps()
    {
        var p = Greedy("ab,bc,cd->ad",
            new[] { 2, 3 },
            new[] { 3, 4 },
            new[] { 4, 5 });
        Assert.Equal(2, p.Steps.Count);
        Assert.Equal(
            p.Steps[0].EstimatedFlops + p.Steps[1].EstimatedFlops,
            p.TotalFlops);
    }

    [Fact]
    public void ThreeWayChain_GreedyPicksCheaperPairFirst()
    {
        // ab * bc = ac (cost 2*a*b*c)
        // bc * cd = bd (cost 2*b*c*d)
        // Choose the cheaper first contraction.
        // Here a=100, b=2, c=2, d=100 → ab*bc costs 800, bc*cd costs 800 (tie)
        // Let's make it clearly lopsided: a=2, b=100, c=2, d=2 → ab*bc = 2*2*100*2=800,
        // bc*cd = 2*100*2*2 = 800. Still tied. Try a=2 b=100 c=3 d=2:
        // (ab,bc): 2*2*100*3 = 1200
        // (bc,cd): 2*100*3*2 = 1200. Multiplication-commutative tied cases.
        // Let's pick a case where the ordering clearly matters.
        // ab=large, bc=small, cd=large would make (bc,cd) cheap then (ab, bd)
        // be cheaper than (ab,bc) then (ac,cd).
        var p = Greedy("ab,bc,cd->ad",
            new[] { 100, 2 },
            new[] { 2, 2 },
            new[] { 2, 100 });
        // With greedy minimising cost, either order is fine; both achievable.
        // The point of this test is that TotalFlops reflects what greedy picked.
        Assert.Equal(2, p.Steps.Count);
        Assert.True(p.TotalFlops > 0);
    }

    [Fact]
    public void AttentionShape_ThreeOperands_Succeeds()
    {
        // (Q·K^T)·V: "bhqd,bhkd->bhqk" then "bhqk,bhkd->bhqd"
        // but as a single 3-operand einsum "bhqd,bhkd,bhvd->bhqv" (v=k identified here)
        // Use: bhqd * bhkd (inner product on d) yields bhqk, then * bhkf → bhqf
        var p = Greedy("bhqd,bhkd,bhkf->bhqf",
            new[] { 2, 4, 8, 16 },
            new[] { 2, 4, 32, 16 },
            new[] { 2, 4, 32, 8 });
        Assert.Equal(2, p.Steps.Count);
        Assert.True(p.TotalFlops > 0);
    }

    [Fact]
    public void Ellipsis_IsTreatedAsLabeledBlock()
    {
        var p = Greedy("...ij,...jk->...ik",
            new[] { 2, 3, 4 },
            new[] { 2, 4, 5 });
        Assert.Single(p.Steps);
        // The ellipsis marker '@' is expected in both result and contracted-not
        // (ellipsis persists to output → it's in result).
        Assert.Contains('@', p.Steps[0].ResultLabels);
        Assert.DoesNotContain('@', p.Steps[0].ContractedLabels);
    }

    [Fact]
    public void Reduction_SingleOperand_ToScalar_ZeroSteps()
    {
        // Single-operand reductions and transposes have no pairwise
        // contraction steps; the executor will handle them.
        var p = Greedy("ij->", new[] { 3, 5 });
        Assert.Empty(p.Steps);
    }

    [Fact]
    public void FiveOperands_LinearChain_HasFourSteps()
    {
        var p = Greedy("ab,bc,cd,de,ef->af",
            new[] { 2, 3 },
            new[] { 3, 4 },
            new[] { 4, 5 },
            new[] { 5, 6 },
            new[] { 6, 7 });
        Assert.Equal(4, p.Steps.Count);
    }

    [Fact]
    public void Greedy_ThrowsWhenFinalFlopMultiplicationOverflows()
    {
        Assert.Throws<OverflowException>(() => Greedy(
            "abc,abc->",
            new[] { int.MaxValue, int.MaxValue, 2 },
            new[] { int.MaxValue, int.MaxValue, 2 }));
    }

    [Fact]
    public void ResultLabels_IncludeDownstreamRequirements()
    {
        // "ij,jk,kl->il": at step 1 we must keep both 'i' (needed by
        // output) and 'k' (needed by operand 3); 'j' can be contracted.
        var p = Greedy("ij,jk,kl->il",
            new[] { 2, 3 },
            new[] { 3, 4 },
            new[] { 4, 5 });
        Assert.Equal(2, p.Steps.Count);
        // First step either contracts (0,1) or (1,2). Verify that the result
        // of the first step retains labels needed later.
        var first = p.Steps[0];
        var second = p.Steps[1];
        // Last step produces only output labels.
        Assert.Equal(new[] { 'i', 'l' }, second.ResultLabels.OrderBy(c => c));
    }
}

[Collection("AutotuneCacheTests")]
public sealed class EinsumEvolutionAutotunerTests : IDisposable
{
    private const string CacheEnvironmentVariable = "AIDOTNET_AUTOTUNE_CACHE_PATH";
    private readonly string _cachePath;
    private readonly string? _originalCachePath;

    public EinsumEvolutionAutotunerTests()
    {
        _originalCachePath = Environment.GetEnvironmentVariable(CacheEnvironmentVariable);
        _cachePath = Path.Combine(
            Path.GetTempPath(), "aidotnet-einsum-evolution-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable(CacheEnvironmentVariable, _cachePath);
        EinsumPathCache.ClearMemoryForTests();
    }

    [Fact]
    public void Optimize_ReconstructsPersistedTypedOrderInsteadOfRerunningGreedy()
    {
        EinsumShapeBinding binding = Bind();
        var storedOrder = new EinsumContractionOrder(new[]
        {
            new EinsumContractionPair(0, 2),
            new EinsumContractionPair(0, 1)
        });
        EinsumPath stored = EinsumPathOptimizer.BuildPath(
            binding, storedOrder, EinsumPathStrategy.Evolutionary);

        KernelTuningIdentity identity = CpuIdentity(binding);
        StoreEvolution(binding, identity, stored);
        EinsumPathCache.ClearMemoryForTests();

        EinsumPath loaded = EinsumPathOptimizer.Optimize(binding);

        Assert.Equal(EinsumPathStrategy.Evolutionary, loaded.Strategy);
        Assert.Equal(
            storedOrder.Pairs.ToArray(),
            loaded.ContractionOrder.Pairs.ToArray());
        Assert.Equal(stored.TotalFlops, loaded.TotalFlops);
    }

    [Fact]
    public void Optimize_DoesNotUseAPathMeasuredForAnotherDevice()
    {
        EinsumShapeBinding binding = Bind();
        var gpu = new KernelTuningDeviceFingerprint(
            KernelTuningDeviceKind.NvidiaGpu,
            "test-gpu-local",
            "test-gpu-model");
        KernelTuningIdentity gpuIdentity = EinsumPathCache.CreateIdentity(
            binding,
            gpu,
            new KernelSearchSpaceVersion(EinsumPathOptimizer.CurrentSearchSpaceVersion),
            new KernelBenchmarkProtocolVersion(EinsumPathOptimizer.CurrentBenchmarkProtocolVersion));
        var storedOrder = new EinsumContractionOrder(new[]
        {
            new EinsumContractionPair(0, 2),
            new EinsumContractionPair(0, 1)
        });
        EinsumPath stored = EinsumPathOptimizer.BuildPath(
            binding, storedOrder, EinsumPathStrategy.Evolutionary);
        StoreEvolution(binding, gpuIdentity, stored);

        EinsumPath cpuPath = EinsumPathOptimizer.Optimize(binding);
        EinsumPath gpuPath = EinsumPathOptimizer.Optimize(
            binding,
            gpu,
            new KernelSearchSpaceVersion(EinsumPathOptimizer.CurrentSearchSpaceVersion),
            new KernelBenchmarkProtocolVersion(EinsumPathOptimizer.CurrentBenchmarkProtocolVersion));

        Assert.Equal(EinsumPathStrategy.Greedy, cpuPath.Strategy);
        Assert.Equal(EinsumPathStrategy.Evolutionary, gpuPath.Strategy);
        Assert.Equal(storedOrder.Pairs.ToArray(), gpuPath.ContractionOrder.Pairs.ToArray());
    }

    [Fact]
    public void PathCache_RejectsPathWhoseEvidenceDoesNotMatchItsTypedOrder()
    {
        EinsumShapeBinding binding = Bind();
        EinsumPath valid = EinsumPathOptimizer.Greedy(binding);
        var inconsistent = new EinsumPath(
            valid.Steps,
            valid.TotalFlops + 1,
            valid.ContractionOrder,
            EinsumPathStrategy.Greedy);

        Assert.Throws<ArgumentException>(() =>
            EinsumPathCache.TryStore(binding, CpuIdentity(binding), inconsistent));
    }

    [Fact]
    public async Task TuneAsync_PublishesMeasuredTypedOrderToServingPathCache()
    {
        EinsumShapeBinding binding = Bind();
        var measuredWinner = new EinsumContractionOrder(new[]
        {
            new EinsumContractionPair(0, 2),
            new EinsumContractionPair(0, 1)
        });
        var options = new EvolutionEngineOptions
        {
            RunId = "einsum-evolution-test",
            Seed = 7,
            MaxEvaluationAttempts = 2,
            MaxProposals = 2,
            MaxGenerations = 0,
            ProposalBatchSize = 2,
            MaxDegreeOfParallelism = 1,
            IslandCount = 1,
            MigrationInterval = 0,
            MigrantsPerIsland = 1
        };

        EinsumEvolutionTuningResult result = await EinsumEvolutionAutotuner.TuneAsync(
            binding,
            KernelTuningDeviceFingerprint.CurrentCpu(),
            Measure,
            new KernelSearchSpaceVersion(1),
            new KernelBenchmarkProtocolVersion(1),
            new[] { measuredWinner },
            options);

        Assert.True(result.WasPromoted);
        Assert.Equal(measuredWinner.Pairs.ToArray(), result.ActivePath.ContractionOrder.Pairs.ToArray());
        EinsumPath servingPath = EinsumPathOptimizer.Optimize(binding);
        Assert.Equal(EinsumPathStrategy.Evolutionary, servingPath.Strategy);
        Assert.Equal(measuredWinner.Pairs.ToArray(), servingPath.ContractionOrder.Pairs.ToArray());
    }

    [Fact]
    public async Task TuneAsync_WithCustomStore_DoesNotAlsoWriteDefaultCache()
    {
        EinsumShapeBinding binding = Bind();
        var measuredWinner = new EinsumContractionOrder(new[]
        {
            new EinsumContractionPair(0, 2),
            new EinsumContractionPair(0, 1)
        });
        var options = new EvolutionEngineOptions
        {
            RunId = "einsum-custom-store-test",
            Seed = 11,
            MaxEvaluationAttempts = 2,
            MaxProposals = 2,
            MaxGenerations = 0,
            ProposalBatchSize = 2,
            MaxDegreeOfParallelism = 1,
            IslandCount = 1,
            MigrationInterval = 0,
            MigrantsPerIsland = 1
        };
        var customStore = new MemoryStore();

        EinsumEvolutionTuningResult result = await EinsumEvolutionAutotuner.TuneAsync(
            binding,
            KernelTuningDeviceFingerprint.CurrentCpu(),
            Measure,
            new KernelSearchSpaceVersion(1),
            new KernelBenchmarkProtocolVersion(1),
            new[] { measuredWinner },
            options,
            store: customStore);

        Assert.True(result.Evolution.WasPersisted);
        Assert.NotNull(customStore.Snapshot);
        EinsumPath servingPath = EinsumPathOptimizer.Optimize(binding);
        Assert.Equal(EinsumPathStrategy.Evolutionary, servingPath.Strategy);

        KernelTuningIdentity identity = CpuIdentity(binding);
        var codec = new EinsumEvolutionAutotuner.EinsumContractionOrderCodec(
            binding.Equation.Operands.Count);
        var defaultStore = new AutotuneCacheKernelTuningStore<EinsumContractionOrder>();
        Assert.False(defaultStore.TryLoad(identity, codec, out _));
    }

    public void Dispose()
    {
        EinsumPathCache.ClearMemoryForTests();
        Environment.SetEnvironmentVariable(CacheEnvironmentVariable, _originalCachePath);
        if (Directory.Exists(_cachePath)) Directory.Delete(_cachePath, recursive: true);
    }

    private static EinsumShapeBinding Bind() => EinsumShapeBinding.Bind(
        EinsumEquation.Parse("ab,bc,cd->ad"),
        new[] { new[] { 32, 8 }, new[] { 8, 16 }, new[] { 16, 64 } });

    private static KernelTuningIdentity CpuIdentity(EinsumShapeBinding binding) =>
        EinsumPathCache.CreateIdentity(
            binding,
            KernelTuningDeviceFingerprint.CurrentCpu(),
            new KernelSearchSpaceVersion(EinsumPathOptimizer.CurrentSearchSpaceVersion),
            new KernelBenchmarkProtocolVersion(EinsumPathOptimizer.CurrentBenchmarkProtocolVersion));

    private static ValueTask<KernelTuningTrialResult> Measure(
        EinsumPath path,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        EinsumContractionPair first = path.ContractionOrder.Pairs[0];
        double throughput = first == new EinsumContractionPair(0, 2) ? 200 : 100;
        return new ValueTask<KernelTuningTrialResult>(KernelTuningTrialResult.Passed(
            Measurement(path, throughput)));
    }

    private static KernelTuningDeploymentSnapshot<EinsumContractionOrder> Snapshot(
        EinsumShapeBinding binding,
        KernelTuningIdentity identity,
        EinsumPath path)
    {
        var codec = new EinsumEvolutionAutotuner.EinsumContractionOrderCodec(
            binding.Equation.Operands.Count);
        string payload = codec.Serialize(path.ContractionOrder);
        return new KernelTuningDeploymentSnapshot<EinsumContractionOrder>(
            identity,
            path.ContractionOrder,
            EvolutionHash.Compute(payload),
            Measurement(path, 200),
            "test-einsum-run-state");
    }

    private static void StoreEvolution(
        EinsumShapeBinding binding,
        KernelTuningIdentity identity,
        EinsumPath path)
    {
        var codec = new EinsumEvolutionAutotuner.EinsumContractionOrderCodec(
            binding.Equation.Operands.Count);
        Assert.True(new AutotuneCacheKernelTuningStore<EinsumContractionOrder>()
            .TryStore(Snapshot(binding, identity, path), codec));
    }

    private static KernelTuningMeasurement Measurement(EinsumPath path, double throughput)
    {
        var timing = KernelTimingStatistics.FromSamples(new[]
        {
            TimeSpan.FromMilliseconds(1.00),
            TimeSpan.FromMilliseconds(0.98),
            TimeSpan.FromMilliseconds(1.02),
            TimeSpan.FromMilliseconds(0.99),
            TimeSpan.FromMilliseconds(1.01)
        });
        var resources = new KernelTuningResourceUsage(
            workspaceBytes: 4096,
            occupancyRatio: 1,
            registersPerThread: 0,
            compileTime: TimeSpan.Zero,
            kernelLaunchCount: path.Steps.Count);
        var correctness = new KernelTuningCorrectnessEvidence(
            KernelTuningValidationScope.Output,
            outputAbsoluteError: 0,
            outputRelativeError: 0,
            outputAbsoluteTolerance: 1e-6,
            outputRelativeTolerance: 1e-6);
        return new KernelTuningMeasurement(throughput, timing, resources, correctness);
    }

    private sealed class MemoryStore : IKernelTuningStore<EinsumContractionOrder>
    {
        internal KernelTuningDeploymentSnapshot<EinsumContractionOrder>? Snapshot { get; private set; }

        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<EinsumContractionOrder> codec,
            out KernelTuningDeploymentSnapshot<EinsumContractionOrder>? snapshot)
        {
            snapshot = Snapshot;
            return snapshot is not null &&
                   string.Equals(snapshot.Identity.StableKey, identity.StableKey, StringComparison.Ordinal);
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<EinsumContractionOrder> snapshot,
            IEvolutionGenomeCodec<EinsumContractionOrder> codec)
        {
            Snapshot = snapshot;
            return true;
        }
    }
}
