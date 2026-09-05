using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public sealed class CodegenTiledContractionEvolutionExplorerTests
{
    [Fact]
    public void Identity_SeparatesEqualGeometryWithDifferentKernelSemantics()
    {
        CodegenKernelSpec activated = (CodegenKernelCatalog.Find("conv2d_1x1_bias_relu") ??
            throw new InvalidOperationException("The test catalog entry is missing.")).Bench;
        var unactivated = new CodegenKernelSpec(
            activated.Name,
            activated.Space,
            activated.Inputs.ToArray(),
            activated.Output,
            activated.ProductInputs.ToArray(),
            activated.Reduce,
            activated.BiasInput,
            activated.ScaleInput,
            CodegenActivationKind.None,
            activated.ReduceScale,
            activated.PreReduce,
            activated.PreBiasInput,
            activated.PreBiasScale,
            activated.Algebra,
            activated.ExtraOutputs.ToArray());
        Assert.NotEqual(activated.Activation, unactivated.Activation);
        Assert.True(CodegenTiledContractionPlan.TryCreate(
            activated, out CodegenTiledContractionPlan? activatedPlan, out string activatedReason),
            activatedReason);
        Assert.True(CodegenTiledContractionPlan.TryCreate(
            unactivated, out CodegenTiledContractionPlan? unactivatedPlan, out string unactivatedReason),
            unactivatedReason);
        var device = new GpuDeviceFingerprint(
            GpuVendorKind.Nvidia, "Test GPU", 8, 6, 550, "test-codegen-identity");
        var kernel = new KernelId("codegen", "same-caller-kernel-id");
        var searchSpace = new KernelSearchSpaceVersion(1);

        KernelTuningIdentity activatedIdentity =
            CodegenTiledContractionEvolutionExplorer.CreateIdentity(
                activated,
                activatedPlan ?? throw new InvalidOperationException("The activated plan is missing."),
                kernel,
                device,
                8,
                6,
                searchSpace);
        KernelTuningIdentity unactivatedIdentity =
            CodegenTiledContractionEvolutionExplorer.CreateIdentity(
                unactivated,
                unactivatedPlan ?? throw new InvalidOperationException("The unactivated plan is missing."),
                kernel,
                device,
                8,
                6,
                searchSpace);

        Assert.Equal(activatedIdentity.Shape, unactivatedIdentity.Shape);
        Assert.NotEqual(activatedIdentity.Kernel, unactivatedIdentity.Kernel);
        Assert.NotEqual(activatedIdentity.StableKey, unactivatedIdentity.StableKey);
    }

    [Fact]
    public async Task ExploreAsync_FiltersInvalidGeometryBeforeGpuEvaluator()
    {
        CodegenKernelSpec spec = (CodegenKernelCatalog.Find("conv2d_1x1_bias_relu") ??
            throw new InvalidOperationException("The test catalog entry is missing.")).Bench;
        var invalid = new CodegenTiledContractionSchedule(64, 112, 64, 8, 4);
        bool invalidReachedEvaluator = false;
        int evaluations = 0;
        var options = new EvolutionEngineOptions
        {
            RunId = "codegen-evolution-explorer-test",
            Seed = 17,
            MaxEvaluationAttempts = 64,
            MaxProposals = 64,
            MaxGenerations = 0,
            ProposalBatchSize = 64,
            MaxDegreeOfParallelism = 1,
            IslandCount = 1,
            MigrationInterval = 0,
            MigrantsPerIsland = 1
        };

        EvolutionKernelTuningResult<CodegenTiledContractionSchedule> result =
            await CodegenTiledContractionEvolutionExplorer.ExploreAsync(
                spec,
                new KernelId("codegen", "conv2d-1x1-evolution-test"),
                new GpuDeviceFingerprint(
                    GpuVendorKind.Nvidia, "Test GPU", 8, 6, 550, "test-codegen-gpu"),
                8,
                6,
                (schedule, plan, context, cancellationToken) =>
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    evaluations++;
                    if (schedule.TileK == invalid.TileK) invalidReachedEvaluator = true;
                    Assert.InRange(plan.BlockThreads, 32, 256);
                    return new ValueTask<KernelTuningTrialResult>(Passed(schedule));
                },
                new KernelSearchSpaceVersion(1),
                new[] { invalid },
                options,
                store: new MemoryStore());

        Assert.True(evaluations > 0);
        Assert.False(invalidReachedEvaluator);
        Assert.True(CodegenTiledContractionPlan.TryCreate(
            spec,
            result.ActiveDeployment.Configuration,
            out CodegenTiledContractionPlan? active,
            out string reason), reason);
        Assert.NotNull(active);
    }

    private static KernelTuningTrialResult Passed(CodegenTiledContractionSchedule schedule)
    {
        var timing = KernelTimingStatistics.FromSamples(new[]
        {
            TimeSpan.FromMilliseconds(1.00),
            TimeSpan.FromMilliseconds(0.99),
            TimeSpan.FromMilliseconds(1.01),
            TimeSpan.FromMilliseconds(0.98),
            TimeSpan.FromMilliseconds(1.02)
        });
        var resources = new KernelTuningResourceUsage(
            workspaceBytes: schedule.TileK * (long)(schedule.TileM + schedule.TileN) * sizeof(float) * 2,
            occupancyRatio: 0.75,
            registersPerThread: schedule.ThreadTileM * schedule.ThreadTileN,
            compileTime: TimeSpan.FromMilliseconds(2));
        var correctness = new KernelTuningCorrectnessEvidence(
            KernelTuningValidationScope.Output,
            outputAbsoluteError: 0,
            outputRelativeError: 0,
            outputAbsoluteTolerance: CodegenMeasurementProtocol.AccumulationTolerance,
            outputRelativeTolerance: CodegenMeasurementProtocol.AccumulationTolerance);
        double throughput = schedule.TileM * schedule.TileN / 100d;
        return KernelTuningTrialResult.Passed(
            new KernelTuningMeasurement(throughput, timing, resources, correctness));
    }

    private sealed class MemoryStore : IKernelTuningStore<CodegenTiledContractionSchedule>
    {
        private KernelTuningDeploymentSnapshot<CodegenTiledContractionSchedule>? _snapshot;

        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<CodegenTiledContractionSchedule> codec,
            out KernelTuningDeploymentSnapshot<CodegenTiledContractionSchedule>? snapshot)
        {
            snapshot = _snapshot;
            return snapshot is not null;
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<CodegenTiledContractionSchedule> snapshot,
            IEvolutionGenomeCodec<CodegenTiledContractionSchedule> codec)
        {
            _snapshot = snapshot;
            return true;
        }
    }
}
