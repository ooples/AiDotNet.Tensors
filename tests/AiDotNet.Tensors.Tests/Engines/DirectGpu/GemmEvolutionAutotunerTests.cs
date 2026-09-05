using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class GemmEvolutionAutotunerTests
{
    [Fact]
    public void HeuristicAndExternalSeeds_AreLocallyValidatedAndDeduplicated()
    {
        var tuner = new GemmAutoTuner();
        GpuCapabilities capabilities = Capabilities(supportsSubgroups: false);
        IReadOnlyList<OpenClGemmConfiguration> baseline =
            tuner.GetEvolutionSeeds(1024, 1024, 1024, capabilities);
        GemmConfig invalidExternal = Copy(
            baseline[0].ToGemmConfig(),
            threadTileM: 32,
            threadTileN: 32,
            useSubgroupOps: true);

        IReadOnlyList<OpenClGemmConfiguration> combined = tuner.GetEvolutionSeeds(
            1024, 1024, 1024, capabilities,
            new[] { baseline[0].ToGemmConfig(), invalidExternal });

        Assert.NotEmpty(baseline);
        Assert.Equal(baseline.Count, combined.Count);
        Assert.All(combined, configuration =>
        {
            GemmConfig candidate = configuration.ToGemmConfig();
            Assert.False(candidate.UseSubgroupOps);
            Assert.True((long)candidate.ThreadTileM * candidate.ThreadTileN <= capabilities.MaxWorkGroupSize);
            Assert.Null(DynamicGemmKernel.ValidateConfig(candidate));
        });
    }

    [Fact]
    public async Task StaticResourceGate_RejectsCandidateBeforeGpuEvaluator()
    {
        var orchestrator = new GemmAutoTuner();
        GpuCapabilities capabilities = Capabilities(supportsSubgroups: false);
        OpenClGemmConfiguration valid = orchestrator.GetEvolutionSeeds(
            512, 512, 512, capabilities)[0];
        OpenClGemmConfiguration invalid = valid with
        {
            ThreadTileM = 32,
            ThreadTileN = 32,
            UseSubgroupOps = true
        };
        int gpuEvaluations = 0;
        EvolutionKernelAutotuner<OpenClGemmConfiguration> tuner = orchestrator.CreateEvolutionTuner(
            512, 512, 512,
            capabilities,
            Fingerprint(),
            (configuration, context, cancellationToken) =>
            {
                gpuEvaluations++;
                return new ValueTask<KernelTuningTrialResult>(Passed(configuration, 100));
            },
            new KernelSearchSpaceVersion(1),
            new KernelBenchmarkProtocolVersion(1),
            EngineOptions(maximumEvaluations: 2),
            deploymentRegistry: new KernelTuningDeploymentRegistry<OpenClGemmConfiguration>(),
            store: new MemoryStore());

        EvolutionKernelTuningResult<OpenClGemmConfiguration> result =
            await tuner.TuneAsync(new[] { invalid, valid });

        Assert.Equal(1, gpuEvaluations);
        Assert.Equal(1, result.Run.Counters.StatusCounts[EvolutionEvaluationStatus.Rejected]);
        Assert.Equal(valid, result.ActiveDeployment.Configuration);
    }

    [Fact]
    public async Task StaticResourceGate_RejectsOverflowingGeometryBeforeGpuEvaluator()
    {
        var orchestrator = new GemmAutoTuner();
        GpuCapabilities capabilities = Capabilities(supportsSubgroups: true);
        OpenClGemmConfiguration valid = orchestrator.GetEvolutionSeeds(
            512, 512, 512, capabilities)[0];
        OpenClGemmConfiguration overflowing = valid with
        {
            TileM = int.MaxValue,
            TileN = int.MaxValue,
            TileK = int.MaxValue,
            ThreadTileM = 1,
            ThreadTileN = 1,
            VectorWidthM = 1,
            VectorWidthN = 1,
            UseDoubleBuffering = true
        };
        int gpuEvaluations = 0;
        EvolutionKernelAutotuner<OpenClGemmConfiguration> tuner = orchestrator.CreateEvolutionTuner(
            512, 512, 512,
            capabilities,
            Fingerprint(),
            (configuration, context, cancellationToken) =>
            {
                gpuEvaluations++;
                return new ValueTask<KernelTuningTrialResult>(Passed(configuration, 100));
            },
            new KernelSearchSpaceVersion(1),
            new KernelBenchmarkProtocolVersion(1),
            EngineOptions(maximumEvaluations: 2),
            deploymentRegistry: new KernelTuningDeploymentRegistry<OpenClGemmConfiguration>(),
            store: new MemoryStore());

        EvolutionKernelTuningResult<OpenClGemmConfiguration> result =
            await tuner.TuneAsync(new[] { overflowing, valid });

        Assert.Equal(1, gpuEvaluations);
        Assert.Equal(1, result.Run.Counters.StatusCounts[EvolutionEvaluationStatus.Rejected]);
        Assert.Equal(valid, result.ActiveDeployment.Configuration);
    }

    [Fact]
    public async Task FixedBudgetRun_UsesSameCorrectnessAndTimingProtocolForAllSeeds()
    {
        var orchestrator = new GemmAutoTuner();
        GpuCapabilities capabilities = Capabilities(supportsSubgroups: true);
        OpenClGemmConfiguration[] seeds = orchestrator.GetEvolutionSeeds(
                1024, 1024, 1024, capabilities)
            .Take(4)
            .ToArray();
        int evaluations = 0;

        EvolutionKernelTuningResult<OpenClGemmConfiguration> result = await
            orchestrator.TuneWithEvolutionAsync(
                1024, 1024, 1024,
                capabilities,
                Fingerprint(),
                (configuration, context, cancellationToken) =>
                {
                    evaluations++;
                    double score = configuration.TileM * configuration.TileN;
                    return new ValueTask<KernelTuningTrialResult>(Passed(configuration, score));
                },
                new KernelSearchSpaceVersion(1),
                new KernelBenchmarkProtocolVersion(1),
                additionalSeeds: seeds.Select(seed => seed.ToGemmConfig()),
                engineOptions: EngineOptions(maximumEvaluations: seeds.Length),
                deploymentRegistry: new KernelTuningDeploymentRegistry<OpenClGemmConfiguration>(),
                store: new MemoryStore());

        Assert.Equal(seeds.Length, evaluations);
        Assert.Equal(seeds.Length, result.Run.Counters.EvaluationAttempts);
        Assert.Equal(seeds.Max(seed => seed.TileM * seed.TileN),
            result.ActiveDeployment.Measurement.ThroughputGflops);
        Assert.Equal(KernelTuningValidationScope.Output,
            result.ActiveDeployment.Measurement.Correctness.Scope);
        Assert.True(result.ActiveDeployment.Measurement.Timing.SampleCount >=
                    KernelTimingStatistics.MinimumSampleCount);
        GemmConfig dispatchChoice = orchestrator.SelectConfig(1024, 1024, 1024, capabilities);
        Assert.Equal(
            result.ActiveDeployment.Configuration.ToGemmConfig().ToKey(),
            dispatchChoice.ToKey());
    }

    [Fact]
    public void DiagnosticName_DoesNotSelectClBlastCodeGenerationTemplate()
    {
        var orchestrator = new GemmAutoTuner();
        OpenClGemmConfiguration typed = orchestrator.GetEvolutionSeeds(
                1024, 1024, 1024, Capabilities(supportsSubgroups: true))
            .First(configuration => configuration.KernelTemplate == GemmKernelTemplate.ClBlastBaselineK0);
        GemmConfig arbitraryName = Copy(typed.ToGemmConfig(), kernelName: "arbitrary-diagnostic-name");
        GemmConfig misleadingName = Copy(typed.ToGemmConfig(), kernelName: "clblast_baseline_k1_misleading");

        Assert.Equal(GemmKernelTemplate.ClBlastBaselineK0, arbitraryName.KernelTemplate);
        Assert.Equal(GemmKernelTemplate.ClBlastBaselineK0, misleadingName.KernelTemplate);
        Assert.Equal(
            DynamicGemmKernel.ValidateConfig(arbitraryName),
            DynamicGemmKernel.ValidateConfig(misleadingName));
        Assert.Equal(arbitraryName.ToKey(), misleadingName.ToKey());
    }

    [Theory]
    [InlineData("AMD", "gfx1012:xnack-", "AMD Radeon RX 5500 XT", GemmKernelTemplate.ClBlastBaselineK0)]
    [InlineData("Apple", "default", "Apple M1", GemmKernelTemplate.ClBlastBaselineK1)]
    public void ClBlastDatabase_PreservesTypedKernelTemplate(
        string vendor,
        string architecture,
        string deviceName,
        GemmKernelTemplate expectedTemplate)
    {
        var device = new ClBlastDeviceInfo("GPU", vendor, architecture, deviceName);

        bool found = ClBlastXgemmDatabase.TryGetConfig(device, out GemmConfig config);

        Assert.True(found);
        Assert.Equal(expectedTemplate, config.KernelTemplate);
        Assert.Null(DynamicGemmKernel.ValidateConfig(config));
    }

    private static EvolutionEngineOptions EngineOptions(int maximumEvaluations) => new()
    {
        RunId = "opencl-gemm-evolution-test",
        Seed = 91,
        MaxEvaluationAttempts = maximumEvaluations,
        MaxProposals = maximumEvaluations,
        MaxGenerations = 0,
        ProposalBatchSize = maximumEvaluations,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1
    };

    private static GpuCapabilities Capabilities(bool supportsSubgroups) => new()
    {
        ComputeUnits = 40,
        GlobalMemoryBytes = 8L * 1024 * 1024 * 1024,
        LocalMemoryBytes = 64 * 1024,
        MaxWorkGroupSize = 256,
        WavefrontSize = 64,
        SupportsSubgroups = supportsSubgroups,
        VendorName = "AMD",
        DeviceName = "Test GPU"
    };

    private static GpuDeviceFingerprint Fingerprint() => new(
        GpuVendorKind.Amd, "Test GPU", 10, 1, 1, "test-opencl-0");

    private static KernelTuningTrialResult Passed(GemmConfig configuration, double throughput)
    {
        var timing = KernelTimingStatistics.FromSamples(new[]
        {
            TimeSpan.FromMilliseconds(1.1),
            TimeSpan.FromMilliseconds(1.0),
            TimeSpan.FromMilliseconds(0.9),
            TimeSpan.FromMilliseconds(1.05),
            TimeSpan.FromMilliseconds(0.95)
        });
        long workspace = (long)configuration.TileK *
                         (configuration.TileM + configuration.TileN) * sizeof(float);
        double occupancy = Math.Min(1d, 256d /
            Math.Max(1d, configuration.ThreadTileM * configuration.ThreadTileN));
        int registers = Math.Max(1,
            configuration.TileM / configuration.ThreadTileM *
            configuration.TileN / configuration.ThreadTileN);
        return KernelTuningTrialResult.Passed(new KernelTuningMeasurement(
            throughput,
            timing,
            new KernelTuningResourceUsage(
                workspace, occupancy, registers, TimeSpan.FromMilliseconds(5)),
            new KernelTuningCorrectnessEvidence(
                KernelTuningValidationScope.Output,
                1e-7, 2e-7, 1e-5, 1e-5)));
    }

    private static GemmConfig Copy(
        GemmConfig source,
        int? threadTileM = null,
        int? threadTileN = null,
        bool? useSubgroupOps = null,
        string? kernelName = null) => new()
    {
        KernelTemplate = source.KernelTemplate,
        KernelName = kernelName ?? source.KernelName,
        TileM = source.TileM,
        TileN = source.TileN,
        TileK = source.TileK,
        ThreadTileM = threadTileM ?? source.ThreadTileM,
        ThreadTileN = threadTileN ?? source.ThreadTileN,
        VectorWidthM = source.VectorWidthM,
        VectorWidthN = source.VectorWidthN,
        UseDoubleBuffering = source.UseDoubleBuffering,
        UseVectorizedLoads = source.UseVectorizedLoads,
        KReg = source.KReg,
        KUnroll = source.KUnroll,
        UseSubgroupOps = useSubgroupOps ?? source.UseSubgroupOps,
        StrideM = source.StrideM,
        StrideN = source.StrideN,
        CacheA = source.CacheA,
        CacheB = source.CacheB,
        MdimaSize = source.MdimaSize,
        NdimbSize = source.NdimbSize,
        UseTrueVectorLDS = source.UseTrueVectorLDS,
        UseColumnMajorA = source.UseColumnMajorA
    };

    private sealed class MemoryStore : IKernelTuningStore<OpenClGemmConfiguration>
    {
        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<OpenClGemmConfiguration> codec,
            out KernelTuningDeploymentSnapshot<OpenClGemmConfiguration>? snapshot)
        {
            snapshot = null;
            return false;
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<OpenClGemmConfiguration> snapshot,
            IEvolutionGenomeCodec<OpenClGemmConfiguration> codec) => true;
    }
}
