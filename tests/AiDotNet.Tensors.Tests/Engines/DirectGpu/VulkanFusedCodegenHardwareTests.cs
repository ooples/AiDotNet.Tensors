using System.Globalization;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class VulkanFusedCodegenHardwareTests
{
    private const string ProofEnvironmentVariable = "AIDOTNET_RUN_VULKAN_CODEGEN_PROOF";
    private readonly ITestOutputHelper _output;

    public VulkanFusedCodegenHardwareTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "HardwareProof")]
    public async Task AmdVulkan_EvolvesFusedPipelineFromMeasuredThreeDispatchBaselineAndReplaysIt()
    {
        if (!string.Equals(
                Environment.GetEnvironmentVariable(ProofEnvironmentVariable),
                "1",
                StringComparison.Ordinal))
        {
            _output.WriteLine($"Set {ProofEnvironmentVariable}=1 to run the AMD Vulkan SPIR-V proof.");
            return;
        }

        const int elementCount = 1 << 20;
        VulkanBackend backend = VulkanBackend.Instance;
        Assert.True(backend.Initialize(), "The Vulkan backend did not initialize on the selected AMD GPU.");
        Assert.True(backend.IsGlslCompilerAvailable, "The Vulkan GLSL-to-SPIR-V compiler is unavailable.");
        Assert.Contains("AMD", backend.DeviceVendor, StringComparison.OrdinalIgnoreCase);

        GpuSourceKernel fused = EmitChain(
            backend, CodegenOpKind.Negate, CodegenOpKind.Exp, CodegenOpKind.Sqrt);
        GpuSourceKernel negate = EmitChain(backend, CodegenOpKind.Negate);
        GpuSourceKernel exp = EmitChain(backend, CodegenOpKind.Exp);
        GpuSourceKernel sqrt = EmitChain(backend, CodegenOpKind.Sqrt);
        float[] input = Enumerable.Range(0, elementCount)
            .Select(index => (index % 257 - 128) / 128f)
            .ToArray();

        using IGpuBuffer deviceInput = backend.AllocateBuffer(input);
        using IGpuBuffer fusedOutput = backend.AllocateBuffer(elementCount);
        using IGpuBuffer intermediateA = backend.AllocateBuffer(elementCount);
        using IGpuBuffer intermediateB = backend.AllocateBuffer(elementCount);
        using IGpuBuffer composedOutput = backend.AllocateBuffer(elementCount);

        CodegenGraph productionGraph = CodegenLowering.LowerUnaryChain<float>(
            new[] { CodegenOpKind.Negate, CodegenOpKind.Exp, CodegenOpKind.Sqrt },
            new[] { elementCount });
        CodegenEmitResult production = await backend.EmitAndExecuteCodegenKernelAsync(
            productionGraph,
            CodegenElementType.Float32,
            new[] { deviceInput },
            new[] { fusedOutput },
            elementCount);
        Assert.False(production.Declined, production.DeclineReason);

        VulkanFusedKernelExecutionEvidence first = Profile(
            backend, fused, deviceInput, fusedOutput, elementCount);
        Assert.Equal(KernelTuningBackend.Vulkan, first.Backend);
        Assert.Equal(CodegenNativeCompilationKind.ShadercToVulkanPipeline, first.Compilation);
        Assert.Equal(CodegenNativeArtifactKind.VulkanComputePipeline, first.NativeArtifact);
        Assert.Contains("AMD", first.DeviceVendor, StringComparison.OrdinalIgnoreCase);
        Assert.True(first.SpirvSizeBytes > 0);
        Assert.Equal(1, first.NativeLaunchCount);
        Assert.InRange(first.LocalWorkgroupSize, 1, checked((int)backend.MaxWorkgroupSize));
        Assert.True(first.SynchronizedLaunchDuration > TimeSpan.Zero);

        var experimentBackend = new VulkanExecutionPlanExperimentBackend(
            backend,
            fused,
            negate,
            exp,
            sqrt,
            deviceInput,
            fusedOutput,
            intermediateA,
            intermediateB,
            composedOutput,
            input);
        var experiment = new KernelTuningExperiment<VulkanExecutionPlan>(
            experimentBackend,
            new StopwatchKernelTuningTimer(),
            VulkanExecutionPlan.SeparateDispatches,
            new KernelTuningWorkload(elementCount, KernelTuningWorkUnit.Elements),
            warmupCount: 2,
            searchSampleCount: 7,
            holdoutSampleCount: 15);
        var store = new VulkanExecutionPlanMemoryStore();
        string fingerprint = EvolutionHash.Combine(new[]
        {
            backend.DeviceVendor,
            backend.DeviceName,
            KernelTuningBackend.Vulkan.ToString()
        });
        var identity = new KernelTuningIdentity(
            new KernelId("codegen", "unary-chain-negate-exp-sqrt"),
            new ShapeProfile(elementCount),
            new KernelTuningDeviceFingerprint(KernelTuningDeviceKind.AmdGpu, fingerprint, fingerprint),
            KernelTuningBackend.Vulkan,
            new KernelSearchSpaceVersion(1),
            new KernelBenchmarkProtocolVersion(1));
        var tuner = new EvolutionKernelAutotuner<VulkanExecutionPlan>(
            identity,
            new VulkanExecutionPlanCodec(),
            new VulkanExecutionPlanVariation(),
            experiment.EvaluateAsync,
            experiment,
            new EvolutionEngineOptions
            {
                RunId = "amd-vulkan-fusion-before-after-proof",
                Seed = 2072,
                MaxEvaluationAttempts = 2,
                MaxProposals = 2,
                MaxGenerations = 1,
                ProposalBatchSize = 1,
                MaxDegreeOfParallelism = 1,
                IslandCount = 1,
                MigrationInterval = 0,
                MigrantsPerIsland = 1
            },
            deploymentRegistry: new KernelTuningDeploymentRegistry<VulkanExecutionPlan>(),
            store: store);

        EvolutionKernelTuningResult<VulkanExecutionPlan> tuning = await tuner.TuneAsync(
            new[] { VulkanExecutionPlan.SeparateDispatches });

        Assert.True(tuning.WasPromoted, "The measured fused candidate did not clear the promotion gate.");
        Assert.True(tuning.WasPersisted);
        Assert.Equal(2, tuning.Run.Counters.EvaluationAttempts);
        Assert.Equal(VulkanExecutionPlan.SeparateDispatches, tuning.IncumbentDeployment.Configuration);
        Assert.Equal(VulkanExecutionPlan.FusedDispatch, tuning.ProposedWinner.Configuration);
        Assert.Equal(VulkanExecutionPlan.FusedDispatch, tuning.ActiveDeployment.Configuration);
        Assert.Equal(KernelTuningEvidenceRole.Incumbent, tuning.IncumbentDeployment.EvidenceRole);
        Assert.Equal(KernelTuningEvidenceRole.Candidate, tuning.ProposedWinner.EvidenceRole);
        Assert.Same(
            tuning.IncumbentDeployment.PromotionEvidence,
            tuning.ProposedWinner.PromotionEvidence);
        Assert.True(tuning.IncumbentDeployment.Measurement.Timing.HasRawSamples);
        Assert.True(tuning.ProposedWinner.Measurement.Timing.HasRawSamples);
        Assert.Equal(3, tuning.IncumbentDeployment.Measurement.Resources.KernelLaunchCount);
        Assert.Equal(1, tuning.ProposedWinner.Measurement.Resources.KernelLaunchCount);
        Assert.InRange(
            tuning.IncumbentDeployment.Measurement.Correctness.OutputAbsoluteError,
            0d,
            2e-5d);
        Assert.InRange(
            tuning.ProposedWinner.Measurement.Correctness.OutputAbsoluteError,
            0d,
            2e-5d);
        TimeSpan beforeMedian = tuning.IncumbentDeployment.Measurement.Timing.Median;
        TimeSpan afterMedian = tuning.ProposedWinner.Measurement.Timing.Median;
        Assert.True(
            afterMedian < beforeMedian,
            $"Vulkan after median {afterMedian.TotalMilliseconds:F4} ms was not below " +
            $"the before median {beforeMedian.TotalMilliseconds:F4} ms.");

        var hydratedTuner = new EvolutionKernelAutotuner<VulkanExecutionPlan>(
            identity,
            new VulkanExecutionPlanCodec(),
            new VulkanExecutionPlanVariation(),
            experiment.EvaluateAsync,
            experiment,
            deploymentRegistry: new KernelTuningDeploymentRegistry<VulkanExecutionPlan>(),
            store: store);
        Assert.True(hydratedTuner.TryHydrate());
        Assert.True(hydratedTuner.Deployment.TryGet(out VulkanExecutionPlan hydrated));
        Assert.Equal(VulkanExecutionPlan.FusedDispatch, hydrated);
        KernelTuningCorrectnessEvidence replayCorrectness =
            await experimentBackend.ValidateAsync(hydrated);
        Assert.InRange(replayCorrectness.OutputAbsoluteError, 0d, 2e-5d);

        _output.WriteLine($"AMD Vulkan device: {first.DeviceName} ({first.DeviceVendor})");
        _output.WriteLine($"Generated SPIR-V size: {first.SpirvSizeBytes} bytes");
        _output.WriteLine(
            $"BEFORE: {tuning.IncumbentDeployment.Configuration}, " +
            $"median={beforeMedian.TotalMilliseconds:F4} ms, " +
            $"p95={tuning.IncumbentDeployment.Measurement.Timing.P95.TotalMilliseconds:F4} ms, " +
            $"launches={tuning.IncumbentDeployment.Measurement.Resources.KernelLaunchCount}, " +
            $"maxAbsError={tuning.IncumbentDeployment.Measurement.Correctness.OutputAbsoluteError:R}");
        _output.WriteLine(
            $"AFTER: {tuning.ProposedWinner.Configuration}, " +
            $"median={afterMedian.TotalMilliseconds:F4} ms, " +
            $"p95={tuning.ProposedWinner.Measurement.Timing.P95.TotalMilliseconds:F4} ms, " +
            $"launches={tuning.ProposedWinner.Measurement.Resources.KernelLaunchCount}, " +
            $"maxAbsError={tuning.ProposedWinner.Measurement.Correctness.OutputAbsoluteError:R}");
        _output.WriteLine(
            $"PROMOTION: median speedup={tuning.ProposedWinner.PromotionEvidence.MedianSpeedup:F3}x, " +
            $"lower bound={tuning.ProposedWinner.PromotionEvidence.LowerSpeedupBound:F3}x, " +
            $"calibrated noise={tuning.ProposedWinner.PromotionEvidence.CalibratedNoiseRatio:F3}x");
        for (int index = 0; index < tuning.ProposedWinner.PromotionEvidence.Samples.Count; index++)
        {
            KernelTuningPairedSample sample = tuning.ProposedWinner.PromotionEvidence.Samples[index];
            _output.WriteLine(
                $"PAIR {index + 1:D2}: before={sample.Incumbent.TotalMilliseconds:F4} ms, " +
                $"after={sample.Candidate.TotalMilliseconds:F4} ms, speedup={sample.Speedup:F3}x");
        }
        _output.WriteLine("The persisted typed winner was hydrated and replayed through the correctness oracle.");
    }

    private static VulkanFusedKernelExecutionEvidence Profile(
        VulkanBackend backend,
        GpuSourceKernel kernel,
        IGpuBuffer input,
        IGpuBuffer output,
        int elementCount) => backend.ExecuteFusedPointwiseProfiled(
            kernel, new[] { input }, new[] { output }, elementCount);

    private static GpuSourceKernel EmitChain(
        INativeGpuCodegenExecutor backend,
        params CodegenOpKind[] operations)
    {
        CodegenGraph graph = CodegenLowering.LowerUnaryChain<float>(operations, new[] { 1 << 20 });
        CodegenEmitResult emitted = backend.EmitCodegenKernel(graph, CodegenElementType.Float32);
        Assert.False(emitted.Declined, emitted.DeclineReason);
        Assert.Equal(backend.NativeCodegenBackend, emitted.Kernel?.RequiredRuntime.Backend);
        return Assert.IsType<GpuSourceKernel>(emitted.Kernel);
    }

    private enum VulkanExecutionPlan
    {
        SeparateDispatches = 0,
        FusedDispatch = 1
    }

    private sealed class VulkanExecutionPlanExperimentBackend :
        IKernelTuningExperimentBackend<VulkanExecutionPlan>
    {
        private const double AbsoluteTolerance = 2e-5d;
        private const double RelativeTolerance = 2e-5d;
        private readonly VulkanBackend _backend;
        private readonly GpuSourceKernel _fused;
        private readonly GpuSourceKernel _negate;
        private readonly GpuSourceKernel _exp;
        private readonly GpuSourceKernel _sqrt;
        private readonly IGpuBuffer _fusedOutput;
        private readonly IGpuBuffer _composedOutput;
        private readonly float[] _input;
        private readonly IReadOnlyList<IGpuBuffer> _fusedInputs;
        private readonly IReadOnlyList<IGpuBuffer> _fusedOutputs;
        private readonly IReadOnlyList<IGpuBuffer> _negateInputs;
        private readonly IReadOnlyList<IGpuBuffer> _negateOutputs;
        private readonly IReadOnlyList<IGpuBuffer> _expInputs;
        private readonly IReadOnlyList<IGpuBuffer> _expOutputs;
        private readonly IReadOnlyList<IGpuBuffer> _sqrtInputs;
        private readonly IReadOnlyList<IGpuBuffer> _sqrtOutputs;

        internal VulkanExecutionPlanExperimentBackend(
            VulkanBackend backend,
            GpuSourceKernel fused,
            GpuSourceKernel negate,
            GpuSourceKernel exp,
            GpuSourceKernel sqrt,
            IGpuBuffer inputBuffer,
            IGpuBuffer fusedOutput,
            IGpuBuffer intermediateA,
            IGpuBuffer intermediateB,
            IGpuBuffer composedOutput,
            float[] input)
        {
            _backend = backend;
            _fused = fused;
            _negate = negate;
            _exp = exp;
            _sqrt = sqrt;
            _fusedOutput = fusedOutput;
            _composedOutput = composedOutput;
            _input = input;
            _fusedInputs = new[] { inputBuffer };
            _fusedOutputs = new[] { fusedOutput };
            _negateInputs = new[] { inputBuffer };
            _negateOutputs = new[] { intermediateA };
            _expInputs = new[] { intermediateA };
            _expOutputs = new[] { intermediateB };
            _sqrtInputs = new[] { intermediateB };
            _sqrtOutputs = new[] { composedOutput };
        }

        public async ValueTask PrepareAsync(
            VulkanExecutionPlan configuration,
            CancellationToken cancellationToken = default)
        {
            await ExecuteAsync(configuration, cancellationToken).ConfigureAwait(false);
        }

        public async ValueTask ExecuteAsync(
            VulkanExecutionPlan configuration,
            CancellationToken cancellationToken = default)
        {
            switch (configuration)
            {
                case VulkanExecutionPlan.SeparateDispatches:
                    await _backend.ExecuteCodegenKernelAsync(
                        _negate, _negateInputs, _negateOutputs, _input.Length,
                        cancellationToken).ConfigureAwait(false);
                    await _backend.ExecuteCodegenKernelAsync(
                        _exp, _expInputs, _expOutputs, _input.Length,
                        cancellationToken).ConfigureAwait(false);
                    await _backend.ExecuteCodegenKernelAsync(
                        _sqrt, _sqrtInputs, _sqrtOutputs, _input.Length,
                        cancellationToken).ConfigureAwait(false);
                    break;
                case VulkanExecutionPlan.FusedDispatch:
                    await _backend.ExecuteCodegenKernelAsync(
                        _fused, _fusedInputs, _fusedOutputs, _input.Length,
                        cancellationToken).ConfigureAwait(false);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(configuration));
            }
        }

        public ValueTask SynchronizeAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _backend.Synchronize();
            return default;
        }

        public async ValueTask<KernelTuningCorrectnessEvidence> ValidateAsync(
            VulkanExecutionPlan configuration,
            CancellationToken cancellationToken = default)
        {
            await ExecuteAsync(configuration, cancellationToken).ConfigureAwait(false);
            await SynchronizeAsync(cancellationToken).ConfigureAwait(false);
            IGpuBuffer output = configuration switch
            {
                VulkanExecutionPlan.SeparateDispatches => _composedOutput,
                VulkanExecutionPlan.FusedDispatch => _fusedOutput,
                _ => throw new ArgumentOutOfRangeException(nameof(configuration))
            };
            float[] actual = _backend.DownloadBuffer(output);
            double maximumAbsoluteError = 0d;
            double maximumRelativeError = 0d;
            for (int index = 0; index < actual.Length; index++)
            {
                float expected = MathF.Sqrt(MathF.Exp(-_input[index]));
                double absoluteError = Math.Abs(actual[index] - expected);
                double relativeError = absoluteError / Math.Max(Math.Abs(expected), 1e-6d);
                maximumAbsoluteError = Math.Max(maximumAbsoluteError, absoluteError);
                maximumRelativeError = Math.Max(maximumRelativeError, relativeError);
            }
            if (maximumAbsoluteError > AbsoluteTolerance && maximumRelativeError > RelativeTolerance)
            {
                throw new KernelTuningValidationException(
                    KernelTuningTrialStatus.OutputMismatch,
                    $"Vulkan {configuration} exceeded the independent CPU oracle tolerances.");
            }
            return new KernelTuningCorrectnessEvidence(
                KernelTuningValidationScope.Output,
                maximumAbsoluteError,
                maximumRelativeError,
                AbsoluteTolerance,
                RelativeTolerance);
        }

        public KernelTuningResourceUsage GetResourceUsage(VulkanExecutionPlan configuration) => new(
            KernelTuningResourceMetric<long>.Measured(configuration switch
            {
                VulkanExecutionPlan.SeparateDispatches => checked(2L * _input.Length * sizeof(float)),
                VulkanExecutionPlan.FusedDispatch => 0L,
                _ => throw new ArgumentOutOfRangeException(nameof(configuration))
            }),
            KernelTuningResourceMetric<double>.Unavailable(),
            KernelTuningResourceMetric<int>.Unavailable(),
            KernelTuningResourceMetric<TimeSpan>.Unavailable(),
            KernelTuningResourceMetric<int>.Measured(configuration switch
            {
                VulkanExecutionPlan.SeparateDispatches => 3,
                VulkanExecutionPlan.FusedDispatch => 1,
                _ => throw new ArgumentOutOfRangeException(nameof(configuration))
            }));
    }

    private sealed class VulkanExecutionPlanCodec : IEvolutionGenomeCodec<VulkanExecutionPlan>
    {
        public string Id => "vulkan-execution-plan";
        public string VersionHash => "v1";

        public string Serialize(VulkanExecutionPlan genome) =>
            ((int)genome).ToString(CultureInfo.InvariantCulture);

        public VulkanExecutionPlan Deserialize(string payload)
        {
            if (!int.TryParse(payload, NumberStyles.None, CultureInfo.InvariantCulture, out int value) ||
                !Enum.IsDefined(typeof(VulkanExecutionPlan), value))
            {
                throw new InvalidDataException("Invalid Vulkan execution plan.");
            }
            return (VulkanExecutionPlan)value;
        }
    }

    private sealed class VulkanExecutionPlanVariation : IVariationOperator<VulkanExecutionPlan>
    {
        public string Id => "vulkan-execution-plan-toggle";
        public string VersionHash => "v1";

        public ValueTask<VulkanExecutionPlan> ProposeAsync(
            EvolutionVariationContext<VulkanExecutionPlan> context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            VulkanExecutionPlan parent = context.Parent.Candidate.CanonicalGenome.Genome;
            return new ValueTask<VulkanExecutionPlan>(parent switch
            {
                VulkanExecutionPlan.SeparateDispatches => VulkanExecutionPlan.FusedDispatch,
                VulkanExecutionPlan.FusedDispatch => VulkanExecutionPlan.SeparateDispatches,
                _ => throw new ArgumentOutOfRangeException(nameof(context))
            });
        }
    }

    private sealed class VulkanExecutionPlanMemoryStore : IKernelTuningStore<VulkanExecutionPlan>
    {
        private KernelTuningDeploymentSnapshot<VulkanExecutionPlan>? _snapshot;

        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<VulkanExecutionPlan> codec,
            out KernelTuningDeploymentSnapshot<VulkanExecutionPlan>? snapshot)
        {
            snapshot = _snapshot;
            return snapshot is not null;
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<VulkanExecutionPlan> snapshot,
            IEvolutionGenomeCodec<VulkanExecutionPlan> codec)
        {
            _snapshot = snapshot;
            return true;
        }
    }
}
