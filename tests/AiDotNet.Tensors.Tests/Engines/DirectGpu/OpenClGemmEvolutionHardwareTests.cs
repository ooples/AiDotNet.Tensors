using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class OpenClGemmEvolutionHardwareTests
{
    private const string ProofEnvironmentVariable = "AIDOTNET_RUN_OPENCL_TUNING_PROOF";
    private readonly ITestOutputHelper _output;

    public OpenClGemmEvolutionHardwareTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "HardwareProof")]
    public async Task AmdDriver_EvolvesValidatesDeploysAndHydratesNativeGemm()
    {
        if (!string.Equals(
                Environment.GetEnvironmentVariable(ProofEnvironmentVariable),
                "1",
                StringComparison.Ordinal))
        {
            _output.WriteLine($"Set {ProofEnvironmentVariable}=1 to run the AMD OpenCL hardware proof.");
            return;
        }

        const int m = 256;
        const int n = 256;
        const int k = 256;
        var store = new MemoryStore();
        var options = new EvolutionEngineOptions
        {
            RunId = "amd-opencl-native-gemm-proof",
            Seed = 2072,
            MaxEvaluationAttempts = 12,
            MaxProposals = 32,
            MaxGenerations = 5,
            ProposalBatchSize = 4,
            MaxDegreeOfParallelism = 1,
            IslandCount = 1,
            MigrationInterval = 0,
            MigrantsPerIsland = 1
        };

        EvolutionKernelTuningResult<OpenClGemmConfiguration> tuning;
        using (var backend = new OpenClBackend(deviceIndex: 0))
        {
            Assert.True(backend.IsAvailable);
            Assert.Equal(KernelTuningDeviceKind.AmdGpu, backend.TuningDeviceKind);
            tuning = await backend.TuneGemmWithEvolutionAsync(
                m, n, k,
                engineOptions: options,
                store: store);

            Assert.True(tuning.Run.Counters.EvaluationAttempts >= 3);
            Assert.True(tuning.WasPersisted);
            Assert.NotNull(store.Snapshot);
            Assert.Equal(
                KernelTuningValidationScope.Output,
                tuning.ActiveDeployment.Measurement.Correctness.Scope);
            Assert.True(tuning.ActiveDeployment.Measurement.Timing.SampleCount >= 3);
            Assert.True(tuning.ActiveDeployment.Measurement.ThroughputGflops > 0);
            Assert.Equal(
                KernelTuningResourceMetricStatus.Unavailable,
                tuning.ActiveDeployment.Measurement.Resources.OccupancyRatioMetric.Status);
            Assert.Equal(
                KernelTuningResourceMetricStatus.Unavailable,
                tuning.ActiveDeployment.Measurement.Resources.RegistersPerThreadMetric.Status);
            Assert.Equal(
                KernelTuningResourceMetricStatus.Unavailable,
                tuning.ActiveDeployment.Measurement.Resources.CompileTimeMetric.Status);
            Assert.Equal(
                KernelTuningResourceMetricStatus.Measured,
                tuning.ActiveDeployment.Measurement.Resources.WorkspaceBytesMetric.Status);
            Assert.Equal(
                KernelTuningResourceMetricStatus.Measured,
                tuning.ActiveDeployment.Measurement.Resources.KernelLaunchCountMetric.Status);
            Assert.True(tuning.ActiveDeployment.Measurement.Resources.KernelLaunchCount > 0);

            ValidateProductionDispatch(backend, m, n, k);
            Assert.Equal(
                OpenClGemmDispatchPath.EvolutionaryNativeKernel,
                backend.LastGemmDispatchPath);
        }

        using (var hydratedBackend = new OpenClBackend(deviceIndex: 0))
        {
            Assert.True(hydratedBackend.TryActivatePersistedGemmEvolution(m, n, k, store));
            ValidateProductionDispatch(hydratedBackend, m, n, k);
            Assert.Equal(
                OpenClGemmDispatchPath.EvolutionaryNativeKernel,
                hydratedBackend.LastGemmDispatchPath);
            Assert.Equal(
                OpenClProgramBuildOrigin.NativeBinaryCache,
                hydratedBackend.LastDynamicGemmProgramOrigin);
            Assert.Equal(
                OpenClProgramBinaryType.Executable,
                hydratedBackend.LastDynamicGemmProgramBinaryType);
        }

        _output.WriteLine(
            $"Active AMD OpenCL GEMM: {tuning.ActiveDeployment.Configuration};");
        _output.WriteLine(
            $"Proposed AMD OpenCL GEMM: {tuning.ProposedWinner.Configuration};");
        _output.WriteLine(
            $"Promotion: {tuning.WasPromoted}; paired median speedup=" +
            $"{tuning.ProposedWinner.PromotionEvidence.MedianSpeedup:F3}x, " +
            $"lower bound={tuning.ProposedWinner.PromotionEvidence.LowerSpeedupBound:F3}x, " +
            $"calibrated noise={tuning.ProposedWinner.PromotionEvidence.CalibratedNoiseRatio:F3}x");
        _output.WriteLine(
            $"Measured production-pipeline median: " +
            $"{tuning.ActiveDeployment.Measurement.Timing.Median.TotalMilliseconds:F4} ms, " +
            $"{tuning.ActiveDeployment.Measurement.ThroughputGflops:F2} GFLOP/s");
        _output.WriteLine(
            $"Measured production resources: " +
            $"workspace={tuning.ActiveDeployment.Measurement.Resources.WorkspaceBytes} bytes, " +
            $"launches={tuning.ActiveDeployment.Measurement.Resources.KernelLaunchCount}");
        _output.WriteLine(
            $"Maximum validated errors: abs=" +
            $"{tuning.ActiveDeployment.Measurement.Correctness.OutputAbsoluteError:R}, rel=" +
            $"{tuning.ActiveDeployment.Measurement.Correctness.OutputRelativeError:R}");
        _output.WriteLine("Fresh backend hydrated the typed deployment and loaded its native driver binary.");
    }

    private static void ValidateProductionDispatch(OpenClBackend backend, int m, int n, int k)
    {
        float[] a = CreateValues(checked(m * k), 17);
        float[] b = CreateValues(checked(k * n), 43);
        float[] expected = ReferenceGemm(a, b, m, n, k);
        using IGpuBuffer deviceA = backend.AllocateBuffer(a);
        using IGpuBuffer deviceB = backend.AllocateBuffer(b);
        using IGpuBuffer deviceC = backend.AllocateBuffer(checked(m * n));
        backend.Gemm(deviceA, deviceB, deviceC, m, n, k);
        backend.Synchronize();
        float[] actual = backend.DownloadBuffer(deviceC);
        for (int i = 0; i < actual.Length; i++)
        {
            double absolute = Math.Abs(actual[i] - expected[i]);
            double relative = absolute / Math.Max(Math.Abs(expected[i]), 1e-6);
            Assert.True(
                absolute <= 2e-3 || relative <= 2e-3,
                $"GEMM mismatch at {i}: expected={expected[i]:R}, actual={actual[i]:R}.");
        }
    }

    private static float[] CreateValues(int length, int salt)
    {
        var result = new float[length];
        for (int i = 0; i < result.Length; i++)
            result[i] = (((i * 37 + salt * 17) % 127) - 63) / 256f;
        return result;
    }

    private static float[] ReferenceGemm(float[] a, float[] b, int m, int n, int k)
    {
        var result = new float[checked(m * n)];
        for (int row = 0; row < m; row++)
        {
            for (int column = 0; column < n; column++)
            {
                double sum = 0;
                for (int inner = 0; inner < k; inner++)
                    sum += (double)a[row * k + inner] * b[inner * n + column];
                result[row * n + column] = (float)sum;
            }
        }
        return result;
    }

    private sealed class MemoryStore : IKernelTuningStore<OpenClGemmConfiguration>
    {
        internal KernelTuningDeploymentSnapshot<OpenClGemmConfiguration>? Snapshot { get; private set; }

        public bool TryLoad(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<OpenClGemmConfiguration> codec,
            out KernelTuningDeploymentSnapshot<OpenClGemmConfiguration>? snapshot)
        {
            snapshot = Snapshot;
            return snapshot is not null;
        }

        public bool TryStore(
            KernelTuningDeploymentSnapshot<OpenClGemmConfiguration> snapshot,
            IEvolutionGenomeCodec<OpenClGemmConfiguration> codec)
        {
            Snapshot = snapshot;
            return true;
        }
    }
}
