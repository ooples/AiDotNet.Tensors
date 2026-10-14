#if NET6_0_OR_GREATER
using System;
using System.Threading.Tasks;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// End-to-end evolution of the HIP CLBlast GEMM (#1266): tune one shape on the device, check the deployment is what
/// Gemm runs, then hydrate it into a fresh backend from the store without benchmarking. Minutes of GPU time, so it runs
/// only when asked, like the OpenCL proof.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HipGemmEvolutionHardwareTests
{
    private const string ProofEnvironmentVariable = "AIDOTNET_RUN_HIP_TUNING_PROOF";
    private readonly ITestOutputHelper _output;

    public HipGemmEvolutionHardwareTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    [Trait("Category", "HardwareProof")]
    public async Task Hip_EvolvesValidatesDeploysAndHydratesClBlastGemm()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable(ProofEnvironmentVariable), "1", StringComparison.Ordinal),
            $"Set {ProofEnvironmentVariable}=1 to run the HIP tuning proof.");

        const int m = 1024, n = 1024, k = 1024;
        var store = new MemoryStore();
        var options = new EvolutionEngineOptions
        {
            RunId = "hip-clblast-gemm-proof",
            Seed = 1266,
            MaxEvaluationAttempts = 24,
            MaxProposals = 48,
            MaxGenerations = 6,
            ProposalBatchSize = 4,
            MaxDegreeOfParallelism = 1,
            IslandCount = 1,
            MigrationInterval = 0,
            MigrantsPerIsland = 1
        };

        EvolutionKernelTuningResult<OpenClGemmConfiguration> tuning;
        GemmConfig deployed;
        using (var backend = new HipBackend())
        {
            Skip.IfNot(backend.IsAvailable && !backend.IsHipBlasAvailable && backend.EnsureClBlastIndirectGemm(),
                "A HIP device with the CLBlast indirect GEMM and no hipBLAS is required.");
            tuning = await backend.TuneGemmWithEvolutionAsync(m, n, k, engineOptions: options, store: store);

            Assert.True(tuning.Run.Counters.EvaluationAttempts >= 3);
            Assert.True(tuning.WasPersisted);
            Assert.NotNull(store.Snapshot);
            Assert.Equal(KernelTuningValidationScope.Output, tuning.ActiveDeployment.Measurement.Correctness.Scope);
            Assert.True(tuning.ActiveDeployment.Measurement.Timing.SampleCount >= 3);
            Assert.True(tuning.ActiveDeployment.Measurement.ThroughputGflops > 0);
            Assert.Equal(KernelTuningBackend.Hip, store.Snapshot?.Identity.Backend);

            deployed = tuning.ActiveDeployment.Configuration.ToGemmConfig();
            Assert.Equal(deployed, backend.ClBlastConfigFor(m, n, k));
            ValidateProductionDispatch(backend, m, n, k);
        }

        using (var hydrated = new HipBackend())
        {
            Assert.True(hydrated.TryActivatePersistedGemmEvolution(m, n, k, store));
            Assert.Equal(deployed, hydrated.ClBlastConfigFor(m, n, k));
            ValidateProductionDispatch(hydrated, m, n, k);
        }

        _output.WriteLine($"Active HIP GEMM: {tuning.ActiveDeployment.Configuration}");
        _output.WriteLine($"Proposed HIP GEMM: {tuning.ProposedWinner.Configuration}");
        _output.WriteLine(
            $"Promotion: {tuning.WasPromoted}; paired median speedup={tuning.ProposedWinner.PromotionEvidence.MedianSpeedup:F3}x, " +
            $"lower bound={tuning.ProposedWinner.PromotionEvidence.LowerSpeedupBound:F3}x, " +
            $"calibrated noise={tuning.ProposedWinner.PromotionEvidence.CalibratedNoiseRatio:F3}x, " +
            $"p95 latency ratio={tuning.ProposedWinner.PromotionEvidence.P95LatencyRatio:F3}");
        _output.WriteLine($"Deployed throughput: {tuning.ActiveDeployment.Measurement.ThroughputGflops:F0} GFLOP/s");
    }

    private static void ValidateProductionDispatch(HipBackend backend, int m, int n, int k)
    {
        float[] a = CreateValues(checked(m * k), 17);
        float[] b = CreateValues(checked(k * n), 43);
        using IGpuBuffer deviceA = backend.AllocateBuffer(a);
        using IGpuBuffer deviceB = backend.AllocateBuffer(b);
        using IGpuBuffer deviceC = backend.AllocateBuffer(checked(m * n));
        long before = backend.ClBlastIndirectGemmRuns;
        backend.Gemm(deviceA, deviceB, deviceC, m, n, k);
        Assert.Equal(before + 1, backend.ClBlastIndirectGemmRuns);
        float[] actual = backend.DownloadBuffer(deviceC);
        for (int row = 0; row < m; row += 61)
        {
            for (int column = 0; column < n; column++)
            {
                double sum = 0;
                for (int inner = 0; inner < k; inner++) sum += (double)a[row * k + inner] * b[inner * n + column];
                double absolute = Math.Abs(actual[row * n + column] - sum);
                Assert.True(absolute <= 2e-3 || absolute / Math.Max(Math.Abs(sum), 1e-6) <= 2e-3,
                    $"GEMM mismatch at ({row},{column}): expected={sum:R}, actual={actual[row * n + column]:R}.");
            }
        }
    }

    private static float[] CreateValues(int length, int salt)
    {
        var result = new float[length];
        for (int i = 0; i < result.Length; i++)
            result[i] = (((i * 37 + salt * 17) % 127) - 63) / 256f;
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
#endif