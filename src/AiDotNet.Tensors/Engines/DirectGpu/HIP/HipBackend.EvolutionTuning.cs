using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// Evolution tuning of the CLBlast GEMM on HIP (#1266). The search space, variation and validation are OpenCL's
// (GemmAutoTuner over CLBlast GEMMK 0 configurations); the identity, measurement and deployment are HIP's. hiprtc
// rewards different parameters than the OpenCL driver: on an RX 5500 XT the database configuration with only its
// STRM / STRN flags swapped runs 19% faster at 1024^3.
public sealed partial class HipBackend
{
    private static readonly KernelId HipClBlastGemmKernelId = new("gemm", "hip-clblast-evolution");
    private static readonly KernelSearchSpaceVersion HipClBlastSearchSpace = new(1);
    private static readonly KernelBenchmarkProtocolVersion HipClBlastBenchmarkProtocol = new(1);

    /// <summary>
    /// Runs correctness-gated evolutionary tuning of the CLBlast GEMM for one shape on this device, seeded by the
    /// database configuration, and deploys the winner so <see cref="Gemm"/> runs that shape with it.
    /// </summary>
    public async Task<EvolutionKernelTuningResult<OpenClGemmConfiguration>> TuneGemmWithEvolutionAsync(
        int m,
        int n,
        int k,
        IEnumerable<GemmConfig>? additionalSeeds = null,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<OpenClGemmConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<OpenClGemmConfiguration>? store = null,
        CancellationToken cancellationToken = default)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        GemmConfig incumbent = ClBlastDefaultConfigOrThrow();

        GpuCapabilities capabilities = CreateHipEvolutionCapabilities();
        GemmAutoTuner orchestrator = CreateHipGemmOrchestrator();
        var seeds = new List<GemmConfig> { incumbent };
        if (additionalSeeds is not null) seeds.AddRange(additionalSeeds);

        using var experimentBackend = new HipClBlastGemmExperimentBackend(this, m, n, k);
        var experiment = new KernelTuningExperiment<OpenClGemmConfiguration>(
            experimentBackend,
            // HIP events on the stream the packing and GEMM launch on: device time only. A host stopwatch around a 1 ms
            // dispatch measured 25% calibrated noise on a busy machine, more than the speedups the search finds.
            new DeviceEventKernelTuningTimer(new GpuKernelTuningDeviceClock(this)),
            OpenClGemmConfiguration.FromGemmConfig(incumbent),
            new KernelTuningWorkload(checked(2d * m * n * k * experimentBackend.Repetitions), KernelTuningWorkUnit.FloatingPointOperations),
            KernelTuningTimingScope.SteadyStateExecution);

        EvolutionKernelTuningResult<OpenClGemmConfiguration> result = await orchestrator.TuneWithEvolutionAsync(
            m,
            n,
            k,
            capabilities,
            CreateHipTuningFingerprint(),
            experiment.EvaluateAsync,
            experiment,
            HipClBlastSearchSpace,
            HipClBlastBenchmarkProtocol,
            seeds,
            GemmKernelTemplate.ClBlastBaselineK0,
            engineOptions,
            tuningOptions,
            checkpointStore,
            deploymentRegistry,
            store,
            cancellationToken).ConfigureAwait(false);

        if (!DeployClBlastGemm(m, n, k, result.ActiveDeployment.Configuration.ToGemmConfig()))
            throw new InvalidOperationException("The validated HIP GEMM deployment could not be built for dispatch.");
        return result;
    }

    /// <summary>
    /// Loads a persisted, identity-matched deployment for one shape without benchmarking and deploys it. Identity covers
    /// the HIP backend, the device, its gcnArchName and runtime, the shape, the search space and the measurement protocol.
    /// </summary>
    public bool TryActivatePersistedGemmEvolution(
        int m,
        int n,
        int k,
        IKernelTuningStore<OpenClGemmConfiguration> store,
        KernelTuningDeploymentRegistry<OpenClGemmConfiguration>? deploymentRegistry = null)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (store is null) throw new ArgumentNullException(nameof(store));
        if (!EnsureClBlastIndirectGemm()) return false;

        EvolutionKernelAutotuner<OpenClGemmConfiguration> tuner = CreateHipGemmOrchestrator().CreateEvolutionTuner(
            m,
            n,
            k,
            CreateHipEvolutionCapabilities(),
            CreateHipTuningFingerprint(),
            UnexpectedHipHydrationBenchmark,
            HipHydrationOnlyFinalistEvaluator.Instance,
            HipClBlastSearchSpace,
            HipClBlastBenchmarkProtocol,
            GemmKernelTemplate.ClBlastBaselineK0,
            deploymentRegistry: deploymentRegistry,
            store: store);
        return tuner.TryHydrate() &&
            tuner.Deployment.TryGet(out OpenClGemmConfiguration configuration) &&
            DeployClBlastGemm(m, n, k, configuration.ToGemmConfig());
    }

    private GemmConfig ClBlastDefaultConfigOrThrow()
    {
        if (!IsAvailable || !EnsureClBlastIndirectGemm() || ClBlastDefaultConfig is not GemmConfig config)
            throw new GpuCapabilityUnavailableException("The HIP CLBlast GEMM is not available for evolutionary tuning.");
        return config;
    }

    private static GemmAutoTuner CreateHipGemmOrchestrator() => new()
    {
        EvolutionKernel = HipClBlastGemmKernelId,
        TuningBackend = KernelTuningBackend.Hip,
        VariationFactory = (capabilities, _) => new HipClBlastGemmVariation(capabilities)
    };

    private GpuCapabilities CreateHipEvolutionCapabilities() => new()
    {
        ComputeUnits = _deviceProps.MultiProcessorCount,
        GlobalMemoryBytes = (long)(ulong)_deviceProps.TotalGlobalMem,
        LocalMemoryBytes = checked((int)(ulong)_deviceProps.SharedMemPerBlock),
        MaxWorkGroupSize = _deviceProps.MaxThreadsPerBlock,
        WavefrontSize = _deviceProps.WarpSize,
        VendorName = "AMD",
        DeviceName = DeviceName
    };

    private KernelTuningDeviceFingerprint CreateHipTuningFingerprint()
    {
        string modelKey = EvolutionHash.Combine(new[]
        {
            "hip-model-v1",
            _deviceProps.GcnArchName ?? _architectureTarget,
            DeviceName,
            _deviceProps.TotalGlobalMem.ToString(),
            HipRuntimeVersionText()
        });
        string localKey = EvolutionHash.Combine(new[]
        {
            "hip-device-v1",
            modelKey,
            _deviceProps.PciBusID.ToString(CultureInfo.InvariantCulture),
            _deviceProps.PciDeviceID.ToString(CultureInfo.InvariantCulture)
        });
        return new KernelTuningDeviceFingerprint(KernelTuningDeviceKind.AmdGpu, localKey, modelKey);
    }

    private static string HipRuntimeVersionText() =>
        HipNativeBindings.hipRuntimeGetVersion(out int version) == HipError.Success
            ? version.ToString(CultureInfo.InvariantCulture)
            : "unknown";

    private static ValueTask<KernelTuningTrialResult> UnexpectedHipHydrationBenchmark(
        OpenClGemmConfiguration configuration,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken) =>
        throw new InvalidOperationException("Persisted deployment hydration must not benchmark a kernel.");

    /// <summary>Runs candidates on the production HIP path (same packing, same launch) against a double oracle.</summary>
    private sealed class HipClBlastGemmExperimentBackend :
        IKernelTuningExperimentBackend<OpenClGemmConfiguration>, IDisposable
    {
        private const double AbsoluteTolerance = 2e-3;
        private const double RelativeTolerance = 2e-3;
        private readonly HipBackend _owner;
        private readonly int _m;
        private readonly int _n;
        private readonly int _k;
        private readonly float[] _expected;
        private readonly IGpuBuffer _deviceA;
        private readonly IGpuBuffer _deviceB;
        private readonly IGpuBuffer _deviceC;
        private readonly HashSet<OpenClGemmConfiguration> _prepared = new();
        private bool _disposed;

        // Each timed execution repeats the GEMM to about this much work. One 1024^3 GEMM is ~1 ms, and the GPU drops its
        // clocks between samples while the host validates and compiles: single-GEMM samples calibrated 25-29% noise
        // (the worst of nine incumbent-vs-incumbent pairs), more than the 15-20% the search finds.
        private const double TargetFlopsPerExecution = 2e10;
        private const int MaximumRepetitions = 64;

        internal HipClBlastGemmExperimentBackend(HipBackend owner, int m, int n, int k)
        {
            _owner = owner;
            _m = m;
            _n = n;
            _k = k;
            Repetitions = (int)Math.Min(MaximumRepetitions, Math.Max(1, Math.Ceiling(TargetFlopsPerExecution / (2d * m * n * k))));
            float[] inputA = CreateValues(checked(m * k), 17);
            float[] inputB = CreateValues(checked(k * n), 43);
            _expected = ComputeReference(inputA, inputB, m, n, k);
            IGpuBuffer? deviceA = null, deviceB = null;
            try
            {
                deviceA = owner.AllocateBuffer(inputA);
                deviceB = owner.AllocateBuffer(inputB);
                _deviceC = owner.AllocateBuffer(checked(m * n));
                _deviceA = deviceA;
                _deviceB = deviceB;
            }
            catch
            {
                deviceB?.Dispose();
                deviceA?.Dispose();
                throw;
            }
        }

        /// <summary>How many GEMMs one timed execution runs (the workload counts all of them).</summary>
        internal int Repetitions { get; }

        public ValueTask PrepareAsync(OpenClGemmConfiguration configuration, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();
            if (_prepared.Contains(configuration)) return default;
            GemmConfig config = configuration.ToGemmConfig();
            if (!HipClBlastGemmKernels.Supports(config))
            {
                throw new KernelTuningValidationException(
                    KernelTuningTrialStatus.InvalidConfiguration,
                    "HIP builds the CLBlast GEMM only for GEMMK 0 with vector widths 1, 2 or 4.");
            }
            if (!_owner.TryPrepareClBlastGemm(config))
            {
                throw new KernelTuningValidationException(
                    KernelTuningTrialStatus.CompilationFailed,
                    "The HIP CLBlast GEMM candidate did not compile.");
            }
            _prepared.Add(configuration);
            return default;
        }

        public ValueTask ExecuteAsync(OpenClGemmConfiguration configuration, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();
            if (!_prepared.Contains(configuration))
                throw new InvalidOperationException("The HIP GEMM candidate was not prepared before execution.");
            GemmConfig config = configuration.ToGemmConfig();
            for (int repetition = 0; repetition < Repetitions; repetition++)
            {
                if (!_owner.TryExecuteClBlastIndirectGemm(config, _deviceA, _deviceB, _deviceC, _m, _n, _k))
                {
                    throw new KernelTuningValidationException(
                        KernelTuningTrialStatus.BenchmarkFailed,
                        "The HIP CLBlast GEMM candidate failed during execution.");
                }
            }
            return default;
        }

        public ValueTask SynchronizeAsync(CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();
            _owner.Synchronize();
            return default;
        }

        public async ValueTask<KernelTuningCorrectnessEvidence> ValidateAsync(
            OpenClGemmConfiguration configuration, CancellationToken cancellationToken = default)
        {
            await ExecuteAsync(configuration, cancellationToken).ConfigureAwait(false);
            await SynchronizeAsync(cancellationToken).ConfigureAwait(false);
            float[] actual = _owner.DownloadBuffer(_deviceC);
            double maximumAbsoluteError = 0, maximumRelativeError = 0;
            for (int i = 0; i < _expected.Length; i++)
            {
                double absoluteError = Math.Abs(actual[i] - _expected[i]);
                maximumAbsoluteError = Math.Max(maximumAbsoluteError, absoluteError);
                maximumRelativeError = Math.Max(maximumRelativeError, absoluteError / Math.Max(Math.Abs(_expected[i]), 1e-6));
            }
            if (maximumAbsoluteError > AbsoluteTolerance && maximumRelativeError > RelativeTolerance)
            {
                throw new KernelTuningValidationException(
                    KernelTuningTrialStatus.OutputMismatch,
                    $"HIP GEMM output exceeded tolerance: abs={maximumAbsoluteError:R}, rel={maximumRelativeError:R}.");
            }
            return new KernelTuningCorrectnessEvidence(
                KernelTuningValidationScope.Output,
                maximumAbsoluteError,
                maximumRelativeError,
                AbsoluteTolerance,
                RelativeTolerance);
        }

        public KernelTuningResourceUsage GetResourceUsage(OpenClGemmConfiguration configuration)
        {
            ThrowIfDisposed();
            GemmConfig c = configuration.ToGemmConfig();
            int kUnit = c.TileK * Math.Max(1, c.KReg);
            long mCeiled = CeilTo(_n, c.TileM), nCeiled = CeilTo(_m, c.TileN), kCeiled = CeilTo(_k, kUnit);
            bool aPads = _n != mCeiled || _k != kCeiled, cPads = _n != mCeiled || _m != nCeiled;
            long workspace = sizeof(float) * (nCeiled * kCeiled + (aPads ? mCeiled * kCeiled : 0) + (cPads ? mCeiled * nCeiled : 0));
            int launches = 2 + (aPads ? 1 : 0) + (cPads ? 1 : 0);
            return new KernelTuningResourceUsage(
                KernelTuningResourceMetric<long>.Measured(workspace),
                KernelTuningResourceMetric<double>.Unavailable(),
                KernelTuningResourceMetric<int>.Unavailable(),
                KernelTuningResourceMetric<TimeSpan>.Unavailable(),
                KernelTuningResourceMetric<int>.Measured(launches));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _prepared.Clear();
            _deviceC.Dispose();
            _deviceB.Dispose();
            _deviceA.Dispose();
        }

        private static float[] CreateValues(int length, int salt)
        {
            var result = new float[length];
            for (int i = 0; i < result.Length; i++)
                result[i] = (((i * 37 + salt * 17) % 127) - 63) / 256f;
            return result;
        }

        private static float[] ComputeReference(float[] a, float[] b, int m, int n, int k)
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

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HipClBlastGemmExperimentBackend));
        }
    }

    private sealed class HipHydrationOnlyFinalistEvaluator : IKernelTuningFinalistEvaluator<OpenClGemmConfiguration>
    {
        internal static HipHydrationOnlyFinalistEvaluator Instance { get; } = new();

        public ValueTask<KernelTuningFinalistReplay<OpenClGemmConfiguration>> ReplayAsync(
            KernelTuningIdentity identity,
            OpenClGemmConfiguration candidate,
            KernelTuningDeploymentSnapshot<OpenClGemmConfiguration>? activeDeployment,
            CancellationToken cancellationToken = default) =>
            throw new InvalidOperationException("Persisted deployment hydration must not replay finalists.");
    }
}