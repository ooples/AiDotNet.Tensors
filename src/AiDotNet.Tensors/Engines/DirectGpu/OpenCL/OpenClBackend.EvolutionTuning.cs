using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>Typed production GEMM route selected by the most recent dispatch.</summary>
public enum OpenClGemmDispatchPath
{
    /// <summary>No GEMM has been dispatched, or the current dispatch has not selected a route yet.</summary>
    None = 0,
    /// <summary>A correctness-gated evolutionary deployment compiled by the OpenCL driver.</summary>
    EvolutionaryNativeKernel = 1,
    /// <summary>The native CLBlast library.</summary>
    ClBlastLibrary = 2,
    /// <summary>The generated CLBlast-compatible baseline kernel.</summary>
    ClBlastGeneratedKernel = 3,
    /// <summary>The legacy tuned dynamic kernel.</summary>
    LegacyTunedNativeKernel = 4,
    /// <summary>A built-in fallback kernel.</summary>
    BuiltInKernel = 5
}

public sealed partial class OpenClBackend
{
    private readonly Dictionary<(int M, int N, int K), GemmConfig> _evolutionaryGemmDeployments = new();
    private readonly object _evolutionaryGemmDeploymentLock = new();
    private int _lastGemmDispatchPath;

    /// <summary>Gets the typed route used by the most recent production GEMM call.</summary>
    public OpenClGemmDispatchPath LastGemmDispatchPath =>
        (OpenClGemmDispatchPath)Volatile.Read(ref _lastGemmDispatchPath);

    /// <summary>Gets the native artifact origin used by the last dynamic GEMM selection.</summary>
    public OpenClProgramBuildOrigin? LastDynamicGemmProgramOrigin =>
        _dynamicGemm?.LastProgramBuildOrigin;

    /// <summary>Gets the driver-reported binary type used by the last dynamic GEMM selection.</summary>
    public OpenClProgramBinaryType? LastDynamicGemmProgramBinaryType =>
        _dynamicGemm?.LastProgramBinaryType;

    private void RecordGemmDispatch(OpenClGemmDispatchPath path) =>
        Volatile.Write(ref _lastGemmDispatchPath, (int)path);

    /// <summary>
    /// Runs correctness-gated evolutionary GEMM tuning on this OpenCL device and installs the
    /// validated deployment into the production dispatch path. Candidate execution, oracle
    /// comparison, warmup, repeated measurement, paired finalist replay, and persistence all use
    /// the first-party tuning scaffold.
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
        if (_context is null || _dynamicGemm is null || !IsAvailable)
            throw new InvalidOperationException("OpenCL is not available for evolutionary GEMM tuning.");

        GpuCapabilities capabilities = CreateEvolutionCapabilities();
        var orchestrator = new GemmAutoTuner();
        IReadOnlyList<OpenClGemmConfiguration> seeds = orchestrator.GetEvolutionSeeds(
            m, n, k, capabilities, additionalSeeds);
        OpenClGemmConfiguration incumbent = seeds[0];
        GemmKernelTemplate requiredTemplate = incumbent.KernelTemplate;
        seeds = orchestrator.GetEvolutionSeeds(
            m, n, k, capabilities, additionalSeeds, requiredTemplate);
        incumbent = seeds[0];

        using var experimentBackend = new OpenClGemmExperimentBackend(this, m, n, k);
        var experiment = new KernelTuningExperiment<OpenClGemmConfiguration>(
            experimentBackend,
            new StopwatchKernelTuningTimer(),
            incumbent,
            new KernelTuningWorkload(
                checked(2d * m * n * k),
                KernelTuningWorkUnit.FloatingPointOperations),
            KernelTuningTimingScope.SteadyStateExecution);

        EvolutionKernelTuningResult<OpenClGemmConfiguration> result =
            await orchestrator.TuneWithEvolutionAsync(
                m,
                n,
                k,
                capabilities,
                CreateOpenClTuningFingerprint(),
                experiment.EvaluateAsync,
                experiment,
                new KernelSearchSpaceVersion(3),
                new KernelBenchmarkProtocolVersion(2),
                additionalSeeds,
                requiredTemplate,
                engineOptions,
                tuningOptions,
                checkpointStore,
                deploymentRegistry,
                store,
                cancellationToken).ConfigureAwait(false);

        GemmConfig deployed = result.ActiveDeployment.Configuration.ToGemmConfig();
        lock (_evolutionaryGemmDeploymentLock)
            _evolutionaryGemmDeployments[(m, n, k)] = deployed;
        return result;
    }

    /// <summary>
    /// Loads a persisted, identity-matched evolutionary GEMM deployment without benchmarking and
    /// installs it into production dispatch. Identity covers the OpenCL backend, physical device,
    /// board, driver, OpenCL version, shape, search space, and measurement protocol.
    /// </summary>
    public bool TryActivatePersistedGemmEvolution(
        int m,
        int n,
        int k,
        IKernelTuningStore<OpenClGemmConfiguration> store,
        IEnumerable<GemmConfig>? additionalSeeds = null,
        KernelTuningDeploymentRegistry<OpenClGemmConfiguration>? deploymentRegistry = null)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (store is null) throw new ArgumentNullException(nameof(store));
        if (_context is null || _dynamicGemm is null || !IsAvailable) return false;

        GpuCapabilities capabilities = CreateEvolutionCapabilities();
        var orchestrator = new GemmAutoTuner();
        IReadOnlyList<OpenClGemmConfiguration> seeds = orchestrator.GetEvolutionSeeds(
            m, n, k, capabilities, additionalSeeds);
        GemmKernelTemplate requiredTemplate = seeds[0].KernelTemplate;
        EvolutionKernelAutotuner<OpenClGemmConfiguration> tuner = orchestrator.CreateEvolutionTuner(
            m,
            n,
            k,
            capabilities,
            CreateOpenClTuningFingerprint(),
            UnexpectedHydrationBenchmark,
            HydrationOnlyFinalistEvaluator.Instance,
            new KernelSearchSpaceVersion(3),
            new KernelBenchmarkProtocolVersion(2),
            requiredTemplate,
            deploymentRegistry: deploymentRegistry,
            store: store);
        if (!tuner.TryHydrate() ||
            !tuner.Deployment.TryGet(out OpenClGemmConfiguration configuration))
        {
            return false;
        }

        lock (_evolutionaryGemmDeploymentLock)
            _evolutionaryGemmDeployments[(m, n, k)] = configuration.ToGemmConfig();
        return true;
    }

    private static ValueTask<KernelTuningTrialResult> UnexpectedHydrationBenchmark(
        OpenClGemmConfiguration configuration,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken) =>
        throw new InvalidOperationException("Persisted deployment hydration must not benchmark a kernel.");

    private GpuCapabilities CreateEvolutionCapabilities()
    {
        DirectOpenClContext context = _context ??
            throw new InvalidOperationException("OpenCL context not available.");
        return GpuCapabilities.Detect(
            ComputeUnits,
            GlobalMemoryBytes,
            checked((int)LocalMemoryBytes),
            checked((int)_maxWorkGroupSize),
            DeviceVendor,
            DeviceName,
            context.Extensions,
            context.MaxWorkItemSizes);
    }

    private KernelTuningDeviceFingerprint CreateOpenClTuningFingerprint()
    {
        if (_context is null)
            throw new InvalidOperationException("OpenCL context not available.");
        KernelTuningDeviceKind kind = ParseTuningDeviceKind(DeviceVendor);
        string boardName = string.IsNullOrWhiteSpace(_context.DeviceBoardName)
            ? DeviceName
            : _context.DeviceBoardName;
        string modelKey = EvolutionHash.Combine(new[]
        {
            "opencl-model-v1",
            DeviceVendor,
            DeviceName,
            boardName,
            _context.DriverVersion,
            _context.OpenClVersion
        });
        string localKey = EvolutionHash.Combine(new[]
        {
            "opencl-device-v1",
            modelKey,
            _deviceIndex.ToString(System.Globalization.CultureInfo.InvariantCulture)
        });
        return new KernelTuningDeviceFingerprint(kind, localKey, modelKey);
    }

    /// <summary>Gets the typed physical device family used for tuning and cache isolation.</summary>
    public KernelTuningDeviceKind TuningDeviceKind => ParseTuningDeviceKind(DeviceVendor);

    private static KernelTuningDeviceKind ParseTuningDeviceKind(string vendor)
    {
        if (vendor.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) >= 0)
            return KernelTuningDeviceKind.NvidiaGpu;
        if (vendor.IndexOf("AMD", StringComparison.OrdinalIgnoreCase) >= 0 ||
            vendor.IndexOf("Advanced Micro Devices", StringComparison.OrdinalIgnoreCase) >= 0)
            return KernelTuningDeviceKind.AmdGpu;
        if (vendor.IndexOf("Intel", StringComparison.OrdinalIgnoreCase) >= 0)
            return KernelTuningDeviceKind.IntelGpu;
        if (vendor.IndexOf("Apple", StringComparison.OrdinalIgnoreCase) >= 0)
            return KernelTuningDeviceKind.AppleGpu;
        return KernelTuningDeviceKind.OtherAccelerator;
    }

    private bool TryGetEvolutionaryGemmDeployment(int m, int n, int k, out GemmConfig configuration)
    {
        lock (_evolutionaryGemmDeploymentLock)
            return _evolutionaryGemmDeployments.TryGetValue((m, n, k), out configuration);
    }

    private sealed class OpenClGemmExperimentBackend :
        IKernelTuningExperimentBackend<OpenClGemmConfiguration>, IDisposable
    {
        private const double AbsoluteTolerance = 2e-3;
        private const double RelativeTolerance = 2e-3;
        private readonly OpenClBackend _owner;
        private readonly int _m;
        private readonly int _n;
        private readonly int _k;
        private readonly float[] _expected;
        private readonly IGpuBuffer _deviceA;
        private readonly IGpuBuffer _deviceB;
        private readonly IGpuBuffer _deviceC;
        private readonly HashSet<OpenClGemmConfiguration> _prepared = new();
        private bool _disposed;

        internal OpenClGemmExperimentBackend(OpenClBackend owner, int m, int n, int k)
        {
            _owner = owner;
            _m = m;
            _n = n;
            _k = k;
            float[] inputA = CreateValues(checked(m * k), 17);
            float[] inputB = CreateValues(checked(k * n), 43);
            _expected = ComputeReference(inputA, inputB, m, n, k);

            IGpuBuffer? deviceA = null;
            IGpuBuffer? deviceB = null;
            IGpuBuffer? deviceC = null;
            try
            {
                deviceA = owner.AllocateBuffer(inputA);
                deviceB = owner.AllocateBuffer(inputB);
                deviceC = owner.AllocateBuffer(checked(m * n));
                _deviceA = deviceA ?? throw new InvalidOperationException("OpenCL returned no A buffer.");
                _deviceB = deviceB ?? throw new InvalidOperationException("OpenCL returned no B buffer.");
                _deviceC = deviceC ?? throw new InvalidOperationException("OpenCL returned no C buffer.");
            }
            catch
            {
                deviceC?.Dispose();
                deviceB?.Dispose();
                deviceA?.Dispose();
                throw;
            }
        }

        public ValueTask PrepareAsync(
            OpenClGemmConfiguration configuration,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();
            if (_prepared.Contains(configuration)) return default;
            GemmConfig config = configuration.ToGemmConfig();
            if (!_owner.TryExecutePackedDynamicGemm(
                    _deviceA,
                    _deviceB,
                    _deviceC,
                    _m,
                    _n,
                    _k,
                    1.0f,
                    0.0f,
                    config,
                    requireExactConfiguration: true))
            {
                throw new KernelTuningValidationException(
                    KernelTuningTrialStatus.CompilationFailed,
                    "The exact production OpenCL GEMM candidate could not be prepared.");
            }
            _owner.Synchronize();
            _prepared.Add(configuration);
            return default;
        }

        public ValueTask ExecuteAsync(
            OpenClGemmConfiguration configuration,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();
            EnsurePrepared(configuration);
            if (!_owner.TryExecutePackedDynamicGemm(
                    _deviceA,
                    _deviceB,
                    _deviceC,
                    _m,
                    _n,
                    _k,
                    1.0f,
                    0.0f,
                    configuration.ToGemmConfig(),
                    requireExactConfiguration: true))
            {
                throw new KernelTuningValidationException(
                    KernelTuningTrialStatus.BenchmarkFailed,
                    "The exact production OpenCL GEMM candidate failed during execution.");
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
            OpenClGemmConfiguration configuration,
            CancellationToken cancellationToken = default)
        {
            await ExecuteAsync(configuration, cancellationToken).ConfigureAwait(false);
            await SynchronizeAsync(cancellationToken).ConfigureAwait(false);
            float[] actual = _owner.DownloadBuffer(_deviceC);
            double maximumAbsoluteError = 0;
            double maximumRelativeError = 0;
            for (int row = 0; row < _m; row++)
            {
                for (int column = 0; column < _n; column++)
                {
                    int index = row * _n + column;
                    double absoluteError = Math.Abs(actual[index] - _expected[index]);
                    double relativeError = absoluteError / Math.Max(Math.Abs(_expected[index]), 1e-6);
                    maximumAbsoluteError = Math.Max(maximumAbsoluteError, absoluteError);
                    maximumRelativeError = Math.Max(maximumRelativeError, relativeError);
                }
            }
            if (maximumAbsoluteError > AbsoluteTolerance && maximumRelativeError > RelativeTolerance)
            {
                throw new KernelTuningValidationException(
                    KernelTuningTrialStatus.OutputMismatch,
                    $"OpenCL GEMM output exceeded tolerance: abs={maximumAbsoluteError:R}, rel={maximumRelativeError:R}.");
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
            GemmConfig config = configuration.ToGemmConfig();
            EnsurePrepared(configuration);
            ProductionResourceUsage resources = CalculateProductionResources(config, _m, _n, _k);
            return new KernelTuningResourceUsage(
                KernelTuningResourceMetric<long>.Measured(resources.WorkspaceBytes),
                // Core OpenCL exposes neither resident-wave occupancy nor register allocation.
                // Configuration-derived proxies are not measurements and are too vendor-specific
                // to compare across the AMD, NVIDIA, Intel, and other OpenCL implementations.
                KernelTuningResourceMetric<double>.Unavailable(),
                KernelTuningResourceMetric<int>.Unavailable(),
                // Exact preparation compiles and executes the production pipeline together, so
                // core OpenCL cannot isolate compilation latency without fabricating a value.
                KernelTuningResourceMetric<TimeSpan>.Unavailable(),
                KernelTuningResourceMetric<int>.Measured(resources.KernelLaunchCount));
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
            {
                int centered = ((i * 37 + salt * 17) % 127) - 63;
                result[i] = centered / 256f;
            }
            return result;
        }

        private static float[] ComputeReference(float[] a, float[] b, int m, int n, int k)
        {
            var result = new float[checked(m * n)];
            for (int row = 0; row < m; row++)
            {
                int aOffset = row * k;
                int cOffset = row * n;
                for (int column = 0; column < n; column++)
                {
                    double sum = 0;
                    for (int inner = 0; inner < k; inner++)
                        sum += (double)a[aOffset + inner] * b[inner * n + column];
                    result[cOffset + column] = (float)sum;
                }
            }
            return result;
        }

        private static ProductionResourceUsage CalculateProductionResources(
            GemmConfig config,
            int m,
            int n,
            int k)
        {
            int kRegister = Math.Max(1, config.KReg);
            int kUnit = checked(config.TileK * kRegister);

            if (config.KernelTemplate == GemmKernelTemplate.ClBlastBaselineK0)
            {
                int mPadded = RoundUp(n, config.TileM);
                int nPadded = RoundUp(m, config.TileN);
                int kPadded = RoundUp(k, kUnit);
                bool needsA = n != mPadded || k != kPadded;
                bool needsC = n != mPadded || m != nPadded;
                long elements = (long)nPadded * kPadded;
                if (needsA) elements = checked(elements + (long)mPadded * kPadded);
                if (needsC) elements = checked(elements + (long)mPadded * nPadded);
                return new ProductionResourceUsage(
                    checked(elements * sizeof(float)),
                    2 + (needsA ? 1 : 0) + (needsC ? 1 : 0));
            }

            if (config.KernelTemplate == GemmKernelTemplate.ClBlastBaselineK1)
            {
                int mPadded = RoundUp(m, config.TileN);
                int nPadded = RoundUp(n, config.TileM);
                int kPadded = RoundUp(k, kUnit);
                bool needsA = k != kPadded || m != mPadded;
                bool needsB = n != nPadded || k != kPadded;
                bool needsC = n != nPadded || m != mPadded;
                long elements = 0;
                if (needsA) elements = checked(elements + (long)kPadded * mPadded);
                if (needsB) elements = checked(elements + (long)nPadded * kPadded);
                if (needsC) elements = checked(elements + (long)nPadded * mPadded);
                return new ProductionResourceUsage(
                    checked(elements * sizeof(float)),
                    1 + (needsA ? 1 : 0) + (needsB ? 1 : 0) + (needsC ? 1 : 0));
            }

            int genericM = RoundUp(m, config.TileM);
            int genericN = RoundUp(n, config.TileN);
            int genericK = RoundUp(k, kUnit);
            bool needsPacking = genericM != m || genericN != n || genericK != k ||
                                config.UseColumnMajorA;
            if (!needsPacking) return new ProductionResourceUsage(0, 1);
            long workspaceElements = checked(
                (long)genericM * genericK +
                (long)genericK * genericN +
                (long)genericM * genericN);
            return new ProductionResourceUsage(
                checked(workspaceElements * sizeof(float)),
                5);
        }

        private static int RoundUp(int value, int multiple) =>
            checked(((value + multiple - 1) / multiple) * multiple);

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(OpenClGemmExperimentBackend));
        }

        private void EnsurePrepared(OpenClGemmConfiguration configuration)
        {
            if (!_prepared.Contains(configuration))
                throw new InvalidOperationException("The OpenCL GEMM candidate was not prepared before execution.");
        }

        private readonly record struct ProductionResourceUsage(
            long WorkspaceBytes,
            int KernelLaunchCount);
    }

    private sealed class HydrationOnlyFinalistEvaluator :
        IKernelTuningFinalistEvaluator<OpenClGemmConfiguration>
    {
        internal static HydrationOnlyFinalistEvaluator Instance { get; } = new();

        public ValueTask<KernelTuningFinalistReplay<OpenClGemmConfiguration>> ReplayAsync(
            KernelTuningIdentity identity,
            OpenClGemmConfiguration candidate,
            KernelTuningDeploymentSnapshot<OpenClGemmConfiguration>? activeDeployment,
            CancellationToken cancellationToken = default) =>
            throw new InvalidOperationException("Persisted deployment hydration must not replay finalists.");
    }
}
