using System.Collections.ObjectModel;
using AiDotNet.Evolution;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>Stable identity of one kernel-tuning problem on one GPU and input shape.</summary>
public sealed class KernelTuningIdentity
{
    /// <summary>Creates a kernel-tuning identity.</summary>
    public KernelTuningIdentity(
        KernelId kernel,
        ShapeProfile shape,
        GpuDeviceFingerprint device,
        string searchSpaceVersion,
        string benchmarkVersion)
    {
        if (string.IsNullOrWhiteSpace(kernel.Category))
            throw new ArgumentException("A kernel category is required.", nameof(kernel));
        if (string.IsNullOrWhiteSpace(kernel.Name))
            throw new ArgumentException("A kernel name is required.", nameof(kernel));
        Kernel = kernel;
        Shape = shape is null
            ? throw new ArgumentNullException(nameof(shape))
            : new ShapeProfile(shape.Dimensions);
        if (string.IsNullOrWhiteSpace(device.UniqueId))
            throw new ArgumentException("A valid GPU fingerprint is required.", nameof(device));
        Device = device;
        SearchSpaceVersion = RequireVersion(searchSpaceVersion, nameof(searchSpaceVersion));
        BenchmarkVersion = RequireVersion(benchmarkVersion, nameof(benchmarkVersion));
        StableKey = EvolutionHash.Combine(new[]
        {
            "tensor-kernel-tuning-identity-v1",
            Kernel.ToFileStem(),
            Shape.ToFileStem(),
            Device.LocalKey,
            SearchSpaceVersion,
            BenchmarkVersion
        });
    }

    /// <summary>Gets the tuned kernel family.</summary>
    public KernelId Kernel { get; }

    /// <summary>Gets an immutable copy of the input shape profile.</summary>
    public ShapeProfile Shape { get; }

    /// <summary>Gets the physical GPU and driver identity.</summary>
    public GpuDeviceFingerprint Device { get; }

    /// <summary>Gets the version of the typed candidate space and mutation rules.</summary>
    public string SearchSpaceVersion { get; }

    /// <summary>Gets the version of the benchmark protocol and correctness oracle.</summary>
    public string BenchmarkVersion { get; }

    /// <summary>Gets a stable hash covering the kernel, shape, GPU, search space, and benchmark.</summary>
    public string StableKey { get; }

    private static string RequireVersion(string value, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException("A non-blank version is required.", parameterName);
        return value.Trim();
    }
}

/// <summary>One valid on-device benchmark measurement for a typed kernel configuration.</summary>
public sealed class KernelTuningMeasurement
{
    private readonly ReadOnlyDictionary<string, double> _additionalMetrics;

    /// <summary>Creates a validated kernel measurement.</summary>
    public KernelTuningMeasurement(
        double throughputGflops,
        TimeSpan elapsed,
        long workspaceBytes = 0,
        IReadOnlyDictionary<string, double>? additionalMetrics = null)
    {
        if (!IsFinite(throughputGflops) || throughputGflops <= 0)
            throw new ArgumentOutOfRangeException(nameof(throughputGflops));
        if (elapsed < TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(elapsed));
        if (workspaceBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(workspaceBytes));

        var metrics = new Dictionary<string, double>(StringComparer.Ordinal);
        if (additionalMetrics is not null)
        {
            foreach (KeyValuePair<string, double> metric in additionalMetrics)
            {
                if (string.IsNullOrWhiteSpace(metric.Key))
                    throw new ArgumentException("Metric names cannot be blank.", nameof(additionalMetrics));
                if (!IsFinite(metric.Value))
                    throw new ArgumentOutOfRangeException(nameof(additionalMetrics));
                string name = metric.Key.Trim();
                if (IsReservedMetric(name) || metrics.ContainsKey(name))
                    throw new ArgumentException($"Metric '{name}' is reserved or duplicated.", nameof(additionalMetrics));
                metrics.Add(name, metric.Value);
            }
        }

        ThroughputGflops = throughputGflops;
        Elapsed = elapsed;
        WorkspaceBytes = workspaceBytes;
        _additionalMetrics = new ReadOnlyDictionary<string, double>(metrics);
    }

    /// <summary>Gets measured throughput in billions of floating-point operations per second.</summary>
    public double ThroughputGflops { get; }

    /// <summary>Gets elapsed time for the benchmark protocol.</summary>
    public TimeSpan Elapsed { get; }

    /// <summary>Gets temporary device memory required by the candidate.</summary>
    public long WorkspaceBytes { get; }

    /// <summary>Gets additional finite reporting metrics.</summary>
    public IReadOnlyDictionary<string, double> AdditionalMetrics => _additionalMetrics;

    private static bool IsReservedMetric(string name) =>
        string.Equals(name, EvolutionKernelAutotunerMetrics.ThroughputGflops, StringComparison.Ordinal) ||
        string.Equals(name, EvolutionKernelAutotunerMetrics.LogThroughputGflops, StringComparison.Ordinal) ||
        string.Equals(name, EvolutionKernelAutotunerMetrics.ElapsedMilliseconds, StringComparison.Ordinal) ||
        string.Equals(name, EvolutionKernelAutotunerMetrics.WorkspaceBytes, StringComparison.Ordinal);

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}

/// <summary>Stable metric names emitted by <see cref="EvolutionKernelAutotuner{TConfiguration}"/>.</summary>
public static class EvolutionKernelAutotunerMetrics
{
    /// <summary>The quality and archive descriptor optimized by the tuner.</summary>
    public const string ThroughputGflops = "throughput-gflops";

    /// <summary>
    /// Base-10 logarithm of throughput, used only to place measurements into a bounded MAP-Elites axis.
    /// </summary>
    public const string LogThroughputGflops = "log10-throughput-gflops";

    /// <summary>Benchmark elapsed time in milliseconds.</summary>
    public const string ElapsedMilliseconds = "elapsed-milliseconds";

    /// <summary>Temporary device memory in bytes.</summary>
    public const string WorkspaceBytes = "workspace-bytes";
}

/// <summary>An immutable, typed configuration activated for production dispatch.</summary>
public sealed class KernelTuningDeploymentSnapshot<TConfiguration>
    where TConfiguration : notnull
{
    internal KernelTuningDeploymentSnapshot(
        KernelTuningIdentity identity,
        TConfiguration configuration,
        string genomeId,
        double measuredGflops,
        string runStateHash)
    {
        Identity = identity;
        Configuration = configuration;
        GenomeId = genomeId;
        MeasuredGflops = measuredGflops;
        RunStateHash = runStateHash;
    }

    /// <summary>Gets the exact kernel, device, and shape identity.</summary>
    public KernelTuningIdentity Identity { get; }

    /// <summary>Gets the typed launch configuration used by dispatch.</summary>
    public TConfiguration Configuration { get; }

    /// <summary>Gets the canonical configuration identity selected by evolution.</summary>
    public string GenomeId { get; }

    /// <summary>Gets the measured throughput of the winner.</summary>
    public double MeasuredGflops { get; }

    /// <summary>Gets the deterministic state hash of the run that selected the winner.</summary>
    public string RunStateHash { get; }
}

/// <summary>
/// Lock-free single-target deployment slot for a typed tuned configuration.
/// </summary>
/// <remarks>
/// Run tuning during startup or on a background worker, then let the kernel hot path call <see cref="TryGet"/>.
/// A hit performs one volatile reference read and one typed assignment: no file I/O, parsing, reflection, hashing,
/// or string-based dispatch occurs while the kernel is serving work.
/// </remarks>
public sealed class KernelTuningDeployment<TConfiguration>
    where TConfiguration : notnull
{
    private KernelTuningDeploymentSnapshot<TConfiguration>? _current;

    /// <summary>Gets the currently deployed snapshot, or <c>null</c> before a configuration is activated.</summary>
    public KernelTuningDeploymentSnapshot<TConfiguration>? Current => Volatile.Read(ref _current);

    /// <summary>Reads the active typed configuration through the hot-path-safe deployment slot.</summary>
    public bool TryGet(out TConfiguration configuration)
    {
        KernelTuningDeploymentSnapshot<TConfiguration>? snapshot = Volatile.Read(ref _current);
        if (snapshot is null)
        {
            configuration = default!;
            return false;
        }

        configuration = snapshot.Configuration;
        return true;
    }

    internal void Publish(KernelTuningDeploymentSnapshot<TConfiguration> snapshot)
    {
        if (snapshot is null) throw new ArgumentNullException(nameof(snapshot));
        Volatile.Write(ref _current, snapshot);
    }
}

/// <summary>The engine result and typed configuration published by a completed kernel-tuning run.</summary>
public sealed class EvolutionKernelTuningResult<TConfiguration>
    where TConfiguration : notnull
{
    internal EvolutionKernelTuningResult(
        EvolutionRunResult<TConfiguration> run,
        KernelTuningDeploymentSnapshot<TConfiguration> deployment)
    {
        Run = run;
        Deployment = deployment;
    }

    /// <summary>Gets the complete quality-diversity run result.</summary>
    public EvolutionRunResult<TConfiguration> Run { get; }

    /// <summary>Gets the winner atomically published for production dispatch.</summary>
    public KernelTuningDeploymentSnapshot<TConfiguration> Deployment { get; }
}

/// <summary>
/// Evolves immutable, typed kernel configurations outside the serving hot path and publishes the measured winner.
/// </summary>
/// <typeparam name="TConfiguration">An immutable launch-configuration type, normally a record struct with enums.</typeparam>
public sealed class EvolutionKernelAutotuner<TConfiguration>
    where TConfiguration : notnull
{
    private readonly KernelTuningIdentity _identity;
    private readonly IEvolutionGenomeCodec<TConfiguration> _codec;
    private readonly IVariationOperator<TConfiguration> _variation;
    private readonly EvolutionEngineOptions _options;
    private readonly Func<TConfiguration, EvolutionEvaluationContext, CancellationToken,
        ValueTask<KernelTuningMeasurement>> _benchmark;
    private readonly Func<int, IEvolutionArchive<TConfiguration>> _archiveFactory;
    private readonly IEvolutionCheckpointStore? _checkpointStore;
    private readonly KernelTuningDeployment<TConfiguration> _deployment;

    /// <summary>Creates a typed evolutionary kernel tuner.</summary>
    public EvolutionKernelAutotuner(
        KernelTuningIdentity identity,
        IEvolutionGenomeCodec<TConfiguration> codec,
        IVariationOperator<TConfiguration> variation,
        Func<TConfiguration, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningMeasurement>> benchmark,
        EvolutionEngineOptions? options = null,
        Func<int, IEvolutionArchive<TConfiguration>>? archiveFactory = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeployment<TConfiguration>? deployment = null)
    {
        _identity = identity ?? throw new ArgumentNullException(nameof(identity));
        _codec = codec ?? throw new ArgumentNullException(nameof(codec));
        _variation = variation ?? throw new ArgumentNullException(nameof(variation));
        _benchmark = benchmark ?? throw new ArgumentNullException(nameof(benchmark));
        _options = (options ?? CreateDefaultOptions(identity)).SnapshotAndValidate();
        _archiveFactory = archiveFactory ?? (_ => CreateDefaultArchive());
        _checkpointStore = checkpointStore;
        _deployment = deployment ?? new KernelTuningDeployment<TConfiguration>();
    }

    /// <summary>Gets the lock-free deployment slot updated after a successful run.</summary>
    public KernelTuningDeployment<TConfiguration> Deployment => _deployment;

    /// <summary>Runs tuning on the calling workflow, suitable for offline jobs and startup warmup.</summary>
    public async Task<EvolutionKernelTuningResult<TConfiguration>> TuneAsync(
        IEnumerable<TConfiguration> seeds,
        CancellationToken cancellationToken = default)
    {
        if (seeds is null) throw new ArgumentNullException(nameof(seeds));
        TConfiguration[] seedSnapshot = seeds.ToArray();
        if (seedSnapshot.Length == 0)
            throw new ArgumentException("At least one typed kernel configuration is required.", nameof(seeds));
        for (int i = 0; i < seedSnapshot.Length; i++)
        {
            if (_codec.Serialize(seedSnapshot[i]) is null)
            {
                throw new InvalidOperationException(
                    $"The kernel configuration codec returned a null payload for seed {i}.");
            }
        }

        var task = new KernelTuningTask(_identity, _codec, _benchmark);
        var engine = new EvolutionEngine<TConfiguration>(
            task,
            _variation,
            _archiveFactory,
            _options,
            checkpointStore: _checkpointStore,
            genomeCodec: _codec);

        EvolutionRunResult<TConfiguration> run =
            await engine.RunAsync(seedSnapshot, cancellationToken).ConfigureAwait(false);
        EvolutionArchiveEntry<TConfiguration> best = run.Best ?? throw new InvalidOperationException(
            "Kernel tuning completed without a valid measured configuration.");
        double measuredGflops = best.Evaluation.Quality ?? throw new InvalidOperationException(
            "The winning kernel configuration has no throughput measurement.");
        var snapshot = new KernelTuningDeploymentSnapshot<TConfiguration>(
            _identity,
            best.Candidate.CanonicalGenome.Genome,
            best.Evaluation.GenomeId,
            measuredGflops,
            run.StateHash);
        _deployment.Publish(snapshot);
        return new EvolutionKernelTuningResult<TConfiguration>(run, snapshot);
    }

    /// <summary>
    /// Starts tuning on a worker task so production dispatch can keep using the current deployment until publication.
    /// </summary>
    /// <remarks>The returned task owns all failures; callers should observe or await it during shutdown.</remarks>
    public Task<EvolutionKernelTuningResult<TConfiguration>> TuneInBackgroundAsync(
        IEnumerable<TConfiguration> seeds,
        CancellationToken cancellationToken = default)
    {
        if (seeds is null) throw new ArgumentNullException(nameof(seeds));
        TConfiguration[] seedSnapshot = seeds.ToArray();
        if (seedSnapshot.Length == 0)
            throw new ArgumentException("At least one typed kernel configuration is required.", nameof(seeds));
        return Task.Run(() => TuneAsync(seedSnapshot, cancellationToken), cancellationToken);
    }

    private static EvolutionEngineOptions CreateDefaultOptions(KernelTuningIdentity identity) => new()
    {
        RunId = "kernel-" + identity.StableKey,
        MaxEvaluationAttempts = 64,
        MaxProposals = 512,
        MaxGenerations = 512,
        ProposalBatchSize = 1,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1
    };

    private static IEvolutionArchive<TConfiguration> CreateDefaultArchive() =>
        new MapElitesArchive<TConfiguration>(new[]
        {
            new EvolutionDescriptorDefinition(
                EvolutionKernelAutotunerMetrics.LogThroughputGflops,
                -12,
                12,
                96,
                EvolutionOutOfRangePolicy.Grow)
        }, EvolutionOptimizationDirection.Maximize, maximumGridCells: 1_000_000);

    private sealed class KernelTuningTask : IEvolutionTask<TConfiguration>
    {
        private readonly IEvolutionGenomeCodec<TConfiguration> _codec;
        private readonly Func<TConfiguration, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningMeasurement>> _benchmark;

        public KernelTuningTask(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<TConfiguration> codec,
            Func<TConfiguration, EvolutionEvaluationContext, CancellationToken,
                ValueTask<KernelTuningMeasurement>> benchmark)
        {
            _codec = codec;
            _benchmark = benchmark;
            Id = "tensor-kernel-" + identity.Kernel.ToFileStem();
            VersionHash = EvolutionHash.Combine(new[]
            {
                "tensor-kernel-tuning-task-v1",
                identity.StableKey,
                codec.Id,
                codec.VersionHash
            });
            EvaluatorVersionHash = EvolutionHash.Combine(new[]
            {
                "tensor-kernel-benchmark-v1",
                identity.BenchmarkVersion,
                identity.Device.LocalKey
            });
        }

        public string Id { get; }

        public string VersionHash { get; }

        public string EvaluatorVersionHash { get; }

        public ValueTask<EvolutionCanonicalGenome<TConfiguration>> CanonicalizeAsync(
            TConfiguration genome,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string payload = _codec.Serialize(genome) ?? throw new InvalidOperationException(
                "The kernel configuration codec returned a null payload.");
            return new ValueTask<EvolutionCanonicalGenome<TConfiguration>>(
                new EvolutionCanonicalGenome<TConfiguration>(genome, EvolutionHash.Compute(payload)));
        }

        public async ValueTask<EvolutionTaskResult> EvaluateAsync(
            EvolutionCandidate<TConfiguration> candidate,
            EvolutionEvaluationContext context,
            CancellationToken cancellationToken = default)
        {
            KernelTuningMeasurement measurement = await _benchmark(
                candidate.CanonicalGenome.Genome,
                context,
                cancellationToken).ConfigureAwait(false) ?? throw new InvalidOperationException(
                    "The kernel benchmark returned no measurement.");

            var descriptors = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                [EvolutionKernelAutotunerMetrics.LogThroughputGflops] = Math.Log10(measurement.ThroughputGflops)
            };
            var metrics = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                [EvolutionKernelAutotunerMetrics.ThroughputGflops] = measurement.ThroughputGflops,
                [EvolutionKernelAutotunerMetrics.ElapsedMilliseconds] = measurement.Elapsed.TotalMilliseconds,
                [EvolutionKernelAutotunerMetrics.WorkspaceBytes] = measurement.WorkspaceBytes
            };
            foreach (KeyValuePair<string, double> metric in measurement.AdditionalMetrics)
                metrics.Add(metric.Key, metric.Value);
            return new EvolutionTaskResult(
                EvolutionEvaluationStatus.Completed,
                measurement.ThroughputGflops,
                EvolutionOptimizationDirection.Maximize,
                descriptors,
                costUnits: measurement.Elapsed.TotalMilliseconds,
                metrics: metrics);
        }
    }
}
