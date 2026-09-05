using AiDotNet.Evolution;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>The engine result, proposed winner, and active deployment from a tuning run.</summary>
public sealed class EvolutionKernelTuningResult<TConfiguration>
    where TConfiguration : notnull
{
    internal EvolutionKernelTuningResult(
        EvolutionRunResult<TConfiguration> run,
        KernelTuningDeploymentSnapshot<TConfiguration> proposedWinner,
        KernelTuningDeploymentSnapshot<TConfiguration> activeDeployment,
        bool wasPromoted,
        bool wasPersisted)
    {
        Run = run;
        ProposedWinner = proposedWinner;
        ActiveDeployment = activeDeployment;
        WasPromoted = wasPromoted;
        WasPersisted = wasPersisted;
    }

    /// <summary>Gets the complete quality-diversity run result.</summary>
    public EvolutionRunResult<TConfiguration> Run { get; }
    /// <summary>Gets the best candidate measured by this run.</summary>
    public KernelTuningDeploymentSnapshot<TConfiguration> ProposedWinner { get; }
    /// <summary>Gets the configuration left active after promotion policy was applied.</summary>
    public KernelTuningDeploymentSnapshot<TConfiguration> ActiveDeployment { get; }
    /// <summary>Gets whether the proposed winner cleared the promotion threshold.</summary>
    public bool WasPromoted { get; }
    /// <summary>Gets whether a promoted winner was persisted successfully.</summary>
    public bool WasPersisted { get; }
}

/// <summary>Evolves typed kernel schedules off-path and publishes only locally validated winners.</summary>
public sealed class EvolutionKernelAutotuner<TConfiguration>
    where TConfiguration : notnull
{
    private const string TimingSampleCountMetric = "timing-sample-count";
    private const string ValidationScopeMetric = "validation-scope";
    private const string OutputAbsoluteErrorMetric = "output-absolute-error";
    private const string OutputRelativeErrorMetric = "output-relative-error";
    private const string OutputAbsoluteToleranceMetric = "output-absolute-tolerance";
    private const string OutputRelativeToleranceMetric = "output-relative-tolerance";
    private const string GradientAbsoluteErrorMetric = "gradient-absolute-error";
    private const string GradientRelativeErrorMetric = "gradient-relative-error";
    private const string GradientAbsoluteToleranceMetric = "gradient-absolute-tolerance";
    private const string GradientRelativeToleranceMetric = "gradient-relative-tolerance";

    private readonly KernelTuningIdentity _identity;
    private readonly IEvolutionGenomeCodec<TConfiguration> _codec;
    private readonly IVariationOperator<TConfiguration> _variation;
    private readonly EvolutionEngineOptions _engineOptions;
    private readonly KernelTuningOptions _tuningOptions;
    private readonly Func<TConfiguration, EvolutionEvaluationContext, CancellationToken,
        ValueTask<KernelTuningTrialResult>> _evaluator;
    private readonly Func<int, IEvolutionArchive<TConfiguration>> _archiveFactory;
    private readonly IEvolutionCheckpointStore? _checkpointStore;
    private readonly KernelTuningDeployment<TConfiguration> _deployment;
    private readonly IKernelTuningStore<TConfiguration> _store;
    private readonly Func<TConfiguration, bool> _deploymentValidator;

    /// <summary>Creates a correctness-first evolutionary kernel tuner.</summary>
    public EvolutionKernelAutotuner(
        KernelTuningIdentity identity,
        IEvolutionGenomeCodec<TConfiguration> codec,
        IVariationOperator<TConfiguration> variation,
        Func<TConfiguration, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningTrialResult>> evaluator,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        Func<int, IEvolutionArchive<TConfiguration>>? archiveFactory = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<TConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<TConfiguration>? store = null,
        Func<TConfiguration, bool>? deploymentValidator = null)
    {
        _identity = identity ?? throw new ArgumentNullException(nameof(identity));
        _codec = codec ?? throw new ArgumentNullException(nameof(codec));
        _variation = variation ?? throw new ArgumentNullException(nameof(variation));
        _evaluator = evaluator ?? throw new ArgumentNullException(nameof(evaluator));
        _engineOptions = (engineOptions ?? CreateDefaultEngineOptions(identity)).SnapshotAndValidate();
        if (_engineOptions.MaxEvaluationAttempts == 0)
            throw new ArgumentException(
                "Kernel tuning requires a positive evaluation-attempt budget.", nameof(engineOptions));
        if (_engineOptions.MaxProposals == 0)
            throw new ArgumentException(
                "Kernel tuning requires a positive proposal budget.", nameof(engineOptions));
        _tuningOptions = (tuningOptions ?? new KernelTuningOptions()).SnapshotAndValidate();
        _archiveFactory = archiveFactory ?? (_ => CreateDefaultArchive(_tuningOptions.ArchiveDescriptors));
        _checkpointStore = checkpointStore;
        _deployment = (deploymentRegistry ?? new KernelTuningDeploymentRegistry<TConfiguration>()).GetOrCreate(identity);
        _store = store ?? new AutotuneCacheKernelTuningStore<TConfiguration>();
        _deploymentValidator = deploymentValidator ?? (_ => true);
    }

    /// <summary>Gets the pre-resolved lock-free deployment handle.</summary>
    public KernelTuningDeployment<TConfiguration> Deployment => _deployment;

    internal int MaximumProposals => _engineOptions.MaxProposals;

    /// <summary>Hydrates a locally persisted winner after fully validating its typed payload and evidence.</summary>
    public bool TryHydrate()
    {
        KernelTuningDeploymentSnapshot<TConfiguration>? current = _deployment.Current;
        if (current is not null) return CanDeploy(current.Configuration);
        if (!TryLoadValidSnapshot(out KernelTuningDeploymentSnapshot<TConfiguration>? snapshot) ||
            snapshot is null) return false;
        if (_deployment.TryPublishIfEmpty(snapshot)) return true;

        current = _deployment.Current;
        return current is not null && CanDeploy(current.Configuration);
    }

    private bool TryLoadValidSnapshot(
        out KernelTuningDeploymentSnapshot<TConfiguration>? snapshot)
    {
        snapshot = null;
        try
        {
            if (!_store.TryLoad(
                    _identity,
                    _codec,
                    out KernelTuningDeploymentSnapshot<TConfiguration>? loaded) ||
                loaded is null ||
                !string.Equals(
                    loaded.Identity.StableKey,
                    _identity.StableKey,
                    StringComparison.Ordinal) ||
                !CanDeploy(loaded.Configuration))
            {
                return false;
            }

            string? payload = _codec.Serialize(loaded.Configuration);
            if (payload is null ||
                !string.Equals(
                    EvolutionHash.Compute(payload),
                    loaded.GenomeId,
                    StringComparison.Ordinal))
            {
                return false;
            }

            snapshot = loaded;
            return true;
        }
        catch
        {
            snapshot = null;
            return false;
        }
    }

    private bool CanDeploy(TConfiguration configuration)
    {
        try
        {
            return _deploymentValidator(configuration);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Runs tuning during an explicit offline or startup workflow.</summary>
    public async Task<EvolutionKernelTuningResult<TConfiguration>> TuneAsync(
        IEnumerable<TConfiguration> seeds,
        CancellationToken cancellationToken = default)
    {
        TConfiguration[] seedSnapshot = ValidateAndSnapshotSeeds(seeds);
        using IDisposable deviceLease = await KernelTuningCoordinator.EnterAsync(
            _identity.Device, cancellationToken).ConfigureAwait(false);
        TryHydrate();
        KernelTuningDeploymentSnapshot<TConfiguration>? existing = _deployment.Current;
        if (existing is not null && !CanDeploy(existing.Configuration)) existing = null;
        if (existing is not null) seedSnapshot = MergePersistedWinner(seedSnapshot, existing.Configuration);

        var task = new KernelTuningTask(
            _identity, _codec, _evaluator, _tuningOptions.ArchiveDescriptors, CanDeploy);
        var engine = new EvolutionEngine<TConfiguration>(
            task,
            _variation,
            _archiveFactory,
            _engineOptions,
            checkpointStore: _checkpointStore,
            genomeCodec: _codec);

        EvolutionRunResult<TConfiguration> run =
            await engine.RunAsync(seedSnapshot, cancellationToken).ConfigureAwait(false);
        EvolutionArchiveEntry<TConfiguration> best = run.Best ?? throw new InvalidOperationException(
            "Kernel tuning completed without a locally valid measured configuration.");
        TConfiguration bestConfiguration = best.Candidate.CanonicalGenome.Genome;
        if (!CanDeploy(bestConfiguration))
            throw new InvalidOperationException(
                "Kernel tuning selected a configuration that failed its deployment invariant.");
        KernelTuningMeasurement measurement = KernelTuningTask.ReadMeasurement(best.Evaluation);
        var proposed = new KernelTuningDeploymentSnapshot<TConfiguration>(
            _identity,
            bestConfiguration,
            best.Evaluation.GenomeId,
            measurement,
            run.StateHash);

        if (existing is not null &&
            proposed.Measurement.ThroughputGflops <
            existing.Measurement.ThroughputGflops * _tuningOptions.MinimumPromotionRatio)
        {
            return new EvolutionKernelTuningResult<TConfiguration>(run, proposed, existing, false, false);
        }

        _deployment.Publish(proposed);
        bool persisted = TryPersist(proposed);
        return new EvolutionKernelTuningResult<TConfiguration>(run, proposed, proposed, true, persisted);
    }

    private bool TryPersist(KernelTuningDeploymentSnapshot<TConfiguration> snapshot)
    {
        try
        {
            return _store.TryStore(snapshot, _codec);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Runs on a worker only after the caller-supplied gate reports that user work will not be displaced.
    /// </summary>
    public Task<EvolutionKernelTuningResult<TConfiguration>> TuneInBackgroundAsync(
        IEnumerable<TConfiguration> seeds,
        IKernelTuningIdleGate idleGate,
        CancellationToken cancellationToken = default)
    {
        if (idleGate is null) throw new ArgumentNullException(nameof(idleGate));
        TConfiguration[] snapshot = ValidateAndSnapshotSeeds(seeds);
        return Task.Run(async () =>
        {
            await idleGate.WaitUntilIdleAsync(_identity, cancellationToken).ConfigureAwait(false);
            return await TuneAsync(snapshot, cancellationToken).ConfigureAwait(false);
        }, cancellationToken);
    }

    private TConfiguration[] ValidateAndSnapshotSeeds(IEnumerable<TConfiguration> seeds)
    {
        if (seeds is null) throw new ArgumentNullException(nameof(seeds));
        TConfiguration[] snapshot = seeds.ToArray();
        if (snapshot.Length == 0)
            throw new ArgumentException("At least one typed kernel configuration is required.", nameof(seeds));
        if (snapshot.Length > _engineOptions.MaxProposals)
            throw new ArgumentException(
                "Kernel configuration seeds cannot exceed the run's proposal budget.", nameof(seeds));
        for (int i = 0; i < snapshot.Length; i++)
        {
            if (snapshot[i] is null)
                throw new ArgumentException($"Kernel configuration seed {i} is null.", nameof(seeds));
            string? payload = _codec.Serialize(snapshot[i]);
            if (payload is null)
                throw new InvalidOperationException($"The kernel configuration codec returned null for seed {i}.");
        }
        return snapshot;
    }

    private TConfiguration[] MergePersistedWinner(
        IReadOnlyList<TConfiguration> seeds,
        TConfiguration persisted)
    {
        string? persistedPayload = _codec.Serialize(persisted);
        if (persistedPayload is null)
            throw new InvalidOperationException("The kernel configuration codec returned a null payload.");
        for (int i = 0; i < seeds.Count; i++)
        {
            if (string.Equals(_codec.Serialize(seeds[i]), persistedPayload, StringComparison.Ordinal))
                return seeds.ToArray();
        }
        if (seeds.Count >= _engineOptions.MaxProposals) return seeds.ToArray();

        var result = new TConfiguration[seeds.Count + 1];
        for (int i = 0; i < seeds.Count; i++) result[i] = seeds[i];
        result[result.Length - 1] = persisted;
        return result;
    }

    private static EvolutionEngineOptions CreateDefaultEngineOptions(KernelTuningIdentity identity) => new()
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

    private static IEvolutionArchive<TConfiguration> CreateDefaultArchive(
        IReadOnlyList<KernelTuningDescriptorDefinition> descriptors) =>
        new MapElitesArchive<TConfiguration>(
            descriptors.Select(descriptor => descriptor.ToEvolutionDefinition()).ToArray(),
            EvolutionOptimizationDirection.Maximize,
            maximumGridCells: 1_000_000);

    private sealed class KernelTuningTask : IEvolutionTask<TConfiguration>
    {
        private readonly IEvolutionGenomeCodec<TConfiguration> _codec;
        private readonly Func<TConfiguration, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningTrialResult>> _evaluator;
        private readonly KernelTuningDescriptorDefinition[] _descriptors;
        private readonly Func<TConfiguration, bool> _deploymentValidator;

        internal KernelTuningTask(
            KernelTuningIdentity identity,
            IEvolutionGenomeCodec<TConfiguration> codec,
            Func<TConfiguration, EvolutionEvaluationContext, CancellationToken,
                ValueTask<KernelTuningTrialResult>> evaluator,
            IReadOnlyList<KernelTuningDescriptorDefinition> descriptors,
            Func<TConfiguration, bool> deploymentValidator)
        {
            _codec = codec;
            _evaluator = evaluator;
            _descriptors = descriptors.ToArray();
            _deploymentValidator = deploymentValidator;
            Id = "tensor-kernel-" + identity.Kernel.ToFileStem();
            VersionHash = EvolutionHash.Combine(new[]
            {
                "tensor-kernel-tuning-task-v2",
                identity.StableKey,
                codec.Id,
                codec.VersionHash
            });
            EvaluatorVersionHash = EvolutionHash.Combine(new[]
            {
                "tensor-kernel-benchmark-v2",
                identity.BenchmarkProtocolVersion.ToString(),
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
            if (!_deploymentValidator(candidate.CanonicalGenome.Genome))
            {
                return FailureResult(KernelTuningTrialResult.Rejected(
                    KernelTuningTrialStatus.InvalidConfiguration,
                    "The candidate failed the typed deployment invariant."));
            }

            KernelTuningTrialResult trial = await _evaluator(
                candidate.CanonicalGenome.Genome, context, cancellationToken).ConfigureAwait(false)
                ?? throw new InvalidOperationException("The kernel evaluator returned no result.");
            if (trial.Status != KernelTuningTrialStatus.Passed)
                return FailureResult(trial);
            KernelTuningMeasurement measurement = trial.Measurement ?? throw new InvalidOperationException(
                "A passed kernel trial must carry a measurement.");

            var descriptors = new Dictionary<string, double>(StringComparer.Ordinal);
            for (int i = 0; i < _descriptors.Length; i++)
            {
                KernelTuningDescriptorDefinition descriptor = _descriptors[i];
                descriptors.Add(
                    KernelTuningMetricNames.Get(descriptor.Metric),
                    measurement.GetMetric(descriptor.Metric));
            }
            var metrics = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (KernelTuningMetric metric in Enum.GetValues(typeof(KernelTuningMetric)))
                metrics.Add(KernelTuningMetricNames.Get(metric), measurement.GetMetric(metric));
            metrics.Add(TimingSampleCountMetric, measurement.Timing.SampleCount);
            metrics.Add(ValidationScopeMetric, (int)measurement.Correctness.Scope);
            metrics.Add(OutputAbsoluteErrorMetric, measurement.Correctness.OutputAbsoluteError);
            metrics.Add(OutputRelativeErrorMetric, measurement.Correctness.OutputRelativeError);
            metrics.Add(OutputAbsoluteToleranceMetric, measurement.Correctness.OutputAbsoluteTolerance);
            metrics.Add(OutputRelativeToleranceMetric, measurement.Correctness.OutputRelativeTolerance);
            metrics.Add(GradientAbsoluteErrorMetric, measurement.Correctness.GradientAbsoluteError);
            metrics.Add(GradientRelativeErrorMetric, measurement.Correctness.GradientRelativeError);
            metrics.Add(GradientAbsoluteToleranceMetric, measurement.Correctness.GradientAbsoluteTolerance);
            metrics.Add(GradientRelativeToleranceMetric, measurement.Correctness.GradientRelativeTolerance);
            return new EvolutionTaskResult(
                EvolutionEvaluationStatus.Completed,
                measurement.ThroughputGflops,
                EvolutionOptimizationDirection.Maximize,
                descriptors,
                costUnits: measurement.Resources.CompileTime.TotalMilliseconds +
                           measurement.Timing.P95.TotalMilliseconds * measurement.Timing.SampleCount,
                metrics: metrics);
        }

        internal static KernelTuningMeasurement ReadMeasurement(EvolutionEvaluation evaluation)
        {
            IReadOnlyDictionary<string, double> metrics = evaluation.Metrics;
            double Metric(KernelTuningMetric metric) => metrics[KernelTuningMetricNames.Get(metric)];
            var timing = KernelTimingStatistics.FromSummary(
                ReadExactInt32(metrics, TimingSampleCountMetric),
                Metric(KernelTuningMetric.MedianLatencyMilliseconds),
                Metric(KernelTuningMetric.P95LatencyMilliseconds));
            var resources = new KernelTuningResourceUsage(
                ReadExactInt64(metrics, KernelTuningMetricNames.Get(KernelTuningMetric.WorkspaceBytes)),
                Metric(KernelTuningMetric.OccupancyRatio),
                ReadExactInt32(metrics, KernelTuningMetricNames.Get(KernelTuningMetric.RegistersPerThread)),
                TimeSpan.FromMilliseconds(Metric(KernelTuningMetric.CompileMilliseconds)),
                ReadExactInt32(metrics, KernelTuningMetricNames.Get(KernelTuningMetric.KernelLaunchCount)));
            var correctness = new KernelTuningCorrectnessEvidence(
                (KernelTuningValidationScope)ReadExactInt32(metrics, ValidationScopeMetric),
                metrics[OutputAbsoluteErrorMetric],
                metrics[OutputRelativeErrorMetric],
                metrics[OutputAbsoluteToleranceMetric],
                metrics[OutputRelativeToleranceMetric],
                metrics[GradientAbsoluteErrorMetric],
                metrics[GradientRelativeErrorMetric],
                metrics[GradientAbsoluteToleranceMetric],
                metrics[GradientRelativeToleranceMetric]);
            return new KernelTuningMeasurement(
                Metric(KernelTuningMetric.ThroughputGflops), timing, resources, correctness);
        }

        private static int ReadExactInt32(
            IReadOnlyDictionary<string, double> metrics,
            string name)
        {
            double value = metrics[name];
            if (!KernelTuningMeasurement.IsFinite(value) || value != Math.Truncate(value))
                throw new InvalidDataException($"Kernel metric '{name}' must be a finite integer.");
            try
            {
                return checked((int)value);
            }
            catch (OverflowException exception)
            {
                throw new InvalidDataException($"Kernel metric '{name}' is outside the Int32 range.", exception);
            }
        }

        private static long ReadExactInt64(
            IReadOnlyDictionary<string, double> metrics,
            string name)
        {
            double value = metrics[name];
            if (!KernelTuningMeasurement.IsFinite(value) || value != Math.Truncate(value))
                throw new InvalidDataException($"Kernel metric '{name}' must be a finite integer.");
            try
            {
                return checked((long)value);
            }
            catch (OverflowException exception)
            {
                throw new InvalidDataException($"Kernel metric '{name}' is outside the Int64 range.", exception);
            }
        }

        private static EvolutionTaskResult FailureResult(KernelTuningTrialResult trial)
        {
            EvolutionEvaluationStatus status = trial.Status == KernelTuningTrialStatus.BenchmarkFailed
                ? EvolutionEvaluationStatus.Failed
                : EvolutionEvaluationStatus.Rejected;
            return new EvolutionTaskResult(
                status,
                diagnostics: new[]
                {
                    new EvolutionDiagnostic(
                        FailureCode(trial.Status),
                        string.IsNullOrWhiteSpace(trial.Diagnostic)
                            ? "The candidate did not pass the typed kernel evaluation gate."
                            : trial.Diagnostic)
                });
        }

        private static string FailureCode(KernelTuningTrialStatus status) => status switch
        {
            KernelTuningTrialStatus.InvalidConfiguration => "invalid_configuration",
            KernelTuningTrialStatus.ResourceLimitExceeded => "resource_limit_exceeded",
            KernelTuningTrialStatus.CompilationFailed => "compilation_failed",
            KernelTuningTrialStatus.OutputMismatch => "output_mismatch",
            KernelTuningTrialStatus.GradientMismatch => "gradient_mismatch",
            KernelTuningTrialStatus.BenchmarkFailed => "benchmark_failed",
            _ => throw new ArgumentOutOfRangeException(nameof(status))
        };
    }
}
