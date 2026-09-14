using System.Text.Json;
using AiDotNet.Evolution;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>Application-owned applicability checks, coalesced bounded retuning, and monitored quarantine/rollback.</summary>
/// <remarks>
/// Call selection at controlled batch boundaries, not inside a kernel: quarantine admission may read the journal.
/// Applications must dispatch through this controller to enforce its envelope. In-flight operations are not revoked.
/// Retuning is queued by selection but drained explicitly through an idle gate, never launched by a hidden timer.
/// All allowances are controller-lifetime, not a distributed monetary budget. Use a private tuner/registry per envelope.
/// </remarks>
public sealed class KernelTuningLifecycle<TConfiguration> where TConfiguration : notnull
{
    private readonly object _gate = new();
    private readonly SemaphoreSlim _operation = new(1, 1);
    private readonly KernelTuningArtifactRegistry<TConfiguration> _registry;
    private readonly IEvolutionGenomeCodec<TConfiguration> _codec;
    private readonly KernelTuningLifecyclePolicy _policy;
    private readonly KernelTuningArtifactPromotionPolicy _promotion;
    private readonly Func<KernelTuningApplicabilityEnvelope, TConfiguration> _fallback;
    private EvolutionKernelAutotuner<TConfiguration> _tuner;
    private KernelTuningApplicabilityEnvelope _envelope;
    private KernelTuningApplicabilityEnvelope? _pending;
    private string _observedKey;
    private long _epoch;
    private int _admitted, _breaches;
    private DateTimeOffset? _lastAdmission, _lastObservation;
    private KernelTuningDeploymentSnapshot<TConfiguration>? _monitored;
    private readonly List<(DateTimeOffset At, long[] Ticks)> _violatingWindows = new();
    private bool _evidenceFailure;

    /// <summary>Creates an opt-in controller with an application-supplied valid fallback for each observed environment.</summary>
    public KernelTuningLifecycle(EvolutionKernelAutotuner<TConfiguration> tuner,
        KernelTuningApplicabilityEnvelope envelope, KernelTuningArtifactRegistry<TConfiguration> registry,
        IEvolutionGenomeCodec<TConfiguration> codec, KernelTuningLifecyclePolicy policy,
        KernelTuningArtifactPromotionPolicy promotion,
        Func<KernelTuningApplicabilityEnvelope, TConfiguration> knownValidFallback)
    {
        _tuner = tuner ?? throw new ArgumentNullException(nameof(tuner));
        _envelope = envelope ?? throw new ArgumentNullException(nameof(envelope));
        if (tuner.Identity.StableKey != envelope.Identity.StableKey) throw new ArgumentException("Tuner identity differs from its envelope.");
        _registry = registry ?? throw new ArgumentNullException(nameof(registry));
        _codec = codec ?? throw new ArgumentNullException(nameof(codec));
        _policy = policy ?? throw new ArgumentNullException(nameof(policy));
        _promotion = promotion ?? throw new ArgumentNullException(nameof(promotion));
        _fallback = knownValidFallback ?? throw new ArgumentNullException(nameof(knownValidFallback));
        _observedKey = envelope.StableKey;
    }

    /// <summary>Gets lifetime retune admissions, including failed, canceled and abandoned work.</summary>
    public int AdmittedRetunes { get { lock (_gate) return _admitted; } }
    /// <summary>Gets whether the single coalesced request slot contains work.</summary>
    public bool RetuneRequested { get { lock (_gate) return _pending is not null; } }

    /// <summary>Checks exact applicability and current quarantine admission, otherwise returns the valid fallback and queues bounded work.</summary>
    public TConfiguration Select(KernelTuningApplicabilityEnvelope observed)
    {
        if (observed is null) throw new ArgumentNullException(nameof(observed));
        EvolutionKernelAutotuner<TConfiguration> tuner;
        long epoch;
        bool compatible;
        lock (_gate)
        {
            if (_observedKey != observed.StableKey) { _observedKey = observed.StableKey; _epoch++; }
            epoch = _epoch;
            tuner = _tuner;
            compatible = !_evidenceFailure && _envelope.StableKey == observed.StableKey;
        }
        // Do not run caller codecs, fallback functions or disk admission checks while holding the state lock.
        bool admitted = compatible && tuner.TryHydrate();
        var active = tuner.Deployment.Current;
        lock (_gate)
        {
            if (_epoch == epoch && admitted && active is not null && ReferenceEquals(tuner, _tuner))
            { _pending = null; return active.Configuration; }
            if (_epoch == epoch && !_evidenceFailure) _pending = observed;
        }
        return _fallback(observed) ?? throw new InvalidOperationException("The application supplied no valid fallback.");
    }

    /// <summary>Drains at most one request using explicit idle admission and verified search caps.</summary>
    /// <remarks>
    /// The factory must return a private tuner for the supplied envelope and must honor the supplied options.
    /// Search evaluation/proposal limits are checked on the returned tuner. Finalist replay is one separate provider
    /// operation with its own measurement limits. Timeout is cooperative; abandoned work keeps the only work slot
    /// until it settles and can never be adopted by this controller after the caller stopped waiting.
    /// </remarks>
    public async Task<KernelTuningRetuneStatus> RetunePendingAsync(
        Func<KernelTuningApplicabilityEnvelope, EvolutionEngineOptions, EvolutionKernelAutotuner<TConfiguration>> factory,
        IReadOnlyList<TConfiguration> seeds, IKernelTuningIdleGate idleGate,
        CancellationToken cancellationToken = default)
    {
        if (factory is null) throw new ArgumentNullException(nameof(factory));
        if (seeds is null || seeds.Count == 0 || seeds.Count > _policy.ProposalsPerRetune)
            throw new ArgumentException("Retuning requires a bounded nonempty seed set.", nameof(seeds));
        if (idleGate is null) throw new ArgumentNullException(nameof(idleGate));
        cancellationToken.ThrowIfCancellationRequested();
        if (!await _operation.WaitAsync(0, cancellationToken).ConfigureAwait(false)) return KernelTuningRetuneStatus.Busy;
        bool deferredRelease = false;
        var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        Task<(EvolutionKernelAutotuner<TConfiguration> Tuner, EvolutionKernelTuningResult<TConfiguration> Result)>? work = null;
        try
        {
            KernelTuningApplicabilityEnvelope requested;
            long epoch;
            lock (_gate)
            {
                if (_pending is null || _evidenceFailure) return KernelTuningRetuneStatus.NotRequested;
                var now = DateTimeOffset.UtcNow;
                if (_admitted >= _policy.MaximumRetunes || _lastAdmission is { } last && now - last < _policy.Cooldown)
                    return KernelTuningRetuneStatus.BudgetDenied;
                requested = _pending;
                _pending = null;
                epoch = _epoch;
                _admitted++;
                _lastAdmission = now;
            }
            var seedCopy = seeds.ToArray();
            deadline.CancelAfter(_policy.Timeout);
            work = Task.Run(async () =>
            {
                var options = new EvolutionEngineOptions
                {
                    RunId = "retune-" + Guid.NewGuid().ToString("N"), MaxEvaluationAttempts = _policy.EvaluationsPerRetune,
                    MaxProposals = _policy.ProposalsPerRetune, MaxGenerations = _policy.ProposalsPerRetune,
                    MaxDegreeOfParallelism = 1, ProposalBatchSize = 1, TimeLimit = _policy.Timeout,
                    EvaluationTimeout = _policy.Timeout, EvaluationGracePeriod = _policy.GracePeriod
                };
                var next = factory(requested, options) ?? throw new InvalidOperationException("Retuning factory returned no tuner.");
                lock (_gate)
                    if (ReferenceEquals(next, _tuner) || ReferenceEquals(next.Deployment, _tuner.Deployment))
                        throw new InvalidOperationException("Retuning requires a private deployment handle.");
                if (next.Identity.StableKey != requested.Identity.StableKey ||
                    next.MaximumEvaluationAttempts > _policy.EvaluationsPerRetune || next.MaximumProposals > _policy.ProposalsPerRetune)
                    throw new InvalidOperationException("Retuning factory changed applicability or exceeded the admitted search budget.");
                var result = await next.TuneInBackgroundAsync(seedCopy, idleGate, deadline.Token).ConfigureAwait(false);
                return (next, result);
            }, deadline.Token);
            using var wait = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            wait.CancelAfter(_policy.Timeout + _policy.GracePeriod);
            try
            {
                if (await Task.WhenAny(work, Task.Delay(System.Threading.Timeout.Infinite, wait.Token)).ConfigureAwait(false) != work)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    return KernelTuningRetuneStatus.Abandoned;
                }
                var completed = await work.ConfigureAwait(false);
                deadline.Token.ThrowIfCancellationRequested();
                var approval = _promotion.Options.SnapshotAndValidate(requested.Identity.Device.Kind);
                var artifact = _registry.Register(completed.Result.ActiveDeployment, requested, _codec);
                if (completed.Result.WasPromoted && !approval.QualifiesForPromotion(completed.Result.ProposedWinner.PromotionEvidence) ||
                    _promotion.RequireDurableArtifact && !artifact.IsDurable) return KernelTuningRetuneStatus.Rejected;
                deadline.Token.ThrowIfCancellationRequested();
                lock (_gate)
                {
                    if (_epoch != epoch || _evidenceFailure) return KernelTuningRetuneStatus.Stale;
                    _tuner = completed.Tuner;
                    _envelope = requested;
                    _pending = null;
                    _monitored = null;
                    _breaches = 0;
                    _violatingWindows.Clear();
                    return KernelTuningRetuneStatus.Completed;
                }
            }
            finally { wait.Cancel(); }
        }
        finally
        {
            if (work is { IsCompleted: false } pending)
            {
                deferredRelease = true;
                _ = pending.ContinueWith(finished => { _ = finished.Exception; deadline.Dispose(); _operation.Release(); },
                    CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
            }
            if (!deferredRelease) { _ = work?.Exception; deadline.Dispose(); _operation.Release(); }
        }
    }

    /// <summary>Measures a predeclared raw-timing window and quarantines after the configured consecutive P95 breaches.</summary>
    /// <remarks>Times must represent the exact observed deployment and its measurement units/scope. This is an application monitor, not a statistical superiority test.</remarks>
    public async Task<KernelTuningQuarantineResult<TConfiguration>?> ObserveLatencyAsync(
        KernelTuningDeploymentSnapshot<TConfiguration> observed, IReadOnlyList<TimeSpan> samples,
        DateTimeOffset observedAt, string? priorArtifactId = null, CancellationToken cancellationToken = default)
    {
        if (observed is null) throw new ArgumentNullException(nameof(observed));
        if (samples is null || samples.Count != _policy.MonitoringSamples) throw new ArgumentException("Unexpected monitoring window size.", nameof(samples));
        if (observedAt == default) throw new ArgumentOutOfRangeException(nameof(observedAt));
        TimeSpan[] raw = samples.ToArray();
        var timing = KernelTimingStatistics.FromSamples(raw);
        double threshold = observed.Measurement.Timing.P95.TotalMilliseconds * _policy.MaximumP95RegressionRatio;
        if (!KernelTuningMeasurement.IsFinite(threshold))
            throw new ArgumentException("Regression threshold exceeds the supported measurement range.", nameof(observed));
        await _operation.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            EvolutionKernelAutotuner<TConfiguration> tuner;
            KernelTuningApplicabilityEnvelope envelope;
            lock (_gate)
            {
                tuner = _tuner;
                envelope = _envelope;
                if (!ReferenceEquals(tuner.Deployment.Current, observed) || _observedKey != envelope.StableKey) return null;
                if (!tuner.SupportsQuarantine) throw new InvalidOperationException("Monitored rollback requires a quarantined store.");
                if (!ReferenceEquals(_monitored, observed))
                { _monitored = observed; _breaches = 0; _lastObservation = null; _violatingWindows.Clear(); }
                if (_lastObservation is { } last && observedAt <= last) throw new ArgumentException("Monitoring windows must advance in time.", nameof(observedAt));
                _lastObservation = observedAt;
                if (timing.P95.TotalMilliseconds <= threshold)
                { _breaches = 0; _violatingWindows.Clear(); return null; }
                if (_violatingWindows.Count == _policy.ConsecutiveBreaches) _violatingWindows.RemoveAt(0);
                _violatingWindows.Add((observedAt.ToUniversalTime(), raw.Select(value => value.Ticks).ToArray()));
                if (++_breaches < _policy.ConsecutiveBreaches) return null;
            }
            byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(new
            {
                SchemaVersion = 1, Envelope = envelope.StableKey, observed.GenomeId, observed.RunStateHash,
                ObservedAtUtc = observedAt.ToUniversalTime(),
                Windows = _violatingWindows.Select(window => new { ObservedAtUtc = window.At, SampleTicks = window.Ticks }).ToArray(),
                RequestedPriorArtifactId = priorArtifactId is { Length: > 64 } ? "[invalid digest]" : priorArtifactId,
                BaselineP95Ticks = observed.Measurement.Timing.P95.Ticks, _policy.MaximumP95RegressionRatio,
                _policy.ConsecutiveBreaches, _policy.MonitoringSamples
            });
            KernelTuningArtifactReceipt receipt;
            try { receipt = _registry.RetainEvidence(bytes); }
            catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException)
            {
                tuner.Deployment.TryDeactivate(observed);
                lock (_gate) _evidenceFailure = true; // No reload/retune after evidence loss; operator intervention is required.
                throw;
            }
            var evidence = new KernelTuningRegressionEvidence(KernelTuningRegressionReason.Latency,
                "kernel-lifecycle-p95-ms-v1", receipt.ArtifactId, timing.P95.TotalMilliseconds,
                threshold, observedAt);
            KernelTuningDeploymentSnapshot<TConfiguration>? prior = null;
            if (priorArtifactId is not null)
            {
                try { prior = _registry.Load(priorArtifactId, envelope, _codec); }
                catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or JsonException)
                { /* An unavailable prior must not prevent quarantine of the observed regression. */ }
            }
            var result = await tuner.QuarantineAsync(observed, evidence, prior, cancellationToken).ConfigureAwait(false);
            if (result.WasApplied && result.RollbackDeployment is null)
                lock (_gate) { if (_observedKey == envelope.StableKey) _pending = envelope; }
            return result;
        }
        finally { _operation.Release(); }
    }
}
