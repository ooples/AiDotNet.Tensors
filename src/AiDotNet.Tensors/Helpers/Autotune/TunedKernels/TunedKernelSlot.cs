using System.Collections.Concurrent;
using System.Diagnostics;

namespace AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

/// <summary>
/// The dispatch seam for one op family on one device. Engine code builds a <see cref="TunedShape"/> for the call and
/// asks <see cref="Resolve"/> which candidate to run; the answer is cached per shape class, so the steady-state
/// cost is one dictionary probe.
/// </summary>
/// <remarks>
/// <para>Resolution order for an unseen shape: an override (<see cref="TunedKernelPolicy.OverrideFor"/>), then the
/// only applicable candidate, then a shipped or persisted profile entry, then — in <see cref="TunedKernelMode.Tune"/>
/// and while the process tuning budget lasts — the evidence gate, else the reference.</para>
/// <para>The evidence gate: every candidate is first run on the live arguments and compared against the reference's
/// output within the harness tolerance; only a correct candidate is timed. Timing is paired and interleaved on the
/// device clock (candidate and reference alternate, order flipping each pair), after a reference-versus-reference
/// control replay that measures the noise floor. A candidate becomes the default for the shape only when its
/// median within-pair speedup clears both 2% and the noise floor AND it won every pair but at most one. A candidate that
/// clears 2% by median but fails only the noise floor or the pair test leaves the measurement inconclusive: the slot
/// measures once more, and if it still cannot tell it serves the reference without persisting that verdict
/// (<see cref="TunedKernelDecisionReason.Inconclusive"/>). A burst of device noise during tuning used to persist a
/// genuine "reference won" for routes that win in every quiet run (measured: four Winograd convolutions on an RTX 3080
/// within eight seconds of one process, ~6% of a ResNet-18 step for every later process). One pair is
/// forgiven, in the noise floor too, because a single sample can stall for reasons unrelated to the kernel (a host
/// preemption, a display-driver scheduling hiccup on a desktop GPU): measured on an RTX 3080 under Windows, a kernel
/// 1.1-1.4x faster by median lost its decision to one pair at 0.1x in most attempts. Winning 8 of 9 pairs is still
/// a sign test at p ~ 0.02 against "no faster".</para>
/// <para>When numerical determinism is requested (<c>requireDeterministic</c> returns true) only candidates that
/// declare <see cref="ITunedKernelCandidate{TArgs}.IsDeterministic"/> are eligible, and those decisions are cached
/// separately from the non-deterministic ones.</para>
/// </remarks>
public sealed class TunedKernelSlot<TArgs> : ITunedKernelSlotInfo where TArgs : struct
{
    private const int ControlPairs = 7;
    private const int CandidatePairs = 9;
    private const double MinimumSpeedup = 1.02;
    private const double TargetSampleMilliseconds = 0.25;
    private const int MaximumRepetitions = 64;

    private readonly ITunedKernelHarness<TArgs> _harness;
    private readonly Func<bool> _requireDeterministic;
    private readonly object _sync = new();
    private readonly ConcurrentDictionary<TunedShape, Entry> _decisions = new();
    private readonly ConcurrentDictionary<TunedShape, Entry> _deterministicDecisions = new();
    private volatile ITunedKernelCandidate<TArgs>[] _candidates;
    private volatile string _poolKey;
    // Candidates that threw at dispatch after being chosen (an evicted module under capture, a failed pin): never chosen
    // again in this process, so a cached tuned choice cannot keep failing the engine op.
    private readonly HashSet<string> _failedCandidates = new(StringComparer.Ordinal);

    /// <summary>Creates a slot. The first candidate is the reference: the established, always-correct path.</summary>
    public TunedKernelSlot(TunedKernelOp op, string device, ITunedKernelHarness<TArgs> harness,
        Func<bool>? requireDeterministic, params ITunedKernelCandidate<TArgs>[] candidates)
    {
        if (string.IsNullOrWhiteSpace(device)) throw new ArgumentException("A device key is required.", nameof(device));
        if (candidates is null || candidates.Length == 0)
            throw new ArgumentException("At least the reference candidate is required.", nameof(candidates));
        Op = op;
        Device = device;
        _harness = harness ?? throw new ArgumentNullException(nameof(harness));
        _requireDeterministic = requireDeterministic ?? (() => false);
        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var c in candidates)
        {
            if (c is null) throw new ArgumentException("Null candidate.", nameof(candidates));
            if (!ids.Add(c.Id)) throw new ArgumentException("Duplicate candidate id " + c.Id, nameof(candidates));
        }
        _candidates = (ITunedKernelCandidate<TArgs>[])candidates.Clone();
        _poolKey = PoolKey(_candidates);
        TunedKernelRegistry.Track(this);
    }

    /// <inheritdoc />
    public TunedKernelOp Op { get; }

    /// <inheritdoc />
    public string Device { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> CandidateIds => _candidates.Select(c => c.Id).ToArray();

    /// <inheritdoc />
    public IReadOnlyList<TunedKernelDecision> CachedDecisions =>
        _decisions.Values.Concat(_deterministicDecisions.Values).Select(e => e.Decision).ToArray();

    /// <summary>
    /// Adds a candidate (generated, evolved or external). Cached decisions are dropped so every shape is
    /// re-resolved — and re-gated — with the new candidate in the pool. A duplicate id replaces the earlier one.
    /// </summary>
    public void AddCandidate(ITunedKernelCandidate<TArgs> candidate)
    {
        if (candidate is null) throw new ArgumentNullException(nameof(candidate));
        lock (_sync)
        {
            var list = _candidates.Where(c => !string.Equals(c.Id, candidate.Id, StringComparison.Ordinal)).ToList();
            if (list.Count == 0) throw new InvalidOperationException("The reference candidate cannot be replaced.");
            if (!string.Equals(list[0].Id, _candidates[0].Id, StringComparison.Ordinal))
                throw new InvalidOperationException("The reference candidate cannot be replaced.");
            list.Add(candidate);
            _candidates = list.ToArray();
            _poolKey = PoolKey(_candidates);
            _decisions.Clear();
            _deterministicDecisions.Clear();
        }
    }

    /// <summary>The registered candidate with this id, or null.</summary>
    public ITunedKernelCandidate<TArgs>? Candidate(string id) =>
        _candidates.FirstOrDefault(c => string.Equals(c.Id, id, StringComparison.Ordinal));

    /// <summary>Drops every cached decision (tests, or after a profile reload).</summary>
    public void InvalidateDecisions()
    {
        lock (_sync)
        {
            _decisions.Clear();
            _deterministicDecisions.Clear();
        }
    }

    /// <summary>
    /// Returns the candidate to run for <paramref name="shape"/>, or null when no registered candidate applies
    /// (the caller then runs its own fallback). <paramref name="args"/> is only used when the shape is being
    /// measured, in which case candidates run on it and the caller's subsequent execution of the returned
    /// candidate overwrites the outputs with the selected result.
    /// </summary>
    public ITunedKernelCandidate<TArgs>? Resolve(in TunedShape shape, in TArgs args)
    {
        bool deterministic = _requireDeterministic();
        var map = deterministic ? _deterministicDecisions : _decisions;
        if (map.TryGetValue(shape, out var hit)) return hit.Candidate;
        return ResolveSlow(shape, args, deterministic, map);
    }

    /// <summary>Resolves and runs. Returns false when no candidate applies.</summary>
    public bool TryExecute(in TunedShape shape, in TArgs args)
    {
        var c = Resolve(shape, args);
        if (c is null) return false;
        try
        {
            c.Execute(args);
            return true;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException && !IsReference(c))
        {
            // A tuned candidate failed where the reference would not: retire it, drop the decisions that chose it,
            // and let the caller run its own reference path for this call.
            RetireFailedCandidate(c, ex);
            return false;
        }
        catch (Engines.DirectGpu.GpuOutOfMemoryException ex)
        {
            // The reference itself ran out of device memory (an im2col reference's column buffer): the caller's own path
            // (a direct kernel with no scratch) still runs. Nothing is retired; a later call with memory free uses it.
            System.Diagnostics.Trace.TraceWarning($"Tuned kernel reference {c.Id} for {Op} ran out of device memory: {ex.Message}");
            return false;
        }
    }

    private bool IsReference(ITunedKernelCandidate<TArgs> candidate) =>
        string.Equals(candidate.Id, _candidates[0].Id, StringComparison.Ordinal);

    private void RetireFailedCandidate(ITunedKernelCandidate<TArgs> candidate, Exception ex)
    {
        lock (_sync)
        {
            _failedCandidates.Add(candidate.Id);
            foreach (var map in new[] { _decisions, _deterministicDecisions })
                foreach (var kv in map.ToArray())
                    if (ReferenceEquals(kv.Value.Candidate, candidate)) map.TryRemove(kv.Key, out _);
        }
        Trace.TraceWarning($"Tuned kernel {candidate.Id} for {Op} on {Device} failed at dispatch and is retired: {ex.GetType().Name}: {ex.Message}");
    }

    private ITunedKernelCandidate<TArgs>? ResolveSlow(in TunedShape shape, in TArgs args, bool deterministic,
        ConcurrentDictionary<TunedShape, Entry> map)
    {
        lock (_sync)
        {
            if (map.TryGetValue(shape, out var hit)) return hit.Candidate;

            var applicable = new List<ITunedKernelCandidate<TArgs>>();
            var rejected = new List<string>();
            foreach (var c in _candidates)
            {
                bool ok;
                try { ok = c.IsApplicable(shape); }
                catch (Exception ex) { ok = false; rejected.Add($"{c.Id}: applicability check threw {ex.GetType().Name}"); }
                if (!ok) continue;
                if (_failedCandidates.Contains(c.Id))
                {
                    rejected.Add($"{c.Id}: failed at dispatch earlier in this process");
                    continue;
                }
                if (deterministic && !c.IsDeterministic)
                {
                    rejected.Add($"{c.Id}: non-deterministic under deterministic mode");
                    continue;
                }
                applicable.Add(c);
            }
            if (applicable.Count == 0) return null;
            var reference = applicable[0];

            string? overrideId = TunedKernelPolicy.OverrideFor(Op);
            if (overrideId is not null)
            {
                var named = applicable.FirstOrDefault(c => string.Equals(c.Id, overrideId, StringComparison.Ordinal));
                if (named is not null)
                    return Decide(map, shape, named, TunedKernelDecisionReason.Override, reference, rejected);
                rejected.Add($"override {overrideId}: not registered or not applicable");
            }

            if (applicable.Count == 1)
                return Decide(map, shape, reference, TunedKernelDecisionReason.OnlyCandidate, reference, rejected);

            if (TunedKernelProfiles.TryLookup(Op, Device, shape, deterministic, out string? profiled, _poolKey) &&
                profiled is not null)
            {
                var named = applicable.FirstOrDefault(c => string.Equals(c.Id, profiled, StringComparison.Ordinal));
                if (named is not null)
                    return Decide(map, shape, named, TunedKernelDecisionReason.Profile, reference, rejected);
                rejected.Add($"profile {profiled}: not registered or not applicable");
            }

            if (TunedKernelPolicy.Mode != TunedKernelMode.Tune)
                return Decide(map, shape, reference, TunedKernelDecisionReason.Pinned, reference, rejected);

            // Measurement needs the device idle-able and outside capture; until then serve the reference
            // without caching, so the shape is tuned on the first eager call.
            if (!_harness.CanMeasure(args)) return reference;

            if (!TunedKernelPolicy.BudgetRemaining)
            {
                // An unmeasured shape gets the slot's shape-aware default when it matches the reference, not the reference
                // itself: the reference is the established, always-correct path, often the slowest (a direct convolution).
                if (TryUntunedDefault(shape, args, reference, applicable, rejected) is { } preferred)
                    return Decide(map, shape, preferred, TunedKernelDecisionReason.Heuristic, reference, rejected);
                return Decide(map, shape, reference, TunedKernelDecisionReason.BudgetExhausted, reference, rejected);
            }

            long start = Stopwatch.GetTimestamp();
            try
            {
                var (chosen, decision) = Gate(shape, args, reference, applicable, rejected);
                // An inconclusive measurement (noise, not a loss) gets one fresh measurement before it is accepted.
                if (decision?.Reason == TunedKernelDecisionReason.Inconclusive)
                {
                    var retryRejected = new List<string>(rejected.Count);
                    (chosen, decision) = Gate(shape, args, reference, applicable, retryRejected);
                }
                // No decision: the reference output was not finite, so nothing could be compared. Serve the reference
                // uncached and unpersisted, and tune this shape on a later call with usable data.
                if (decision is null) return chosen;
                map[shape] = new Entry(chosen, decision);
                TunedKernelRegistry.Record(decision);
                // Still inconclusive: the reference for this process, but a later process measures again.
                if (decision.Reason != TunedKernelDecisionReason.Inconclusive)
                    TunedKernelProfiles.Persist(decision, deterministic, _poolKey);
                return chosen;
            }
            finally
            {
                TunedKernelPolicy.Charge(Stopwatch.GetTimestamp() - start);
                TunedKernelRegistry.NoteMeasurement();
            }
        }
    }

    /// <summary>
    /// Candidate ids, most preferred first, for a shape the process has no budget left to measure: the first applicable
    /// one other than the reference is used if one run of it matches the reference within the harness tolerance. Null
    /// (the default) serves the reference, as before.
    /// </summary>
    internal Func<TunedShape, IReadOnlyList<string>?>? UntunedPreference { get; set; }

    private ITunedKernelCandidate<TArgs>? TryUntunedDefault(in TunedShape shape, in TArgs args,
        ITunedKernelCandidate<TArgs> reference, List<ITunedKernelCandidate<TArgs>> applicable, List<string> rejected)
    {
        var preference = UntunedPreference?.Invoke(shape);
        if (preference is null) return null;
        ITunedKernelCandidate<TArgs>? preferred = null;
        foreach (var id in preference)
        {
            preferred = applicable.FirstOrDefault(c => string.Equals(c.Id, id, StringComparison.Ordinal));
            if (preferred is not null) break;
        }
        if (preferred is null || ReferenceEquals(preferred, reference)) return null;
        try
        {
            using (Engines.DirectGpu.GpuLaunchProbe.TuningScope())
                reference.Execute(args);
            float[] expected = _harness.SnapshotOutput(args);
            double scale = 0;
            foreach (float v in expected)
            {
                double a = Math.Abs(v);
                if (double.IsNaN(a) || double.IsInfinity(a)) return null;   // nothing to compare against
                if (a > scale) scale = a;
            }
            if (scale <= 0) scale = 1;   // every |v| was zero
            _harness.PoisonOutput(args);
            using (Engines.DirectGpu.GpuLaunchProbe.TuningScope())
                preferred.Execute(args);
            double error = MaxRelativeError(expected, _harness.SnapshotOutput(args), scale);
            if (error <= _harness.RelativeTolerance) return preferred;
            rejected.Add($"{preferred.Id}: untuned default error {error:E2} exceeds {_harness.RelativeTolerance:E1}");
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            rejected.Add($"{preferred.Id}: untuned default threw {ex.GetType().Name}: {ex.Message}");
        }
        return null;
    }
    private ITunedKernelCandidate<TArgs> Decide(ConcurrentDictionary<TunedShape, Entry> map, in TunedShape shape,
        ITunedKernelCandidate<TArgs> chosen, TunedKernelDecisionReason reason, ITunedKernelCandidate<TArgs> reference,
        List<string> rejected)
    {
        var decision = new TunedKernelDecision(Op, Device, shape, chosen.Id, chosen.Origin, reason, reference.Id,
            rejected: rejected.ToArray());
        map[shape] = new Entry(chosen, decision);
        TunedKernelRegistry.Record(decision);
        return chosen;
    }

    private (ITunedKernelCandidate<TArgs> Chosen, TunedKernelDecision? Decision) Gate(in TunedShape shape, in TArgs args,
        ITunedKernelCandidate<TArgs> reference, List<ITunedKernelCandidate<TArgs>> applicable, List<string> rejected)
    {
        // The correctness runs below are the registry's evidence, like its snapshots and timings: TryExecute runs the
        // chosen candidate again for the caller's result. Their transfers (an im2col candidate's shape-table upload)
        // were charged to the op, so a residency check failed whenever a shape class was tuned in-process.
        using (Engines.DirectGpu.GpuLaunchProbe.TuningScope())
            reference.Execute(args);
        float[] expected = _harness.SnapshotOutput(args);
        double scale = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = Math.Abs(expected[i]);
            if (double.IsNaN(a) || double.IsInfinity(a))
            {
                // Not a decision: caching or persisting ReferenceWon here would pin the reference for this shape in
                // every later process because of one batch of non-finite data.
                return (reference, null);
            }
            if (a > scale) scale = a;
        }
        if (scale == 0) scale = 1;

        double one = _harness.MeasureMilliseconds(reference, args, 1);
        int reps = one <= 0 ? MaximumRepetitions
            : Math.Max(1, Math.Min(MaximumRepetitions, (int)Math.Ceiling(TargetSampleMilliseconds / one)));

        var control = new List<KernelTuningPairedSample>(ControlPairs);
        for (int i = 0; i < ControlPairs; i++)
            control.Add(new KernelTuningPairedSample(Span(_harness.MeasureMilliseconds(reference, args, reps)),
                Span(_harness.MeasureMilliseconds(reference, args, reps))));
        double noise = RobustNoiseRatio(control);

        ITunedKernelCandidate<TArgs> best = reference;
        KernelTuningPairedEvidence? bestEvidence = null;
        bool inconclusive = false;
        double bestError = 0;
        for (int k = 1; k < applicable.Count; k++)
        {
            var c = applicable[k];
            double error;
            try
            {
                _harness.PoisonOutput(args);
                using (Engines.DirectGpu.GpuLaunchProbe.TuningScope())
                    c.Execute(args);
                float[] actual = _harness.SnapshotOutput(args);
                error = MaxRelativeError(expected, actual, scale);
            }
            catch (Exception ex)
            {
                rejected.Add($"{c.Id}: threw {ex.GetType().Name}: {ex.Message}");
                continue;
            }
            if (!(error <= _harness.RelativeTolerance))
            {
                rejected.Add($"{c.Id}: error {error:E2} exceeds {_harness.RelativeTolerance:E1}");
                continue;
            }

            KernelTuningPairedEvidence evidence;
            bool remeasured = false;
        measure:
            try
            {
                if (!remeasured) _harness.MeasureMilliseconds(c, args, reps); // warm caches, module loads and workspaces
                var pairs = new List<KernelTuningPairedSample>(CandidatePairs);
                for (int i = 0; i < CandidatePairs; i++)
                {
                    double tc, tr;
                    if ((i & 1) == 0)
                    {
                        tc = _harness.MeasureMilliseconds(c, args, reps);
                        tr = _harness.MeasureMilliseconds(reference, args, reps);
                    }
                    else
                    {
                        tr = _harness.MeasureMilliseconds(reference, args, reps);
                        tc = _harness.MeasureMilliseconds(c, args, reps);
                    }
                    pairs.Add(new KernelTuningPairedSample(Span(tc), Span(tr)));
                }
                evidence = new KernelTuningPairedEvidence(pairs, noise);
            }
            catch (Exception ex)
            {
                rejected.Add($"{c.Id}: timing threw {ex.GetType().Name}: {ex.Message}");
                continue;
            }

            double secondLowest = SecondLowestSpeedup(evidence);
            bool wins = evidence.MedianSpeedup >= Math.Max(MinimumSpeedup, noise) && secondLowest > 1.0;
            // Faster by a median that clears the bar but refused for two lost pairs: a burst of device noise during its
            // nine pairs. Measure it once more before refusing it, or a slower candidate that happened to measure
            // cleanly wins and is persisted (ResNet-18's 512@4 weight gradient tuned r8 at 1.25x over the 1.6x
            // Winograd route about one run in four, 28 us a step).
            if (!wins && !remeasured && evidence.MedianSpeedup >= Math.Max(MinimumSpeedup, noise))
            {
                remeasured = true;
                rejected.Add($"{c.Id}: re-measured after {secondLowest:F3}x second-lowest pair at {evidence.MedianSpeedup:F3}x median");
                goto measure;
            }
            if (!wins)
            {
                // Faster by median, refused only for consistency or for the noise floor: undecided, not a loss.
                if (evidence.MedianSpeedup >= MinimumSpeedup) inconclusive = true;
                rejected.Add($"{c.Id}: {evidence.MedianSpeedup:F3}x median, {evidence.LowerSpeedupBound:F3}x min pair, " +
                             $"{secondLowest:F3}x second-lowest pair (noise {noise:F3})");
                continue;
            }
            if (bestEvidence is null || evidence.MedianSpeedup > bestEvidence.MedianSpeedup)
            {
                if (bestEvidence is not null)
                    rejected.Add($"{best.Id}: passed at {bestEvidence.MedianSpeedup:F3}x but a faster candidate won");
                best = c;
                bestEvidence = evidence;
                bestError = error;
            }
            else
            {
                rejected.Add($"{c.Id}: passed at {evidence.MedianSpeedup:F3}x but a faster candidate won");
            }
        }

        if (bestEvidence is null)
            return (reference, new TunedKernelDecision(Op, Device, shape, reference.Id, reference.Origin,
                inconclusive ? TunedKernelDecisionReason.Inconclusive : TunedKernelDecisionReason.ReferenceWon,
                reference.Id, noiseRatio: noise, rejected: rejected.ToArray()));

        return (best, new TunedKernelDecision(Op, Device, shape, best.Id, best.Origin, TunedKernelDecisionReason.Tuned,
            reference.Id,
            referenceMs: bestEvidence.IncumbentTiming.Median.TotalMilliseconds / reps,
            candidateMs: bestEvidence.CandidateTiming.Median.TotalMilliseconds / reps,
            medianSpeedup: bestEvidence.MedianSpeedup, lowerSpeedup: bestEvidence.LowerSpeedupBound,
            noiseRatio: noise, maxRelativeError: bestError, rejected: rejected.ToArray()));
    }

    /// <summary>The second-lowest within-pair speedup: above one when the candidate won every pair but at most one.</summary>
    internal static double SecondLowestSpeedup(KernelTuningPairedEvidence evidence)
    {
        var speedups = evidence.Samples.Select(s => s.Speedup).OrderBy(v => v).ToArray();
        return speedups.Length > 1 ? speedups[1] : speedups.Length == 1 ? speedups[0] : 0.0;   // no pairs: never a win
    }

    /// <summary>
    /// The control replay's noise floor with its single worst pair dropped: the largest remaining reference-versus-
    /// reference deviation (max(r, 1/r)). One stalled control sample otherwise sets the floor (a 10x stall made every
    /// candidate's median fail it).
    /// </summary>
    internal static double RobustNoiseRatio(IReadOnlyList<KernelTuningPairedSample> control)
    {
        var deviations = control.Select(s => Math.Max(s.Speedup, 1.0 / s.Speedup)).OrderBy(v => v).ToArray();
        return deviations.Length > 1 ? deviations[deviations.Length - 2] : deviations.Length == 1 ? deviations[0] : 1.0;
    }

    // 32-bit FNV-1a.
    private const uint FnvOffsetBasis = 2166136261;
    private const uint FnvPrime = 16777619;

    // FNV-1a over the ordered candidate ids, as 8 hex digits: a short, stable name for the candidate set. It holds across
    // package upgrades on purpose: keying it by build (tried in #1432) discarded every persisted decision on each upgrade,
    // and until the tuning budget re-covered every shape over several processes the uncovered ones served the reference
    // (ResNet-50 predict on OpenCL: 8 ms with the kept decisions, 184 ms in the first process after a rebuild).
    private static string PoolKey(ITunedKernelCandidate<TArgs>[] candidates)
    {
        uint h = FnvOffsetBasis;
        foreach (var c in candidates)
        {
            foreach (char ch in c.Id) { h ^= ch; h *= FnvPrime; }
            h ^= '|'; h *= FnvPrime;
        }
        return h.ToString("x8", System.Globalization.CultureInfo.InvariantCulture);
    }
    internal static double MaxRelativeError(float[] expected, float[] actual, double scale)
    {
        if (actual.Length != expected.Length) return double.PositiveInfinity;
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs((double)actual[i] - expected[i]);
            if (double.IsNaN(d) || double.IsInfinity(d)) return double.PositiveInfinity;
            if (d > worst) worst = d;
        }
        return worst / scale;
    }

    // TimeSpan.FromMilliseconds rounds to whole milliseconds on .NET Framework; build from ticks instead.
    private static TimeSpan Span(double milliseconds) =>
        TimeSpan.FromTicks(Math.Max(1L, (long)Math.Round(milliseconds * TimeSpan.TicksPerMillisecond)));

    private sealed class Entry
    {
        internal Entry(ITunedKernelCandidate<TArgs> candidate, TunedKernelDecision decision)
        {
            Candidate = candidate;
            Decision = decision;
        }

        internal ITunedKernelCandidate<TArgs> Candidate { get; }
        internal TunedKernelDecision Decision { get; }
    }
}
