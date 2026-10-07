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
/// median within-pair speedup clears both 2% and the noise floor AND it won every single pair.</para>
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
        c.Execute(args);
        return true;
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
                return Decide(map, shape, reference, TunedKernelDecisionReason.BudgetExhausted, reference, rejected);

            long start = Stopwatch.GetTimestamp();
            try
            {
                var decision = Gate(shape, args, reference, applicable, rejected);
                map[shape] = new Entry(decision.Item1, decision.Item2);
                TunedKernelRegistry.Record(decision.Item2);
                TunedKernelProfiles.Persist(decision.Item2, deterministic, _poolKey);
                return decision.Item1;
            }
            finally
            {
                TunedKernelPolicy.Charge(Stopwatch.GetTimestamp() - start);
            }
        }
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

    private (ITunedKernelCandidate<TArgs>, TunedKernelDecision) Gate(in TunedShape shape, in TArgs args,
        ITunedKernelCandidate<TArgs> reference, List<ITunedKernelCandidate<TArgs>> applicable, List<string> rejected)
    {
        reference.Execute(args);
        float[] expected = _harness.SnapshotOutput(args);
        double scale = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = Math.Abs(expected[i]);
            if (double.IsNaN(a) || double.IsInfinity(a))
            {
                rejected.Add("reference produced a non-finite output; the shape is not tuned");
                return (reference, new TunedKernelDecision(Op, Device, shape, reference.Id, reference.Origin,
                    TunedKernelDecisionReason.ReferenceWon, reference.Id, rejected: rejected.ToArray()));
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
        double noise = KernelTuningPairedEvidence.SymmetricNoiseRatio(control);

        ITunedKernelCandidate<TArgs> best = reference;
        KernelTuningPairedEvidence? bestEvidence = null;
        double bestError = 0;
        for (int k = 1; k < applicable.Count; k++)
        {
            var c = applicable[k];
            double error;
            try
            {
                _harness.PoisonOutput(args);
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
            try
            {
                _harness.MeasureMilliseconds(c, args, reps); // warm caches, module loads and workspaces
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

            bool wins = evidence.MedianSpeedup >= Math.Max(MinimumSpeedup, noise) && evidence.LowerSpeedupBound > 1.0;
            if (!wins)
            {
                rejected.Add($"{c.Id}: {evidence.MedianSpeedup:F3}x median, {evidence.LowerSpeedupBound:F3}x min pair (noise {noise:F3})");
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
                TunedKernelDecisionReason.ReferenceWon, reference.Id, noiseRatio: noise, rejected: rejected.ToArray()));

        return (best, new TunedKernelDecision(Op, Device, shape, best.Id, best.Origin, TunedKernelDecisionReason.Tuned,
            reference.Id,
            referenceMs: bestEvidence.IncumbentTiming.Median.TotalMilliseconds / reps,
            candidateMs: bestEvidence.CandidateTiming.Median.TotalMilliseconds / reps,
            medianSpeedup: bestEvidence.MedianSpeedup, lowerSpeedup: bestEvidence.LowerSpeedupBound,
            noiseRatio: noise, maxRelativeError: bestError, rejected: rejected.ToArray()));
    }

    // FNV-1a over the ordered candidate ids, as 8 hex digits: a short, stable name for the candidate set.
    private static string PoolKey(ITunedKernelCandidate<TArgs>[] candidates)
    {
        uint h = 2166136261;
        foreach (var c in candidates)
        {
            foreach (char ch in c.Id) { h ^= ch; h *= 16777619; }
            h ^= '|'; h *= 16777619;
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
