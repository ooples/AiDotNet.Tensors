using System.Linq;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>Annealing shape of the second phase of <see cref="LrSchedule.OneCycleWithAnneal"/>.</summary>
public enum OneCycleAnneal
{
    /// <summary>Half cosine from the peak to the final rate.</summary>
    Cosine,
    /// <summary>Linear from the peak to the final rate.</summary>
    Linear
}

/// <summary>Amplitude policy of <see cref="LrSchedule.Cyclic(double, double, int, int, CyclicLrMode, double)"/>.</summary>
public enum CyclicLrMode
{
    /// <summary>The same amplitude every cycle.</summary>
    Triangular,
    /// <summary>The amplitude halves every cycle.</summary>
    Triangular2,
    /// <summary>The amplitude scales by gamma^step.</summary>
    ExponentialRange
}

/// <summary>
/// A schedule stepped once per optimizer step from a starting rate: step 1 runs at <see cref="Initial"/>, step N at the
/// rate after N - 1 completed steps, never below <see cref="Floor"/>. This is the per-batch convention of every
/// AiDotNet learning-rate scheduler (the optimizer reads the scheduler's rate, then steps it), so each subclass is the
/// scheduler's formula operation for operation and the fused sequence is bit-identical to the eager one.
/// </summary>
internal abstract class CompletedStepLr : LrSchedule
{
    public sealed override double GetLr(int step) => step <= 1 ? Initial : System.Math.Max(Floor, RateAfter(step - 1));

    protected abstract double Initial { get; }

    protected virtual double Floor => 0.0;

    /// <summary>The rate after <paramref name="t"/> completed steps, before the floor.</summary>
    internal abstract double RateAfter(int t);

    internal abstract FusedLrScheduleCheckpoint Checkpoint();

    internal sealed override FusedLrScheduleCheckpoint? TryCaptureCheckpoint() => Checkpoint();

    protected static void Require(bool condition, string parameter, string message)
    {
        if (!condition) throw new ArgumentOutOfRangeException(parameter, message);
    }

    protected static void RequireRate(double rate, string parameter)
        => Require(!double.IsNaN(rate) && !double.IsInfinity(rate) && rate >= 0, parameter, "Rates must be finite and not negative.");

    /// <summary>
    /// Rebuilds a schedule of this family from its checkpoint through the public factory, so a restored schedule
    /// passes the same validation as a new one and runs the same code.
    /// </summary>
    internal static LrSchedule Restore(FusedLrScheduleKind kind, double[] d, int[] i)
    {
        try
        {
            return kind switch
            {
                FusedLrScheduleKind.WarmupStableDecay => new WarmupStableDecayLr(d[0], i[0], i[1], i[2], (StableDecayShape)i[3], d[1]),
                FusedLrScheduleKind.Polynomial => new PolynomialLr(d[0], i[0], d[1], d[2]),
                FusedLrScheduleKind.MultiStep => new MultiStepLr(d[0], i, d[1], d[2]),
                FusedLrScheduleKind.TriStage => new TriStageLr(d[0], i[0], i[1], i[2], d[1], d[2]),
                FusedLrScheduleKind.NoamHoldAnnealing => new NoamHoldAnnealingLr(d[0], i[0], i[1], d[1]),
                FusedLrScheduleKind.CosineWarmRestarts => new CosineWarmRestartsLr(d[0], i[0], i[1], d[1]),
                FusedLrScheduleKind.OneCycleWithAnneal => new OneCycleWithAnnealLr(d[0], i[0], d[1], d[2], d[3], (OneCycleAnneal)i[1]),
                FusedLrScheduleKind.CyclicRange => new CyclicRangeLr(d[0], d[1], i[0], i[1], (CyclicLrMode)i[2], d[2]),
                FusedLrScheduleKind.Sequential => SequentialLr.FromCheckpoint(d, i),
                _ => throw new System.IO.InvalidDataException($"Serialized LR schedule kind {kind} is not a completed-step schedule."),
            };
        }
        catch (Exception ex) when (ex is ArgumentException or IndexOutOfRangeException)
        {
            throw new System.IO.InvalidDataException($"Serialized LR schedule kind {kind} is invalid: {ex.Message}", ex);
        }
    }

    internal static bool IsFamilyKind(FusedLrScheduleKind kind) => kind >= FusedLrScheduleKind.WarmupStableDecay;
}

internal sealed class WarmupStableDecayLr : CompletedStepLr
{
    private readonly double _lrMax, _endLr;
    private readonly int _warmup, _decayStart, _decaySteps;
    private readonly StableDecayShape _shape;

    public WarmupStableDecayLr(double lrMax, int warmupSteps, int decayStartStep, int decaySteps, StableDecayShape shape, double endLr)
    {
        RequireRate(lrMax, nameof(lrMax));
        Require(warmupSteps >= 0, nameof(warmupSteps), "Warmup steps cannot be negative.");
        Require(decayStartStep >= warmupSteps, nameof(decayStartStep), "Decay cannot start before warmup ends.");
        Require(decaySteps >= 0, nameof(decaySteps), "Decay steps cannot be negative.");
        Require(Enum.IsDefined(typeof(StableDecayShape), shape), nameof(shape), "Undefined decay shape.");
        Require(endLr >= 0 && endLr <= lrMax, nameof(endLr), "End rate must lie in [0, lrMax].");
        _lrMax = lrMax; _warmup = warmupSteps; _decayStart = decayStartStep; _decaySteps = decaySteps; _shape = shape; _endLr = endLr;
    }

    protected override double Initial => RateAfter(0);

    internal override double RateAfter(int t)
    {
        if (t < _warmup) return _lrMax * t / _warmup;
        if (t < _decayStart) return _lrMax;
        if (_decaySteps == 0) return _endLr;
        double progress = System.Math.Min(1.0, (double)(t - _decayStart) / _decaySteps);
        double factor = _shape switch
        {
            StableDecayShape.Cosine => (1 + System.Math.Cos(System.Math.PI * progress)) / 2,
            StableDecayShape.OneMinusSqrt => 1 - System.Math.Sqrt(progress),
            _ => 1 - progress
        };
        return _endLr + (_lrMax - _endLr) * factor;
    }

    internal override FusedLrScheduleCheckpoint Checkpoint()
        => new(FusedLrScheduleKind.WarmupStableDecay, new[] { _lrMax, _endLr }, new[] { _warmup, _decayStart, _decaySteps, (int)_shape });
}

internal sealed class PolynomialLr : CompletedStepLr
{
    private readonly double _lr0, _power, _endLr;
    private readonly int _total;

    public PolynomialLr(double lr0, int totalSteps, double power, double endLr)
    {
        RequireRate(lr0, nameof(lr0));
        RequireRate(endLr, nameof(endLr));
        Require(totalSteps > 0, nameof(totalSteps), "Total steps must be positive.");
        Require(power > 0 && !double.IsInfinity(power), nameof(power), "Power must be positive and finite.");
        _lr0 = lr0; _total = totalSteps; _power = power; _endLr = endLr;
    }

    protected override double Initial => _lr0;
    protected override double Floor => _endLr;

    internal override double RateAfter(int t)
    {
        if (t >= _total) return _endLr;
        double progress = 1.0 - (double)t / _total;
        return (_lr0 - _endLr) * System.Math.Pow(progress, _power) + _endLr;
    }

    internal override FusedLrScheduleCheckpoint Checkpoint()
        => new(FusedLrScheduleKind.Polynomial, new[] { _lr0, _power, _endLr }, new[] { _total });
}

internal sealed class MultiStepLr : CompletedStepLr
{
    private readonly double _lr0, _gamma, _min;
    private readonly int[] _milestones;

    public MultiStepLr(double lr0, int[] milestones, double gamma, double minLr)
    {
        RequireRate(lr0, nameof(lr0));
        RequireRate(minLr, nameof(minLr));
        Require(milestones is { Length: > 0 }, nameof(milestones), "Milestones cannot be empty.");
        for (int k = 1; k < milestones!.Length; k++)
            Require(milestones[k] > milestones[k - 1], nameof(milestones), "Milestones must be strictly increasing.");
        Require(gamma > 0 && gamma <= 1, nameof(gamma), "Gamma must be in (0, 1].");
        _lr0 = lr0; _milestones = milestones.ToArray(); _gamma = gamma; _min = minLr;
    }

    protected override double Initial => _lr0;
    protected override double Floor => _min;

    internal override double RateAfter(int t)
    {
        int decayCount = 0;
        foreach (int m in _milestones) if (t >= m) decayCount++;
        return _lr0 * System.Math.Pow(_gamma, decayCount);
    }

    internal override FusedLrScheduleCheckpoint Checkpoint()
        => new(FusedLrScheduleKind.MultiStep, new[] { _lr0, _gamma, _min }, _milestones.ToArray());
}

internal sealed class TriStageLr : CompletedStepLr
{
    private readonly double _lr0, _min, _power;
    private readonly int _warmup, _hold, _total;

    public TriStageLr(double lr0, int warmupSteps, int holdSteps, int totalSteps, double minLr, double decayPower)
    {
        RequireRate(lr0, nameof(lr0));
        RequireRate(minLr, nameof(minLr));
        Require(warmupSteps >= 0, nameof(warmupSteps), "Warmup steps cannot be negative.");
        Require(holdSteps >= 0, nameof(holdSteps), "Hold steps cannot be negative.");
        Require(totalSteps > 0, nameof(totalSteps), "Total steps must be positive.");
        Require((long)warmupSteps + holdSteps <= totalSteps, nameof(holdSteps), "Warmup and hold exceed the run.");
        Require(decayPower > 0 && !double.IsInfinity(decayPower), nameof(decayPower), "Decay power must be positive and finite.");
        _lr0 = lr0; _warmup = warmupSteps; _hold = holdSteps; _total = totalSteps; _min = minLr; _power = decayPower;
    }

    protected override double Initial => System.Math.Max(_min, RateAfter(0));
    protected override double Floor => _min;

    internal override double RateAfter(int t)
    {
        if (t < _warmup) return _min + (_lr0 - _min) * (t + 1) / _warmup;
        int decayStart = _warmup + _hold;
        if (t < decayStart) return _lr0;
        int decaySteps = _total - decayStart;
        if (decaySteps <= 0 || t >= _total) return _min;
        double progress = (double)(t - decayStart) / decaySteps;
        double remaining = System.Math.Pow(1.0 - progress, _power);
        return _min + (_lr0 - _min) * remaining;
    }

    internal override FusedLrScheduleCheckpoint Checkpoint()
        => new(FusedLrScheduleKind.TriStage, new[] { _lr0, _min, _power }, new[] { _warmup, _hold, _total });
}

internal sealed class NoamHoldAnnealingLr : CompletedStepLr
{
    private readonly double _peak, _decayRate;
    private readonly int _warmup, _hold;

    public NoamHoldAnnealingLr(double peak, int warmupSteps, int holdSteps, double decayRate)
    {
        Require(!double.IsNaN(peak) && !double.IsInfinity(peak) && peak > 0, nameof(peak), "Peak rate must be finite and positive.");
        Require(warmupSteps > 0, nameof(warmupSteps), "Warmup steps must be positive.");
        Require(holdSteps >= 0, nameof(holdSteps), "Hold steps must be non-negative.");
        Require(!double.IsNaN(decayRate) && !double.IsInfinity(decayRate) && decayRate > 0, nameof(decayRate), "Decay rate must be finite and positive.");
        _peak = peak; _warmup = warmupSteps; _hold = holdSteps; _decayRate = decayRate;
    }

    protected override double Initial => RateAfter(0);

    internal override double RateAfter(int t)
    {
        double trainingStep = t + 1.0;
        if (trainingStep < _warmup) return _peak * trainingStep / _warmup;
        if (trainingStep < _warmup + _hold) return _peak;
        double decayStep = trainingStep - _hold;
        return _peak * System.Math.Pow(_warmup / decayStep, _decayRate);
    }

    internal override FusedLrScheduleCheckpoint Checkpoint()
        => new(FusedLrScheduleKind.NoamHoldAnnealing, new[] { _peak, _decayRate }, new[] { _warmup, _hold });
}

internal sealed class CosineWarmRestartsLr : CompletedStepLr
{
    private readonly double _lr0, _etaMin;
    private readonly int _t0, _tMult;

    public CosineWarmRestartsLr(double lr0, int t0, int tMult, double etaMin)
    {
        RequireRate(lr0, nameof(lr0));
        RequireRate(etaMin, nameof(etaMin));
        Require(t0 > 0, nameof(t0), "T_0 must be positive.");
        Require(tMult >= 1, nameof(tMult), "T_mult must be >= 1.");
        _lr0 = lr0; _t0 = t0; _tMult = tMult; _etaMin = etaMin;
    }

    protected override double Initial => _lr0;
    // The eager schedule returns its formula unfloored; etaMin + a non-negative term never rounds below etaMin, so
    // the floor is the identity here and only states the bound.
    protected override double Floor => _etaMin;

    internal override double RateAfter(int t)
    {
        int cycle = 0, period = _t0, accumulated = 0;
        while (accumulated + period <= t)
        {
            accumulated += period;
            cycle++;
            period = _t0 * (int)System.Math.Pow(_tMult, cycle);
        }
        int position = t - accumulated;
        int currentPeriod = _t0 * (int)System.Math.Pow(_tMult, cycle);
        double cosine = System.Math.Cos(System.Math.PI * position / currentPeriod);
        return _etaMin + 0.5 * (_lr0 - _etaMin) * (1 + cosine);
    }

    internal override FusedLrScheduleCheckpoint Checkpoint()
        => new(FusedLrScheduleKind.CosineWarmRestarts, new[] { _lr0, _etaMin }, new[] { _t0, _tMult });
}

internal sealed class OneCycleWithAnnealLr : CompletedStepLr
{
    private readonly double _lrMax, _pctStart, _divFactor, _finalDivFactor, _initialLr, _finalLr;
    private readonly int _total, _warmupSteps, _annealSteps;
    private readonly OneCycleAnneal _anneal;

    public OneCycleWithAnnealLr(double lrMax, int totalSteps, double pctStart, double divFactor, double finalDivFactor, OneCycleAnneal anneal)
    {
        Require(!double.IsNaN(lrMax) && !double.IsInfinity(lrMax) && lrMax > 0, nameof(lrMax), "Max rate must be finite and positive.");
        Require(totalSteps > 0, nameof(totalSteps), "Total steps must be positive.");
        Require(pctStart >= 0 && pctStart < 1, nameof(pctStart), "pctStart must be in [0, 1).");
        Require(divFactor > 0 && !double.IsInfinity(divFactor), nameof(divFactor), "divFactor must be positive.");
        Require(finalDivFactor > 0 && !double.IsInfinity(finalDivFactor), nameof(finalDivFactor), "finalDivFactor must be positive.");
        Require(Enum.IsDefined(typeof(OneCycleAnneal), anneal), nameof(anneal), "Undefined anneal shape.");
        _lrMax = lrMax; _total = totalSteps; _pctStart = pctStart; _divFactor = divFactor; _finalDivFactor = finalDivFactor; _anneal = anneal;
        _warmupSteps = (int)(totalSteps * pctStart);
        _annealSteps = totalSteps - _warmupSteps;
        _initialLr = lrMax / divFactor;
        _finalLr = _initialLr / finalDivFactor;
    }

    protected override double Initial => _initialLr;

    internal override double RateAfter(int t)
    {
        if (t >= _total) return _finalLr;
        if (t < _warmupSteps)
        {
            double progress = (double)t / _warmupSteps;
            return _initialLr + (_lrMax - _initialLr) * progress;
        }
        double annealProgress = (double)(t - _warmupSteps) / _annealSteps;
        if (_anneal == OneCycleAnneal.Cosine)
        {
            double cosineValue = (1 + System.Math.Cos(System.Math.PI * annealProgress)) / 2;
            return _finalLr + (_lrMax - _finalLr) * cosineValue;
        }
        return _lrMax - (_lrMax - _finalLr) * annealProgress;
    }

    internal override FusedLrScheduleCheckpoint Checkpoint()
        => new(FusedLrScheduleKind.OneCycleWithAnneal, new[] { _lrMax, _pctStart, _divFactor, _finalDivFactor }, new[] { _total, (int)_anneal });
}

internal sealed class CyclicRangeLr : CompletedStepLr
{
    private readonly double _base, _max, _gamma;
    private readonly int _up, _down;
    private readonly CyclicLrMode _mode;

    public CyclicRangeLr(double lrBase, double lrMax, int stepSizeUp, int stepSizeDown, CyclicLrMode mode, double gamma)
    {
        RequireRate(lrBase, nameof(lrBase));
        Require(!double.IsNaN(lrMax) && !double.IsInfinity(lrMax) && lrMax > lrBase, nameof(lrMax), "Max rate must exceed the base rate.");
        Require(stepSizeUp > 0, nameof(stepSizeUp), "Step size up must be positive.");
        Require(stepSizeDown > 0, nameof(stepSizeDown), "Step size down must be positive.");
        Require(Enum.IsDefined(typeof(CyclicLrMode), mode), nameof(mode), "Undefined cyclic mode.");
        Require(gamma > 0 && gamma <= 1, nameof(gamma), "Gamma must be in (0, 1].");
        _base = lrBase; _max = lrMax; _up = stepSizeUp; _down = stepSizeDown; _mode = mode; _gamma = gamma;
    }

    protected override double Initial => _base;

    internal override double RateAfter(int t)
    {
        int cycleLength = _up + _down;
        int cycle = t / cycleLength;
        int position = t % cycleLength;
        double scale = position < _up
            ? (double)position / _up
            : 1.0 - (double)(position - _up) / _down;
        scale = System.Math.Max(0.0, System.Math.Min(1.0, scale));
        double amplitude = _max - _base;
        if (_mode == CyclicLrMode.Triangular2) amplitude = amplitude / System.Math.Pow(2, cycle);
        else if (_mode == CyclicLrMode.ExponentialRange) amplitude = amplitude * System.Math.Pow(_gamma, t);
        double rate = _base + amplitude * scale;
        return System.Math.Max(_base, System.Math.Min(_max, rate));
    }

    internal override FusedLrScheduleCheckpoint Checkpoint()
        => new(FusedLrScheduleKind.CyclicRange, new[] { _base, _max, _gamma }, new[] { _up, _down, (int)_mode });
}

/// <summary>
/// Runs schedule k after milestone k - 1: the first schedule from the start, each later one from its own step 1 again,
/// as the eager sequential scheduler steps only its active child.
/// </summary>
internal sealed class SequentialLr : LrSchedule
{
    private readonly LrSchedule[] _schedules;
    private readonly int[] _milestones;

    public SequentialLr(LrSchedule[] schedules, int[] milestones)
    {
        if (schedules is null || schedules.Length == 0) throw new ArgumentOutOfRangeException(nameof(schedules), "Schedules cannot be empty.");
        if (schedules.Any(s => s is null)) throw new ArgumentOutOfRangeException(nameof(schedules), "Schedules cannot contain null.");
        if (milestones is null || milestones.Length != schedules.Length - 1)
            throw new ArgumentOutOfRangeException(nameof(milestones), "Milestones must number one fewer than the schedules.");
        for (int k = 1; k < milestones.Length; k++)
            if (milestones[k] <= milestones[k - 1])
                throw new ArgumentOutOfRangeException(nameof(milestones), "Milestones must be strictly increasing.");
        _schedules = schedules.ToArray();
        _milestones = milestones.ToArray();
    }

    public override double GetLr(int step)
    {
        int completed = step <= 1 ? 0 : step - 1;
        if (completed == 0) return _schedules[0].GetLr(1);
        int index = 0;
        while (index < _milestones.Length && completed > _milestones[index]) index++;
        int start = index == 0 ? 0 : _milestones[index - 1];
        return _schedules[index].GetLr(completed - start + 1);
    }

    // Layout: Ints = [count, milestones (count - 1), then per child: kind, doubles, ints, child ints],
    // Doubles = the children's doubles in order.
    internal override FusedLrScheduleCheckpoint? TryCaptureCheckpoint()
    {
        var ints = new System.Collections.Generic.List<int> { _schedules.Length };
        ints.AddRange(_milestones);
        var doubles = new System.Collections.Generic.List<double>();
        foreach (var child in _schedules)
        {
            var c = child.TryCaptureCheckpoint();
            if (c is null) return null;
            ints.Add((int)c.Kind); ints.Add(c.Doubles.Length); ints.Add(c.Ints.Length);
            ints.AddRange(c.Ints);
            doubles.AddRange(c.Doubles);
        }
        return new FusedLrScheduleCheckpoint(FusedLrScheduleKind.Sequential, doubles.ToArray(), ints.ToArray());
    }

    internal static SequentialLr FromCheckpoint(double[] d, int[] i)
    {
        int count = i[0];
        if (count <= 0 || count > i.Length) throw new ArgumentOutOfRangeException(nameof(i), "Bad sequential schedule count.");
        var milestones = i.Skip(1).Take(count - 1).ToArray();
        int ip = count, dp = 0;
        var children = new LrSchedule[count];
        for (int k = 0; k < count; k++)
        {
            var kind = (FusedLrScheduleKind)i[ip];
            int nd = i[ip + 1], ni = i[ip + 2];
            ip += 3;
            if (nd < 0 || ni < 0 || ip + ni > i.Length || dp + nd > d.Length)
                throw new ArgumentOutOfRangeException(nameof(i), "Sequential child payload overruns the checkpoint.");
            children[k] = new FusedLrScheduleCheckpoint(kind, d.Skip(dp).Take(nd).ToArray(), i.Skip(ip).Take(ni).ToArray()).ToSchedule();
            ip += ni; dp += nd;
        }
        if (ip != i.Length || dp != d.Length) throw new ArgumentOutOfRangeException(nameof(i), "Sequential checkpoint has trailing data.");
        return new SequentialLr(children, milestones);
    }
}
