using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;

namespace AiDotNet.Tensors.Engines.Optimization.Schedulers;

/// <summary>Cyclic-LR mode: triangular wave (Smith, 2017).</summary>
public enum CyclicMode
{
    /// <summary>Plain triangular wave between base_lr and max_lr.</summary>
    Triangular,
    /// <summary>Triangular but max_lr halved every cycle.</summary>
    Triangular2,
    /// <summary>Triangular with exponential decay <c>γ^iteration</c>.</summary>
    ExpRange,
}

/// <summary>CyclicLr: cyclical LR schedule between <c>baseLr</c> and <c>maxLr</c> (Smith, 2017).</summary>
public sealed class CyclicLr : LrScheduler
{
    /// <summary>Lower bound of the cycle.</summary>
    public double BaseLr { get; }
    /// <summary>Upper bound of the cycle.</summary>
    public double MaxLr { get; }
    /// <summary>Iterations from base to max.</summary>
    public int StepSizeUp { get; }
    /// <summary>Iterations from max back to base.</summary>
    public int StepSizeDown { get; }
    /// <summary>Wave shape.</summary>
    public CyclicMode Mode { get; }
    /// <summary>Decay base for <see cref="CyclicMode.ExpRange"/>.</summary>
    public double Gamma { get; }

    /// <summary>Build a CyclicLr scheduler.</summary>
    public CyclicLr(IOptimizer optimizer,
                    double baseLr, double maxLr,
                    int stepSizeUp = 2000, int? stepSizeDown = null,
                    CyclicMode mode = CyclicMode.Triangular,
                    double gamma = 1.0,
                    int lastEpoch = -1)
        : base(optimizer, lastEpoch)
    {
        if (stepSizeUp <= 0) throw new ArgumentOutOfRangeException(nameof(stepSizeUp));
        if (stepSizeDown.HasValue && stepSizeDown.Value <= 0)
            throw new ArgumentOutOfRangeException(nameof(stepSizeDown));
        BaseLr = baseLr; MaxLr = maxLr;
        StepSizeUp = stepSizeUp;
        StepSizeDown = stepSizeDown ?? stepSizeUp;
        Mode = mode; Gamma = gamma;

        // Override base_lrs to baseLr (PyTorch lets users set base_lr per group; we keep parity by single value).
        for (int i = 0; i < BaseLrs.Length; i++) BaseLrs[i] = baseLr;
        ApplyInitialLrs();
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        int cycleSize = StepSizeUp + StepSizeDown;
        int cycle = LastEpoch / cycleSize;
        int x = LastEpoch - cycle * cycleSize;
        double scale;
        if (x <= StepSizeUp) scale = (double)x / StepSizeUp;
        else scale = 1.0 - (double)(x - StepSizeUp) / StepSizeDown;
        if (scale < 0.0) scale = 0.0;

        double amplitude = MaxLr - BaseLr;
        double cycleScale = Mode switch
        {
            CyclicMode.Triangular  => 1.0,
            CyclicMode.Triangular2 => 1.0 / Math.Pow(2, cycle),
            CyclicMode.ExpRange    => Math.Pow(Gamma, LastEpoch),
            _ => 1.0,
        };
        double lr = BaseLr + amplitude * scale * cycleScale;
        var lrs = new double[BaseLrs.Length];
        for (int i = 0; i < lrs.Length; i++) lrs[i] = lr;
        return lrs;
    }
}

/// <summary>OneCycleLr: 1cycle policy (Smith &amp; Topin, 2017). Linear or cosine annealing variant.</summary>
public sealed class OneCycleLr : LrScheduler
{
    /// <summary>Peak LR.</summary>
    public double MaxLr { get; }
    /// <summary>Total iterations across the cycle.</summary>
    public int TotalSteps { get; }
    /// <summary>Fraction of total spent rising to max_lr.</summary>
    public double PctStart { get; }
    /// <summary>"linear" or "cos" annealing.</summary>
    public bool UseCosineAnnealing { get; }
    /// <summary>Initial LR = max_lr / div_factor.</summary>
    public double DivFactor { get; }
    /// <summary>Final LR = initial / final_div_factor.</summary>
    public double FinalDivFactor { get; }

    private readonly double _initLr;
    private readonly double _finalLr;
    private readonly int _stepUp;
    private readonly int _stepDown;

    /// <summary>Build a OneCycleLr scheduler.</summary>
    public OneCycleLr(IOptimizer optimizer, double maxLr, int totalSteps,
                      double pctStart = 0.3, bool cosineAnneal = true,
                      double divFactor = 25.0, double finalDivFactor = 1e4,
                      int lastEpoch = -1)
        : base(optimizer, lastEpoch)
    {
        if (totalSteps <= 0) throw new ArgumentOutOfRangeException(nameof(totalSteps));
        if (pctStart <= 0 || pctStart >= 1) throw new ArgumentOutOfRangeException(nameof(pctStart));
        if (divFactor <= 0)
            throw new ArgumentOutOfRangeException(nameof(divFactor),
                "divFactor must be > 0 (initial_lr = max_lr / divFactor).");
        if (finalDivFactor <= 0)
            throw new ArgumentOutOfRangeException(nameof(finalDivFactor),
                "finalDivFactor must be > 0 (final_lr = initial_lr / finalDivFactor).");
        MaxLr = maxLr; TotalSteps = totalSteps;
        PctStart = pctStart; UseCosineAnnealing = cosineAnneal;
        DivFactor = divFactor; FinalDivFactor = finalDivFactor;
        _initLr = maxLr / divFactor;
        _finalLr = _initLr / finalDivFactor;
        _stepUp = (int)Math.Round(pctStart * totalSteps) - 1;
        if (_stepUp < 0) _stepUp = 0;
        _stepDown = totalSteps - _stepUp - 1;

        for (int i = 0; i < BaseLrs.Length; i++) BaseLrs[i] = _initLr;
        for (int i = 0; i < Optimizer.ParamGroups.Count; i++)
            Optimizer.ParamGroups[i].LearningRate = _initLr;
        ApplyInitialLrs();
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        double lr;
        int t = LastEpoch;
        if (t <= _stepUp)
        {
            double x = _stepUp == 0 ? 1.0 : (double)t / _stepUp;
            lr = UseCosineAnnealing
                ? CosineInterp(_initLr, MaxLr, x)
                : LinearInterp(_initLr, MaxLr, x);
        }
        else if (t <= TotalSteps - 1)
        {
            double x = _stepDown == 0 ? 1.0 : (double)(t - _stepUp) / _stepDown;
            lr = UseCosineAnnealing
                ? CosineInterp(MaxLr, _finalLr, x)
                : LinearInterp(MaxLr, _finalLr, x);
        }
        else
        {
            lr = _finalLr;
        }
        var lrs = new double[BaseLrs.Length];
        for (int i = 0; i < lrs.Length; i++) lrs[i] = lr;
        return lrs;
    }

    private static double LinearInterp(double a, double b, double x) => a + (b - a) * x;
    private static double CosineInterp(double a, double b, double x) =>
        b + (a - b) * (1.0 + Math.Cos(Math.PI * x)) / 2.0;
}

/// <summary>ReduceLrOnPlateau: reduce LR when a tracked metric stops improving.</summary>
public sealed class ReduceLrOnPlateau
{
    /// <summary>"min" reduces LR when the metric stops decreasing; "max" when it stops increasing.</summary>
    public string Mode { get; }
    /// <summary>Multiplier applied when patience is exhausted.</summary>
    public double Factor { get; }
    /// <summary>Number of bad epochs to wait before reducing.</summary>
    public int Patience { get; }
    /// <summary>Threshold of significant change.</summary>
    public double Threshold { get; }
    /// <summary>"rel" or "abs" threshold mode.</summary>
    public string ThresholdMode { get; }
    /// <summary>Cooldown (in epochs) after a reduction during which no further reductions can fire.</summary>
    public int Cooldown { get; }
    /// <summary>Minimum LR floor — never goes below this.</summary>
    public double MinLr { get; }
    /// <summary>Tiny additive change to the LR; reductions less than eps are skipped.</summary>
    public double Eps { get; }

    /// <summary>Optimizer the scheduler controls.</summary>
    public IOptimizer Optimizer { get; }

    private double _best;
    private int _numBadEpochs;
    private int _cooldownCounter;

    /// <summary>Build a ReduceLrOnPlateau scheduler.</summary>
    public ReduceLrOnPlateau(IOptimizer optimizer,
                             string mode = "min", double factor = 0.1, int patience = 10,
                             double threshold = 1e-4, string thresholdMode = "rel",
                             int cooldown = 0, double minLr = 0.0, double eps = 1e-8)
    {
        if (mode != "min" && mode != "max") throw new ArgumentException("mode must be 'min' or 'max'.", nameof(mode));
        if (thresholdMode != "rel" && thresholdMode != "abs")
            throw new ArgumentException("thresholdMode must be 'rel' or 'abs'.", nameof(thresholdMode));
        if (factor <= 0.0 || factor >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(factor),
                "factor must satisfy 0 < factor < 1 (it is a multiplicative LR reduction).");
        if (patience < 0)
            throw new ArgumentOutOfRangeException(nameof(patience), "patience must be >= 0.");
        if (cooldown < 0)
            throw new ArgumentOutOfRangeException(nameof(cooldown), "cooldown must be >= 0.");
        if (threshold < 0.0)
            throw new ArgumentOutOfRangeException(nameof(threshold), "threshold must be >= 0.");
        if (minLr < 0.0)
            throw new ArgumentOutOfRangeException(nameof(minLr), "minLr must be >= 0.");
        if (eps < 0.0)
            throw new ArgumentOutOfRangeException(nameof(eps), "eps must be >= 0.");
        Optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        Mode = mode; Factor = factor; Patience = patience;
        Threshold = threshold; ThresholdMode = thresholdMode;
        Cooldown = cooldown; MinLr = minLr; Eps = eps;
        _best = mode == "min" ? double.PositiveInfinity : double.NegativeInfinity;
    }

    /// <summary>Step with the latest metric value (e.g. validation loss).</summary>
    public void Step(double metric)
    {
        if (IsBetter(metric, _best))
        {
            _best = metric;
            _numBadEpochs = 0;
        }
        else
        {
            _numBadEpochs++;
        }

        if (_cooldownCounter > 0)
        {
            _cooldownCounter--;
            _numBadEpochs = 0;
        }

        if (_numBadEpochs > Patience)
        {
            ReduceLr();
            _cooldownCounter = Cooldown;
            _numBadEpochs = 0;
        }
    }

    private bool IsBetter(double a, double best)
    {
        if (Mode == "min")
        {
            if (ThresholdMode == "rel") return a < best * (1.0 - Threshold);
            return a < best - Threshold;
        }
        if (ThresholdMode == "rel") return a > best * (1.0 + Threshold);
        return a > best + Threshold;
    }

    private void ReduceLr()
    {
        foreach (var g in Optimizer.ParamGroups)
        {
            double oldLr = g.LearningRate;
            double newLr = Math.Max(oldLr * Factor, MinLr);
            if (oldLr - newLr > Eps)
            {
                g.LearningRate = newLr;
                g.LastLearningRate = newLr;
            }
        }
    }
}

/// <summary>SequentialLr: dispatch to a list of schedulers based on milestone epochs.</summary>
public sealed class SequentialLr : LrScheduler
{
    /// <summary>Sub-schedulers, applied in order.</summary>
    public IReadOnlyList<LrScheduler> Schedulers { get; }
    /// <summary>Milestone epochs at which to switch from scheduler i to i+1.</summary>
    public IReadOnlyList<int> Milestones { get; }

    /// <summary>Build a SequentialLr scheduler.</summary>
    public SequentialLr(IOptimizer optimizer, IReadOnlyList<LrScheduler> schedulers, IReadOnlyList<int> milestones)
        // lastEpoch starts at 0 because the very first child has already applied its
        // epoch-0 LR to the optimizer in its own constructor; we should not double-apply.
        : base(optimizer, lastEpoch: 0)
    {
        if (schedulers == null) throw new ArgumentNullException(nameof(schedulers));
        if (milestones == null) throw new ArgumentNullException(nameof(milestones));
        if (milestones.Count != schedulers.Count - 1)
            throw new ArgumentException("milestones must be one less than the number of schedulers.");
        for (int i = 1; i < milestones.Count; i++)
            if (milestones[i] <= milestones[i - 1])
                throw new ArgumentException("milestones must be strictly increasing.");
        foreach (var s in schedulers)
            if (!ReferenceEquals(s.Optimizer, optimizer))
                throw new ArgumentException("all sub-schedulers must share the same optimizer.");
        Schedulers = schedulers; Milestones = milestones;
        // Pull the active child's LR through to our _lastLrs so GetLastLr() is consistent
        // before any explicit Step() call.
        var first = schedulers[0].GetLastLr();
        for (int i = 0; i < _lastLrs.Length; i++) _lastLrs[i] = first[i];
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        int idx = ActiveIndex();
        return Schedulers[idx].GetLastLr();
    }

    /// <summary>Advance the active scheduler. If <paramref name="epoch"/> is supplied,
    /// the active child is driven to its corresponding LOCAL epoch
    /// (<c>epoch − milestone-of-prev-segment</c>) so resume-from-checkpoint lands on
    /// the correct LR even if it falls past several milestones.</summary>
    public override void Step(int? epoch = null)
    {
        LastEpoch = epoch ?? LastEpoch + 1;
        int idx = ActiveIndex();
        // Local epoch within the active child's segment.
        int localEpoch = idx == 0 ? LastEpoch : LastEpoch - Milestones[idx - 1];
        if (epoch.HasValue)
        {
            // Explicit epoch — drive the child directly to its local epoch.
            Schedulers[idx].Step(localEpoch);
        }
        else
        {
            // Each child scheduler has already applied its epoch-0 LR in its constructor.
            // On the very step we cross a milestone, the new scheduler should apply that
            // already-prepared epoch-0 LR (via Step(0)) instead of advancing to its epoch 1.
            bool justSwitched = idx > 0 && LastEpoch == Milestones[idx - 1];
            if (justSwitched) Schedulers[idx].Step(0);
            else Schedulers[idx].Step();
        }
        // Mirror the active child's last LRs into our own buffer for GetLastLr().
        var lr = Schedulers[idx].GetLastLr();
        for (int i = 0; i < _lastLrs.Length; i++) _lastLrs[i] = lr[i];
    }

    private int ActiveIndex()
    {
        int idx = 0;
        for (int i = 0; i < Milestones.Count; i++) if (LastEpoch >= Milestones[i]) idx = i + 1;
        return idx;
    }
}

/// <summary>ChainedScheduler: compose every sub-scheduler multiplicatively at every step.
///
/// PyTorch parity (<c>torch.optim.lr_scheduler.ChainedScheduler</c>): each child computes
/// an absolute LR from its own <c>BaseLrs</c>, but the chained schedule should compound
/// effects — e.g. <c>ConstantLr(0.5) ∘ ExponentialLr(γ=0.9)</c> at epoch <c>e</c> yields
/// <c>base_lr · 0.5 · 0.9^e</c>, not just <c>base_lr · 0.9^e</c>.
///
/// We achieve composition by stepping each child (which advances its internal state and
/// transiently writes its absolute LR into the optimizer), reading its per-group factor
/// <c>last_lr / base_lr</c>, multiplying those factors together against the shared base
/// LRs, and finally overwriting the optimizer's LR with the composed result.
/// </summary>
public sealed class ChainedScheduler
{
    /// <summary>Sub-schedulers — all stepped together.</summary>
    public IReadOnlyList<LrScheduler> Schedulers { get; }
    /// <summary>Optimizer they all share.</summary>
    public IOptimizer Optimizer { get; }

    private readonly double[] _composedLrs;
    private readonly double[] _baseLrs;

    /// <summary>Build a ChainedScheduler.</summary>
    public ChainedScheduler(IOptimizer optimizer, IReadOnlyList<LrScheduler> schedulers)
    {
        Optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        Schedulers = schedulers ?? throw new ArgumentNullException(nameof(schedulers));
        if (schedulers.Count == 0)
            throw new ArgumentException("at least one sub-scheduler is required.", nameof(schedulers));
        foreach (var s in schedulers)
            if (!ReferenceEquals(s.Optimizer, optimizer))
                throw new ArgumentException("all sub-schedulers must share the same optimizer.");
        // Snapshot the shared base LRs (every child captured the same values at construction).
        _baseLrs = (double[])Schedulers[0].BaseLrs.Clone();
        _composedLrs = (double[])_baseLrs.Clone();

        // Each child's constructor already applied its own absolute epoch-0 LR to the optimizer.
        // Compose those factors so the optimizer reflects the multiplicative starting state.
        ApplyComposedFactors();
    }

    /// <summary>Step every sub-scheduler and write the composed (multiplicative) LR to the optimizer.</summary>
    public void Step()
    {
        foreach (var s in Schedulers) s.Step();
        ApplyComposedFactors();
    }

    private void ApplyComposedFactors()
    {
        int n = Optimizer.ParamGroups.Count;
        for (int i = 0; i < n; i++) _composedLrs[i] = _baseLrs[i];
        foreach (var s in Schedulers)
        {
            var lastLr = s.GetLastLr();
            for (int i = 0; i < n; i++)
            {
                double sBase = s.BaseLrs[i];
                double factor = sBase != 0.0 ? lastLr[i] / sBase : 1.0;
                _composedLrs[i] *= factor;
            }
        }
        for (int i = 0; i < n; i++)
        {
            Optimizer.ParamGroups[i].LearningRate = _composedLrs[i];
            Optimizer.ParamGroups[i].LastLearningRate = _composedLrs[i];
        }
    }

    /// <summary>Final composed LRs after the last <see cref="Step"/>.</summary>
    public IReadOnlyList<double> GetLastLr() => _composedLrs;
}
