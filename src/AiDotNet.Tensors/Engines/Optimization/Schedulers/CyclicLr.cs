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
        if (factor >= 1.0) throw new ArgumentOutOfRangeException(nameof(factor), "factor must be < 1.");
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
public sealed class SequentialLr
{
    /// <summary>Sub-schedulers, applied in order.</summary>
    public IReadOnlyList<LrScheduler> Schedulers { get; }
    /// <summary>Milestone epochs at which to switch from scheduler i to i+1.</summary>
    public IReadOnlyList<int> Milestones { get; }
    /// <summary>The optimizer being controlled.</summary>
    public IOptimizer Optimizer { get; }

    private int _lastEpoch;

    /// <summary>Build a SequentialLr scheduler.</summary>
    public SequentialLr(IOptimizer optimizer, IReadOnlyList<LrScheduler> schedulers, IReadOnlyList<int> milestones)
    {
        if (schedulers == null) throw new ArgumentNullException(nameof(schedulers));
        if (milestones == null) throw new ArgumentNullException(nameof(milestones));
        if (milestones.Count != schedulers.Count - 1)
            throw new ArgumentException("milestones must be one less than the number of schedulers.");
        for (int i = 1; i < milestones.Count; i++)
            if (milestones[i] <= milestones[i - 1])
                throw new ArgumentException("milestones must be strictly increasing.");
        Optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        Schedulers = schedulers; Milestones = milestones;
        _lastEpoch = -1;
    }

    /// <summary>Advance the active scheduler.</summary>
    public void Step()
    {
        _lastEpoch++;
        int idx = 0;
        for (int i = 0; i < Milestones.Count; i++) if (_lastEpoch >= Milestones[i]) idx = i + 1;
        Schedulers[idx].Step();
    }

    /// <summary>Last LRs applied across all groups.</summary>
    public IReadOnlyList<double> GetLastLr()
    {
        int idx = 0;
        for (int i = 0; i < Milestones.Count; i++) if (_lastEpoch >= Milestones[i]) idx = i + 1;
        return Schedulers[idx].GetLastLr();
    }
}

/// <summary>ChainedScheduler: apply every scheduler at every step (composition by multiplication).</summary>
public sealed class ChainedScheduler
{
    /// <summary>Sub-schedulers — all stepped together.</summary>
    public IReadOnlyList<LrScheduler> Schedulers { get; }
    /// <summary>Optimizer they all share.</summary>
    public IOptimizer Optimizer { get; }

    /// <summary>Build a ChainedScheduler.</summary>
    public ChainedScheduler(IOptimizer optimizer, IReadOnlyList<LrScheduler> schedulers)
    {
        Optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        Schedulers = schedulers ?? throw new ArgumentNullException(nameof(schedulers));
        foreach (var s in schedulers)
            if (!ReferenceEquals(s.Optimizer, optimizer))
                throw new ArgumentException("all sub-schedulers must share the same optimizer.");
    }

    /// <summary>Step every sub-scheduler.</summary>
    public void Step()
    {
        foreach (var s in Schedulers) s.Step();
    }

    /// <summary>The LRs from the last sub-scheduler stepped (the one that last wrote into param_groups).</summary>
    public IReadOnlyList<double> GetLastLr() =>
        Schedulers.Count == 0 ? Array.Empty<double>() : Schedulers[Schedulers.Count - 1].GetLastLr();
}
