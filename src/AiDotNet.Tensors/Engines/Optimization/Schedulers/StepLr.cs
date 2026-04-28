using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;

namespace AiDotNet.Tensors.Engines.Optimization.Schedulers;

/// <summary>StepLr: decay LR by <c>γ</c> every <c>stepSize</c> epochs.</summary>
public sealed class StepLr : LrScheduler
{
    /// <summary>Number of epochs between LR drops.</summary>
    public int StepSize { get; }
    /// <summary>Multiplicative decay factor applied every <see cref="StepSize"/> epochs.</summary>
    public double Gamma { get; }

    /// <summary>Build a StepLr scheduler.</summary>
    public StepLr(IOptimizer optimizer, int stepSize, double gamma = 0.1, int lastEpoch = -1)
        : base(optimizer, lastEpoch)
    {
        if (stepSize <= 0) throw new ArgumentOutOfRangeException(nameof(stepSize));
        StepSize = stepSize;
        Gamma = gamma;
        ApplyInitialLrs();
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        var lrs = new double[BaseLrs.Length];
        int drops = LastEpoch / StepSize;
        double factor = Math.Pow(Gamma, drops);
        for (int i = 0; i < BaseLrs.Length; i++) lrs[i] = BaseLrs[i] * factor;
        return lrs;
    }
}

/// <summary>MultiStepLr: decay LR by γ at each milestone epoch.</summary>
public sealed class MultiStepLr : LrScheduler
{
    /// <summary>Sorted list of epochs at which to drop the LR.</summary>
    public IReadOnlyList<int> Milestones { get; }
    /// <summary>Multiplicative decay factor applied at each milestone.</summary>
    public double Gamma { get; }

    /// <summary>Build a MultiStepLr scheduler.</summary>
    public MultiStepLr(IOptimizer optimizer, IEnumerable<int> milestones, double gamma = 0.1, int lastEpoch = -1)
        : base(optimizer, lastEpoch)
    {
        if (milestones == null) throw new ArgumentNullException(nameof(milestones));
        var sorted = new List<int>(milestones);
        sorted.Sort();
        Milestones = sorted;
        Gamma = gamma;
        ApplyInitialLrs();
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        int drops = 0;
        for (int i = 0; i < Milestones.Count; i++) if (LastEpoch >= Milestones[i]) drops++;
        double factor = Math.Pow(Gamma, drops);
        var lrs = new double[BaseLrs.Length];
        for (int i = 0; i < BaseLrs.Length; i++) lrs[i] = BaseLrs[i] * factor;
        return lrs;
    }
}

/// <summary>ExponentialLr: <c>lr = lr0 · γᵉ</c> at every epoch.</summary>
public sealed class ExponentialLr : LrScheduler
{
    /// <summary>Multiplicative decay applied every epoch.</summary>
    public double Gamma { get; }
    /// <summary>Build an ExponentialLr scheduler.</summary>
    public ExponentialLr(IOptimizer optimizer, double gamma, int lastEpoch = -1)
        : base(optimizer, lastEpoch) { Gamma = gamma; ApplyInitialLrs(); }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        var lrs = new double[BaseLrs.Length];
        double factor = Math.Pow(Gamma, LastEpoch);
        for (int i = 0; i < BaseLrs.Length; i++) lrs[i] = BaseLrs[i] * factor;
        return lrs;
    }
}

/// <summary>PolynomialLr: <c>lr = lr0 · (1 − e/E)^p</c>; clamped to 0 after <c>totalIters</c>.</summary>
public sealed class PolynomialLr : LrScheduler
{
    /// <summary>Total number of epochs over which the LR decays from base to 0.</summary>
    public int TotalIters { get; }
    /// <summary>Polynomial power.</summary>
    public double Power { get; }

    /// <summary>Build a PolynomialLr scheduler.</summary>
    public PolynomialLr(IOptimizer optimizer, int totalIters, double power = 1.0, int lastEpoch = -1)
        : base(optimizer, lastEpoch)
    {
        if (totalIters <= 0) throw new ArgumentOutOfRangeException(nameof(totalIters));
        TotalIters = totalIters;
        Power = power;
        ApplyInitialLrs();
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        var lrs = new double[BaseLrs.Length];
        if (LastEpoch == 0)
        {
            for (int i = 0; i < BaseLrs.Length; i++) lrs[i] = BaseLrs[i];
            return lrs;
        }
        if (LastEpoch >= TotalIters)
        {
            for (int i = 0; i < BaseLrs.Length; i++) lrs[i] = 0.0;
            return lrs;
        }
        // PyTorch implementation: incremental scaling factor each step.
        // lr_t / lr_{t-1} = ((1 - t/T) / (1 - (t-1)/T))^power
        // We compute the closed form for stability:
        double frac = 1.0 - (double)LastEpoch / TotalIters;
        double factor = Math.Pow(frac, Power);
        for (int i = 0; i < BaseLrs.Length; i++) lrs[i] = BaseLrs[i] * factor;
        return lrs;
    }
}

/// <summary>CosineAnnealingLr: <c>lr = ηmin + ½(lr0 − ηmin)(1 + cos(π·e/Tmax))</c>.</summary>
public sealed class CosineAnnealingLr : LrScheduler
{
    /// <summary>Half-cycle length in epochs.</summary>
    public int TMax { get; }
    /// <summary>Minimum LR at end of half-cycle.</summary>
    public double EtaMin { get; }

    /// <summary>Build a CosineAnnealingLr scheduler.</summary>
    public CosineAnnealingLr(IOptimizer optimizer, int tMax, double etaMin = 0.0, int lastEpoch = -1)
        : base(optimizer, lastEpoch)
    {
        if (tMax <= 0) throw new ArgumentOutOfRangeException(nameof(tMax));
        TMax = tMax;
        EtaMin = etaMin;
        ApplyInitialLrs();
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        var lrs = new double[BaseLrs.Length];
        for (int i = 0; i < BaseLrs.Length; i++)
        {
            lrs[i] = EtaMin + 0.5 * (BaseLrs[i] - EtaMin) * (1.0 + Math.Cos(Math.PI * LastEpoch / TMax));
        }
        return lrs;
    }
}

/// <summary>CosineAnnealingWarmRestarts: SGDR with restarts every T₀ · T_mult^k epochs.</summary>
public sealed class CosineAnnealingWarmRestarts : LrScheduler
{
    /// <summary>Initial restart period.</summary>
    public int T0 { get; }
    /// <summary>Period multiplier on each restart.</summary>
    public int TMult { get; }
    /// <summary>Minimum LR (asymptote of the cosine).</summary>
    public double EtaMin { get; }

    private int _tCur;
    private int _ti;

    /// <summary>Build a CosineAnnealingWarmRestarts scheduler.</summary>
    public CosineAnnealingWarmRestarts(IOptimizer optimizer, int t0, int tMult = 1, double etaMin = 0.0, int lastEpoch = -1)
        : base(optimizer, lastEpoch)
    {
        if (t0 <= 0) throw new ArgumentOutOfRangeException(nameof(t0));
        if (tMult < 1) throw new ArgumentOutOfRangeException(nameof(tMult));
        T0 = t0; TMult = tMult; EtaMin = etaMin;
        _tCur = lastEpoch == -1 ? 0 : lastEpoch;
        _ti = t0;
        ApplyInitialLrs();
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        var lrs = new double[BaseLrs.Length];
        for (int i = 0; i < BaseLrs.Length; i++)
            lrs[i] = EtaMin + 0.5 * (BaseLrs[i] - EtaMin) * (1.0 + Math.Cos(Math.PI * _tCur / _ti));
        return lrs;
    }

    /// <inheritdoc />
    public override void Step(int? epoch = null)
    {
        if (epoch == null)
        {
            _tCur++;
            if (_tCur >= _ti) { _tCur -= _ti; _ti *= TMult; }
            LastEpoch = LastEpoch + 1;
        }
        else
        {
            int e = epoch.Value;
            if (e < 0) throw new ArgumentOutOfRangeException(nameof(epoch));
            // Solve for which restart segment e lands in.
            if (TMult == 1)
            {
                _tCur = e % T0;
                _ti = T0;
            }
            else
            {
                int n = (int)Math.Floor(Math.Log(e * (TMult - 1) / (double)T0 + 1, TMult));
                _tCur = e - T0 * (int)((Math.Pow(TMult, n) - 1) / (TMult - 1));
                _ti = T0 * (int)Math.Pow(TMult, n);
            }
            LastEpoch = e;
        }
        ApplyLrs(GetLr());
    }
}

/// <summary>ConstantLr: scale base LR by <c>factor</c> for the first <c>totalIters</c> epochs, then back to base.</summary>
public sealed class ConstantLr : LrScheduler
{
    /// <summary>Multiplicative factor applied during warmup.</summary>
    public double Factor { get; }
    /// <summary>Number of epochs to apply <see cref="Factor"/>.</summary>
    public int TotalIters { get; }

    /// <summary>Build a ConstantLr scheduler.</summary>
    public ConstantLr(IOptimizer optimizer, double factor = 1.0/3.0, int totalIters = 5, int lastEpoch = -1)
        : base(optimizer, lastEpoch) { Factor = factor; TotalIters = totalIters; ApplyInitialLrs(); }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        var lrs = new double[BaseLrs.Length];
        double m = LastEpoch < TotalIters ? Factor : 1.0;
        for (int i = 0; i < BaseLrs.Length; i++) lrs[i] = BaseLrs[i] * m;
        return lrs;
    }
}

/// <summary>LinearLr: linearly interpolate between <c>startFactor</c> and <c>endFactor</c> over <c>totalIters</c>, then hold.</summary>
public sealed class LinearLr : LrScheduler
{
    /// <summary>Multiplicative factor at epoch 0.</summary>
    public double StartFactor { get; }
    /// <summary>Multiplicative factor at <see cref="TotalIters"/>.</summary>
    public double EndFactor { get; }
    /// <summary>Number of epochs over which factor moves from start to end.</summary>
    public int TotalIters { get; }

    /// <summary>Build a LinearLr scheduler.</summary>
    public LinearLr(IOptimizer optimizer, double startFactor = 1.0/3.0, double endFactor = 1.0,
                    int totalIters = 5, int lastEpoch = -1)
        : base(optimizer, lastEpoch)
    {
        if (totalIters <= 0) throw new ArgumentOutOfRangeException(nameof(totalIters));
        StartFactor = startFactor; EndFactor = endFactor; TotalIters = totalIters;
        ApplyInitialLrs();
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        double m;
        if (LastEpoch == 0) m = StartFactor;
        else if (LastEpoch >= TotalIters) m = EndFactor;
        else m = StartFactor + (EndFactor - StartFactor) * ((double)LastEpoch / TotalIters);
        var lrs = new double[BaseLrs.Length];
        for (int i = 0; i < BaseLrs.Length; i++) lrs[i] = BaseLrs[i] * m;
        return lrs;
    }
}

/// <summary>LambdaLr: <c>lr = lr0 · λ(epoch)</c> with a user-supplied multiplier function per group.</summary>
public sealed class LambdaLr : LrScheduler
{
    /// <summary>One multiplier function per param group, applied to the base LR.</summary>
    public IReadOnlyList<Func<int, double>> LrLambdas { get; }

    /// <summary>Build a LambdaLr scheduler with one λ per param group.</summary>
    public LambdaLr(IOptimizer optimizer, IReadOnlyList<Func<int, double>> lrLambdas, int lastEpoch = -1)
        : base(optimizer, lastEpoch)
    {
        if (lrLambdas == null) throw new ArgumentNullException(nameof(lrLambdas));
        if (lrLambdas.Count != optimizer.ParamGroups.Count)
            throw new ArgumentException("must supply one λ per param group.");
        LrLambdas = lrLambdas;
        ApplyInitialLrs();
    }

    /// <summary>Build a LambdaLr scheduler with a single λ broadcast to every group.</summary>
    public LambdaLr(IOptimizer optimizer, Func<int, double> lrLambda, int lastEpoch = -1)
        : this(optimizer, BroadcastLambda(optimizer, lrLambda), lastEpoch) { }

    private static IReadOnlyList<Func<int, double>> BroadcastLambda(IOptimizer optimizer, Func<int, double> f)
    {
        var arr = new Func<int, double>[optimizer.ParamGroups.Count];
        for (int i = 0; i < arr.Length; i++) arr[i] = f;
        return arr;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        var lrs = new double[BaseLrs.Length];
        for (int i = 0; i < BaseLrs.Length; i++) lrs[i] = BaseLrs[i] * LrLambdas[i](LastEpoch);
        return lrs;
    }
}

/// <summary>MultiplicativeLr: at each step, multiply current LR by <c>λ(epoch)</c>.</summary>
public sealed class MultiplicativeLr : LrScheduler
{
    /// <summary>Per-group multiplicative function.</summary>
    public IReadOnlyList<Func<int, double>> LrLambdas { get; }

    /// <summary>Build a MultiplicativeLr scheduler with one λ per param group.</summary>
    public MultiplicativeLr(IOptimizer optimizer, IReadOnlyList<Func<int, double>> lrLambdas, int lastEpoch = -1)
        : base(optimizer, lastEpoch)
    {
        if (lrLambdas == null) throw new ArgumentNullException(nameof(lrLambdas));
        if (lrLambdas.Count != optimizer.ParamGroups.Count)
            throw new ArgumentException("must supply one λ per param group.");
        LrLambdas = lrLambdas;
        ApplyInitialLrs();
    }

    /// <summary>Build a MultiplicativeLr scheduler broadcasting a single λ to every group.</summary>
    public MultiplicativeLr(IOptimizer optimizer, Func<int, double> lrLambda, int lastEpoch = -1)
        : this(optimizer, BroadcastLambda(optimizer, lrLambda), lastEpoch) { }

    private static IReadOnlyList<Func<int, double>> BroadcastLambda(IOptimizer optimizer, Func<int, double> f)
    {
        var arr = new Func<int, double>[optimizer.ParamGroups.Count];
        for (int i = 0; i < arr.Length; i++) arr[i] = f;
        return arr;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<double> GetLr()
    {
        var lrs = new double[BaseLrs.Length];
        if (LastEpoch == 0)
        {
            for (int i = 0; i < BaseLrs.Length; i++) lrs[i] = BaseLrs[i];
            return lrs;
        }
        // Compose the multipliers from epoch 1..LastEpoch.
        for (int i = 0; i < BaseLrs.Length; i++)
        {
            double cur = BaseLrs[i];
            for (int e = 1; e <= LastEpoch; e++) cur *= LrLambdas[i](e);
            lrs[i] = cur;
        }
        return lrs;
    }
}
