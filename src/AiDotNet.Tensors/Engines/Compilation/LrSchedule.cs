using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Per-step learning-rate schedule used by the fused-compiled training path.
/// Implementations must be pure and allocation-free — they're called inside
/// the tight <c>Step()</c> loop, once per optimizer step.
///
/// PyTorch's <c>torch.optim.lr_scheduler</c> runs in managed-language code
/// between optimizer steps and pays per-step dispatch overhead. The fused
/// kernels here already take <c>lr</c> as a per-call scalar, so co-locating
/// the schedule formula with the kernel-call wrapper lets schedule evaluation
/// happen inline (3–5 cycles) instead of via a separate Python-level callback.
/// </summary>
public abstract class LrSchedule
{
    /// <summary>
    /// Resolves the learning rate for a 1-indexed optimizer step.
    /// </summary>
    /// <param name="step">1-indexed step counter — matches Adam bias-correction
    /// convention so the same counter feeds both the schedule and the moment
    /// debiasing without an off-by-one trap.</param>
    public abstract double GetLr(int step);

    /// <summary>Constant learning rate. Use this when you don't want a schedule.</summary>
    public static LrSchedule Constant(double lr) => new ConstantLr(lr);

    /// <summary>
    /// Cosine decay from <paramref name="lrMax"/> at step 1 down to
    /// <paramref name="lrMin"/> at step <paramref name="totalSteps"/>.
    /// Matches PyTorch <c>CosineAnnealingLR(eta_min=lrMin, T_max=totalSteps)</c>.
    /// </summary>
    public static LrSchedule Cosine(double lrMax, int totalSteps, double lrMin = 0.0)
        => new CosineLr(lrMax, totalSteps, lrMin);

    /// <summary>
    /// One-cycle schedule (Smith, 2018) — linear warmup from
    /// <c>lrMax / divFactor</c> to <paramref name="lrMax"/> over the first
    /// <paramref name="pctStart"/> fraction of steps, then cosine anneal down
    /// to <c>lrMax / finalDivFactor</c>. Matches PyTorch <c>OneCycleLR</c>
    /// with <c>anneal_strategy="cos"</c>.
    /// </summary>
    public static LrSchedule OneCycle(
        double lrMax, int totalSteps,
        double pctStart = 0.3, double divFactor = 25.0, double finalDivFactor = 1e4)
        => new OneCycleLr(lrMax, totalSteps, pctStart, divFactor, finalDivFactor);

    /// <summary>
    /// Multiplicative exponential decay: <c>lr0 · γ^step</c>.
    /// Matches PyTorch <c>ExponentialLR(gamma=gamma)</c>.
    /// </summary>
    public static LrSchedule Exponential(double lr0, double gamma) => new ExponentialLr(lr0, gamma);

    /// <summary>
    /// Step decay: <c>lr0 · γ^(step / stepSize)</c> (integer division).
    /// Matches PyTorch <c>StepLR(step_size=stepSize, gamma=gamma)</c>.
    /// </summary>
    public static LrSchedule Step(double lr0, int stepSize, double gamma = 0.1)
        => new StepLrImpl(lr0, stepSize, gamma);

    /// <summary>
    /// Cyclic triangular schedule (Smith, 2017) — sawtooths between
    /// <paramref name="lrBase"/> and <paramref name="lrMax"/> with period
    /// <c>2·stepSize</c>. Matches PyTorch <c>CyclicLR(mode="triangular")</c>.
    /// </summary>
    public static LrSchedule Cyclic(double lrBase, double lrMax, int stepSize)
        => new CyclicLr(lrBase, lrMax, stepSize);

    /// <summary>
    /// Linear warmup for <paramref name="warmupSteps"/> from 0 to
    /// <paramref name="lrMax"/>, followed by cosine anneal to
    /// <paramref name="lrMin"/> at step <paramref name="totalSteps"/>.
    /// This is the schedule the original Transformer paper uses and
    /// what most modern foundation-model training recipes follow.
    /// </summary>
    public static LrSchedule LinearWarmupCosine(
        double lrMax, int warmupSteps, int totalSteps, double lrMin = 0.0)
        => new LinearWarmupCosineLr(lrMax, warmupSteps, totalSteps, lrMin);
}

internal sealed class ConstantLr : LrSchedule
{
    private readonly double _lr;
    public ConstantLr(double lr) { _lr = lr; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override double GetLr(int step) => _lr;
}

internal sealed class CosineLr : LrSchedule
{
    private readonly double _lrMax;
    private readonly double _lrMin;
    private readonly int _total;
    public CosineLr(double lrMax, int total, double lrMin)
    {
        if (total < 1) throw new ArgumentOutOfRangeException(nameof(total), "totalSteps must be >= 1.");
        _lrMax = lrMax; _lrMin = lrMin; _total = total;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override double GetLr(int step)
    {
        // Clamp on the right: stepping past the schedule's end pins the lr at
        // lrMin instead of producing oscillation from cos extrapolation.
        int s = step < 1 ? 1 : step > _total ? _total : step;
        double progress = (s - 1) / (double)System.Math.Max(_total - 1, 1);
        double cos = 0.5 * (1.0 + System.Math.Cos(System.Math.PI * progress));
        return _lrMin + (_lrMax - _lrMin) * cos;
    }
}

internal sealed class OneCycleLr : LrSchedule
{
    private readonly double _lrMax;
    private readonly double _lrInitial;
    private readonly double _lrFinal;
    private readonly int _total;
    private readonly int _warmupEnd;
    public OneCycleLr(double lrMax, int total, double pctStart, double divFactor, double finalDivFactor)
    {
        if (total < 2) throw new ArgumentOutOfRangeException(nameof(total), "OneCycle needs at least 2 steps.");
        if (pctStart <= 0.0 || pctStart >= 1.0) throw new ArgumentOutOfRangeException(nameof(pctStart), "pctStart must be in (0, 1).");
        if (divFactor <= 0.0) throw new ArgumentOutOfRangeException(nameof(divFactor));
        if (finalDivFactor <= 0.0) throw new ArgumentOutOfRangeException(nameof(finalDivFactor));
        _lrMax = lrMax;
        _lrInitial = lrMax / divFactor;
        _lrFinal = lrMax / finalDivFactor;
        _total = total;
        _warmupEnd = System.Math.Max(1, (int)System.Math.Round(pctStart * total));
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override double GetLr(int step)
    {
        int s = step < 1 ? 1 : step > _total ? _total : step;
        if (s <= _warmupEnd)
        {
            double t = (s - 1) / (double)System.Math.Max(_warmupEnd - 1, 1);
            return _lrInitial + (_lrMax - _lrInitial) * t;
        }
        double progress = (s - _warmupEnd) / (double)System.Math.Max(_total - _warmupEnd, 1);
        double cos = 0.5 * (1.0 + System.Math.Cos(System.Math.PI * progress));
        return _lrFinal + (_lrMax - _lrFinal) * cos;
    }
}

internal sealed class ExponentialLr : LrSchedule
{
    private readonly double _lr0;
    private readonly double _gamma;
    public ExponentialLr(double lr0, double gamma)
    {
        if (gamma <= 0.0) throw new ArgumentOutOfRangeException(nameof(gamma), "gamma must be > 0.");
        _lr0 = lr0; _gamma = gamma;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override double GetLr(int step) => _lr0 * System.Math.Pow(_gamma, System.Math.Max(0, step - 1));
}

internal sealed class StepLrImpl : LrSchedule
{
    private readonly double _lr0;
    private readonly int _stepSize;
    private readonly double _gamma;
    public StepLrImpl(double lr0, int stepSize, double gamma)
    {
        if (stepSize < 1) throw new ArgumentOutOfRangeException(nameof(stepSize), "stepSize must be >= 1.");
        if (gamma <= 0.0) throw new ArgumentOutOfRangeException(nameof(gamma), "gamma must be > 0.");
        _lr0 = lr0; _stepSize = stepSize; _gamma = gamma;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override double GetLr(int step) => _lr0 * System.Math.Pow(_gamma, System.Math.Max(0, step - 1) / _stepSize);
}

internal sealed class CyclicLr : LrSchedule
{
    private readonly double _base;
    private readonly double _max;
    private readonly int _stepSize;
    public CyclicLr(double lrBase, double lrMax, int stepSize)
    {
        if (stepSize < 1) throw new ArgumentOutOfRangeException(nameof(stepSize), "stepSize must be >= 1.");
        _base = lrBase; _max = lrMax; _stepSize = stepSize;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override double GetLr(int step)
    {
        // Triangular: ascend for stepSize, descend for stepSize, repeat.
        int s = System.Math.Max(0, step - 1);
        int period = 2 * _stepSize;
        int phase = s % period;
        double t = phase < _stepSize
            ? phase / (double)_stepSize
            : 1.0 - (phase - _stepSize) / (double)_stepSize;
        return _base + (_max - _base) * t;
    }
}

internal sealed class LinearWarmupCosineLr : LrSchedule
{
    private readonly double _lrMax;
    private readonly double _lrMin;
    private readonly int _warmup;
    private readonly int _total;
    public LinearWarmupCosineLr(double lrMax, int warmupSteps, int totalSteps, double lrMin)
    {
        if (warmupSteps < 0) throw new ArgumentOutOfRangeException(nameof(warmupSteps));
        if (totalSteps < warmupSteps + 1) throw new ArgumentOutOfRangeException(nameof(totalSteps), "totalSteps must exceed warmupSteps.");
        _lrMax = lrMax; _lrMin = lrMin; _warmup = warmupSteps; _total = totalSteps;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override double GetLr(int step)
    {
        int s = step < 1 ? 1 : step > _total ? _total : step;
        if (_warmup > 0 && s <= _warmup)
            return _lrMax * s / (double)_warmup;
        double progress = (s - _warmup) / (double)System.Math.Max(_total - _warmup, 1);
        double cos = 0.5 * (1.0 + System.Math.Cos(System.Math.PI * progress));
        return _lrMin + (_lrMax - _lrMin) * cos;
    }
}
