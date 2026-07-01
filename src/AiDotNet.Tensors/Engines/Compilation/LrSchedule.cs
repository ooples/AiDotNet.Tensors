using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Decay shape applied AFTER the linear-warmup ramp in <see cref="LrSchedule.LinearWarmup"/>.
/// Mirrors <c>AiDotNet.LearningRateSchedulers.LinearWarmupScheduler.DecayMode</c> so the
/// fused mapping reproduces the eager scheduler exactly.
/// </summary>
public enum WarmupDecayMode
{
    /// <summary>Hold the peak learning rate after warmup.</summary>
    Constant,
    /// <summary>Linearly decay from peak to <c>endLr</c> over the post-warmup steps.</summary>
    Linear,
    /// <summary>Cosine-anneal from peak to <c>endLr</c> over the post-warmup steps.</summary>
    Cosine
}

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

    /// <summary>
    /// Linear warmup from <paramref name="warmupInitLr"/> to <paramref name="lrMax"/>
    /// over <paramref name="warmupSteps"/> steps, then a Constant / Linear / Cosine
    /// decay to <paramref name="endLr"/> per <paramref name="decayMode"/>. This is the
    /// fused-path image of <c>AiDotNet.LearningRateSchedulers.LinearWarmupScheduler</c>:
    /// its per-step sequence is bit-identical to that eager scheduler's per-batch LR
    /// (batch 1 = ctor <c>warmupInitLr</c>; batch n = <c>max(endLr, ComputeLearningRate(n-1))</c>),
    /// so the warmup recipe runs on the captured fast path instead of the eager tape.
    /// Unlike <see cref="LinearWarmupCosine"/> (warmup→cosine only, floor 0), this honors
    /// a non-zero warmup-init, all three decay modes, and a non-zero end LR.
    /// </summary>
    public static LrSchedule LinearWarmup(
        double lrMax, int warmupSteps, int totalSteps,
        double warmupInitLr = 0.0,
        WarmupDecayMode decayMode = WarmupDecayMode.Constant,
        double endLr = 0.0)
        => new LinearWarmupLr(lrMax, warmupSteps, totalSteps, warmupInitLr, decayMode, endLr);

    /// <summary>
    /// Noam schedule from "Attention Is All You Need" (Vaswani et al. 2017,
    /// §5.3): linear warmup then inverse-square-root decay —
    /// <c>lr(t) = factor · d_model^(-0.5) · min(t^(-0.5), t · warmup^(-1.5))</c>,
    /// peaking at <c>t = warmupSteps</c>. <c>t</c> is the 1-indexed optimizer
    /// step (the same 1-based counter <see cref="GetLr"/> receives), so the
    /// fused-path LR sequence is bit-identical to the eager
    /// <c>AiDotNet.LearningRateSchedulers.NoamSchedule</c>: both yield
    /// <c>lr(t=N)</c> on batch N. This lets the default Transformer recipe
    /// (Adam β₂=0.98 + Noam) run on the fused-compiled training path with a
    /// correct per-step ramp instead of falling back to the eager tape.
    /// </summary>
    /// <param name="modelDimension">Transformer model dimension (d_model); must be &gt; 0.</param>
    /// <param name="warmupSteps">Linear-warmup step count; must be &gt; 0.</param>
    /// <param name="factor">Multiplicative scale on the schedule (default 1.0, paper-faithful); must be &gt; 0.</param>
    public static LrSchedule Noam(int modelDimension, int warmupSteps, double factor = 1.0)
        => new NoamLr(modelDimension, warmupSteps, factor);
}

internal sealed class ConstantLr : LrSchedule
{
    private readonly double _lr;
    public ConstantLr(double lr) { _lr = lr; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override double GetLr(int step) => _lr;
}

internal sealed class NoamLr : LrSchedule
{
    // Pre-computed step-invariant coefficients so GetLr is one Math.Sqrt + a
    // couple of multiplies per step (no Math.Pow), matching the eager
    // NoamSchedule's hot-path optimization:
    //   _scaledModelInvSqrt = factor · d_model^(-0.5)
    //   _scaledWarmupTerm   = _scaledModelInvSqrt · warmup^(-1.5)   (coeff on the linear branch)
    private readonly double _scaledModelInvSqrt;
    private readonly double _scaledWarmupTerm;

    public NoamLr(int modelDimension, int warmupSteps, double factor)
    {
        if (modelDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(modelDimension), "modelDimension must be positive.");
        if (warmupSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(warmupSteps), "warmupSteps must be positive.");
        if (factor <= 0)
            throw new ArgumentOutOfRangeException(nameof(factor), "factor must be positive.");

        _scaledModelInvSqrt = factor / System.Math.Sqrt(modelDimension);
        _scaledWarmupTerm = _scaledModelInvSqrt / ((double)warmupSteps * System.Math.Sqrt(warmupSteps));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override double GetLr(int step)
    {
        // step is 1-indexed (CompiledTrainingPlan increments _optimizerStep to 1
        // before the first GetLr call), and the paper's t is also 1-based, so
        // t = step directly — no +1 remap (unlike the eager NoamSchedule, which
        // remaps from a 0-based framework counter). The clamp guards a non-
        // positive t that would make t^(-0.5) blow up.
        int t = step < 1 ? 1 : step;
        double invSqrtBranch = _scaledModelInvSqrt / System.Math.Sqrt(t);
        double warmupBranch = _scaledWarmupTerm * t;
        return System.Math.Min(invSqrtBranch, warmupBranch);
    }
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
        // PyTorch: min_lr = initial_lr / final_div_factor (i.e. lrMax /
        // (divFactor * finalDivFactor)) — NOT lrMax / finalDivFactor.
        // Anchoring the floor to lrMax leaves the schedule divFactor× too
        // high at the end. See docs.pytorch.org OneCycleLR reference.
        _lrFinal = _lrInitial / finalDivFactor;
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
        // With warmupSteps == 0 the cosine half must start AT lrMax on
        // step 1 — same convention as bare Cosine schedule. Using
        // (s - _warmup) / (_total - _warmup) would give progress = 1/_total
        // at step 1 (slightly below lrMax) and lrMin at step _total with
        // _total == 1. (s-1)/(_total-1) is the right base.
        double progress = _warmup == 0
            ? (s - 1) / (double)System.Math.Max(_total - 1, 1)
            : (s - _warmup) / (double)System.Math.Max(_total - _warmup, 1);
        double cos = 0.5 * (1.0 + System.Math.Cos(System.Math.PI * progress));
        return _lrMin + (_lrMax - _lrMin) * cos;
    }
}

internal sealed class LinearWarmupLr : LrSchedule
{
    private readonly double _lrMax;
    private readonly int _warmupSteps;
    private readonly int _totalSteps;
    private readonly double _warmupInitLr;
    private readonly WarmupDecayMode _decayMode;
    private readonly double _endLr;

    public LinearWarmupLr(double lrMax, int warmupSteps, int totalSteps,
        double warmupInitLr, WarmupDecayMode decayMode, double endLr)
    {
        if (warmupSteps < 0) throw new ArgumentOutOfRangeException(nameof(warmupSteps));
        if (!Enum.IsDefined(typeof(WarmupDecayMode), decayMode))
            throw new ArgumentOutOfRangeException(nameof(decayMode), decayMode, "Undefined warmup decay mode.");
        int normalizedTotalSteps = totalSteps > 0 ? totalSteps : warmupSteps;
        if (decayMode != WarmupDecayMode.Constant && normalizedTotalSteps < warmupSteps)
            throw new ArgumentOutOfRangeException(nameof(totalSteps),
                "totalSteps must be >= warmupSteps for decay modes.");
        _lrMax = lrMax;
        _warmupSteps = warmupSteps;
        _totalSteps = normalizedTotalSteps;
        _warmupInitLr = warmupInitLr;
        _decayMode = decayMode;
        _endLr = endLr;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override double GetLr(int step)
    {
        // The eager LinearWarmupScheduler reports its ctor value (warmupInitLr) on
        // batch 1 (unfloored), then on batch n>=2 reports max(endLr, ComputeLearningRate(n-1))
        // — _currentStep is (batch - 1) and Step() floors with _minLearningRate = endLr.
        // The 1-indexed fused step maps batch n → GetLr(n); reproduce both branches so the
        // per-step LR is bit-identical to the eager replay (FusedLrScheduleMappingTests).
        if (step <= 1)
            return _warmupSteps > 0 ? _warmupInitLr : _lrMax;
        double raw = ComputeRaw(step - 1);
        return raw > _endLr ? raw : _endLr;
    }

    // Mirror of LinearWarmupScheduler.ComputeLearningRate(step) with _baseLearningRate = lrMax.
    private double ComputeRaw(int t)
    {
        if (t < _warmupSteps)
        {
            if (_warmupSteps == 0) return _lrMax;
            return _warmupInitLr + (_lrMax - _warmupInitLr) * ((double)t / _warmupSteps);
        }
        if (_decayMode == WarmupDecayMode.Constant) return _lrMax;

        int decaySteps = _totalSteps - _warmupSteps;
        int decayStep = t - _warmupSteps;
        if (decayStep >= decaySteps) return _endLr;
        double progress = (double)decayStep / decaySteps;
        if (_decayMode == WarmupDecayMode.Linear)
            return _lrMax - (_lrMax - _endLr) * progress;
        double cos = (1.0 + System.Math.Cos(System.Math.PI * progress)) / 2.0;
        return _endLr + (_lrMax - _endLr) * cos;
    }
}
