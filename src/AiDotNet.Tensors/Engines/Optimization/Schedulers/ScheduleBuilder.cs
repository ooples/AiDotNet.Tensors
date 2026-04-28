using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;

namespace AiDotNet.Tensors.Engines.Optimization.Schedulers;

/// <summary>
/// Fluent scheduler-composition DSL. Builds a <see cref="SequentialLr"/> chain that stitches
/// common training-recipe stages (warmup → cosine → linear decay, etc.) without forcing the
/// user to construct each constituent scheduler manually.
///
/// Example:
/// <code>
/// var sched = ScheduleBuilder
///     .For(optimizer)
///     .Warmup(1000)
///     .Cosine(90_000)
///     .LinearDecay(5_000, finalFactor: 0.0)
///     .Build();
/// </code>
/// </summary>
public sealed class ScheduleBuilder
{
    private readonly IOptimizer _opt;
    private readonly List<(int durationSteps, Func<IOptimizer, LrScheduler> factory)> _stages = new();

    private ScheduleBuilder(IOptimizer opt) { _opt = opt; }

    /// <summary>Begin building a schedule for <paramref name="optimizer"/>.</summary>
    public static ScheduleBuilder For(IOptimizer optimizer) => new ScheduleBuilder(optimizer);

    /// <summary>Linear warm-up over <paramref name="steps"/> from <paramref name="startFactor"/>·base_lr to base_lr.</summary>
    public ScheduleBuilder Warmup(int steps, double startFactor = 0.0)
    {
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
        _stages.Add((steps, opt => new LinearLr(opt, startFactor: startFactor, endFactor: 1.0, totalIters: steps)));
        return this;
    }

    /// <summary>Half-cycle cosine annealing over <paramref name="steps"/> down to <paramref name="etaMin"/>.</summary>
    public ScheduleBuilder Cosine(int steps, double etaMin = 0.0)
    {
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
        _stages.Add((steps, opt => new CosineAnnealingLr(opt, tMax: steps, etaMin: etaMin)));
        return this;
    }

    /// <summary>Linear decay from current LR to <paramref name="finalFactor"/>·base_lr over <paramref name="steps"/>.</summary>
    public ScheduleBuilder LinearDecay(int steps, double finalFactor = 0.0)
    {
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
        _stages.Add((steps, opt => new LinearLr(opt, startFactor: 1.0, endFactor: finalFactor, totalIters: steps)));
        return this;
    }

    /// <summary>Hold LR constant at <paramref name="factor"/>·base_lr for <paramref name="steps"/>.</summary>
    public ScheduleBuilder Constant(int steps, double factor = 1.0)
    {
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
        _stages.Add((steps, opt => new ConstantLr(opt, factor: factor, totalIters: steps)));
        return this;
    }

    /// <summary>Exponential decay <c>lr ← γ · lr</c> per step for <paramref name="steps"/>.</summary>
    public ScheduleBuilder Exponential(int steps, double gamma)
    {
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
        if (gamma <= 0 || gamma > 1) throw new ArgumentOutOfRangeException(nameof(gamma));
        _stages.Add((steps, opt => new ExponentialLr(opt, gamma)));
        return this;
    }

    /// <summary>Polynomial decay over <paramref name="steps"/> with the given <paramref name="power"/>.</summary>
    public ScheduleBuilder Polynomial(int steps, double power = 1.0)
    {
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
        _stages.Add((steps, opt => new PolynomialLr(opt, totalIters: steps, power: power)));
        return this;
    }

    /// <summary>Build the composed scheduler. Returns a <see cref="SequentialLr"/> if there are 2+ stages,
    /// or the single underlying scheduler if only one stage was configured.</summary>
    public object Build()
    {
        if (_stages.Count == 0) throw new InvalidOperationException("no stages configured.");
        if (_stages.Count == 1) return _stages[0].factory(_opt);

        var schedulers = new LrScheduler[_stages.Count];
        for (int i = 0; i < _stages.Count; i++) schedulers[i] = _stages[i].factory(_opt);

        var milestones = new int[_stages.Count - 1];
        int acc = 0;
        for (int i = 0; i < milestones.Length; i++)
        {
            acc += _stages[i].durationSteps;
            milestones[i] = acc;
        }
        return new SequentialLr(_opt, schedulers, milestones);
    }
}
