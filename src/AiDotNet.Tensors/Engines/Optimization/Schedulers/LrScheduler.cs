using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;

namespace AiDotNet.Tensors.Engines.Optimization.Schedulers;

/// <summary>
/// Base class for learning-rate schedulers.
/// Mirrors PyTorch <c>torch.optim.lr_scheduler._LRScheduler</c>:
///   * holds a reference to the optimizer
///   * <see cref="Step"/> increments the internal epoch counter and writes new LRs into each param group
///   * <see cref="GetLastLr"/> returns the LRs that were applied in the last <see cref="Step"/>
///
/// Concrete subclasses override <see cref="GetLr"/>, which produces one LR per param group at the
/// current <see cref="LastEpoch"/>.
/// </summary>
public abstract class LrScheduler
{
    /// <summary>Optimizer the scheduler controls.</summary>
    public IOptimizer Optimizer { get; }

    /// <summary>Initial (base) learning rate per param group, captured at construction.</summary>
    public double[] BaseLrs { get; }

    /// <summary>The epoch counter as last-stepped (PyTorch parity: −1 means "no steps yet").</summary>
    public int LastEpoch { get; protected set; }

    /// <summary>Last LR each param group received from <see cref="Step"/>.</summary>
    protected double[] _lastLrs;

    /// <summary>Construct a scheduler bound to <paramref name="optimizer"/>; capture base LRs.</summary>
    /// <remarks>
    /// Initial LR application is deferred — calling <see cref="GetLr"/> from this constructor
    /// would observe uninitialised fields on the derived class. Derived classes should call
    /// <see cref="ApplyInitialLrs"/> at the end of their own constructor.
    /// </remarks>
    protected LrScheduler(IOptimizer optimizer, int lastEpoch = -1)
    {
        Optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        BaseLrs = new double[optimizer.ParamGroups.Count];
        _lastLrs = new double[optimizer.ParamGroups.Count];
        for (int i = 0; i < optimizer.ParamGroups.Count; i++)
        {
            BaseLrs[i] = optimizer.ParamGroups[i].LearningRate;
            _lastLrs[i] = BaseLrs[i];
            optimizer.ParamGroups[i].LastLearningRate = BaseLrs[i];
        }
        LastEpoch = lastEpoch;
    }

    /// <summary>
    /// Apply LRs for epoch 0 once derived-class fields have been initialised.
    /// Should be called at the end of each derived constructor when <see cref="LastEpoch"/> is 0.
    /// </summary>
    protected void ApplyInitialLrs()
    {
        if (LastEpoch == -1)
        {
            LastEpoch = 0;
            ApplyLrs(GetLr());
            LastEpoch = 0;
        }
    }

    /// <summary>Return one LR per param group for the current <see cref="LastEpoch"/>.</summary>
    protected abstract IReadOnlyList<double> GetLr();

    /// <summary>Advance the epoch and apply new LRs to the optimizer.</summary>
    public virtual void Step(int? epoch = null)
    {
        LastEpoch = epoch ?? LastEpoch + 1;
        ApplyLrs(GetLr());
    }

    /// <summary>LRs that were actually written by the last <see cref="Step"/>.</summary>
    public IReadOnlyList<double> GetLastLr() => _lastLrs;

    /// <summary>Write LRs into the optimizer's param groups.</summary>
    protected void ApplyLrs(IReadOnlyList<double> lrs)
    {
        for (int i = 0; i < Optimizer.ParamGroups.Count; i++)
        {
            Optimizer.ParamGroups[i].LearningRate = lrs[i];
            Optimizer.ParamGroups[i].LastLearningRate = lrs[i];
            _lastLrs[i] = lrs[i];
        }
    }
}
