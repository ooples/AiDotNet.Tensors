// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Loss scaling for a compiled training plan, PyTorch's <c>torch.amp.GradScaler</c>: the backward starts from
/// <c>scale</c> instead of 1, every gradient is unscaled before the plan's clip and optimizer, a step whose unscaled
/// gradients hold a NaN or infinity is skipped, and (when <see cref="Dynamic"/>) the scale backs off after such a step and
/// grows after <see cref="GrowthInterval"/> finite steps in a row. On CUDA the whole of it runs on the device, inside a
/// captured step graph, with no host read per step.
/// </summary>
public sealed class LossScalingOptions
{
    /// <summary>The first step's scale. Default 65536 (2^16), PyTorch's <c>init_scale</c>.</summary>
    public float InitialScale { get; set; } = 65536f;

    /// <summary>The scale's growth after <see cref="GrowthInterval"/> finite steps. Default 2.</summary>
    public float GrowthFactor { get; set; } = 2f;

    /// <summary>The scale's factor after a non-finite step. Default 0.5.</summary>
    public float BackoffFactor { get; set; } = 0.5f;

    /// <summary>Finite steps in a row before the scale grows. Default 2000.</summary>
    public int GrowthInterval { get; set; } = 2000;

    /// <summary>The scale never backs off below this. Default 1.</summary>
    public float MinScale { get; set; } = 1f;

    /// <summary>The scale never grows above this. Default 2^24.</summary>
    public float MaxScale { get; set; } = 16777216f;

    /// <summary>False keeps the scale constant (non-finite steps are still skipped). Default true.</summary>
    public bool Dynamic { get; set; } = true;

    internal void Validate()
    {
        if (!(InitialScale > 0f) || float.IsInfinity(InitialScale))
            throw new ArgumentOutOfRangeException(nameof(InitialScale), InitialScale, "The scale must be positive and finite.");
        if (!(GrowthFactor >= 1f) || float.IsInfinity(GrowthFactor))
            throw new ArgumentOutOfRangeException(nameof(GrowthFactor), GrowthFactor, "The growth factor must be at least 1.");
        if (!(BackoffFactor > 0f && BackoffFactor <= 1f))
            throw new ArgumentOutOfRangeException(nameof(BackoffFactor), BackoffFactor, "The backoff factor must be in (0, 1].");
        if (GrowthInterval < 1)
            throw new ArgumentOutOfRangeException(nameof(GrowthInterval), GrowthInterval, "The growth interval must be positive.");
        if (!(MinScale > 0f) || !(MaxScale >= MinScale) || float.IsInfinity(MaxScale))
            throw new ArgumentOutOfRangeException(nameof(MinScale), "The scale bounds must satisfy 0 < MinScale <= MaxScale < Infinity.");
        if (InitialScale < MinScale || InitialScale > MaxScale)
            throw new ArgumentOutOfRangeException(nameof(InitialScale), InitialScale,
                $"The initial scale must lie within [MinScale, MaxScale] = [{MinScale}, {MaxScale}].");
    }
}

/// <summary>A loss scaler's state: the scale, the finite steps since it last changed, and its step counts.</summary>
/// <param name="Scale">The scale the next backward starts from.</param>
/// <param name="GrowthTracker">Finite steps in a row since the scale last changed.</param>
/// <param name="OverflowSteps">Steps skipped because a gradient was NaN or infinite.</param>
/// <param name="Steps">Steps the scaler has seen.</param>
public readonly record struct LossScaleState(float Scale, int GrowthTracker, long OverflowSteps, long Steps);

/// <summary>A compiled training plan that can scale its loss (<see cref="LossScalingOptions"/>).</summary>
public interface ICompiledTrainingPlanLossScaling
{
    /// <summary>
    /// Turns loss scaling on with <paramref name="options"/> (null turns it off), starting from
    /// <paramref name="initialState"/> or the options' initial scale. The plan's non-finite gradient policy must discard
    /// a non-finite step (<see cref="Gpu.NonFiniteGradientPolicy.LeaveUnscaled"/> or
    /// <see cref="Gpu.NonFiniteGradientPolicy.Propagate"/>).
    /// </summary>
    void ConfigureLossScaling(LossScalingOptions? options, LossScaleState? initialState = null);

    /// <summary>The scaler's state, or null when loss scaling is off. On the device this reads it back (one
    /// synchronization), so call it for checkpoints and reports, not every step.</summary>
    LossScaleState? ReadLossScaleState();
}
