// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>What the global-norm clip does to the gradients when an element is NaN or infinite. Either way the
/// norm's record counts the NaN and infinite elements and names the first gradient holding one.</summary>
public enum NonFiniteGradientPolicy
{
    /// <summary>Leave every gradient unscaled (coefficient 1), so the step that the finiteness check then discards
    /// keeps its evidence.</summary>
    LeaveUnscaled = 0,

    /// <summary>PyTorch's <c>clip_grad_norm_(error_if_nonfinite=False)</c>: the norm is NaN or +Inf, and the gradients
    /// are multiplied by the resulting coefficient (NaN, or 0 for an infinite norm).</summary>
    Propagate = 1,
}
