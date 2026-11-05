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

    /// <summary>Zero only the NaN and infinite elements and clip every other element by the norm of the finite ones
    /// (the sum the record holds; the record's norm itself stays NaN or +Inf, reporting the gradients as they were).
    /// </summary>
    ZeroNonFinite = 2,

    /// <summary>PyTorch's <c>clip_grad_norm_(error_if_nonfinite=True)</c>: leave the gradients untouched and throw
    /// <see cref="System.ArithmeticException"/> naming the first non-finite gradient. Raising needs the record on the
    /// host, so this policy reads it (one synchronization) after the clip's kernels are queued.</summary>
    Error = 3,
}
