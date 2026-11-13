// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Capability: the batched numeric recurrences behind <see cref="IEngine.TensorTridiagonalSolve{T}"/>,
/// <see cref="IEngine.TensorFritschCarlsonLimit{T}"/>, <see cref="IEngine.TensorCTCForwardBackward{T}"/>,
/// <see cref="IEngine.TensorCTCLabelPosteriors{T}"/> and <see cref="IEngine.TensorLinearSumAssignment{T}"/> as device
/// work, each recurrence inside one kernel (one thread or one work-group per independent problem).
/// </summary>
/// <remarks>
/// Every backend runs <see cref="NumericRecurrenceKernelTemplate"/>'s kernels through
/// <see cref="PortableNumericRecurrence"/>: OpenCL, HIP, Vulkan and Metal in their dialects, CUDA the HIP dialect through
/// NVRTC, WebGPU the WGSL restatement. Integer inputs (targets, lengths) ride as exactly-valued floats. A false return
/// launches nothing and the caller runs the host op.
/// </remarks>
internal interface INumericRecurrenceBackend
{
    /// <summary>x = A^-1 rhs for <paramref name="batch"/> systems of size n (lower and upper [batch, n - 1], diagonal,
    /// rhs and x [batch, n]); <paramref name="workspace"/> holds at least max(1, batch (n - 1)) floats.</summary>
    bool TryTridiagonalSolve(IGpuBuffer lower, IGpuBuffer diagonal, IGpuBuffer upper, IGpuBuffer rhs, IGpuBuffer x,
        IGpuBuffer workspace, int batch, int n);

    /// <summary>The Fritsch-Carlson step-2 limiter on <paramref name="batch"/> series: slopes and limited [batch, n],
    /// secants [batch, n - 1].</summary>
    bool TryFritschCarlsonLimit(IGpuBuffer slopes, IGpuBuffer secants, IGpuBuffer limited, int batch, int n);

    /// <summary>
    /// Log-alpha and log-beta [batch, maxTime, maxStates] and each item's log-likelihood [batch] (at most -1e29 when the
    /// target has no alignment) from time-major log-probabilities [maxTime, batch, classes]. <paramref name="lengths"/>
    /// holds the input lengths, the target lengths and the target offsets, batch floats each.
    /// </summary>
    bool TryCtcForwardBackward(IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer lengths, IGpuBuffer logAlpha,
        IGpuBuffer logBeta, IGpuBuffer logLikelihood, int maxTime, int batch, int classes, int maxStates, int blank);

    /// <summary>gamma [maxTime, batch, classes] from <see cref="TryCtcForwardBackward"/>'s outputs.</summary>
    bool TryCtcLabelPosteriors(IGpuBuffer logAlpha, IGpuBuffer logBeta, IGpuBuffer targets, IGpuBuffer lengths,
        IGpuBuffer logLikelihood, IGpuBuffer gamma, int maxTime, int batch, int classes, int maxStates, int blank);

    /// <summary>The column of each row [batch, size] (as floats) of the minimum-cost assignment of each cost
    /// [batch, size, size]; <paramref name="workspace"/> holds at least batch * 8 * size floats (the solver state).</summary>
    bool TryLinearSumAssignment(IGpuBuffer cost, IGpuBuffer assignment, IGpuBuffer workspace, int batch, int size);
}
