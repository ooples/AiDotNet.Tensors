// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Direct GPU backends that can update a WHOLE LIST of fp32 Adam/AdamW parameter tensors
/// in a SINGLE kernel launch (apex <c>multi_tensor_apply</c> / PyTorch <c>foreach</c> style),
/// instead of one launch per tensor. On the per-tensor path a deep model with hundreds of
/// small parameter tensors pays hundreds of kernel-launch + argument-marshal + scheduling
/// round-trips every optimizer step; those fixed per-launch costs dominate when the tensors
/// are small. Batching them into one launch amortizes that overhead.
/// </summary>
/// <remarks>
/// This is a capability interface (like <see cref="ICompressedMomentGpuOptimizerBackend"/>):
/// callers probe for it with <c>backend is IMultiTensorGpuOptimizerBackend</c> and fall back
/// to the per-tensor <see cref="IDirectGpuBackend.AdamUpdate"/> loop when it is absent, so a
/// backend that does not implement it keeps working unchanged. The batched update is
/// numerically identical to calling the single-tensor kernel once per tensor — only the launch
/// batching changes.
/// </remarks>
public interface IMultiTensorGpuOptimizerBackend : IDirectGpuBackend
{
    /// <summary>
    /// Adam update over a list of fp32 parameter tensors in one launch. The four buffer lists
    /// are parallel: index <c>i</c> gives the param / gradient / first-moment / second-moment
    /// buffers for tensor <c>i</c>, and <paramref name="sizes"/>[i] its element count. Every
    /// buffer is updated in place; the update matches <see cref="IDirectGpuBackend.AdamUpdate"/>
    /// applied per tensor with the same hyperparameters and <paramref name="step"/>.
    /// </summary>
    void AdamMultiTensorUpdate(
        IReadOnlyList<IGpuBuffer> parameters,
        IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments,
        IReadOnlyList<IGpuBuffer> secondMoments,
        IReadOnlyList<int> sizes,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        int step);

    /// <summary>
    /// AdamW (decoupled weight decay) counterpart of <see cref="AdamMultiTensorUpdate"/>.
    /// Matches <see cref="IDirectGpuBackend.AdamWUpdate"/> applied per tensor.
    /// </summary>
    void AdamWMultiTensorUpdate(
        IReadOnlyList<IGpuBuffer> parameters,
        IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments,
        IReadOnlyList<IGpuBuffer> secondMoments,
        IReadOnlyList<int> sizes,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        int step);
}
