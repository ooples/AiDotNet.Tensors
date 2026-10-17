// Copyright (c) AiDotNet. All rights reserved.
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// The compiled plan's fused mean-squared-error node (MseLossFusionPattern) on backends with IFusedMseLossBackend.
public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// <c>mean((predictions - targets)^2)</c> in one launch, shaped <paramref name="outputShape"/> (the fused
    /// <c>ReduceMean</c>'s). Null - nothing launched - when the backend or the tensors do not take the kernel; the caller
    /// then composes the loss. Records nothing: the fused node supplies the backward.
    /// </summary>
    internal Tensor<T>? TryFusedMseLoss<T>(Tensor<T> predictions, Tensor<T> targets, int[] outputShape)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedMseLossBackend mse
            || predictions.Length != targets.Length || predictions.Length == 0
            || !predictions.IsContiguous || !targets.IsContiguous)
            return null;
        using var bufP = GetOrAllocateBuffer(backend, predictions);
        using var bufY = GetOrAllocateBuffer(backend, targets);
        var bufLoss = AllocateFullyWrittenOutputBuffer(backend, 1);
        if (!mse.TryMseLoss(bufP.Buffer, bufY.Buffer, bufLoss.Buffer, predictions.Length))
        {
            bufLoss.Dispose();
            return null;
        }
        return DeferTensorResult<T>(backend, bufLoss.Buffer, 1, (int[])outputShape.Clone());
    }

    /// <summary>
    /// The fused loss's gradients in one launch: <c>2 (p - y) g / n</c> for the predictions, and its negation for the
    /// targets when <paramref name="targetGradient"/>. Null when unavailable; the caller then composes them.
    /// </summary>
    internal Tensor<T>? TryFusedMseLossBackward<T>(Tensor<T> gradLoss, Tensor<T> predictions, Tensor<T> targets,
        bool targetGradient, out Tensor<T>? gradTargets)
    {
        gradTargets = null;
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedMseLossBackend mse
            || predictions.Length != targets.Length || predictions.Length == 0 || gradLoss.Length < 1
            || !predictions.IsContiguous || !targets.IsContiguous || !gradLoss.IsContiguous)
            return null;
        int n = predictions.Length;
        using var bufP = GetOrAllocateBuffer(backend, predictions);
        using var bufY = GetOrAllocateBuffer(backend, targets);
        using var bufG = GetOrAllocateBuffer(backend, gradLoss);
        var bufDp = AllocateFullyWrittenOutputBuffer(backend, n);
        var bufDy = targetGradient ? AllocateFullyWrittenOutputBuffer(backend, n) : default;
        if (!mse.TryMseLossBackward(bufP.Buffer, bufY.Buffer, bufG.Buffer, bufDp.Buffer,
                targetGradient ? bufDy.Buffer : null, n))
        {
            bufDp.Dispose();
            bufDy.Dispose();
            return null;
        }
        if (targetGradient) gradTargets = DeferTensorResult<T>(backend, bufDy.Buffer, n, targets.Shape.ToArray());
        return DeferTensorResult<T>(backend, bufDp.Buffer, n, predictions.Shape.ToArray());
    }
}
