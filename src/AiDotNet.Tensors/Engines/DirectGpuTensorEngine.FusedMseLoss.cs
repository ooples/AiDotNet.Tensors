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

    // x [m, k], w [k, n], b [n] or none, y m * n elements, all contiguous: the fused head's shapes.
    private static bool TryGetLinearMseShape<T>(Tensor<T> x, Tensor<T> w, Tensor<T>? b, Tensor<T> y,
        out int m, out int k, out int n)
    {
        m = k = n = 0;
        if (x.Rank != 2 || w.Rank != 2 || x._shape[1] != w._shape[0]) return false;
        m = x._shape[0]; k = x._shape[1]; n = w._shape[1];
        return m > 0 && k > 0 && n > 0 && y.Length == m * n && (b is null || (b.Rank == 1 && b.Length == n))
            && x.IsContiguous && w.IsContiguous && y.IsContiguous && (b is null || b.IsContiguous);
    }

    /// <summary>
    /// True when this engine's backend runs the fused linear head + loss kernels for an <c>[m, k] x [k, n]</c> head (the
    /// compiled plan fuses only those).
    /// </summary>
    internal bool SupportsFusedLinearMseLoss(int m, int k, int n)
        => TryGetBackend(out var backend) && backend is IFusedLinearMseLossBackend head && head.SupportsLinearMseLoss(m, k, n);

    /// <summary>
    /// <c>mean((x w + b - y)^2)</c> in one launch, shaped <paramref name="outputShape"/>: the compiled plan's fused linear
    /// head and loss (LinearMseLossFusionPattern). The head's output is never materialized. Null - nothing launched -
    /// when the backend or the shapes do not take the kernel. Records nothing: the fused node supplies the backward.
    /// </summary>
    internal Tensor<T>? TryFusedLinearMseLoss<T>(Tensor<T> x, Tensor<T> w, Tensor<T>? b, Tensor<T> y, int[] outputShape)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedLinearMseLossBackend head
            || !TryGetLinearMseShape(x, w, b, y, out int m, out int k, out int n))
            return null;
        using var bufX = GetOrAllocateBuffer(backend, x);
        using var bufW = GetOrAllocateBuffer(backend, w);
        using var bufB = b is not null ? GetOrAllocateBuffer(backend, b) : default;
        using var bufY = GetOrAllocateBuffer(backend, y);
        var bufLoss = AllocateFullyWrittenOutputBuffer(backend, 1);
        if (!head.TryLinearMseLoss(bufX.Buffer, bufW.Buffer, b is not null ? bufB.Buffer : null, bufY.Buffer,
                bufLoss.Buffer, m, k, n))
        {
            bufLoss.Dispose();
            return null;
        }
        return DeferTensorResult<T>(backend, bufLoss.Buffer, 1, (int[])outputShape.Clone());
    }

    /// <summary>
    /// The fused head's gradients in one launch, from the loss's gradient: with <c>dp = 2 (x w + b - y) g / (m n)</c>,
    /// <c>dx = dp w^T</c>, <c>dw = x^T dp</c> and <c>db</c> the column sums of dp (when there is a bias). False when
    /// unavailable; the caller then composes them.
    /// </summary>
    internal bool TryFusedLinearMseLossBackward<T>(Tensor<T> gradLoss, Tensor<T> x, Tensor<T> w, Tensor<T>? b,
        Tensor<T> y, out Tensor<T>? gradX, out Tensor<T>? gradW, out Tensor<T>? gradB)
    {
        gradX = gradW = gradB = null;
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedLinearMseLossBackend head
            || gradLoss.Length < 1 || !gradLoss.IsContiguous || !TryGetLinearMseShape(x, w, b, y, out int m, out int k, out int n))
            return false;
        using var bufX = GetOrAllocateBuffer(backend, x);
        using var bufW = GetOrAllocateBuffer(backend, w);
        using var bufB = b is not null ? GetOrAllocateBuffer(backend, b) : default;
        using var bufY = GetOrAllocateBuffer(backend, y);
        using var bufG = GetOrAllocateBuffer(backend, gradLoss);
        var bufDx = AllocateFullyWrittenOutputBuffer(backend, m * k);
        var bufDw = AllocateFullyWrittenOutputBuffer(backend, k * n);
        var bufDb = b is not null ? AllocateFullyWrittenOutputBuffer(backend, n) : default;
        if (!head.TryLinearMseLossBackward(bufX.Buffer, bufW.Buffer, b is not null ? bufB.Buffer : null, bufY.Buffer,
                bufG.Buffer, bufDx.Buffer, bufDw.Buffer, b is not null ? bufDb.Buffer : null, m, k, n))
        {
            bufDx.Dispose();
            bufDw.Dispose();
            bufDb.Dispose();
            return false;
        }
        gradX = DeferTensorResult<T>(backend, bufDx.Buffer, m * k, x.Shape.ToArray());
        gradW = DeferTensorResult<T>(backend, bufDw.Buffer, k * n, w.Shape.ToArray());
        if (b is not null) gradB = DeferTensorResult<T>(backend, bufDb.Buffer, n, b.Shape.ToArray());
        return true;
    }
}
