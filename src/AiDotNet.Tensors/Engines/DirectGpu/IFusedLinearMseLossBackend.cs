// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: the mean squared error of a linear head's output, <c>mean((x w + b - y)^2)</c>, and every
/// gradient of the head, each in one launch. The compiled training plan fuses a bias-and-identity <c>FusedLinear</c>
/// whose only consumer is the fused mean-squared-error loss into one node that runs these; composed, the head and the
/// loss were seven nodes of a parity LSTM's captured step (the head GEMM, the loss, its gradient, the two gradient
/// GEMMs, the bias's column sum and a copy), ~11 us of a ~63 us replay. False - nothing launched - when the shapes are
/// outside the one-block kernels; the caller then composes FusedLinear and the fused loss.
/// </summary>
[BackendSpecificRoute("CUDA one-launch linear head + mean-squared-error loss and its gradients for the compiled plan's fused head node; every backend composes the head's GEMM, the loss and the gradient GEMMs",
    nameof(IDirectGpuBackend.Gemm), nameof(IDirectGpuBackend.Subtract), nameof(IDirectGpuBackend.MeanAxis))]
internal interface IFusedLinearMseLossBackend
{
    /// <summary>
    /// True when the kernels take a head of <paramref name="m"/> rows, <paramref name="k"/> inputs and
    /// <paramref name="n"/> outputs. The compiled plan fuses only such heads: a fused node whose kernels decline composes
    /// its backward from a recomputed head, slower than the unfused FusedLinear and loss nodes.
    /// </summary>
    bool SupportsLinearMseLoss(int m, int k, int n);

    /// <summary>
    /// Writes <c>mean((x w + b - y)^2)</c> to <paramref name="loss"/>[0] for x <c>[m, k]</c>, w <c>[k, n]</c>, b
    /// <c>[n]</c> (or none) and y <c>[m, n]</c>.
    /// </summary>
    bool TryLinearMseLoss(IGpuBuffer x, IGpuBuffer w, IGpuBuffer? b, IGpuBuffer y, IGpuBuffer loss, int m, int k, int n);

    /// <summary>
    /// With <c>dp = 2 (x w + b - y) g / (m n)</c> and <c>g = gradLoss[0]</c>: writes <c>dx = dp w^T</c>,
    /// <c>dw = x^T dp</c> and, when <paramref name="db"/> is not null, <c>db</c> = the column sums of dp.
    /// </summary>
    bool TryLinearMseLossBackward(IGpuBuffer x, IGpuBuffer w, IGpuBuffer? b, IGpuBuffer y, IGpuBuffer gradLoss,
        IGpuBuffer dx, IGpuBuffer dw, IGpuBuffer? db, int m, int k, int n);
}
