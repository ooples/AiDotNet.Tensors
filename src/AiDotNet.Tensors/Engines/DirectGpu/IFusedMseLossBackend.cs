// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: the mean squared error of two equal-length tensors, and its gradient, each in one launch. The
/// compiled training plan fuses <c>ReduceMean(Square(Subtract(p, y)))</c> - how a mean-squared-error loss is usually
/// written, AiDotNet's own included - into one node that runs these; composed, the loss and its backward were about ten
/// graph nodes (subtract, square, mean, the mean's zero-fill and the gradient's broadcast, two multiplies, an add, a
/// negate and the copies between them). False - nothing launched - when the size is outside the kernels; the caller
/// then composes Subtract, Multiply and MeanAxis.
/// </summary>
[BackendSpecificRoute("CUDA one-launch mean-squared-error loss and gradient for the compiled plan's fused MSE node; every backend composes the loss from Subtract, Multiply and MeanAxis",
    nameof(IDirectGpuBackend.Subtract), nameof(IDirectGpuBackend.MeanAxis))]
internal interface IFusedMseLossBackend
{
    /// <summary>Writes <c>mean((p - y)^2)</c> over <paramref name="length"/> elements to <paramref name="loss"/>[0].</summary>
    bool TryMseLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int length);

    /// <summary>
    /// Writes <c>gradPredictions = 2 (p - y) g / length</c> with <c>g = gradLoss[0]</c>, and its negation to
    /// <paramref name="gradTargets"/> when that is not null. Every element of each output is written.
    /// </summary>
    bool TryMseLossBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradLoss, IGpuBuffer gradPredictions,
        IGpuBuffer? gradTargets, int length);
}
