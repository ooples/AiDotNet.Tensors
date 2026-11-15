// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// The GPU depthwise convolution followed by the GPU inference batch norm with its activation fused into the same
    /// kernel, so the normalized output is the only extra pass. Under a tape, the differentiable eval-mode batch norm and
    /// a recorded activation, as the CPU engine records them (BatchNormInference is not recorded).
    /// </summary>
    public override Tensor<T> DepthwiseConv2DBatchNorm<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding,
        Tensor<T> gamma, Tensor<T> beta, Tensor<T> runningMean, Tensor<T> runningVariance, double epsilon,
        FusedActivationType activation)
    {
        if (input is null) throw new System.ArgumentNullException(nameof(input));
        if (kernel is null) throw new System.ArgumentNullException(nameof(kernel));
        if (stride is null) throw new System.ArgumentNullException(nameof(stride));
        if (padding is null) throw new System.ArgumentNullException(nameof(padding));
        if (gamma is null) throw new System.ArgumentNullException(nameof(gamma));
        if (beta is null) throw new System.ArgumentNullException(nameof(beta));
        if (runningMean is null) throw new System.ArgumentNullException(nameof(runningMean));
        if (runningVariance is null) throw new System.ArgumentNullException(nameof(runningVariance));

        var depthwise = DepthwiseConv2D(input, kernel, stride, padding);
        return IsRecordingPreActivatedConv<T>()
            ? ApplyActivationRecorded(BatchNormAffine(depthwise, gamma, beta, runningMean, runningVariance, epsilon), activation)
            : BatchNormInference(depthwise, gamma, beta, runningMean, runningVariance, epsilon, activation);
    }
}
