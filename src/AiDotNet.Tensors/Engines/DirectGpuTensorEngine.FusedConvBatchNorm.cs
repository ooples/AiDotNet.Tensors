// Copyright (c) AiDotNet. All rights reserved.
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// Folds on the device when any parameter's current value is only there (after on-device training): the host fold
    /// would download each one. Host-authoritative parameters (loaded weights) keep the host fold, whose folded kernel
    /// is host-authoritative too, so it uploads once and backend transforms keyed by host weights (the Winograd filter
    /// cache) still apply. Either way the fold is cached, and the convolution runs with the bias and activation fused.
    /// </summary>
    public override Tensor<T> FusedConv2DBatchNorm<T>(
        Tensor<T> input, Tensor<T> kernel, Tensor<T>? bias,
        Tensor<T> gamma, Tensor<T> beta, Tensor<T> runningMean, Tensor<T> runningVariance, double epsilon,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        bool recording = GraphMode.IsActive || (GradientTape<T>.Current is not null && !NoGradScope<T>.IsSuppressed);
        if (recording || !TryGetBackend(out _)
            || !(DeviceOnly(kernel) || (bias is not null && DeviceOnly(bias)) || DeviceOnly(gamma) || DeviceOnly(beta)
                 || DeviceOnly(runningMean) || DeviceOnly(runningVariance)))
            return base.FusedConv2DBatchNorm(input, kernel, bias, gamma, beta, runningMean, runningVariance, epsilon,
                strideH, strideW, padH, padW, dilationH, dilationW, activation);

        // Invalid shapes take the base path, which reports them.
        if (kernel.Rank != 4 || gamma.Length != kernel.Shape[0] || beta.Length != kernel.Shape[0]
            || runningMean.Length != kernel.Shape[0] || runningVariance.Length != kernel.Shape[0]
            || (bias is not null && bias.Length != kernel.Shape[0]))
            return base.FusedConv2DBatchNorm(input, kernel, bias, gamma, beta, runningMean, runningVariance, epsilon,
                strideH, strideW, padH, padW, dilationH, dilationW, activation);

        var (foldedKernel, foldedBias) = ConvBatchNormFold<T>.Get(kernel, bias, gamma, beta, runningMean,
            runningVariance, epsilon, FoldOnDevice);
        return FusedConv2D(input, foldedKernel, foldedBias, strideH, strideW, padH, padW, dilationH, dilationW, activation);
    }

    private bool DeviceOnly<T>(Tensor<T> tensor)
        => IsDeviceAuthoritative(tensor) || HostSync.IsPending(tensor.DataVector);

    // s = gamma / sqrt(var + eps); kernel' = kernel * s per output channel; bias' = beta + (bias - mean) * s.
    private (Tensor<T> Kernel, Tensor<T> Bias) FoldOnDevice<T>(Tensor<T> kernel, Tensor<T>? bias, Tensor<T> gamma,
        Tensor<T> beta, Tensor<T> mean, Tensor<T> variance, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        using var noGrad = new NoGradScope<T>();
        int outChannels = kernel.Shape[0];
        var scale = TensorDivide(gamma, TensorSqrt(TensorAddScalar(variance, ops.FromDouble(epsilon))));
        var perChannel = TensorBroadcastTo(Reshape(scale, new[] { outChannels, 1, 1, 1 }), kernel.Shape.ToArray());
        var foldedKernel = TensorMultiply(kernel, perChannel);
        var shifted = bias is null ? TensorMultiplyScalar(mean, ops.FromDouble(-1.0)) : TensorSubtract(bias, mean);
        var foldedBias = TensorAdd(beta, TensorMultiply(shifted, scale));
        return (foldedKernel, foldedBias);
    }
}
