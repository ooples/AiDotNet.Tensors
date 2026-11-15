// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> DepthwiseConv2DBatchNorm<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding,
        Tensor<T> gamma, Tensor<T> beta, Tensor<T> runningMean, Tensor<T> runningVariance, double epsilon,
        FusedActivationType activation)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (stride is null) throw new ArgumentNullException(nameof(stride));
        if (padding is null) throw new ArgumentNullException(nameof(padding));
        if (gamma is null) throw new ArgumentNullException(nameof(gamma));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (runningMean is null) throw new ArgumentNullException(nameof(runningMean));
        if (runningVariance is null) throw new ArgumentNullException(nameof(runningVariance));
        ValidateDepthwiseGeometry(input._shape, kernel._shape, null, stride, padding, "output");
        int channels = checked(input._shape[1] * kernel._shape[1]);
        if (gamma.Length != channels || beta.Length != channels
            || runningMean.Length != channels || runningVariance.Length != channels)
            throw new ArgumentException("Batch normalization parameters must match the depthwise output channels.");

        if (typeof(T) == typeof(float) && !SupportsGpu && !IsRecordingPreActivatedConv<T>()
            && (activation == FusedActivationType.ReLU || activation == FusedActivationType.ReLU6)
            && input.Rank == 4 && kernel.Rank == 4 && input.Layout == LinearAlgebra.TensorLayout.Nchw
            && input._shape[1] == kernel._shape[0] && gamma.Length == input._shape[1]
            && Simd.DepthwiseConv3x3.Applies(kernel._shape[1], kernel._shape[2], kernel._shape[3], stride[0], stride[1],
                padding[0], padding[1]))
        {
            return (Tensor<T>)(object)DepthwiseConv3x3BatchNormFloat((Tensor<float>)(object)input,
                (Tensor<float>)(object)kernel, stride[0], (Tensor<float>)(object)gamma, (Tensor<float>)(object)beta,
                (Tensor<float>)(object)runningMean, (Tensor<float>)(object)runningVariance, (float)epsilon,
                activation == FusedActivationType.ReLU6 ? 6f : float.PositiveInfinity);
        }
        var depthwise = DepthwiseConv2D(input, kernel, stride, padding);
        // Recording takes the differentiable eval-mode batch norm and a recorded activation, as PreActivate does;
        // BatchNormInference is not recorded.
        return IsRecordingPreActivatedConv<T>()
            ? ApplyActivationRecorded(BatchNormAffine(depthwise, gamma, beta, runningMean, runningVariance, epsilon), activation)
            : BatchNormInference(depthwise, gamma, beta, runningMean, runningVariance, epsilon, activation);
    }

    // The batch norm and its clamp applied as the depthwise kernel stores each output (DepthwiseConv3x3's epilogue, the
    // scale and shift NchwcBatchNorm packs): MobileNetV2's every depthwise stage wrote its activation, then read it back
    // and wrote the normalized copy, a second pass and a second buffer per layer.
    private static Tensor<float> DepthwiseConv3x3BatchNormFloat(Tensor<float> input, Tensor<float> kernel, int stride,
        Tensor<float> gamma, Tensor<float> beta, Tensor<float> mean, Tensor<float> variance, float epsilon, float high)
    {
        int batch = input._shape[0], channels = input._shape[1], h = input._shape[2], w = input._shape[3];
        int oH = (h - 1) / stride + 1, oW = (w - 1) / stride + 1;
        var result = TensorAllocator.RentUninitialized<float>(new[] { batch, channels, oH, oW });
        var scale = ArrayPool<float>.Shared.Rent(channels);
        var shift = ArrayPool<float>.Shared.Rent(channels);
        try
        {
            Simd.NchwcBatchNorm.PackScaleBias(gamma.AsSpanUnmarked(), beta.AsSpanUnmarked(), mean.AsSpanUnmarked(),
                variance.AsSpanUnmarked(), epsilon, channels, scale, shift);
            using var inputKeep = new KeepAliveScope(input);
            using var kernelKeep = new KeepAliveScope(kernel);
            using var resultKeep = new KeepAliveScope(result);
            Simd.DepthwiseConv3x3.Forward(input.GetFlattenedData(), kernel.GetFlattenedData(), result.GetDataArray(),
                batch, channels, h, w, stride, scale, shift, 0f, high);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(shift);
            ArrayPool<float>.Shared.Return(scale);
        }
        return result;
    }
}
