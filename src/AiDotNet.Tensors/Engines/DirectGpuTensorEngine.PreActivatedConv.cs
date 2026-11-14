// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// On a backend whose convolution applies a per-input-channel affine and ReLU as it loads its input
    /// (<see cref="IConvPreActivationBackend"/>), the batch norm and ReLU never write the normalized activation: on a b16
    /// DenseNet-121 predict that pass was ~1.8 ms of the 58 dense layers' pre-activations, which torch.compile still runs
    /// as its own kernel before the cuDNN convolution. Otherwise, and under a tape or graph capture, the base path.
    /// </summary>
    public override Tensor<T> BatchNormActivationConv2D<T>(
        Tensor<T> input, Tensor<T> preGamma, Tensor<T> preBeta, Tensor<T> preMean, Tensor<T> preVariance, double preEpsilon,
        Tensor<T> kernel, Tensor<T>? bias,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        using var __gpuOp = EnterGpuOp();
        if (input is null) throw new System.ArgumentNullException(nameof(input));
        if (preGamma is null) throw new System.ArgumentNullException(nameof(preGamma));
        if (preBeta is null) throw new System.ArgumentNullException(nameof(preBeta));
        if (preMean is null) throw new System.ArgumentNullException(nameof(preMean));
        if (preVariance is null) throw new System.ArgumentNullException(nameof(preVariance));
        if (kernel is null) throw new System.ArgumentNullException(nameof(kernel));
        if (typeof(T) != typeof(float) || IsRecordingPreActivatedConv<T>()
            || input.Rank != 4 || kernel.Rank != 4
            || strideH <= 0 || strideW <= 0 || padH < 0 || padW < 0 || dilationH <= 0 || dilationW <= 0
            || !TryGetBackend(out var backend) || backend is not IConvPreActivationBackend pre
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return base.BatchNormActivationConv2D(input, preGamma, preBeta, preMean, preVariance, preEpsilon, kernel,
                bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        int batch = input.Shape._dims[0], inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2], inWidth = input.Shape._dims[3];
        int outChannels = kernel.Shape._dims[0], kernelH = kernel.Shape._dims[2], kernelW = kernel.Shape._dims[3];
        long spanH = inHeight + 2L * padH - (kernelH + (long)(kernelH - 1) * (dilationH - 1));
        long spanW = inWidth + 2L * padW - (kernelW + (long)(kernelW - 1) * (dilationW - 1));
        long outHeightL = spanH < 0 ? 0 : spanH / strideH + 1;
        long outWidthL = spanW < 0 ? 0 : spanW / strideW + 1;
        // Invalid or overflowing shapes take the base path, which reports them.
        if (batch <= 0 || inChannels <= 0 || inHeight <= 0 || inWidth <= 0
            || outChannels <= 0 || kernelH <= 0 || kernelW <= 0
            || kernel.Shape._dims[1] != inChannels || preGamma.Length != inChannels || preBeta.Length != inChannels
            || preMean.Length != inChannels || preVariance.Length != inChannels
            || (bias is not null && (bias.Rank != 1 || bias.Length != outChannels))
            || outHeightL <= 0 || outWidthL <= 0 || outHeightL > int.MaxValue || outWidthL > int.MaxValue)
            return base.BatchNormActivationConv2D(input, preGamma, preBeta, preMean, preVariance, preEpsilon, kernel, bias,
                strideH, strideW, padH, padW, dilationH, dilationW, activation);

        int outHeight = (int)outHeightL, outWidth = (int)outWidthL;
        long bc = (long)batch * outChannels;
        long hw = (long)outHeight * outWidth;
        long countL = bc <= int.MaxValue && hw <= int.MaxValue ? bc * hw : (long)int.MaxValue + 1;
        if (countL <= 0 || countL > int.MaxValue)
            return base.BatchNormActivationConv2D(input, preGamma, preBeta, preMean, preVariance, preEpsilon, kernel, bias,
                strideH, strideW, padH, padW, dilationH, dilationW, activation);

        int count = (int)countL;
        var (scale, shift) = BatchNormAffineFold<T>.Get(preGamma, preBeta, preMean, preVariance, preEpsilon);
        using var inputBuffer = GetOrAllocateBuffer(backend, input);
        using var kernelBuffer = GetWeightBufferPreferResident(backend, kernel, PersistentTensorRole.Weights);
        using var scaleBuffer = GetWeightBufferPreferResident(backend, scale, PersistentTensorRole.Biases);
        using var shiftBuffer = GetWeightBufferPreferResident(backend, shift, PersistentTensorRole.Biases);
        using var biasBuffer = bias is null ? default : GetWeightBufferPreferResident(backend, bias, PersistentTensorRole.Biases);
        // The engine's kernels store the whole output.
        using var outputBuffer = AllocateFullyWrittenOutputBuffer(backend, count);
        if (!pre.TryConv2DPreActivatedBiasActivation(inputBuffer.Buffer, scaleBuffer.Buffer, shiftBuffer.Buffer,
                kernelBuffer.Buffer, bias is null ? null : biasBuffer.Buffer, outputBuffer.Buffer, batch, inChannels, inHeight,
                inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH,
                dilationW, activation))
        {
            outputBuffer.Dispose();
            return base.BatchNormActivationConv2D(input, preGamma, preBeta, preMean, preVariance, preEpsilon, kernel, bias,
                strideH, strideW, padH, padW, dilationH, dilationW, activation);
        }
        var result = DeferTensorResult<T>(backend, outputBuffer.Buffer, count, new[] { batch, outChannels, outHeight, outWidth });
        if (ResidentStepActive) BindResidentBuffer(result, outputBuffer.Buffer, backend);
        outputBuffer.RelinquishOwnership();
        return result;
    }

    /// <summary>
    /// The convolution's own batch norm folds into its kernel and bias (on the device when the parameters live there,
    /// <see cref="FoldConvBatchNormForPreActivatedConv{T}"/>), then the pre-activated convolution above runs: one launch
    /// for batch norm, ReLU, convolution, batch norm. Under a tape or graph capture, the recorded unfused ops.
    /// </summary>
    public override Tensor<T> BatchNormActivationConv2DBatchNorm<T>(
        Tensor<T> input, Tensor<T> preGamma, Tensor<T> preBeta, Tensor<T> preMean, Tensor<T> preVariance, double preEpsilon,
        Tensor<T> kernel, Tensor<T>? bias,
        Tensor<T> gamma, Tensor<T> beta, Tensor<T> runningMean, Tensor<T> runningVariance, double epsilon,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        using var __gpuOp = EnterGpuOp();
        return base.BatchNormActivationConv2DBatchNorm(input, preGamma, preBeta, preMean, preVariance, preEpsilon, kernel,
            bias, gamma, beta, runningMean, runningVariance, epsilon, strideH, strideW, padH, padW, dilationH, dilationW,
            activation);
    }

    // The fold FusedConv2DBatchNorm uses: on the device when any parameter's current value is only there.
    private protected override (Tensor<T> Kernel, Tensor<T> Bias) FoldConvBatchNormForPreActivatedConv<T>(Tensor<T> kernel,
        Tensor<T>? bias, Tensor<T> gamma, Tensor<T> beta, Tensor<T> mean, Tensor<T> variance, double epsilon)
        => DeviceOnly(kernel) || (bias is not null && DeviceOnly(bias)) || DeviceOnly(gamma) || DeviceOnly(beta)
           || DeviceOnly(mean) || DeviceOnly(variance)
            ? ConvBatchNormFold<T>.Get(kernel, bias, gamma, beta, mean, variance, epsilon, FoldOnDevice, ReleaseFolded)
            : base.FoldConvBatchNormForPreActivatedConv(kernel, bias, gamma, beta, mean, variance, epsilon);
}
