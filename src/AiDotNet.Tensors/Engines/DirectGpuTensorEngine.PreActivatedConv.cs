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
        if (typeof(T) != typeof(float) || IsRecordingPreActivatedConv<T>() || input is null || kernel is null
            || input.Rank != 4 || kernel.Rank != 4 || preGamma is null || preBeta is null || preMean is null
            || preVariance is null || !TryGetBackend(out var backend) || backend is not IConvPreActivationBackend pre
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return base.BatchNormActivationConv2D(input!, preGamma!, preBeta!, preMean!, preVariance!, preEpsilon, kernel!,
                bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        int batch = input.Shape._dims[0], inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2], inWidth = input.Shape._dims[3];
        int outChannels = kernel.Shape._dims[0], kernelH = kernel.Shape._dims[2], kernelW = kernel.Shape._dims[3];
        int outHeight = (inHeight + 2 * padH - (kernelH + (kernelH - 1) * (dilationH - 1))) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - (kernelW + (kernelW - 1) * (dilationW - 1))) / strideW + 1;
        // Invalid shapes take the base path, which reports them.
        if (kernel.Shape._dims[1] != inChannels || preGamma.Length != inChannels || preBeta.Length != inChannels
            || preMean.Length != inChannels || preVariance.Length != inChannels
            || (bias is not null && (bias.Rank != 1 || bias.Length != outChannels))
            || outHeight <= 0 || outWidth <= 0 || batch <= 0)
            return base.BatchNormActivationConv2D(input, preGamma, preBeta, preMean, preVariance, preEpsilon, kernel, bias,
                strideH, strideW, padH, padW, dilationH, dilationW, activation);

        var (scale, shift) = BatchNormAffineFold<T>.Get(preGamma, preBeta, preMean, preVariance, preEpsilon);
        int count = batch * outChannels * outHeight * outWidth;
        using var inputBuffer = GetOrAllocateBuffer(backend, input);
        using var kernelBuffer = GetWeightBufferPreferResident(backend, kernel, PersistentTensorRole.Weights);
        using var scaleBuffer = GetWeightBufferPreferResident(backend, scale, PersistentTensorRole.Biases);
        using var shiftBuffer = GetWeightBufferPreferResident(backend, shift, PersistentTensorRole.Biases);
        using var biasBuffer = bias is null ? default : GetWeightBufferPreferResident(backend, bias, PersistentTensorRole.Biases);
        // The engine's kernels store the whole output.
        var outputBuffer = AllocateFullyWrittenOutputBuffer(backend, count);
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
