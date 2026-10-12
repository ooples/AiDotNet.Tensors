// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

// No FP16-native activation kernels here: every capability reports false, so the engine keeps activations in FP32.
// The operations still work: PortableFp16Activation widens, runs the FP32 kernel and narrows back.
public sealed partial class OpenClBackend : IFp16ActivationBackend
{
    bool IFp16ActivationBackend.SupportsFp16ActivationStorage => false;

    bool IFp16ActivationBackend.SupportsFp16NativeOps => false;

    bool IFp16ActivationBackend.Fp16BroadcastAddChannelAvailable => false;

    bool IFp16ActivationBackend.Fp16DirectConvAvailable => false;

    bool IFp16ActivationBackend.Fp16GroupNormSwishAvailable => false;

    bool IFp16ActivationBackend.Fp16Im2colFromFp16Available => false;

    void IFp16ActivationBackend.Fp16Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => PortableFp16Activation.Add(this, a, b, output, size);

    void IFp16ActivationBackend.Fp16BroadcastAddChannel(IGpuBuffer ioHalf, IGpuBuffer biasHalf, int batch, int channels, int spatial)
        => PortableFp16Activation.BroadcastAddChannel(this, ioHalf, biasHalf, batch, channels, spatial);

    void IFp16ActivationBackend.Fp16Relu(IGpuBuffer input, IGpuBuffer output, int size)
        => PortableFp16Activation.Unary(this, input, output, size, gelu: false);

    void IFp16ActivationBackend.Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int size)
        => PortableFp16Activation.Unary(this, input, output, size, gelu: true);

    void IFp16ActivationBackend.Fp16ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => PortableFp16Activation.UnaryBackward(this, gradOutput, input, gradInput, size, gelu: false);

    void IFp16ActivationBackend.Fp16GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => PortableFp16Activation.UnaryBackward(this, gradOutput, input, gradInput, size, gelu: true);

    void IFp16ActivationBackend.Fp16SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
        => PortableFp16Activation.SoftmaxBackward(this, gradOutput, output, gradInput, batchSize, features);

    void IFp16ActivationBackend.Fp16LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveMeanFp32, IGpuBuffer saveInvVarFp32, IGpuBuffer gradInput, int rows, int cols)
        => PortableFp16Activation.LayerNormBackward(this, gradOutput, input, gamma, saveMeanFp32, saveInvVarFp32, gradInput, null, null, rows, cols);

    void IFp16ActivationBackend.Fp16LayerNormGradParams(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer saveMeanFp32, IGpuBuffer saveInvVarFp32, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int rows, int cols)
        => PortableFp16Activation.LayerNormBackward(this, gradOutput, input, null, saveMeanFp32, saveInvVarFp32, null, gradGamma, gradBeta, rows, cols);

    void IFp16ActivationBackend.Fp16GroupNormSwish(IGpuBuffer inHalf, IGpuBuffer gammaFp32, IGpuBuffer betaFp32, IGpuBuffer outHalf, int batch, int numGroups, int channels, int spatial, float eps)
        => PortableFp16Activation.GroupNormSwish(this, inHalf, gammaFp32, betaFp32, outHalf, batch, numGroups, channels, spatial, eps);

    void IFp16ActivationBackend.ConvertToFp16Native(IGpuBuffer input, IGpuBuffer output, int size)
        => ConvertToFp16(input, output, size);

    void IFp16ActivationBackend.ConvertToFp32Native(IGpuBuffer input, IGpuBuffer output, int size)
        => ConvertToFp32(input, output, size);

    void IFp16ActivationBackend.Conv2dDirectFp16Hw(IGpuBuffer input, IGpuBuffer weightsHalf, IGpuBuffer output, int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => PortableFp16Activation.Conv2dHalfWeights(this, input, weightsHalf, output, batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);

    void IFp16ActivationBackend.UnfoldKNFp16FromFp16(IGpuBuffer inputHalf, IGpuBuffer outputHalf, int batch, int channels, int height, int width, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => PortableFp16Activation.UnfoldKN(this, inputHalf, outputHalf, batch, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);

    void IFp16ActivationBackend.GemmFp16HalfOut(IGpuBuffer Ahalf, IGpuBuffer Bhalf, IGpuBuffer Chalf, int M, int N, int K, float alpha, float beta)
        => PortableFp16Activation.GemmHalfOut(this, Ahalf, Bhalf, Chalf, M, N, K, alpha, beta);

    void IFp16ActivationBackend.CopyBufferDtoD(IGpuBuffer source, IGpuBuffer destination, long byteCount)
        => PortableFp16Activation.CopyBytes(this, source, destination, byteCount);
}
