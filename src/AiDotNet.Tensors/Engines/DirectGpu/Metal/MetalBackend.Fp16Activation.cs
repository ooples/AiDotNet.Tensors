// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// No FP16-native activation kernels here: every capability reports false, so the engine keeps activations in FP32
// and never calls the operations.
public sealed partial class MetalBackend : IFp16ActivationBackend
{
    private static NotSupportedException NoFp16Activations(string kernel)
        => new($"{kernel}: this backend has no FP16-native activation kernels; check the capability first.");

    bool IFp16ActivationBackend.SupportsFp16ActivationStorage => false;

    bool IFp16ActivationBackend.SupportsFp16NativeOps => false;

    bool IFp16ActivationBackend.Fp16BroadcastAddChannelAvailable => false;

    bool IFp16ActivationBackend.Fp16DirectConvAvailable => false;

    bool IFp16ActivationBackend.Fp16GroupNormSwishAvailable => false;

    bool IFp16ActivationBackend.Fp16Im2colFromFp16Available => false;

    void IFp16ActivationBackend.Fp16Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Fp16Add));

    void IFp16ActivationBackend.Fp16BroadcastAddChannel(IGpuBuffer ioHalf, IGpuBuffer biasHalf, int batch, int channels, int spatial)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Fp16BroadcastAddChannel));

    void IFp16ActivationBackend.Fp16Relu(IGpuBuffer input, IGpuBuffer output, int size)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Fp16Relu));

    void IFp16ActivationBackend.Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int size)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Fp16Gelu));

    void IFp16ActivationBackend.Fp16ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Fp16ReluBackward));

    void IFp16ActivationBackend.Fp16GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Fp16GeluBackward));

    void IFp16ActivationBackend.Fp16SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Fp16SoftmaxBackward));

    void IFp16ActivationBackend.Fp16LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveMeanFp32, IGpuBuffer saveInvVarFp32, IGpuBuffer gradInput, int rows, int cols)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Fp16LayerNormBackward));

    void IFp16ActivationBackend.Fp16LayerNormGradParams(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer saveMeanFp32, IGpuBuffer saveInvVarFp32, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int rows, int cols)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Fp16LayerNormGradParams));

    void IFp16ActivationBackend.Fp16GroupNormSwish(IGpuBuffer inHalf, IGpuBuffer gammaFp32, IGpuBuffer betaFp32, IGpuBuffer outHalf, int batch, int numGroups, int channels, int spatial, float eps)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Fp16GroupNormSwish));

    void IFp16ActivationBackend.ConvertToFp16Native(IGpuBuffer input, IGpuBuffer output, int size)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.ConvertToFp16Native));

    void IFp16ActivationBackend.ConvertToFp32Native(IGpuBuffer input, IGpuBuffer output, int size)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.ConvertToFp32Native));

    void IFp16ActivationBackend.Conv2dDirectFp16Hw(IGpuBuffer input, IGpuBuffer weightsHalf, IGpuBuffer output, int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.Conv2dDirectFp16Hw));

    void IFp16ActivationBackend.UnfoldKNFp16FromFp16(IGpuBuffer inputHalf, IGpuBuffer outputHalf, int batch, int channels, int height, int width, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.UnfoldKNFp16FromFp16));

    void IFp16ActivationBackend.GemmFp16HalfOut(IGpuBuffer Ahalf, IGpuBuffer Bhalf, IGpuBuffer Chalf, int M, int N, int K, float alpha, float beta)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.GemmFp16HalfOut));

    void IFp16ActivationBackend.CopyBufferDtoD(IGpuBuffer source, IGpuBuffer destination, long byteCount)
        => throw NoFp16Activations(nameof(IFp16ActivationBackend.CopyBufferDtoD));
}
