// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// FP16-resident activations (#1154): kernels that read and write half-precision activations directly, so a forward
/// keeps them at half the memory. A backend without them reports every capability false and the engine keeps the
/// activations in FP32; each operation is called only after its capability reported true.
/// </summary>
internal interface IFp16ActivationBackend : IGpuBatchExecution
{
    /// <summary>True when cached FP32 activations may be compressed to FP16 storage (and widened back) here.</summary>
    bool SupportsFp16ActivationStorage { get; }

    /// <summary>True when this backend has FP16-native elementwise, softmax and layer-norm kernels (read and write FP16, compute in FP32).</summary>
    bool SupportsFp16NativeOps { get; }

    /// <summary>True when this backend has the in-place per-channel FP16 broadcast add.</summary>
    bool Fp16BroadcastAddChannelAvailable { get; }

    /// <summary>True when this backend has the direct FP16-weight convolution.</summary>
    bool Fp16DirectConvAvailable { get; }

    /// <summary>True when this backend has the fused FP16 GroupNorm + Swish.</summary>
    bool Fp16GroupNormSwishAvailable { get; }

    /// <summary>True when this backend has the FP16-input im2col.</summary>
    bool Fp16Im2colFromFp16Available { get; }

    /// <summary>The FP16 Fp16Add kernel; only after its capability reported true.</summary>
    void Fp16Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);

    /// <summary>The FP16 Fp16BroadcastAddChannel kernel; only after its capability reported true.</summary>
    void Fp16BroadcastAddChannel(IGpuBuffer ioHalf, IGpuBuffer biasHalf, int batch, int channels, int spatial);

    /// <summary>The FP16 Fp16Relu kernel; only after its capability reported true.</summary>
    void Fp16Relu(IGpuBuffer input, IGpuBuffer output, int size);

    /// <summary>The FP16 Fp16Gelu kernel; only after its capability reported true.</summary>
    void Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int size);

    /// <summary>The FP16 Fp16ReluBackward kernel; only after its capability reported true.</summary>
    void Fp16ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size);

    /// <summary>The FP16 Fp16GeluBackward kernel; only after its capability reported true.</summary>
    void Fp16GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size);

    /// <summary>The FP16 Fp16SoftmaxBackward kernel; only after its capability reported true.</summary>
    void Fp16SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features);

    /// <summary>The FP16 Fp16LayerNormBackward kernel; only after its capability reported true.</summary>
    void Fp16LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveMeanFp32, IGpuBuffer saveInvVarFp32, IGpuBuffer gradInput, int rows, int cols);

    /// <summary>The FP16 Fp16LayerNormGradParams kernel; only after its capability reported true.</summary>
    void Fp16LayerNormGradParams(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer saveMeanFp32, IGpuBuffer saveInvVarFp32, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int rows, int cols);

    /// <summary>The FP16 Fp16GroupNormSwish kernel; only after its capability reported true.</summary>
    void Fp16GroupNormSwish(IGpuBuffer inHalf, IGpuBuffer gammaFp32, IGpuBuffer betaFp32, IGpuBuffer outHalf, int batch, int numGroups, int channels, int spatial, float eps);

    /// <summary>The FP16 ConvertToFp16Native kernel; only after its capability reported true.</summary>
    void ConvertToFp16Native(IGpuBuffer input, IGpuBuffer output, int size);

    /// <summary>The FP16 ConvertToFp32Native kernel; only after its capability reported true.</summary>
    void ConvertToFp32Native(IGpuBuffer input, IGpuBuffer output, int size);

    /// <summary>The FP16 Conv2dDirectFp16Hw kernel; only after its capability reported true.</summary>
    void Conv2dDirectFp16Hw(IGpuBuffer input, IGpuBuffer weightsHalf, IGpuBuffer output, int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW);

    /// <summary>The FP16 UnfoldKNFp16FromFp16 kernel; only after its capability reported true.</summary>
    void UnfoldKNFp16FromFp16(IGpuBuffer inputHalf, IGpuBuffer outputHalf, int batch, int channels, int height, int width, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW);

    /// <summary>The FP16 GemmFp16HalfOut kernel; only after its capability reported true.</summary>
    void GemmFp16HalfOut(IGpuBuffer Ahalf, IGpuBuffer Bhalf, IGpuBuffer Chalf, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);

    /// <summary>Copies byteCount raw bytes device to device (FP16 buffers are byte-sized).</summary>
    void CopyBufferDtoD(IGpuBuffer source, IGpuBuffer destination, long byteCount);
}
