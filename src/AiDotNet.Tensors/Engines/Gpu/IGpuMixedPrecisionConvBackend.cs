// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Issue #337: capability interface for backends that ship
/// mixed-precision cuDNN convolution / batch-norm / softmax kernels.
/// <para>
/// <b>Discovery:</b> consumers reach this surface through
/// <see cref="IDirectGpuBackend.MixedPrecisionConv"/> instead of
/// downcasting. Backends that don't ship cuDNN-equivalent paths return
/// <c>null</c>; the CUDA backend returns <c>this</c>. This follows the
/// established engine-extension pattern (see also PR #344's
/// stream-scheduler promotion) — capability surfaces live on the engine
/// contract, not on standalone probe interfaces.
/// </para>
/// <para>
/// <b>Why a separate interface rather than folded into IDirectGpuBackend
/// methods directly:</b> Vulkan, WebGpu, Metal MPSGraph, HIP, and OpenCL
/// don't have cuDNN-equivalent half/bfloat conv yet. Pushing the methods
/// onto every backend would either force every implementation to throw
/// <c>NotSupportedException</c> or block all backends until they ship
/// their own mixed-precision adapter. The typed accessor lets callers
/// null-check once and skip the whole path cleanly.
/// </para>
/// <para>
/// The element type is passed through as a <see cref="GpuMixedPrecisionDataType"/>
/// (a framework-side enum that maps to <c>CudnnDataType</c> on CUDA, the
/// equivalent metal data type on Metal, etc.) so the interface stays
/// framework-neutral.
/// </para>
/// </summary>
public interface IGpuMixedPrecisionConvBackend
{
    /// <summary>True when this backend ships FP16 cuDNN-equivalent conv/BN/softmax.</summary>
    bool SupportsHalfConv { get; }

    /// <summary>True when this backend ships BF16 cuDNN-equivalent conv/BN/softmax.
    /// On NVIDIA hardware this requires compute capability >= 8.0 (Ampere+).</summary>
    bool SupportsBFloat16Conv { get; }

    /// <summary>
    /// 2D convolution forward with the given mixed-precision element type for
    /// input / filter / output buffers. Compute type stays fp32 — accuracy-
    /// preserving mixed-precision convention.
    /// </summary>
    void Conv2DMixed(
        IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType);

    /// <summary>
    /// 2D convolution backward-data with the given mixed-precision element
    /// type. Produces dX from dY + filter.
    /// </summary>
    void Conv2DBackwardDataMixed(
        IGpuBuffer kernel, IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType);

    /// <summary>
    /// 2D convolution backward-filter with the given mixed-precision element
    /// type. Produces dW from input + dY.
    /// </summary>
    void Conv2DBackwardFilterMixed(
        IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradFilter,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType);
}

/// <summary>
/// Framework-neutral mixed-precision element type for the
/// <see cref="IGpuMixedPrecisionConvBackend"/> capability surface.
/// </summary>
public enum GpuMixedPrecisionDataType
{
    /// <summary>32-bit IEEE float — preserves the pre-#337 default.</summary>
    Float32,
    /// <summary>16-bit IEEE half (5-exp / 10-mant). Requires loss scaling
    /// for training stability on most networks; see <c>GradScaler</c>.</summary>
    Half,
    /// <summary>16-bit bfloat (8-exp / 7-mant). Same exponent range as fp32 so
    /// no loss scaling needed. Ampere+ (NVIDIA compute capability >= 8.0).</summary>
    BFloat16,
}
