// Copyright (c) AiDotNet. All rights reserved.

#if !NET462
using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Issue #337: CudaBackend implementation of <see cref="IGpuMixedPrecisionConvBackend"/>.
/// Routes Conv2D forward + backward to the cuDNN helpers with the requested
/// dtype. Consumers that want mixed-precision conv downcast the backend
/// reference to this interface and dispatch through it.
/// </summary>
public sealed partial class CudaBackend : IGpuMixedPrecisionConvBackend
{
    /// <summary>
    /// IDirectGpuBackend.MixedPrecisionConv: CUDA ships the cuDNN-backed
    /// mixed-precision conv surface, so the typed accessor returns
    /// `this`. Backends without cuDNN-equivalent paths return null.
    /// </summary>
    public IGpuMixedPrecisionConvBackend? MixedPrecisionConv => this;

    /// <summary>
    /// True when cuDNN is available and the cuDNN convolution helper has
    /// been initialised. FP16 conv is supported on every cuDNN release —
    /// no compute-capability floor here, unlike Hgemm's Maxwell+ floor.
    /// </summary>
    bool IGpuMixedPrecisionConvBackend.SupportsHalfConv
        => CudaDispatchPolicy.UseCudnnForConv;

    /// <summary>
    /// True when cuDNN BF16 conv is supported — requires compute capability
    /// 8.0 (Ampere) or higher, where the BF16 tensor cores live.
    /// </summary>
    bool IGpuMixedPrecisionConvBackend.SupportsBFloat16Conv
        => CudaDispatchPolicy.UseCudnnForConv && _ccMajor >= 8;

    private static CuDnnNative.CudnnDataType MapDataType(GpuMixedPrecisionDataType dataType)
        => dataType switch
        {
            GpuMixedPrecisionDataType.Float32 => CuDnnNative.CudnnDataType.Float,
            GpuMixedPrecisionDataType.Half => CuDnnNative.CudnnDataType.Half,
            GpuMixedPrecisionDataType.BFloat16 => CuDnnNative.CudnnDataType.BFloat16,
            _ => throw new ArgumentOutOfRangeException(nameof(dataType), dataType,
                "Unsupported mixed-precision data type."),
        };

    void IGpuMixedPrecisionConvBackend.Conv2DMixed(
        IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (!CudaDispatchPolicy.UseCudnnForConv)
            throw new InvalidOperationException(
                "Mixed-precision Conv2D requires cuDNN, which is disabled or unavailable on this backend.");

        var cudnnDataType = MapDataType(dataType);
        if (dataType == GpuMixedPrecisionDataType.BFloat16 && _ccMajor < 8)
            throw new NotSupportedException(
                $"BFloat16 convolution requires compute capability >= 8.0 (Ampere+); this device reports {_ccMajor}.{_ccMinor}.");

        using var _ctx = PushContext();
        EnsureCudnnConv();
        CuDnnContext.CheckStatus(
            CuDnnNative.cudnnSetStream(_cudnnContext!.Handle, _stream),
            "cudnnSetStream");
        _cudnnConv!.Conv2DForwardGpu(
            inputDevPtr: input.Handle,
            filterDevPtr: kernel.Handle,
            outputDevPtr: output.Handle,
            n: batch, c: inChannels, h: inHeight, w: inWidth,
            k: outChannels, filterH: kernelH, filterW: kernelW,
            outputHeight: outHeight, outputWidth: outWidth,
            padH: padH, padW: padW,
            strideH: strideH, strideW: strideW,
            dilationH: dilationH, dilationW: dilationW,
            dataType: cudnnDataType);
    }

    void IGpuMixedPrecisionConvBackend.Conv2DBackwardDataMixed(
        IGpuBuffer kernel, IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType)
    {
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        if (!CudaDispatchPolicy.UseCudnnForConv)
            throw new InvalidOperationException(
                "Mixed-precision Conv2D requires cuDNN, which is disabled or unavailable on this backend.");

        var cudnnDataType = MapDataType(dataType);
        if (dataType == GpuMixedPrecisionDataType.BFloat16 && _ccMajor < 8)
            throw new NotSupportedException(
                $"BFloat16 convolution requires compute capability >= 8.0 (Ampere+); this device reports {_ccMajor}.{_ccMinor}.");

        using var _ctx = PushContext();
        EnsureCudnnConv();
        CuDnnContext.CheckStatus(
            CuDnnNative.cudnnSetStream(_cudnnContext!.Handle, _stream),
            "cudnnSetStream");
        _cudnnConv!.Conv2DBackwardDataGpu(
            filterDevPtr: kernel.Handle,
            gradOutputDevPtr: gradOutput.Handle,
            gradInputDevPtr: gradInput.Handle,
            n: batch, c: inChannels, h: inHeight, w: inWidth,
            k: outChannels, filterH: kernelH, filterW: kernelW,
            outputHeight: outHeight, outputWidth: outWidth,
            padH: padH, padW: padW,
            strideH: strideH, strideW: strideW,
            dilationH: dilationH, dilationW: dilationW,
            dataType: cudnnDataType);
    }

    void IGpuMixedPrecisionConvBackend.Conv2DBackwardFilterMixed(
        IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradFilter,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (gradFilter is null) throw new ArgumentNullException(nameof(gradFilter));
        if (!CudaDispatchPolicy.UseCudnnForConv)
            throw new InvalidOperationException(
                "Mixed-precision Conv2D requires cuDNN, which is disabled or unavailable on this backend.");

        var cudnnDataType = MapDataType(dataType);
        if (dataType == GpuMixedPrecisionDataType.BFloat16 && _ccMajor < 8)
            throw new NotSupportedException(
                $"BFloat16 convolution requires compute capability >= 8.0 (Ampere+); this device reports {_ccMajor}.{_ccMinor}.");

        using var _ctx = PushContext();
        EnsureCudnnConv();
        CuDnnContext.CheckStatus(
            CuDnnNative.cudnnSetStream(_cudnnContext!.Handle, _stream),
            "cudnnSetStream");
        _cudnnConv!.Conv2DBackwardFilterGpu(
            inputDevPtr: input.Handle,
            gradOutputDevPtr: gradOutput.Handle,
            gradFilterDevPtr: gradFilter.Handle,
            n: batch, c: inChannels, h: inHeight, w: inWidth,
            k: outChannels, filterH: kernelH, filterW: kernelW,
            outputHeight: outHeight, outputWidth: outWidth,
            padH: padH, padW: padW,
            strideH: strideH, strideW: strideW,
            dilationH: dilationH, dilationW: dilationW,
            dataType: cudnnDataType);
    }
}
#endif
