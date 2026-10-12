// Copyright (c) AiDotNet. All rights reserved.
// HIP IGpuMixedPrecisionConvBackend (#1142): 16-bit storage over the backend's own FP32 convolution.
using AiDotNet.Tensors.Engines.Gpu;
using P = AiDotNet.Tensors.Engines.DirectGpu.PortableMixedPrecisionConv;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IGpuMixedPrecisionConvBackend
{
    /// <summary>Mixed-precision convolution runs here through <see cref="PortableMixedPrecisionConv"/>.</summary>
    public IGpuMixedPrecisionConvBackend? MixedPrecisionConv => this;

    bool IGpuMixedPrecisionConvBackend.SupportsHalfConv => true;

    bool IGpuMixedPrecisionConvBackend.SupportsBFloat16Conv => true;

    void IGpuMixedPrecisionConvBackend.Conv2DMixed(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType)
        => P.Forward(this, this, input, kernel, output, batch, inChannels, inHeight, inWidth, outChannels, outHeight,
            outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, dataType);

    void IGpuMixedPrecisionConvBackend.Conv2DBackwardDataMixed(IGpuBuffer kernel, IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType)
        => P.BackwardData(this, this, kernel, gradOutput, gradInput, batch, inChannels, inHeight, inWidth, outChannels,
            outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, dataType);

    void IGpuMixedPrecisionConvBackend.Conv2DBackwardFilterMixed(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradFilter,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType)
        => P.BackwardFilter(this, this, input, gradOutput, gradFilter, batch, inChannels, inHeight, inWidth, outChannels,
            outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, dataType);
}
