// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// NCHW convolutions: staging a mixed-precision convolution channels-last would only add passes.
public sealed partial class HipBackend : IChannelsLastMixedConvBackend
{
    bool IChannelsLastMixedConvBackend.SupportsChannelsLastMixedConv(GpuMixedPrecisionDataType dataType) => false;

    void IChannelsLastMixedConvBackend.NarrowToChannelsLast(IGpuBuffer source, IGpuBuffer destination, int batch,
        int channels, int spatial, GpuMixedPrecisionDataType dataType)
        => PortableChannelsLastMixedConv.Narrow(this, source, destination, batch, channels, spatial, dataType);

    void IChannelsLastMixedConvBackend.WidenFromChannelsLast(IGpuBuffer source, IGpuBuffer destination, int batch,
        int channels, int spatial, GpuMixedPrecisionDataType dataType)
        => PortableChannelsLastMixedConv.Widen(this, source, destination, batch, channels, spatial, dataType);

    bool IChannelsLastMixedConvBackend.TryConv2DMixedChannelsLast(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH,
        int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType, out bool shapeUnsupported)
    {
        shapeUnsupported = true;   // never staged channels-last here
        return false;
    }
}
