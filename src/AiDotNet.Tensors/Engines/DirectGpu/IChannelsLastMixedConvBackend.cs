// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// An autocast convolution staged channels-last. The engine's FP32 activations are NCHW; a 16-bit tensor-core
/// convolution (cuDNN's) reads and writes channels-last natively and transposes NCHW operands itself, in extra passes
/// on every call. A backend with this capability narrows FP32 NCHW straight into 16-bit NHWC (and widens back) in one
/// pass per operand, so the convolution runs on its native layout and the layout change costs nothing beyond the
/// conversion the autocast path makes anyway.
/// </summary>
/// <remarks>
/// <para>Layouts: activations [n, h, w, c]; filters [k, r, s, c] (the filter as a [k][c][r*s] transpose, the same pass
/// as an activation with batch k).</para>
/// <para>Only backends whose convolutions are channels-last-native answer true in
/// <see cref="SupportsChannelsLastMixedConv"/>; the others' convolutions are NCHW, where staging channels-last would only
/// add passes, and they leave the autocast route on <see cref="IGpuMixedPrecisionConvBackend"/>.</para>
/// </remarks>
internal interface IChannelsLastMixedConvBackend
{
    /// <summary>Whether this backend runs <paramref name="dataType"/> convolutions channels-last.</summary>
    bool SupportsChannelsLastMixedConv(GpuMixedPrecisionDataType dataType);

    /// <summary>
    /// Narrows FP32 [batch][channels][spatial] into 16-bit [batch][spatial][channels] (round to nearest even).
    /// </summary>
    void NarrowToChannelsLast(IGpuBuffer source, IGpuBuffer destination, int batch, int channels, int spatial,
        GpuMixedPrecisionDataType dataType);

    /// <summary>Widens 16-bit [batch][spatial][channels] into FP32 [batch][channels][spatial] (exact).</summary>
    void WidenFromChannelsLast(IGpuBuffer source, IGpuBuffer destination, int batch, int channels, int spatial,
        GpuMixedPrecisionDataType dataType);

    /// <summary>
    /// One 16-bit convolution pass with every operand channels-last, FP32 accumulation, output overwritten.
    /// <paramref name="pass"/> 0: a = x, b = w, c = y. 1: a = w, b = dy, c = dx. 2: a = x, b = dy, c = dw. False when
    /// it did not run; the caller then takes the NCHW route. <paramref name="shapeUnsupported"/> tells a decline that
    /// holds for every later call of the shape (no kernel for it in this layout) from one that does not (a graph
    /// capture, where no workspace or scratch can be allocated).
    /// </summary>
    bool TryConv2DMixedChannelsLast(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int inChannels,
        int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH,
        int strideW, int padH, int padW, int dilationH, int dilationW, GpuMixedPrecisionDataType dataType,
        out bool shapeUnsupported);
}
