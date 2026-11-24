// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The autocast convolution staged channels-last (<see cref="IChannelsLastMixedConvBackend"/>). cuDNN's 16-bit
/// tensor-core convolutions are channels-last-native: given NCHW operands they run nchwToNhwc / nhwcToNchw passes of
/// their own on every call (measured on the BF16 DCGAN step, RTX 3090: 1.2 ms of a 9.9 ms step). Narrowing FP32 NCHW
/// straight into 16-bit NHWC, and widening back, folds that layout change into the conversion the autocast path runs
/// anyway.
/// </summary>
public sealed partial class CudaBackend : IChannelsLastMixedConvBackend
{
    /// <summary>Whether autocast convolutions stage channels-last where they can (tests compare both routes).</summary>
    internal static bool s_channelsLastMixedConv = true;

    /// <summary>Channels-last mixed-precision convolution passes launched (tests, diagnostics).</summary>
    internal static long ChannelsLastMixedConvolutions;

    private static FftElementType StorageOf(GpuMixedPrecisionDataType dataType) => dataType switch
    {
        GpuMixedPrecisionDataType.Half => FftElementType.Float16,
        GpuMixedPrecisionDataType.BFloat16 => FftElementType.BFloat16,
        _ => throw new ArgumentOutOfRangeException(nameof(dataType), dataType, "Not a 16-bit storage type."),
    };

    bool IChannelsLastMixedConvBackend.SupportsChannelsLastMixedConv(GpuMixedPrecisionDataType dataType)
    {
        if (!s_channelsLastMixedConv || !IsAvailable || !CudaDispatchPolicy.UseCudnnForConv) return false;
        if (dataType == GpuMixedPrecisionDataType.BFloat16 ? _ccMajor < 8
            : dataType != GpuMixedPrecisionDataType.Half || _ccMajor < 7)
            return false;   // channels-last pays only on the tensor cores that read it natively
        var storage = StorageOf(dataType);
        return SupportsFftElementType(storage)
               && _kernelCache.ContainsKey("fftg_transpose_from_f32" + storage.KernelSuffix())
               && _kernelCache.ContainsKey("fftg_transpose_to_f32" + storage.KernelSuffix())
               && CudnnConvLoaded();
    }

    void IChannelsLastMixedConvBackend.NarrowToChannelsLast(IGpuBuffer source, IGpuBuffer destination, int batch,
        int channels, int spatial, GpuMixedPrecisionDataType dataType)
        => LaunchTransposingConversion("fftg_transpose_from_f32", source, destination, batch, rows: channels,
            cols: spatial, dataType, narrowing: true);

    void IChannelsLastMixedConvBackend.WidenFromChannelsLast(IGpuBuffer source, IGpuBuffer destination, int batch,
        int channels, int spatial, GpuMixedPrecisionDataType dataType)
        => LaunchTransposingConversion("fftg_transpose_to_f32", source, destination, batch, rows: spatial,
            cols: channels, dataType, narrowing: false);

    // [batch][rows][cols] -> [batch][cols][rows], converting between FP32 and the 16-bit storage type.
    private unsafe void LaunchTransposingConversion(string baseName, IGpuBuffer source, IGpuBuffer destination,
        int batch, int rows, int cols, GpuMixedPrecisionDataType dataType, bool narrowing)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (batch < 1 || rows < 1 || cols < 1) throw new ArgumentOutOfRangeException(nameof(batch), "Extents must be positive.");
        long count = (long)batch * rows * cols;
        if (count > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(batch), "Too many elements for one pass.");
        // Buffer sizes are in FP32 slots; two 16-bit elements fit in one.
        long wideSlots = count, narrowSlots = (count + 1) / 2;
        if ((narrowing ? source : destination).Size < wideSlots || (narrowing ? destination : source).Size < narrowSlots)
            throw new ArgumentException("A buffer is smaller than the conversion it was given.");
        long gridX = (cols + 31) / 32, gridY = (rows + 31) / 32;
        if (gridY > MaxGridDimY || batch > MaxGridDimY)
            throw new ArgumentOutOfRangeException(nameof(rows), "The conversion's grid exceeds the device's limits.");
        var storage = StorageOf(dataType);
        IntPtr kernel = RequireGenericFftKernel(baseName + storage.KernelSuffix(), storage);
        IntPtr src = source.Handle, dst = destination.Handle;
        void** args = stackalloc void*[4];
        args[0] = &src; args[1] = &dst; args[2] = &rows; args[3] = &cols;
        using var _ = PushContext();
        LaunchKernel2D(kernel, (uint)gridX, (uint)gridY, (uint)batch, 32, 8, args);
    }

    // Scratch for operands re-strided to padded channel counts (grown outside captures only; see TryConvScratch).
    private IGpuBuffer? _clMixedPadA, _clMixedPadB, _clMixedPadC;

    // cuDNN's 16-bit tensor-core convolutions need channel counts in multiples of 8 (16 bytes); other counts fall to a
    // generic engine, which cost 0.8 ms per DCGAN step on the stem (3 image channels) and head (1 output channel).
    private static int PadChannels(int channels) => (channels + 7) & ~7;

    bool IChannelsLastMixedConvBackend.TryConv2DMixedChannelsLast(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH,
        int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType, out bool shapeUnsupported)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (c is null) throw new ArgumentNullException(nameof(c));
        if (pass is < 0 or > 2) throw new ArgumentOutOfRangeException(nameof(pass));
        shapeUnsupported = false;
        if (!((IChannelsLastMixedConvBackend)this).SupportsChannelsLastMixedConv(dataType)) return false;
        using var _ctx = PushContext();
        // Inside a capture nothing can be allocated (an uncached search, a workspace or scratch to grow): a decline
        // there says nothing about the shape. Outside one, a decline is cuDNN finding no kernel for it.
        bool launched = TryConv2DMixedChannelsLastCore(pass, a, b, c, batch, inChannels, inHeight, inWidth, outChannels,
            outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, dataType);
        shapeUnsupported = !launched && !IsStreamCapturing();
        return launched;
    }

    private bool TryConv2DMixedChannelsLastCore(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch,
        int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH,
        int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType)
    {
        int cP = PadChannels(inChannels), kP = PadChannels(outChannels);
        if (cP == inChannels && kP == outChannels)
            return LaunchChannelsLastMixed(pass, a, b, c, batch, inChannels, inHeight, inWidth, outChannels, outHeight,
                outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, dataType);

        // Each operand as rows of channels: x [n*h*w][c], w [k*r*s][c] (its k rows padded too), y [n*oh*ow][k]. Zero
        // channels and zero filters add nothing to any sum, so the padded convolution's values are the unpadded ones.
        long xRows = (long)batch * inHeight * inWidth, yRows = (long)batch * outHeight * outWidth;
        int filterSpatial = kernelH * kernelW;
        var x = (Rows: xRows, PaddedRows: xRows, Channels: inChannels, Padded: cP);
        var w = (Rows: (long)outChannels * filterSpatial, PaddedRows: (long)kP * filterSpatial, Channels: inChannels,
            Padded: cP);
        var y = (Rows: yRows, PaddedRows: yRows, Channels: outChannels, Padded: kP);
        var (aShape, bShape, cShape) = pass switch { 0 => (x, w, y), 1 => (w, y, x), _ => (x, y, w) };

        IGpuBuffer? Stage(ref IGpuBuffer? slot, IGpuBuffer source,
            (long Rows, long PaddedRows, int Channels, int Padded) s, bool output)
        {
            if (s.Channels == s.Padded && s.Rows == s.PaddedRows) return source;
            var scratch = TryConvScratch(ref slot, (s.PaddedRows * s.Padded + 1) / 2);
            if (scratch is not null && !output)
                LaunchChannelPad(source, scratch, s.PaddedRows, s.Rows, s.Channels, s.Padded, dataType);
            return scratch;
        }

        var aPadded = Stage(ref _clMixedPadA, a, aShape, output: false);
        var bPadded = Stage(ref _clMixedPadB, b, bShape, output: false);
        var cPadded = Stage(ref _clMixedPadC, c, cShape, output: true);
        if (aPadded is null || bPadded is null || cPadded is null) return false;   // undersized in a capture: NCHW route
        if (!LaunchChannelsLastMixed(pass, aPadded, bPadded, cPadded, batch, cP, inHeight, inWidth, kP, outHeight,
                outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, dataType))
            return false;
        if (!ReferenceEquals(cPadded, c))   // drop the padding: rows past the real ones (padded filters) are not copied
            LaunchChannelPad(cPadded, c, cShape.Rows, cShape.Rows, cShape.Padded, cShape.Channels, dataType);
        return true;
    }

    private bool LaunchChannelsLastMixed(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int inChannels,
        int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH,
        int strideW, int padH, int padW, int dilationH, int dilationW, GpuMixedPrecisionDataType dataType)
    {
        var context = _cudnnContext!;
        bool launched;
        lock (context.StreamBindingLock)
        {
            context.BindStream(_stream);
            launched = _cudnnConv!.ConvolutionChannelsLast(pass, a.Handle, b.Handle, c.Handle, batch, inChannels, inHeight,
                inWidth, outChannels, kernelH, kernelW, outHeight, outWidth, padH, padW, strideH, strideW, dilationH,
                dilationW, inputNhwc: true, outputNhwc: true, CuDnnNative.CudnnMathType.TensorOp, GpuDeterminism.IsActive,
                IsStreamCapturing(), filterNhwc: true, MapDataType(dataType));
        }
        if (launched) System.Threading.Interlocked.Increment(ref ChannelsLastMixedConvolutions);
        return launched;
    }

    // dst[r][c] = src[r][c] for r < srcRows and c < srcC, else 0 (16-bit elements, rows of channels).
    private unsafe void LaunchChannelPad(IGpuBuffer source, IGpuBuffer destination, long dstRows, long srcRows, int srcC,
        int dstC, GpuMixedPrecisionDataType dataType)
    {
        long count = dstRows * dstC;
        if (count > int.MaxValue || srcRows * srcC > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(dstRows), "Too many elements for one pass.");
        var storage = StorageOf(dataType);
        IntPtr kernel = RequireGenericFftKernel("fftg_channel_pad" + storage.KernelSuffix(), storage);
        IntPtr src = source.Handle, dst = destination.Handle;
        int dr = (int)dstRows, sr = (int)srcRows, total = (int)count;
        void** args = stackalloc void*[6];
        args[0] = &src; args[1] = &dst; args[2] = &dr; args[3] = &sr; args[4] = &srcC; args[5] = &dstC;
        LaunchKernel(kernel, GridFor(total), DefaultBlockSize, args);
    }
}
