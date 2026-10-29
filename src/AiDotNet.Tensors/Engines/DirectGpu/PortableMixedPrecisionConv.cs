// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IGpuMixedPrecisionConvBackend"/> for every backend without cuDNN (#1142): 16-bit storage, FP32 compute,
/// the convention cuDNN's mixed-precision convolution follows. Operands arrive as contiguous IEEE half or bfloat16
/// elements (NCHW / OIHW, as cuDNN reads them); each is widened into an FP32 workspace with the backend's
/// <see cref="IGenericFftBackend"/> storage conversion, the backend's own FP32 convolution runs, and the result is
/// narrowed back with round-to-nearest-even. The output is overwritten, as cuDNN with beta = 0.
/// </summary>
internal static class PortableMixedPrecisionConv
{
    private static FftElementType Storage(GpuMixedPrecisionDataType type) => type switch
    {
        GpuMixedPrecisionDataType.Float32 => FftElementType.Float32,
        GpuMixedPrecisionDataType.Half => FftElementType.Float16,
        GpuMixedPrecisionDataType.BFloat16 => FftElementType.BFloat16,
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unsupported mixed-precision data type."),
    };

    private readonly struct Shape
    {
        public readonly int Input, Kernel, Output;

        public Shape(int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
            int kernelH, int kernelW)
        {
            if (batch < 1 || inChannels < 1 || inHeight < 1 || inWidth < 1 || outChannels < 1 || outHeight < 1 ||
                outWidth < 1 || kernelH < 1 || kernelW < 1)
                throw new ArgumentOutOfRangeException(nameof(batch), "Every convolution extent must be positive.");
            Input = checked(batch * inChannels * inHeight * inWidth);
            Kernel = checked(outChannels * inChannels * kernelH * kernelW);
            Output = checked(batch * outChannels * outHeight * outWidth);
        }
    }

    private static IGpuBuffer Widen(IDirectGpuBackend backend, IGenericFftBackend convert, IGpuBuffer source, int count,
        FftElementType type)
    {
        var wide = backend.AllocateBuffer(count);
        try { convert.ConvertFftStorageToFloat(source, wide, count, type); }
        catch { wide.Dispose(); throw; }
        return wide;
    }

    public static void Forward(IDirectGpuBackend backend, IGenericFftBackend convert,
        IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (output is null) throw new ArgumentNullException(nameof(output));
        var type = Storage(dataType);
        if (type == FftElementType.Float32)
        {
            backend.Conv2D(input, kernel, output, batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
            return;
        }
        var shape = new Shape(batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW);
        using var x = Widen(backend, convert, input, shape.Input, type);
        using var w = Widen(backend, convert, kernel, shape.Kernel, type);
        using var y = backend.AllocateBuffer(shape.Output);
        backend.Conv2D(x, w, y, batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
        convert.ConvertFloatToFftStorage(y, output, shape.Output, type);
    }

    public static void BackwardData(IDirectGpuBackend backend, IGenericFftBackend convert,
        IGpuBuffer kernel, IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType)
    {
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        var type = Storage(dataType);
        if (type == FftElementType.Float32)
        {
            backend.Conv2DBackwardInput(gradOutput, kernel, gradInput, batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
            return;
        }
        var shape = new Shape(batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW);
        using var w = Widen(backend, convert, kernel, shape.Kernel, type);
        using var dy = Widen(backend, convert, gradOutput, shape.Output, type);
        using var dx = backend.AllocateBuffer(shape.Input);
        backend.Conv2DBackwardInput(dy, w, dx, batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
        convert.ConvertFloatToFftStorage(dx, gradInput, shape.Input, type);
    }

    public static void BackwardFilter(IDirectGpuBackend backend, IGenericFftBackend convert,
        IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradFilter,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        GpuMixedPrecisionDataType dataType)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (gradFilter is null) throw new ArgumentNullException(nameof(gradFilter));
        var type = Storage(dataType);
        if (type == FftElementType.Float32)
        {
            backend.Conv2DBackwardKernel(input, gradOutput, gradFilter, batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
            return;
        }
        var shape = new Shape(batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW);
        using var x = Widen(backend, convert, input, shape.Input, type);
        using var dy = Widen(backend, convert, gradOutput, shape.Output, type);
        using var dw = backend.AllocateBuffer(shape.Kernel);
        backend.Conv2DBackwardKernel(x, dy, dw, batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
        convert.ConvertFloatToFftStorage(dw, gradFilter, shape.Kernel, type);
    }
}
