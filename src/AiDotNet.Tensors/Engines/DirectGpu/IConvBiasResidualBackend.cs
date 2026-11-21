namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Inference convolution with an independently owned residual sum and optional activated carry.</summary>
[BackendSpecificRoute("Convolution/bias/residual composition on all six backends, with a vendor fused CUDA store",
    nameof(IDirectGpuBackend.Conv2D), nameof(IDirectGpuBackend.LeakyRelu))]
internal interface IConvBiasResidualBackend
{
    bool TryConvBiasResidual(in ConvBiasResidualArgs args);
}

internal readonly struct ConvBiasResidualArgs
{
    internal readonly IGpuBuffer Input, Kernel, Bias, Residual, Output, Activated;
    internal readonly int Batch, InChannels, InHeight, InWidth, OutChannels, OutHeight, OutWidth;
    internal readonly int KernelH, KernelW, StrideH, StrideW, PadH, PadW, DilationH, DilationW;
    internal readonly float Alpha;
    internal readonly bool WriteActivation;

    internal ConvBiasResidualArgs(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer bias, IGpuBuffer residual,
        IGpuBuffer output, IGpuBuffer activated, int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, float alpha, bool writeActivation)
    {
        Input = input; Kernel = kernel; Bias = bias; Residual = residual; Output = output; Activated = activated;
        Batch = batch; InChannels = inChannels; InHeight = inHeight; InWidth = inWidth;
        OutChannels = outChannels; OutHeight = outHeight; OutWidth = outWidth;
        KernelH = kernelH; KernelW = kernelW; StrideH = strideH; StrideW = strideW;
        PadH = padH; PadW = padW; DilationH = dilationH; DilationW = dilationW;
        Alpha = alpha; WriteActivation = writeActivation;
    }

    internal bool TryCount(out int count)
    {
        count = 0;
        if (Batch <= 0 || InChannels <= 0 || InHeight <= 0 || InWidth <= 0 || OutChannels <= 0
            || OutHeight <= 0 || OutWidth <= 0 || KernelH <= 0 || KernelW <= 0 || StrideH <= 0 || StrideW <= 0
            || PadH < 0 || PadW < 0 || DilationH <= 0 || DilationW <= 0 || !(Alpha > 0) || float.IsInfinity(Alpha)) return false;
        long height = (long)InHeight + 2L * PadH - (long)DilationH * (KernelH - 1) - 1;
        long width = (long)InWidth + 2L * PadW - (long)DilationW * (KernelW - 1) - 1;
        if (height < 0 || width < 0 || height / StrideH + 1 != OutHeight || width / StrideW + 1 != OutWidth) return false;
        long inputCount, kernelCount, outputCount;
        try
        {
            inputCount = checked((long)Batch * InChannels * InHeight * InWidth);
            kernelCount = checked((long)OutChannels * InChannels * KernelH * KernelW);
            outputCount = checked((long)Batch * OutChannels * OutHeight * OutWidth);
        }
        catch (OverflowException) { return false; }
        if (inputCount > int.MaxValue || kernelCount > int.MaxValue || outputCount > int.MaxValue
            || !Fits(Input, inputCount) || !Fits(Kernel, kernelCount) || !Fits(Bias, OutChannels)
            || !Fits(Residual, outputCount) || !Fits(Output, outputCount)
            || WriteActivation && !Fits(Activated, outputCount)) return false;
        if (Aliases(Output, Input) || Aliases(Output, Kernel) || Aliases(Output, Bias) || Aliases(Output, Residual)) return false;
        if (WriteActivation && (Aliases(Activated, Input) || Aliases(Activated, Kernel) || Aliases(Activated, Bias)
            || Aliases(Activated, Residual) || Aliases(Activated, Output))) return false;
        count = (int)outputCount;
        return true;
    }

    private static bool Fits(IGpuBuffer buffer, long count)
        => buffer is not null && GpuBufferViewRoot.IsBacked(buffer) && buffer.Size >= count && buffer.SizeInBytes >= count * sizeof(float);
    private static bool Aliases(IGpuBuffer left, IGpuBuffer right) => GpuBufferViewRoot.SameAllocation(left, right);
}

internal static class ConvBiasResidualComposition
{
    internal static bool TryExecute(IDirectGpuBackend backend, in ConvBiasResidualArgs args)
    {
        if (!args.TryCount(out int count) || backend is not IChannelBiasResidualBackend epilogue) return false;
        var convolution = backend is IUninitializedGpuAllocation uninitialized
            ? uninitialized.AllocateBufferUninitialized(count) : backend.AllocateBuffer(count);
        try
        {
            backend.Conv2D(args.Input, args.Kernel, convolution, args.Batch, args.InChannels, args.InHeight, args.InWidth,
                args.OutChannels, args.OutHeight, args.OutWidth, args.KernelH, args.KernelW,
                args.StrideH, args.StrideW, args.PadH, args.PadW, args.DilationH, args.DilationW);
            return epilogue.TryChannelBiasResidual(convolution, args.Bias, args.Residual, args.Output, args.Activated,
                args.Batch, args.OutChannels, checked(args.OutHeight * args.OutWidth), args.Alpha, args.WriteActivation);
        }
        finally
        {
            if (backend is IQueuedBufferRelease queued) queued.ReleaseAfterQueuedWork(convolution);
            else convolution.Dispose();
        }
    }
}
