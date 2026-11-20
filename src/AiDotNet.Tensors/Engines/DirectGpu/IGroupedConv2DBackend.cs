using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Ordinary compact-kernel NCHW convolution; groups do not mix channels.</summary>
[BackendSpecificRoute("All six ordinary grouped convolution passes; engines retain composition for custom implementations",
    nameof(IDirectGpuBackend.Conv2D), nameof(IDirectGpuBackend.Conv2DBackwardInput), nameof(IDirectGpuBackend.Conv2DBackwardKernel))]
internal interface IGroupedConv2DBackend
{
    bool TryGroupedConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output, in GroupedConv2DGeometry geometry);
    bool TryGroupedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        in GroupedConv2DGeometry geometry, bool accumulate);
    bool TryGroupedConv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        in GroupedConv2DGeometry geometry, bool accumulate);
}

/// <summary>Validated per-axis geometry shared by the recorded operation and native passes.</summary>
internal readonly struct GroupedConv2DGeometry
{
    internal readonly int Batch, InChannels, InHeight, InWidth, OutChannels, KernelH, KernelW;
    internal readonly int StrideH, StrideW, PadH, PadW, DilationH, DilationW, Groups, OutHeight, OutWidth;
    internal readonly int InputCount, KernelCount, OutputCount;
    internal int InChannelsPerGroup => InChannels / Groups;
    internal int OutChannelsPerGroup => OutChannels / Groups;

    internal GroupedConv2DGeometry(int batch, int inChannels, int inHeight, int inWidth, int outChannels,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int groups)
    {
        if (batch < 0 || inChannels <= 0 || inHeight <= 0 || inWidth <= 0 || outChannels <= 0
            || kernelH <= 0 || kernelW <= 0 || strideH <= 0 || strideW <= 0 || padH < 0 || padW < 0
            || dilationH <= 0 || dilationW <= 0 || groups <= 0 || inChannels % groups != 0 || outChannels % groups != 0)
            throw new ArgumentException("Grouped convolution requires valid NCHW geometry and channel-dividing groups.");
        long effectiveH = (long)dilationH * (kernelH - 1) + 1, effectiveW = (long)dilationW * (kernelW - 1) + 1;
        long paddedH = (long)inHeight + 2L * padH, paddedW = (long)inWidth + 2L * padW;
        if (paddedH < effectiveH || paddedW < effectiveW)
            throw new ArgumentException("Grouped convolution has no valid output positions.");
        Batch = batch; InChannels = inChannels; InHeight = inHeight; InWidth = inWidth; OutChannels = outChannels;
        KernelH = kernelH; KernelW = kernelW; StrideH = strideH; StrideW = strideW; PadH = padH; PadW = padW;
        DilationH = dilationH; DilationW = dilationW; Groups = groups;
        OutHeight = checked((int)((paddedH - effectiveH) / strideH + 1));
        OutWidth = checked((int)((paddedW - effectiveW) / strideW + 1));
        InputCount = checked((int)((long)batch * inChannels * inHeight * inWidth));
        KernelCount = checked((int)((long)outChannels * (inChannels / groups) * kernelH * kernelW));
        OutputCount = checked((int)((long)batch * outChannels * OutHeight * OutWidth));
    }

    internal int[] InputShape => new[] { Batch, InChannels, InHeight, InWidth };
    internal int[] KernelShape => new[] { OutChannels, InChannelsPerGroup, KernelH, KernelW };
    internal int[] OutputShape => new[] { Batch, OutChannels, OutHeight, OutWidth };
    internal int[] Stride => new[] { StrideH, StrideW };
    internal int[] Padding => new[] { PadH, PadW };
    internal int[] Dilation => new[] { DilationH, DilationW };

    internal bool Accepts(IGpuBuffer first, int firstCount, IGpuBuffer second, int secondCount,
        IGpuBuffer destination, int destinationCount)
    {
        if (Batch <= 0 || firstCount <= 0 || secondCount <= 0 || destinationCount <= 0) return false;
        if (first.Size < firstCount || second.Size < secondCount || destination.Size < destinationCount
            || first.SizeInBytes < (long)firstCount * sizeof(float)
            || second.SizeInBytes < (long)secondCount * sizeof(float)
            || destination.SizeInBytes < (long)destinationCount * sizeof(float)) return false;
        var a = GpuBufferViewRoot.Of(first); var b = GpuBufferViewRoot.Of(second); var d = GpuBufferViewRoot.Of(destination);
        return a.Handle != IntPtr.Zero && b.Handle != IntPtr.Zero && d.Handle != IntPtr.Zero
            && !ReferenceEquals(a, d) && !ReferenceEquals(b, d) && a.Handle != d.Handle && b.Handle != d.Handle;
    }
}
