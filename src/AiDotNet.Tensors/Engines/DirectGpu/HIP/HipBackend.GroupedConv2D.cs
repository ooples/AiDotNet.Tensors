namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IGroupedConv2DBackend
{
    bool IGroupedConv2DBackend.TryGroupedConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        in GroupedConv2DGeometry g)
    {
        if (!g.Accepts(input, g.InputCount, kernel, g.KernelCount, output, g.OutputCount)) return false;
        LaunchBatch(BatchExecutionCudaKernels.GroupedConv2D, g.OutputCount, input, kernel, output,
            g.Batch, g.InChannels, g.InHeight, g.InWidth, g.OutChannels, g.OutHeight, g.OutWidth,
            g.KernelH, g.KernelW, g.StrideH, g.StrideW, g.PadH, g.PadW, g.DilationH, g.DilationW, g.Groups);
        return true;
    }

    bool IGroupedConv2DBackend.TryGroupedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        in GroupedConv2DGeometry g, bool accumulate)
    {
        if (!g.Accepts(gradOutput, g.OutputCount, kernel, g.KernelCount, gradInput, g.InputCount)) return false;
        LaunchBatch(BatchExecutionCudaKernels.GroupedConv2DInputGrad, g.InputCount, gradOutput, kernel, gradInput,
            g.Batch, g.InChannels, g.InHeight, g.InWidth, g.OutChannels, g.OutHeight, g.OutWidth,
            g.KernelH, g.KernelW, g.StrideH, g.StrideW, g.PadH, g.PadW, g.DilationH, g.DilationW, g.Groups, accumulate ? 1 : 0);
        return true;
    }

    bool IGroupedConv2DBackend.TryGroupedConv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        in GroupedConv2DGeometry g, bool accumulate)
    {
        if (!g.Accepts(input, g.InputCount, gradOutput, g.OutputCount, gradKernel, g.KernelCount)) return false;
        LaunchBatch(BatchExecutionCudaKernels.GroupedConv2DKernelGrad, g.KernelCount, input, gradOutput, gradKernel,
            g.Batch, g.InChannels, g.InHeight, g.InWidth, g.OutChannels, g.OutHeight, g.OutWidth,
            g.KernelH, g.KernelW, g.StrideH, g.StrideW, g.PadH, g.PadW, g.DilationH, g.DilationW, g.Groups, accumulate ? 1 : 0);
        return true;
    }
}

