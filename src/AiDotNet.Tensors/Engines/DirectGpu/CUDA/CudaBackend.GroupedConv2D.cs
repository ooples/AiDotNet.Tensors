namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGroupedConv2DBackend
{
    bool IGroupedConv2DBackend.TryGroupedConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        in GroupedConv2DGeometry geometry)
    {
        if (!geometry.Accepts(input, geometry.InputCount, kernel, geometry.KernelCount, output, geometry.OutputCount)) return false;
        if (TryGroupedCudnn(0, input, kernel, output, geometry, false)) return true;
        LaunchBatchKernel("bx_grouped_conv2d", geometry.OutputCount, input, kernel, output,
            geometry.Batch, geometry.InChannels, geometry.InHeight, geometry.InWidth, geometry.OutChannels,
            geometry.OutHeight, geometry.OutWidth, geometry.KernelH, geometry.KernelW, geometry.StrideH, geometry.StrideW,
            geometry.PadH, geometry.PadW, geometry.DilationH, geometry.DilationW, geometry.Groups);
        return true;
    }

    bool IGroupedConv2DBackend.TryGroupedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        in GroupedConv2DGeometry geometry, bool accumulate)
    {
        if (!geometry.Accepts(gradOutput, geometry.OutputCount, kernel, geometry.KernelCount, gradInput, geometry.InputCount)) return false;
        if (TryGroupedCudnn(1, gradOutput, kernel, gradInput, geometry, accumulate)) return true;
        LaunchBatchKernel("bx_grouped_conv2d_dx", geometry.InputCount, gradOutput, kernel, gradInput,
            geometry.Batch, geometry.InChannels, geometry.InHeight, geometry.InWidth, geometry.OutChannels,
            geometry.OutHeight, geometry.OutWidth, geometry.KernelH, geometry.KernelW, geometry.StrideH, geometry.StrideW,
            geometry.PadH, geometry.PadW, geometry.DilationH, geometry.DilationW, geometry.Groups, accumulate ? 1 : 0);
        return true;
    }

    bool IGroupedConv2DBackend.TryGroupedConv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        in GroupedConv2DGeometry geometry, bool accumulate)
    {
        if (!geometry.Accepts(input, geometry.InputCount, gradOutput, geometry.OutputCount, gradKernel, geometry.KernelCount)) return false;
        if (TryGroupedCudnn(2, input, gradOutput, gradKernel, geometry, accumulate)) return true;
        LaunchBatchKernel("bx_grouped_conv2d_dw", geometry.KernelCount, input, gradOutput, gradKernel,
            geometry.Batch, geometry.InChannels, geometry.InHeight, geometry.InWidth, geometry.OutChannels,
            geometry.OutHeight, geometry.OutWidth, geometry.KernelH, geometry.KernelW, geometry.StrideH, geometry.StrideW,
            geometry.PadH, geometry.PadW, geometry.DilationH, geometry.DilationW, geometry.Groups, accumulate ? 1 : 0);
        return true;
    }

    private bool TryGroupedCudnn(int pass, IGpuBuffer first, IGpuBuffer second, IGpuBuffer destination,
        in GroupedConv2DGeometry geometry, bool accumulate)
    {
        if (!CudaDispatchPolicy.UseCudnnForConv) return false;
        using var contextScope = PushContext();
        EnsureCudnnConv();
        var context = _cudnnContext ?? throw new InvalidOperationException("cuDNN grouped context was not initialized.");
        var convolution = _cudnnConv ?? throw new InvalidOperationException("cuDNN grouped convolution was not initialized.");
        lock (context.StreamBindingLock)
        {
            context.BindStream(_stream);
            // This operation has its own group-aware descriptors. The ungrouped tuned-slot keys cannot represent it.
            bool success = pass switch
            {
                0 => convolution.TryGroupedConv2DForwardGpu(first.Handle, second.Handle, destination.Handle, geometry, CudnnConvMath),
                1 => convolution.TryGroupedConv2DBackwardDataGpu(first.Handle, second.Handle, destination.Handle, geometry, accumulate, CudnnConvMath),
                _ => convolution.TryGroupedConv2DBackwardFilterGpu(first.Handle, second.Handle, destination.Handle, geometry, accumulate, CudnnConvMath),
            };
            if (success && CudnnExecutionDiagnosticsEnabled)
            {
                if (pass == 0) System.Threading.Interlocked.Increment(ref _cudnnForwardExecutions);
                else if (pass == 1) System.Threading.Interlocked.Increment(ref _cudnnBackwardDataExecutions);
                else System.Threading.Interlocked.Increment(ref _cudnnBackwardFilterExecutions);
            }
            return success;
        }
    }
}
