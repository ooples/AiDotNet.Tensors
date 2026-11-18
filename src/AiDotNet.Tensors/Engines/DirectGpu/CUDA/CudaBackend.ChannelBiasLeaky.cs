namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IChannelBiasLeakyBackend
{
    bool IChannelBiasLeakyBackend.TryChannelBiasLeaky(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer output,
        int batch, int channels, int spatial, float alpha)
    {
        if (!ChannelBiasLeakyGeometry.TryCount(input, bias, output, batch, channels, spatial, alpha, out int count)) return false;
        LaunchBatchKernel(BatchExecutionCudaKernels.ChannelBiasLeaky, count, input, bias, output, batch, channels, spatial, alpha);
        return true;
    }
}

