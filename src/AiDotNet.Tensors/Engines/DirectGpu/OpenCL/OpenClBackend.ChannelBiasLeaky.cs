namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IChannelBiasLeakyBackend
{
    bool IChannelBiasLeakyBackend.TryChannelBiasLeaky(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer output,
        int batch, int channels, int spatial, float alpha)
    {
        if (!ChannelBiasLeakyGeometry.TryCount(input, bias, output, batch, channels, spatial, alpha, out int count)) return false;
        LaunchBatch(BatchExecutionCudaKernels.ChannelBiasLeaky, count, input, bias, output, batch, channels, spatial, alpha);
        return true;
    }
}

