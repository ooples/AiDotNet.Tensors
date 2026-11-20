namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IChannelBiasResidualBackend
{
    bool IChannelBiasResidualBackend.TryChannelBiasResidual(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer residual,
        IGpuBuffer output, IGpuBuffer activated, int batch, int channels, int spatial, float alpha, bool writeActivation)
    {
        if (!ChannelBiasResidualGeometry.TryCount(input, bias, residual, output, activated,
            batch, channels, spatial, alpha, writeActivation, out int count)) return false;
        LaunchBatch(BatchExecutionCudaKernels.ChannelBiasResidual, count, input, bias, residual, output, activated,
            batch, channels, spatial, alpha, writeActivation ? 1 : 0);
        return true;
    }
}

