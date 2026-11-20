#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IChannelBiasResidualBackend
{
    bool IChannelBiasResidualBackend.TryChannelBiasResidual(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer residual,
        IGpuBuffer output, IGpuBuffer activated, int batch, int channels, int spatial, float alpha, bool writeActivation)
    {
        if (!ChannelBiasResidualGeometry.TryCount(input, bias, residual, output, activated,
            batch, channels, spatial, alpha, writeActivation, out int count)) return false;
        if (writeActivation)
            LaunchBatch(BatchExecutionCudaKernels.ChannelBiasResidual, count, input, bias, residual, output, activated,
                batch, channels, spatial, alpha, 1);
        else
            LaunchBatch(WebGpuBatchExecutionKernels.ChannelBiasResidualOnlyName, count, input, bias, residual, output,
                batch, channels, spatial);
        return true;
    }
}
#endif

