namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Read-only convolution epilogue preserving the separately rounded bias and residual additions.</summary>
[BackendSpecificRoute("All six native bias/residual epilogues; unsupported contexts retain separate operations",
    nameof(IDirectGpuBackend.Conv2DBiasAdd), nameof(IDirectGpuBackend.Add), nameof(IDirectGpuBackend.LeakyRelu))]
internal interface IChannelBiasResidualBackend
{
    bool TryChannelBiasResidual(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer residual,
        IGpuBuffer output, IGpuBuffer activated, int batch, int channels, int spatial, float alpha, bool writeActivation);
}

internal static class ChannelBiasResidualGeometry
{
    internal static bool TryCount(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer residual,
        IGpuBuffer output, IGpuBuffer activated, int batch, int channels, int spatial, float alpha,
        bool writeActivation, out int count)
    {
        count = 0;
        if (!ChannelBiasLeakyGeometry.TryCount(input, bias, output, batch, channels, spatial, alpha, out int total)
            || residual.Size < total || residual.SizeInBytes < (long)total * sizeof(float)) return false;
        var residualRoot = GpuBufferViewRoot.Of(residual);
        var outputRoot = GpuBufferViewRoot.Of(output);
        if (!GpuBufferViewRoot.IsLive(residualRoot) || SameRoot(residualRoot, outputRoot)) return false;
        if (writeActivation)
        {
            if (!ChannelBiasLeakyGeometry.TryCount(input, bias, activated, batch, channels, spatial, alpha, out _)) return false;
            if (GpuBufferViewRoot.SameAllocation(activated, residual) || GpuBufferViewRoot.SameAllocation(activated, output))
                return false;
        }
        count = total;
        return true;
    }

    private static bool SameRoot(IGpuBuffer left, IGpuBuffer right) => GpuBufferViewRoot.Same(left, right);
}
