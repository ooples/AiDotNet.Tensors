namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>One out-of-place NCHW channel-bias pass followed by the existing LeakyReLU branch.</summary>
[BackendSpecificRoute("All six native channel-bias/Leaky epilogues; unsupported contexts keep the two native operations",
    nameof(IDirectGpuBackend.Conv2DBiasAdd), nameof(IDirectGpuBackend.LeakyRelu))]
internal interface IChannelBiasLeakyBackend
{
    /// <summary>False means no output was written. Inputs stay read-only; finite positive slopes only.</summary>
    bool TryChannelBiasLeaky(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer output,
        int batch, int channels, int spatial, float alpha);
}

internal static class ChannelBiasLeakyGeometry
{
    internal static bool TryCount(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer output,
        int batch, int channels, int spatial, float alpha, out int count)
    {
        count = 0;
        if (batch <= 0 || channels <= 0 || spatial <= 0 || !(alpha > 0) || float.IsInfinity(alpha)) return false;
        long planes = (long)batch * channels;
        if (planes > int.MaxValue / spatial) return false;
        long total = planes * spatial;
        if (input.Size < total || output.Size < total || bias.Size < channels
            || input.SizeInBytes < total * sizeof(float) || output.SizeInBytes < total * sizeof(float)
            || bias.SizeInBytes < (long)channels * sizeof(float)) return false;
        if (!GpuBufferViewRoot.IsBacked(output) || !GpuBufferViewRoot.IsBacked(input) || !GpuBufferViewRoot.IsBacked(bias)
            || GpuBufferViewRoot.SameAllocation(output, input) || GpuBufferViewRoot.SameAllocation(output, bias)) return false;
        count = (int)total;
        return true;
    }
}

