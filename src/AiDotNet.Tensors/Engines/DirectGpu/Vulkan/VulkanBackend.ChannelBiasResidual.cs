namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static partial class VulkanBatchExecutionKernels
{
    internal static readonly string ChannelBiasResidualShader = Shader(new[] { "r:inp", "r:bias", "r:residual", "w:outp", "w:activated" },
        "int batch; int channels; int spatial; float alpha; int writeActivation;", @"
void main() {
    int i = bx_gid(); if (i >= batch * channels * spatial) return;
    precise float biased = inp[i] + bias[(i / spatial) % channels];
    precise float value = biased + residual[i];
    outp[i] = value;
    if (writeActivation != 0) activated[i] = value >= 0.0 ? value : alpha * value;
}
");
}

public sealed partial class VulkanBackend : IChannelBiasResidualBackend
{
    bool IChannelBiasResidualBackend.TryChannelBiasResidual(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer residual,
        IGpuBuffer output, IGpuBuffer activated, int batch, int channels, int spatial, float alpha, bool writeActivation)
    {
        if (!ChannelBiasResidualGeometry.TryCount(input, bias, residual, output, activated,
            batch, channels, spatial, alpha, writeActivation, out int count)) return false;
        GlslNaryOp(VulkanBatchExecutionKernels.ChannelBiasResidualShader, new[] { input, bias, residual, output, activated }, count,
            new[] { Bits(batch), Bits(channels), Bits(spatial), FloatBits(alpha), Bits(writeActivation ? 1 : 0) });
        return true;
    }
}

