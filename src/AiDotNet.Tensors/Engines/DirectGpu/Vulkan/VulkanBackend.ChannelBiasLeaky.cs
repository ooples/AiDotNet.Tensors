namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static partial class VulkanBatchExecutionKernels
{
    internal static readonly string ChannelBiasLeakyShader = Shader(new[] { "r:inp", "r:bias", "w:outp" },
        "int batch; int channels; int spatial; float alpha;", @"
void main() {
    int i = bx_gid(); if (i >= batch * channels * spatial) return;
    precise float rounded = inp[i] + bias[(i / spatial) % channels];
    outp[i] = rounded >= 0.0 ? rounded : alpha * rounded;
}
");
}

public sealed partial class VulkanBackend : IChannelBiasLeakyBackend
{
    bool IChannelBiasLeakyBackend.TryChannelBiasLeaky(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer output,
        int batch, int channels, int spatial, float alpha)
    {
        if (!ChannelBiasLeakyGeometry.TryCount(input, bias, output, batch, channels, spatial, alpha, out int count)) return false;
        GlslNaryOp(VulkanBatchExecutionKernels.ChannelBiasLeakyShader, new[] { input, bias, output }, count,
            new[] { Bits(batch), Bits(channels), Bits(spatial), FloatBits(alpha) });
        return true;
    }
}

