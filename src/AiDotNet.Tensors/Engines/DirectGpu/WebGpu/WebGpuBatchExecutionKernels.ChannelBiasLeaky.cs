#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static partial class WebGpuBatchExecutionKernels
{
    internal static readonly string ChannelBiasLeaky = Shader(new[] { "r:inp", "r:bias", "rw:outp" },
        "batch: i32, channels: i32, spatial: i32, alpha: f32", Main + @"
    if (i >= p.batch * p.channels * p.spatial) { return; }
    let rounded = inp[i] + bias[(i / p.spatial) % p.channels];
    outp[i] = select(p.alpha * rounded, rounded, rounded > 0.0);
}");
}
#endif

