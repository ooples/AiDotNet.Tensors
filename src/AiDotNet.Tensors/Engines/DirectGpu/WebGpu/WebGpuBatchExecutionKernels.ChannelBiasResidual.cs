#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static partial class WebGpuBatchExecutionKernels
{
    internal static readonly string ChannelBiasResidual = Shader(new[] { "r:inp", "r:bias", "r:residual", "rw:outp", "rw:activated" },
        "batch: i32, channels: i32, spatial: i32, alpha: f32, writeActivation: i32", Main + @"
    if (i >= p.batch * p.channels * p.spatial) { return; }
    let biased = inp[i] + bias[(i / p.spatial) % p.channels];
    let value = biased + residual[i];
    outp[i] = value;
    if (p.writeActivation != 0) { activated[i] = select(p.alpha * value, value, value > 0.0); }
}");
}
#endif

