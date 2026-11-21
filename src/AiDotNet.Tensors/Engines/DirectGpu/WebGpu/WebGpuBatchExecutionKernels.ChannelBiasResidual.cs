#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static partial class WebGpuBatchExecutionKernels
{
    internal const string ChannelBiasResidualOnlyName = "bx_channel_bias_residual_only";
    private const string ResidualSum = @"
    if (i >= p.batch * p.channels * p.spatial) { return; }
    let biased = inp[i] + bias[(i / p.spatial) % p.channels];
    let value = biased + residual[i];
    outp[i] = value;
";

    internal static readonly string ChannelBiasResidualOnly = Shader(new[] { "r:inp", "r:bias", "r:residual", "rw:outp" },
        "batch: i32, channels: i32, spatial: i32", Main + ResidualSum + "}");

    internal static readonly string ChannelBiasResidual = Shader(new[] { "r:inp", "r:bias", "r:residual", "rw:outp", "rw:activated" },
        "batch: i32, channels: i32, spatial: i32, alpha: f32, writeActivation: i32", Main + ResidualSum + @"
    if (p.writeActivation != 0) { activated[i] = select(p.alpha * value, value, value > 0.0); }
}");
}
#endif

