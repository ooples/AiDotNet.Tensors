namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    internal const string ChannelBiasLeaky = "bx_channel_bias_leaky";
    // Keep the separately rounded addition before the predicate and multiplication.
    // CUDA uses its ordinary activation module's fast-math/FTZ policy, with explicit
    // round-to-nearest intrinsics preventing algebraic contraction across the old boundary.
    internal const string ChannelBiasLeakySource = @"
extern ""C"" __global__ __launch_bounds__(256) void bx_channel_bias_leaky(
    const float* input, const float* bias, float* output,
    int batch, int channels, int spatial, float alpha) {
    int i = bx_gid(); if (i >= batch * channels * spatial) return;
    int channel = (i / spatial) % channels;
#ifdef __CUDA_ARCH__
    float rounded = __fadd_rn(input[i], bias[channel]);
    output[i] = rounded > 0.0f ? rounded : __fmul_rn(alpha, rounded);
#else
    float rounded = input[i] + bias[channel];
    output[i] = rounded > 0.0f ? rounded : alpha * rounded;
#endif
}
";
}

