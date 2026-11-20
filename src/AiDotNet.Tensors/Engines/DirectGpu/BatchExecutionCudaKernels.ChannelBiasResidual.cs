namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    internal const string ChannelBiasResidual = "bx_channel_bias_residual";
    internal const string ChannelBiasResidualSource = @"
extern ""C"" __global__ __launch_bounds__(256) void bx_channel_bias_residual(
    const float* input, const float* bias, const float* residual, float* output, float* activated,
    int batch, int channels, int spatial, float alpha, int writeActivation) {
    int i = bx_gid(); if (i >= batch * channels * spatial) return;
    int channel = (i / spatial) % channels;
#ifdef __CUDA_ARCH__
    float biased = __fadd_rn(input[i], bias[channel]);
    float value = __fadd_rn(biased, residual[i]);
    output[i] = value;
    if (writeActivation) activated[i] = value > 0.0f ? value : __fmul_rn(alpha, value);
#else
    float biased = input[i] + bias[channel];
    float value = biased + residual[i];
    output[i] = value;
    if (writeActivation) activated[i] = value > 0.0f ? value : alpha * value;
#endif
}
";
}
