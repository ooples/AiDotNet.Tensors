// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    internal const string RfftMagnitude = "bx_rfft_magnitude";
    internal const string RfftMagnitudeSeed = "bx_rfft_magnitude_seed";
    internal const string RfftRealGradient = "bx_rfft_real_gradient";
    internal static readonly string[] RfftMagnitudeKernelNames = { RfftMagnitude, RfftMagnitudeSeed, RfftRealGradient };
    internal static string RfftMagnitudeCudaSource =>
        "__device__ __forceinline__ int bx_gid() { return blockIdx.x * blockDim.x + threadIdx.x; }\n" +
        RfftMagnitudeSource.Replace("re * re + im * im", "__fadd_rn(__fmul_rn(re, re), __fmul_rn(im, im))");

    // This module is compiled without CUDA fast math: the mel loss depends on
    // small magnitudes and must retain sqrt/division and subnormal semantics.
    internal const string RfftMagnitudeSource = @"
extern ""C"" __global__ __launch_bounds__(256) void bx_rfft_magnitude(const float* spectrum, float* output,
    int rows, int n, int spectrumStride, float epsilon) {
    int i = bx_gid(); int bins = n / 2 + 1; if (i >= rows * bins) return;
    int source = 2 * ((i / bins) * spectrumStride + i % bins);
    float re = spectrum[source]; float im = spectrum[source + 1];
    output[i] = sqrtf((re * re + im * im) + epsilon);
}
extern ""C"" __global__ __launch_bounds__(256) void bx_rfft_magnitude_seed(const float* spectrum,
    const float* upstream, float* seed, int rows, int n, int spectrumStride, float epsilon) {
    int i = bx_gid(); if (i >= rows * n) return;
    int k = i % n; int bins = n / 2 + 1;
    float re = 0.0f; float im = 0.0f;
    if (k < bins) {
        int source = 2 * ((i / n) * spectrumStride + k);
        re = spectrum[source]; im = spectrum[source + 1];
        float magnitude = sqrtf((re * re + im * im) + epsilon);
        float scale = upstream[(i / n) * bins + k] / magnitude;
        re = re * scale; im = im * scale;
    }
    seed[2 * i] = re; seed[2 * i + 1] = im;
}
extern ""C"" __global__ __launch_bounds__(256) void bx_rfft_real_gradient(const float* complex,
    float* output, int count, float scale, int accumulate) {
    int i = bx_gid(); if (i >= count) return;
    float value = complex[2 * i] * scale;
    output[i] = accumulate != 0 ? output[i] + value : value;
}
";
}
