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
        "#define BX_RFFT_NORM2(x,y) __fadd_rn(__fmul_rn(x,x), __fmul_rn(y,y))\n" + RfftMagnitudeBody;

    internal const string RfftMagnitudeSource = "#define BX_RFFT_NORM2(x,y) ((x)*(x) + (y)*(y))\n" + RfftMagnitudeBody;

    private const string RfftMagnitudeBody = @"
__device__ __forceinline__ float bx_rfft_norm(float re, float im, float epsilon) {
    float peak = fmaxf(fabsf(re), fabsf(im));
    if (peak > 1e19f || (peak > 0.0f && peak < 1e-18f) || epsilon > 1e37f) {
        peak = fmaxf(peak, sqrtf(epsilon));
        float x = re / peak, y = im / peak, e = sqrtf(epsilon) / peak;
        return peak * sqrtf(BX_RFFT_NORM2(x,y) + e * e);
    }
    return sqrtf(BX_RFFT_NORM2(re,im) + epsilon);
}
extern ""C"" __global__ __launch_bounds__(256) void bx_rfft_magnitude(const float* spectrum, float* output,
    int rows, int n, int spectrumStride, float epsilon) {
    int i = bx_gid(); int bins = n / 2 + 1; if (i >= rows * bins) return;
    int source = 2 * ((i / bins) * spectrumStride + i % bins);
    float re = spectrum[source]; float im = spectrum[source + 1];
    output[i] = bx_rfft_norm(re, im, epsilon);
}
extern ""C"" __global__ __launch_bounds__(256) void bx_rfft_magnitude_seed(const float* spectrum,
    const float* upstream, float* seed, int rows, int n, int spectrumStride, float epsilon) {
    int i = bx_gid(); if (i >= rows * n) return;
    int k = i % n; int bins = n / 2 + 1;
    float re = 0.0f; float im = 0.0f;
    if (k < bins) {
        int source = 2 * ((i / n) * spectrumStride + k);
        re = spectrum[source]; im = spectrum[source + 1];
        float magnitude = bx_rfft_norm(re, im, epsilon);
        float grad = upstream[(i / n) * bins + k];
        re = (re / magnitude) * grad; im = (im / magnitude) * grad;
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
