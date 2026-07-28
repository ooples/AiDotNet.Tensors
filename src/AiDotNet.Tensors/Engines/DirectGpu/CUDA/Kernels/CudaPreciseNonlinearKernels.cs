// Copyright (c) AiDotNet. All rights reserved.
// Accuracy-critical activation kernels, compiled WITHOUT --use_fast_math.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>Nonlinear kernels whose public parity contracts require accurate libm functions.</summary>
    internal static class CudaPreciseNonlinearKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

extern ""C"" __global__ __launch_bounds__(256) void tanh_activation(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = fminf(fmaxf(input[idx], -20.0f), 20.0f);
    output[idx] = tanhf(x);
}

extern ""C"" __global__ __launch_bounds__(256) void tanh_activation_vec4(
    const float* __restrict__ input, float* __restrict__ output, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 x = reinterpret_cast<const float4*>(input)[idx];
    float4 result;
    result.x = tanhf(fminf(fmaxf(x.x, -20.0f), 20.0f));
    result.y = tanhf(fminf(fmaxf(x.y, -20.0f), 20.0f));
    result.z = tanhf(fminf(fmaxf(x.z, -20.0f), 20.0f));
    result.w = tanhf(fminf(fmaxf(x.w, -20.0f), 20.0f));
    reinterpret_cast<float4*>(output)[idx] = result;
}

__device__ __forceinline__ float precise_softplus(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

extern ""C"" __global__ __launch_bounds__(256) void mish(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x * tanhf(precise_softplus(x));
}

extern ""C"" __global__ __launch_bounds__(256) void mish_backward(
    const float* __restrict__ gradOutput, const float* __restrict__ input,
    float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    float tanhSoftplus = tanhf(precise_softplus(x));
    float sigmoid;
    if (x >= 0.0f) {
        float expNeg = expf(-x);
        sigmoid = 1.0f / (1.0f + expNeg);
    } else {
        float expPos = expf(x);
        sigmoid = expPos / (1.0f + expPos);
    }
    float derivative = tanhSoftplus + x * (1.0f - tanhSoftplus * tanhSoftplus) * sigmoid;
    gradInput[idx] = gradOutput[idx] * derivative;
}
";
        }

        public static string[] GetKernelNames() => new[]
        {
            "tanh_activation",
            "tanh_activation_vec4",
            "mish",
            "mish_backward",
        };
    }
}
