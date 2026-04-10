// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for native Complex<T> tensor operations.
// HIP is source-compatible with CUDA device code.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipComplexKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "split_complex_multiply", "split_complex_conjugate", "split_complex_magnitude",
        "split_complex_magnitude_squared", "split_complex_phase", "split_complex_from_polar",
        "split_complex_scale", "split_complex_add", "split_complex_cross_spectral"
    };

    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

#define PI 3.14159265358979323846f

extern ""C"" __global__ void split_complex_multiply(
    const float* aReal, const float* aImag,
    const float* bReal, const float* bImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float ar = aReal[idx], ai = aImag[idx];
    float br = bReal[idx], bi = bImag[idx];
    outReal[idx] = ar * br - ai * bi;
    outImag[idx] = ar * bi + ai * br;
}

extern ""C"" __global__ void split_complex_conjugate(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}

extern ""C"" __global__ void split_complex_magnitude(
    const float* inReal, const float* inImag,
    float* outMag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMag[idx] = sqrtf(re * re + im * im);
}

extern ""C"" __global__ void split_complex_magnitude_squared(
    const float* inReal, const float* inImag,
    float* outMagSq, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}

extern ""C"" __global__ void split_complex_phase(
    const float* inReal, const float* inImag,
    float* outPhase, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outPhase[idx] = atan2f(inImag[idx], inReal[idx]);
}

extern ""C"" __global__ void split_complex_from_polar(
    const float* mag, const float* phase,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float m = mag[idx], p = phase[idx];
    outReal[idx] = m * cosf(p);
    outImag[idx] = m * sinf(p);
}

extern ""C"" __global__ void split_complex_scale(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = inReal[idx] * scalar;
    outImag[idx] = inImag[idx] * scalar;
}

extern ""C"" __global__ void split_complex_add(
    const float* aReal, const float* aImag,
    const float* bReal, const float* bImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}

extern ""C"" __global__ void split_complex_cross_spectral(
    const float* xReal, const float* xImag,
    const float* yReal, const float* yImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xr = xReal[idx], xi = xImag[idx];
    float yr = yReal[idx], yi = yImag[idx];
    outReal[idx] = xr * yr + xi * yi;
    outImag[idx] = xi * yr - xr * yi;
}
";
    }
}
