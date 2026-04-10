// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language (MSL) compute kernels for native Complex<T> tensor operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

internal static class MetalComplexKernels
{
    public static string GetSource() => @"
#include <metal_stdlib>
using namespace metal;

kernel void split_complex_multiply(
    device const float* aReal [[buffer(0)]],
    device const float* aImag [[buffer(1)]],
    device const float* bReal [[buffer(2)]],
    device const float* bImag [[buffer(3)]],
    device float* outReal [[buffer(4)]],
    device float* outImag [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float ar = aReal[idx], ai = aImag[idx];
    float br = bReal[idx], bi = bImag[idx];
    outReal[idx] = ar * br - ai * bi;
    outImag[idx] = ar * bi + ai * br;
}

kernel void split_complex_conjugate(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outReal [[buffer(2)]],
    device float* outImag [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}

kernel void split_complex_magnitude(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outMag [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMag[idx] = sqrt(re * re + im * im);
}

kernel void split_complex_magnitude_squared(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outMagSq [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}

kernel void split_complex_phase(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outPhase [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    outPhase[idx] = atan2(inImag[idx], inReal[idx]);
}

kernel void split_complex_from_polar(
    device const float* mag [[buffer(0)]],
    device const float* phase [[buffer(1)]],
    device float* outReal [[buffer(2)]],
    device float* outImag [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float m = mag[idx], p = phase[idx];
    outReal[idx] = m * cos(p);
    outImag[idx] = m * sin(p);
}

kernel void split_complex_scale(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outReal [[buffer(2)]],
    device float* outImag [[buffer(3)]],
    constant float& scalar [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    outReal[idx] = inReal[idx] * scalar;
    outImag[idx] = inImag[idx] * scalar;
}

kernel void split_complex_add(
    device const float* aReal [[buffer(0)]],
    device const float* aImag [[buffer(1)]],
    device const float* bReal [[buffer(2)]],
    device const float* bImag [[buffer(3)]],
    device float* outReal [[buffer(4)]],
    device float* outImag [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}

kernel void split_complex_cross_spectral(
    device const float* xReal [[buffer(0)]],
    device const float* xImag [[buffer(1)]],
    device const float* yReal [[buffer(2)]],
    device const float* yImag [[buffer(3)]],
    device float* outReal [[buffer(4)]],
    device float* outImag [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float xr = xReal[idx], xi = xImag[idx];
    float yr = yReal[idx], yi = yImag[idx];
    outReal[idx] = xr * yr + xi * yi;
    outImag[idx] = xi * yr - xr * yi;
}
";
}
