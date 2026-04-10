// OpenCL kernels for native Complex<T> tensor operations.
// Uses split real/imaginary buffers for coalesced memory access.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

public static class ComplexKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "split_complex_multiply", "split_complex_conjugate",
        "split_complex_magnitude", "split_complex_magnitude_squared",
        "split_complex_phase", "split_complex_from_polar",
        "split_complex_scale", "split_complex_add",
        "split_complex_cross_spectral"
    };

    public static string GetSource() => @"
__kernel void split_complex_multiply(
    __global const float* aReal, __global const float* aImag,
    __global const float* bReal, __global const float* bImag,
    __global float* outReal, __global float* outImag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float ar = aReal[idx], ai = aImag[idx];
    float br = bReal[idx], bi = bImag[idx];
    outReal[idx] = ar * br - ai * bi;
    outImag[idx] = ar * bi + ai * br;
}

__kernel void split_complex_conjugate(
    __global const float* inReal, __global const float* inImag,
    __global float* outReal, __global float* outImag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}

__kernel void split_complex_magnitude(
    __global const float* inReal, __global const float* inImag,
    __global float* outMag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMag[idx] = sqrt(re * re + im * im);
}

__kernel void split_complex_magnitude_squared(
    __global const float* inReal, __global const float* inImag,
    __global float* outMagSq, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}

__kernel void split_complex_phase(
    __global const float* inReal, __global const float* inImag,
    __global float* outPhase, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    outPhase[idx] = atan2(inImag[idx], inReal[idx]);
}

__kernel void split_complex_from_polar(
    __global const float* mag, __global const float* phase,
    __global float* outReal, __global float* outImag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float m = mag[idx], p = phase[idx];
    outReal[idx] = m * cos(p);
    outImag[idx] = m * sin(p);
}

__kernel void split_complex_scale(
    __global const float* inReal, __global const float* inImag,
    __global float* outReal, __global float* outImag,
    const float scalar, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    outReal[idx] = inReal[idx] * scalar;
    outImag[idx] = inImag[idx] * scalar;
}

__kernel void split_complex_add(
    __global const float* aReal, __global const float* aImag,
    __global const float* bReal, __global const float* bImag,
    __global float* outReal, __global float* outImag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}

__kernel void split_complex_cross_spectral(
    __global const float* xReal, __global const float* xImag,
    __global const float* yReal, __global const float* yImag,
    __global float* outReal, __global float* outImag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float xr = xReal[idx], xi = xImag[idx];
    float yr = yReal[idx], yi = yImag[idx];
    outReal[idx] = xr * yr + xi * yi;
    outImag[idx] = xi * yr - xr * yi;
}
";
}
