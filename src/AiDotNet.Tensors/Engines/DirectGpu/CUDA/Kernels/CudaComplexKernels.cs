// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for native Complex<T> tensor operations.
// These operate on split real/imaginary GPU buffers for coalesced memory access.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    internal static class CudaComplexKernels
    {
        public static string GetSource()
        {
            return @"
// Element-wise complex multiplication: (a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)
extern ""C"" __global__ __launch_bounds__(256) void complex_multiply(
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

// Element-wise complex conjugate: (re, -im)
extern ""C"" __global__ __launch_bounds__(256) void complex_conjugate(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}

// Complex magnitude: sqrt(re^2 + im^2)
extern ""C"" __global__ __launch_bounds__(256) void complex_magnitude(
    const float* inReal, const float* inImag,
    float* outMag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float re = inReal[idx], im = inImag[idx];
    outMag[idx] = sqrtf(re * re + im * im);
}

// Complex magnitude squared: re^2 + im^2 (no sqrt for performance)
extern ""C"" __global__ __launch_bounds__(256) void complex_magnitude_squared(
    const float* inReal, const float* inImag,
    float* outMagSq, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float re = inReal[idx], im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}

// Complex phase: atan2(im, re)
extern ""C"" __global__ __launch_bounds__(256) void complex_phase(
    const float* inReal, const float* inImag,
    float* outPhase, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    outPhase[idx] = atan2f(inImag[idx], inReal[idx]);
}

// Complex from polar: (mag*cos(phase), mag*sin(phase))
extern ""C"" __global__ __launch_bounds__(256) void complex_from_polar(
    const float* mag, const float* phase,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float m = mag[idx], p = phase[idx];
    outReal[idx] = m * cosf(p);
    outImag[idx] = m * sinf(p);
}

// Complex scale by real scalar: (re*s, im*s)
extern ""C"" __global__ __launch_bounds__(256) void complex_scale(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    outReal[idx] = inReal[idx] * scalar;
    outImag[idx] = inImag[idx] * scalar;
}

// Complex addition: (a.re+b.re, a.im+b.im)
extern ""C"" __global__ __launch_bounds__(256) void complex_add(
    const float* aReal, const float* aImag,
    const float* bReal, const float* bImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}

// Cross-spectral density: X * conj(Y) = (xr*yr + xi*yi, xi*yr - xr*yi)
extern ""C"" __global__ __launch_bounds__(256) void complex_cross_spectral(
    const float* xReal, const float* xImag,
    const float* yReal, const float* yImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float xr = xReal[idx], xi = xImag[idx];
    float yr = yReal[idx], yi = yImag[idx];
    outReal[idx] = xr * yr + xi * yi;  // X * conj(Y) real part
    outImag[idx] = xi * yr - xr * yi;  // X * conj(Y) imag part
}
";
        }
    }
}
