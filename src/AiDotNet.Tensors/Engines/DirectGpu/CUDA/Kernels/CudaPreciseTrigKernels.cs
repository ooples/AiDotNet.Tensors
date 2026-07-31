// Copyright (c) AiDotNet. All rights reserved.
// Accuracy-critical trigonometry, compiled WITHOUT --use_fast_math.
//
// These kernels were part of CudaActivationKernels, whose module is compiled with --use_fast_math.
// That flag silently remaps sinf/cosf to the __sinf/__cosf intrinsics, which are far less accurate.
// PositionalEncoding builds a double-precision-equivalent angle with a Dekker two-product split and
// the angle-addition identities, then called backend.Sin/Cos -- and the fast intrinsics threw that
// accuracy straight back away. Measured: the GPU sat at 379 ULP against a 64 ULP tolerance, WORSE
// than naive all-float (18 ULP) and far worse than the algorithm itself (1 ULP on CPU float).
// That "worse than doing nothing clever" gap is what proved it was a defect, not a precision limit.
//
// They live in their own module so the fix costs nothing elsewhere: expf/tanhf/sigmoid/sqrtf stay in
// the fast-math activation module, since those are the NN hot path and standard practice is to use
// the fast intrinsics there. Mirrors the existing parity210 and audio modules, which opt out of
// fast-math for the same reason (the audio note calls out the same silent __sinf/__cosf remap).
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>Trig kernels that require IEEE-accurate libm, not the fast intrinsics.</summary>
    internal static class CudaPreciseTrigKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

extern ""C"" __global__ __launch_bounds__(256) void sin_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = sinf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void cos_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = cosf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void tan_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = tanf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void asin_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = asinf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void acos_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = acosf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void atan_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = atanf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void sinh_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = sinhf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void cosh_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = coshf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void asinh_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = asinhf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void acosh_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = acoshf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void atanh_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = atanhf(A[idx]);
}
";
        }

        public static string[] GetKernelNames() => new[]
        {
            "sin_vector",
            "cos_vector",
            "tan_vector",
            "asin_vector",
            "acos_vector",
            "atan_vector",
            "sinh_vector",
            "cosh_vector",
            "asinh_vector",
            "acosh_vector",
            "atanh_vector",
        };
    }
}
