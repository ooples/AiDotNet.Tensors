// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for native Complex<T> split-buffer tensor operations.
// Uses separate real/imaginary GPU buffers for coalesced memory access.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    internal static class CudaComplexKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "split_complex_multiply", "split_complex_conjugate",
            "split_complex_magnitude", "split_complex_magnitude_squared",
            "split_complex_phase", "split_complex_from_polar",
            "split_complex_scale", "split_complex_add",
            "split_complex_cross_spectral",
            "split_complex_topk", "softmax_rows"
        };

        public static string GetSource()
        {
            return @"
extern ""C"" __global__ __launch_bounds__(256) void split_complex_multiply(
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

extern ""C"" __global__ __launch_bounds__(256) void split_complex_conjugate(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_magnitude(
    const float* inReal, const float* inImag,
    float* outMag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMag[idx] = sqrtf(re * re + im * im);
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_magnitude_squared(
    const float* inReal, const float* inImag,
    float* outMagSq, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_phase(
    const float* inReal, const float* inImag,
    float* outPhase, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outPhase[idx] = atan2f(inImag[idx], inReal[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_from_polar(
    const float* mag, const float* phase,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float m = mag[idx], p = phase[idx];
    outReal[idx] = m * cosf(p);
    outImag[idx] = m * sinf(p);
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_scale(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = inReal[idx] * scalar;
    outImag[idx] = inImag[idx] * scalar;
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_add(
    const float* aReal, const float* aImag,
    const float* bReal, const float* bImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_cross_spectral(
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

// Top-K by magnitude: zero elements not in top-K.
// Step 1: compute magnitude squared into scratch buffer (caller does this).
// Step 2: this kernel zeros elements below the K-th largest magnitude.
// Caller provides the threshold magnitude squared value.
extern ""C"" __global__ __launch_bounds__(256) void split_complex_topk(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag,
    float thresholdMagSq, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    float magSq = re * re + im * im;
    if (magSq >= thresholdMagSq) {
        outReal[idx] = re;
        outImag[idx] = im;
    } else {
        outReal[idx] = 0.0f;
        outImag[idx] = 0.0f;
    }
}

// Per-row softmax: each block handles one row.
// Shared memory used for reduction (max and sum).
extern ""C"" __global__ void softmax_rows(
    const float* input, float* output, int rows, int cols)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* rowIn = input + row * cols;
    float* rowOut = output + row * cols;

    // Step 1: find max (reduction)
    float maxVal = -1e30f;
    for (int c = tid; c < cols; c += blockDim.x)
        maxVal = fmaxf(maxVal, rowIn[c]);
    sdata[tid] = maxVal;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    maxVal = sdata[0];

    // Step 2: exp and sum
    float sumExp = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float e = expf(rowIn[c] - maxVal);
        rowOut[c] = e;
        sumExp += e;
    }
    sdata[tid] = sumExp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sumExp = sdata[0];

    // Step 3: normalize
    for (int c = tid; c < cols; c += blockDim.x)
        rowOut[c] /= sumExp;
}
";
        }
    }
}
