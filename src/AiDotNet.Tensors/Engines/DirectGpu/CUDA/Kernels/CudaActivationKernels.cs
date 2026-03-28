// Copyright (c) AiDotNet. All rights reserved.
// CUDA activation kernels and simple elementwise ops.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    internal static class CudaActivationKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

extern ""C"" __global__ __launch_bounds__(256) void relu(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x > 0.0f ? x : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void sigmoid(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = 1.0f / (1.0f + expf(-x));
}

extern ""C"" __global__ __launch_bounds__(256) void tanh_activation(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // Clamp to avoid NaN on some GPU drivers; tanh saturates to +/-1 for |x| > ~10
    float x = fminf(fmaxf(input[idx], -20.0f), 20.0f);
    output[idx] = tanhf(x);
}

extern ""C"" __global__ __launch_bounds__(256) void gelu(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float sqrt2OverPi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x = input[idx];
    float x3 = x * x * x;
    float inner = sqrt2OverPi * (x + coeff * x3);
    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}

extern ""C"" __global__ __launch_bounds__(256) void swish(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x / (1.0f + expf(-x));
}

// ===========================================================================
// ADDITIONAL ACTIVATION KERNELS (Forward)
// ===========================================================================

// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
// Uses numerically stable softplus to prevent overflow for large x
extern ""C"" __global__ __launch_bounds__(256) void mish(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    // Numerically stable softplus: for large x, softplus(x) ≈ x
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = expf(x);
    } else {
        sp = log1pf(expf(x));
    }
    output[idx] = x * tanhf(sp);
}

// Softplus: ln(1 + e^x)
extern ""C"" __global__ __launch_bounds__(256) void softplus(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    // Numerically stable: for large x, softplus(x) ≈ x
    if (x > 20.0f) {
        output[idx] = x;
    } else if (x < -20.0f) {
        output[idx] = expf(x);
    } else {
        output[idx] = log1pf(expf(x));
    }
}

// Hardswish: x * relu6(x + 3) / 6
extern ""C"" __global__ __launch_bounds__(256) void hardswish(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    if (x <= -3.0f) {
        output[idx] = 0.0f;
    } else if (x >= 3.0f) {
        output[idx] = x;
    } else {
        output[idx] = x * (x + 3.0f) / 6.0f;
    }
}

// SELU: scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
// scale = 1.0507009873554804934193349852946, alpha = 1.6732632423543772848170429916717
extern ""C"" __global__ __launch_bounds__(256) void selu(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    float x = input[idx];
    output[idx] = scale * (x > 0.0f ? x : alpha * expm1f(x));
}

// Hardsigmoid: relu6(x + 3) / 6 = clip((x + 3) / 6, 0, 1)
extern ""C"" __global__ __launch_bounds__(256) void hardsigmoid(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float result = (x + 3.0f) / 6.0f;
    output[idx] = fminf(fmaxf(result, 0.0f), 1.0f);
}

// Hardtanh: clip(x, min_val, max_val) - default min=-1, max=1
extern ""C"" __global__ __launch_bounds__(256) void hardtanh(const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = fminf(fmaxf(x, -1.0f), 1.0f);
}

// Leaky ReLU: x > 0 ? x : alpha * x
extern ""C"" __global__ __launch_bounds__(256) void leaky_relu(const float* __restrict__ input, float* __restrict__ output, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x > 0.0f ? x : alpha * x;
}

// ===========================================================================
// ACTIVATION BACKWARD KERNELS
// ===========================================================================

// ReLU backward: grad * (x > 0 ? 1 : 0)
extern ""C"" __global__ __launch_bounds__(256) void relu_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradInput[idx] = input[idx] > 0.0f ? gradOutput[idx] : 0.0f;
}

// Leaky ReLU backward: grad * (x > 0 ? 1 : alpha)
extern ""C"" __global__ __launch_bounds__(256) void leaky_relu_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    gradInput[idx] = gradOutput[idx] * (x > 0.0f ? 1.0f : alpha);
}

// Sigmoid backward: grad * sigmoid(x) * (1 - sigmoid(x))
extern ""C"" __global__ __launch_bounds__(256) void sigmoid_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float sig = 1.0f / (1.0f + expf(-input[idx]));
    gradInput[idx] = gradOutput[idx] * sig * (1.0f - sig);
}

// Tanh backward: grad * (1 - tanh(x)^2)
extern ""C"" __global__ __launch_bounds__(256) void tanh_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float t = tanhf(input[idx]);
    gradInput[idx] = gradOutput[idx] * (1.0f - t * t);
}

// GELU backward (approximation)
extern ""C"" __global__ __launch_bounds__(256) void gelu_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float sqrt2OverPi = 0.7978845608f;
    const float coeff = 0.044715f;

    float x = input[idx];
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = sqrt2OverPi * (x + coeff * x3);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float dInner = sqrt2OverPi * (1.0f + 3.0f * coeff * x2);

    float dgelu = 0.5f * (1.0f + t) + 0.5f * x * sech2 * dInner;
    gradInput[idx] = gradOutput[idx] * dgelu;
}

// Swish backward: grad * (swish(x) + sigmoid(x) * (1 - swish(x)))
extern ""C"" __global__ __launch_bounds__(256) void swish_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    float sig = 1.0f / (1.0f + expf(-x));
    float swish_val = x * sig;
    float dswish = swish_val + sig * (1.0f - swish_val);
    gradInput[idx] = gradOutput[idx] * dswish;
}

// Mish backward: d/dx[x * tanh(softplus(x))]
// Uses numerically stable softplus and sigmoid to prevent overflow
extern ""C"" __global__ __launch_bounds__(256) void mish_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    // Numerically stable softplus
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = expf(x);
    } else {
        sp = log1pf(expf(x));
    }
    float tsp = tanhf(sp);
    // Numerically stable sigmoid
    float sigmoid_x;
    if (x >= 0.0f) {
        float ex = expf(-x);
        sigmoid_x = 1.0f / (1.0f + ex);
    } else {
        float ex = expf(x);
        sigmoid_x = ex / (1.0f + ex);
    }
    float sech2_sp = 1.0f - tsp * tsp;
    // d(mish)/dx = tanh(sp) + x * sech^2(sp) * sigmoid(x)
    float grad = tsp + x * sech2_sp * sigmoid_x;
    gradInput[idx] = gradOutput[idx] * grad;
}

// Softplus backward: d/dx[ln(1 + e^x)] = sigmoid(x)
extern ""C"" __global__ __launch_bounds__(256) void softplus_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    gradInput[idx] = gradOutput[idx] * sigmoid_x;
}

// Hardswish backward
extern ""C"" __global__ __launch_bounds__(256) void hardswish_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float grad;
    if (x <= -3.0f) {
        grad = 0.0f;
    } else if (x >= 3.0f) {
        grad = 1.0f;
    } else {
        // d/dx[x * (x + 3) / 6] = (2x + 3) / 6
        grad = (2.0f * x + 3.0f) / 6.0f;
    }
    gradInput[idx] = gradOutput[idx] * grad;
}

// SELU backward
extern ""C"" __global__ __launch_bounds__(256) void selu_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    float x = input[idx];
    float grad = x > 0.0f ? scale : scale * alpha * expf(x);
    gradInput[idx] = gradOutput[idx] * grad;
}

// Hardsigmoid backward
extern ""C"" __global__ __launch_bounds__(256) void hardsigmoid_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float grad = (x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f;
    gradInput[idx] = gradOutput[idx] * grad;
}

// Hardtanh backward
extern ""C"" __global__ __launch_bounds__(256) void hardtanh_backward(const float* __restrict__ gradOutput, const float* __restrict__ input, float* __restrict__ gradInput, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float grad = (x > -1.0f && x < 1.0f) ? 1.0f : 0.0f;
    gradInput[idx] = gradOutput[idx] * grad;
}

// Warp-level reduction helpers for maximum performance
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Parallel softmax: 1 block per row, 256 threads cooperate on reduction
// Uses warp shuffles for the last 32 elements, shared memory tree for inter-warp
extern ""C"" __global__ __launch_bounds__(256) void softmax(const float* __restrict__ input, float* __restrict__ output, int batchSize, int features)
{
    int batch = blockIdx.x;
    if (batch >= batchSize) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int baseIdx = batch * features;

    // Phase 1: Find max (parallel reduction with warp shuffles)
    float localMax = -INFINITY;
    for (int f = tid; f < features; f += blockDim.x) {
        localMax = fmaxf(localMax, input[baseIdx + f]);
    }
    // Warp-level max reduction
    localMax = warpReduceMax(localMax);
    int warpId = tid / 32;
    int lane = tid % 32;
    if (lane == 0) smem[warpId] = localMax;
    __syncthreads();
    // Final reduction across warps (first warp only)
    int numWarps = (blockDim.x + 31) / 32;
    if (warpId == 0) {
        localMax = (lane < numWarps) ? smem[lane] : -INFINITY;
        localMax = warpReduceMax(localMax);
    }
    if (tid == 0) smem[0] = localMax;
    __syncthreads();
    float maxVal = smem[0];

    // Phase 2: Compute sum of exp(x - max) (parallel reduction with warp shuffles)
    float localSum = 0.0f;
    for (int f = tid; f < features; f += blockDim.x) {
        float expVal = expf(input[baseIdx + f] - maxVal);
        output[baseIdx + f] = expVal;
        localSum += expVal;
    }
    // Warp-level sum reduction
    localSum = warpReduceSum(localSum);
    if (lane == 0) smem[warpId] = localSum;
    __syncthreads();
    if (warpId == 0) {
        localSum = (lane < numWarps) ? smem[lane] : 0.0f;
        localSum = warpReduceSum(localSum);
    }
    if (tid == 0) smem[0] = localSum;
    __syncthreads();
    float invSum = (smem[0] > 0.0f) ? (1.0f / smem[0]) : 1.0f;

    // Phase 3: Normalize
    for (int f = tid; f < features; f += blockDim.x) {
        output[baseIdx + f] *= invSum;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void add_vectors(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = A[idx] + B[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void subtract_vectors(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = A[idx] - B[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void multiply_vectors(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = A[idx] * B[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void divide_vectors(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float b = B[idx];
    C[idx] = (b != 0.0f) ? (A[idx] / b) : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void min_vectors(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float a = A[idx];
    float b = B[idx];
    C[idx] = a < b ? a : b;
}

extern ""C"" __global__ __launch_bounds__(256) void max_vectors(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float a = A[idx];
    float b = B[idx];
    C[idx] = a > b ? a : b;
}

extern ""C"" __global__ __launch_bounds__(256) void scale_vector(const float* __restrict__ A, float* __restrict__ B, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = A[idx] * scalar;
}

extern ""C"" __global__ __launch_bounds__(256) void abs_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = A[idx];
    B[idx] = x < 0.0f ? -x : x;
}

extern ""C"" __global__ __launch_bounds__(256) void exp_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = expf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void log_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = logf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void log2_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = log2f(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void exp2_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = exp2f(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void exp10_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = powf(10.0f, A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void expm1_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = expm1f(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void log1p_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = log1pf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void sqrt_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = sqrtf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void sign_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = A[idx];
    B[idx] = x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
}

extern ""C"" __global__ __launch_bounds__(256) void power_scalar(const float* __restrict__ A, float* __restrict__ B, float exponent, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = powf(A[idx], exponent);
}
extern ""C"" __global__ __launch_bounds__(256) void reduce_sum(const float* __restrict__ input, float* __restrict__ output, int size)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < (unsigned int)size) ? __ldg(&input[idx]) : 0.0f;

    // Warp-level reduction first (no __syncthreads needed within a warp)
    unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);

    // Write warp results to shared memory
    unsigned int lane = tid & 31;
    unsigned int warpId = tid >> 5;
    if (lane == 0)
        scratch[warpId] = val;
    __syncthreads();

    // Final reduction across warps (only first warp)
    unsigned int numWarps = (blockDim.x + 31) >> 5;
    if (tid < numWarps)
    {
        val = scratch[tid];
        unsigned int warp_mask = (numWarps >= 32) ? 0xFFFFFFFF : ((1u << numWarps) - 1);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(warp_mask, val, offset);

        if (tid == 0)
            output[blockIdx.x] = val;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_max(const float* __restrict__ input, float* __restrict__ output, int size)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < (unsigned int)size) ? __ldg(&input[idx]) : -INFINITY;

    // Warp-level reduction first (no __syncthreads needed within a warp)
    unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(mask, val, offset);
        val = fmaxf(val, other);
    }

    // Write warp results to shared memory
    unsigned int lane = tid & 31;
    unsigned int warpId = tid >> 5;
    if (lane == 0)
        scratch[warpId] = val;
    __syncthreads();

    // Final reduction across warps (only first warp)
    unsigned int numWarps = (blockDim.x + 31) >> 5;
    if (tid < numWarps)
    {
        val = scratch[tid];
        unsigned int warp_mask = (numWarps >= 32) ? 0xFFFFFFFF : ((1u << numWarps) - 1);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_down_sync(warp_mask, val, offset);
            val = fmaxf(val, other);
        }

        if (tid == 0)
            output[blockIdx.x] = val;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_min(const float* __restrict__ input, float* __restrict__ output, int size)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < (unsigned int)size) ? __ldg(&input[idx]) : INFINITY;

    // Warp-level reduction first (no __syncthreads needed within a warp)
    unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(mask, val, offset);
        val = fminf(val, other);
    }

    // Write warp results to shared memory
    unsigned int lane = tid & 31;
    unsigned int warpId = tid >> 5;
    if (lane == 0)
        scratch[warpId] = val;
    __syncthreads();

    // Final reduction across warps (only first warp)
    unsigned int numWarps = (blockDim.x + 31) >> 5;
    if (tid < numWarps)
    {
        val = scratch[tid];
        unsigned int warp_mask = (numWarps >= 32) ? 0xFFFFFFFF : ((1u << numWarps) - 1);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_down_sync(warp_mask, val, offset);
            val = fminf(val, other);
        }

        if (tid == 0)
            output[blockIdx.x] = val;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void sum_axis(const float* __restrict__ input, float* __restrict__ output, int outerSize, int reduceSize)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned int)outerSize) return;

    float sum = 0.0f;
    unsigned int baseOffset = idx * (unsigned int)reduceSize;
    for (int i = 0; i < reduceSize; ++i)
        sum += input[baseOffset + (unsigned int)i];
    output[idx] = sum;
}

extern ""C"" __global__ __launch_bounds__(256) void bias_add(float* __restrict__ data, const float* __restrict__ bias, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= cols || row >= rows) return;
    data[row * cols + col] += __ldg(&bias[col]);
}

// Bias add with separate output: C[i,j] = A[i,j] + bias[j]
extern ""C"" __global__ __launch_bounds__(256) void bias_add_out(const float* __restrict__ A, const float* __restrict__ bias, float* __restrict__ C, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= cols || row >= rows) return;
    int idx = row * cols + col;
    C[idx] = A[idx] + __ldg(&bias[col]);
}

// Conv2D bias add in NCHW format: output[b,c,h,w] += bias[c]
// Uses 2D grid to eliminate integer division/modulo:
//   blockIdx.y = batch * channel index, blockIdx.x * blockDim.x + threadIdx.x = spatial index
extern ""C"" __global__ __launch_bounds__(256) void conv2d_bias_add(float* __restrict__ output, const float* __restrict__ bias, int batch, int channels, int spatialSize)
{
    int spatialIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int bc = blockIdx.y;
    if (spatialIdx >= spatialSize || bc >= batch * channels) return;
    int channel = bc % channels;
    output[bc * spatialSize + spatialIdx] += __ldg(&bias[channel]);
}

// ===========================================================================
// TRIGONOMETRIC KERNELS
// ===========================================================================

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

// ===========================================================================
// HYPERBOLIC KERNELS
// ===========================================================================

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

// ===========================================================================
// ADDITIONAL UNARY KERNELS
// ===========================================================================

extern ""C"" __global__ __launch_bounds__(256) void reciprocal_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = 1.0f / A[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void cbrt_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = cbrtf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void log10_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = log10f(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void negate_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = -A[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void floor_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = floorf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void ceil_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = ceilf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void round_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = roundf(A[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void trunc_vector(const float* __restrict__ A, float* __restrict__ B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = truncf(A[idx]);
}

// ===========================================================================
// VECTORIZED (float4) UNARY KERNELS - 128-bit loads/stores
// Each thread processes 4 elements. size4 = originalSize / 4.
// ===========================================================================

extern ""C"" __global__ __launch_bounds__(256) void relu_vec4(const float* __restrict__ input, float* __restrict__ output, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 v = reinterpret_cast<const float4*>(input)[idx];
    float4 r;
    r.x = v.x > 0.0f ? v.x : 0.0f;
    r.y = v.y > 0.0f ? v.y : 0.0f;
    r.z = v.z > 0.0f ? v.z : 0.0f;
    r.w = v.w > 0.0f ? v.w : 0.0f;
    reinterpret_cast<float4*>(output)[idx] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void sigmoid_vec4(const float* __restrict__ input, float* __restrict__ output, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 v = reinterpret_cast<const float4*>(input)[idx];
    float4 r;
    r.x = 1.0f / (1.0f + expf(-v.x));
    r.y = 1.0f / (1.0f + expf(-v.y));
    r.z = 1.0f / (1.0f + expf(-v.z));
    r.w = 1.0f / (1.0f + expf(-v.w));
    reinterpret_cast<float4*>(output)[idx] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void tanh_activation_vec4(const float* __restrict__ input, float* __restrict__ output, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 v = reinterpret_cast<const float4*>(input)[idx];
    float4 r;
    r.x = tanhf(v.x);
    r.y = tanhf(v.y);
    r.z = tanhf(v.z);
    r.w = tanhf(v.w);
    reinterpret_cast<float4*>(output)[idx] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void gelu_vec4(const float* __restrict__ input, float* __restrict__ output, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    const float sqrt2OverPi = 0.7978845608f;
    const float coeff = 0.044715f;
    float4 v = reinterpret_cast<const float4*>(input)[idx];
    float4 r;
    { float x = v.x; float x3 = x*x*x; float inner = sqrt2OverPi * (x + coeff * x3); r.x = 0.5f * x * (1.0f + tanhf(inner)); }
    { float x = v.y; float x3 = x*x*x; float inner = sqrt2OverPi * (x + coeff * x3); r.y = 0.5f * x * (1.0f + tanhf(inner)); }
    { float x = v.z; float x3 = x*x*x; float inner = sqrt2OverPi * (x + coeff * x3); r.z = 0.5f * x * (1.0f + tanhf(inner)); }
    { float x = v.w; float x3 = x*x*x; float inner = sqrt2OverPi * (x + coeff * x3); r.w = 0.5f * x * (1.0f + tanhf(inner)); }
    reinterpret_cast<float4*>(output)[idx] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void swish_vec4(const float* __restrict__ input, float* __restrict__ output, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 v = reinterpret_cast<const float4*>(input)[idx];
    float4 r;
    r.x = v.x / (1.0f + expf(-v.x));
    r.y = v.y / (1.0f + expf(-v.y));
    r.z = v.z / (1.0f + expf(-v.z));
    r.w = v.w / (1.0f + expf(-v.w));
    reinterpret_cast<float4*>(output)[idx] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void abs_vector_vec4(const float* __restrict__ A, float* __restrict__ B, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 v = reinterpret_cast<const float4*>(A)[idx];
    float4 r;
    r.x = fabsf(v.x);
    r.y = fabsf(v.y);
    r.z = fabsf(v.z);
    r.w = fabsf(v.w);
    reinterpret_cast<float4*>(B)[idx] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void exp_vector_vec4(const float* __restrict__ A, float* __restrict__ B, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 v = reinterpret_cast<const float4*>(A)[idx];
    float4 r;
    r.x = expf(v.x); r.y = expf(v.y); r.z = expf(v.z); r.w = expf(v.w);
    reinterpret_cast<float4*>(B)[idx] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void log_vector_vec4(const float* __restrict__ A, float* __restrict__ B, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 v = reinterpret_cast<const float4*>(A)[idx];
    float4 r;
    r.x = logf(v.x); r.y = logf(v.y); r.z = logf(v.z); r.w = logf(v.w);
    reinterpret_cast<float4*>(B)[idx] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void sqrt_vector_vec4(const float* __restrict__ A, float* __restrict__ B, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 v = reinterpret_cast<const float4*>(A)[idx];
    float4 r;
    r.x = sqrtf(v.x); r.y = sqrtf(v.y); r.z = sqrtf(v.z); r.w = sqrtf(v.w);
    reinterpret_cast<float4*>(B)[idx] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void negate_vector_vec4(const float* __restrict__ A, float* __restrict__ B, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 v = reinterpret_cast<const float4*>(A)[idx];
    float4 r;
    r.x = -v.x; r.y = -v.y; r.z = -v.z; r.w = -v.w;
    reinterpret_cast<float4*>(B)[idx] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void scale_vector_vec4(const float* __restrict__ A, float* __restrict__ B, float scalar, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 v = reinterpret_cast<const float4*>(A)[idx];
    float4 r;
    r.x = v.x * scalar; r.y = v.y * scalar; r.z = v.z * scalar; r.w = v.w * scalar;
    reinterpret_cast<float4*>(B)[idx] = r;
}

// ===========================================================================
// VECTORIZED (float4) BINARY KERNELS
// ===========================================================================

extern ""C"" __global__ __launch_bounds__(256) void add_vectors_vec4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 a = reinterpret_cast<const float4*>(A)[idx];
    float4 b = reinterpret_cast<const float4*>(B)[idx];
    float4 c;
    c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.w = a.w + b.w;
    reinterpret_cast<float4*>(C)[idx] = c;
}

extern ""C"" __global__ __launch_bounds__(256) void subtract_vectors_vec4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 a = reinterpret_cast<const float4*>(A)[idx];
    float4 b = reinterpret_cast<const float4*>(B)[idx];
    float4 c;
    c.x = a.x - b.x; c.y = a.y - b.y; c.z = a.z - b.z; c.w = a.w - b.w;
    reinterpret_cast<float4*>(C)[idx] = c;
}

extern ""C"" __global__ __launch_bounds__(256) void multiply_vectors_vec4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 a = reinterpret_cast<const float4*>(A)[idx];
    float4 b = reinterpret_cast<const float4*>(B)[idx];
    float4 c;
    c.x = a.x * b.x; c.y = a.y * b.y; c.z = a.z * b.z; c.w = a.w * b.w;
    reinterpret_cast<float4*>(C)[idx] = c;
}

extern ""C"" __global__ __launch_bounds__(256) void divide_vectors_vec4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 a = reinterpret_cast<const float4*>(A)[idx];
    float4 b = reinterpret_cast<const float4*>(B)[idx];
    float4 c;
    c.x = (b.x != 0.0f) ? (a.x / b.x) : 0.0f;
    c.y = (b.y != 0.0f) ? (a.y / b.y) : 0.0f;
    c.z = (b.z != 0.0f) ? (a.z / b.z) : 0.0f;
    c.w = (b.w != 0.0f) ? (a.w / b.w) : 0.0f;
    reinterpret_cast<float4*>(C)[idx] = c;
}

extern ""C"" __global__ __launch_bounds__(256) void min_vectors_vec4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 a = reinterpret_cast<const float4*>(A)[idx];
    float4 b = reinterpret_cast<const float4*>(B)[idx];
    float4 c;
    c.x = fminf(a.x, b.x); c.y = fminf(a.y, b.y); c.z = fminf(a.z, b.z); c.w = fminf(a.w, b.w);
    reinterpret_cast<float4*>(C)[idx] = c;
}

extern ""C"" __global__ __launch_bounds__(256) void max_vectors_vec4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size4) return;
    float4 a = reinterpret_cast<const float4*>(A)[idx];
    float4 b = reinterpret_cast<const float4*>(B)[idx];
    float4 c;
    c.x = fmaxf(a.x, b.x); c.y = fmaxf(a.y, b.y); c.z = fmaxf(a.z, b.z); c.w = fmaxf(a.w, b.w);
    reinterpret_cast<float4*>(C)[idx] = c;
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                // Activations
                "relu",
                "sigmoid",
                "tanh_activation",
                "gelu",
                "swish",
                "mish",
                "softplus",
                "hardswish",
                "selu",
                "hardsigmoid",
                "hardtanh",
                "leaky_relu",
                "softmax",
                // Activation backward
                "relu_backward",
                "leaky_relu_backward",
                "sigmoid_backward",
                "tanh_backward",
                "gelu_backward",
                "swish_backward",
                "mish_backward",
                "softplus_backward",
                "hardswish_backward",
                "selu_backward",
                "hardsigmoid_backward",
                "hardtanh_backward",
                // Element-wise binary
                "add_vectors",
                "subtract_vectors",
                "multiply_vectors",
                "divide_vectors",
                "min_vectors",
                "max_vectors",
                // Scalar ops
                "scale_vector",
                "power_scalar",
                // Unary math
                "abs_vector",
                "exp_vector",
                "log_vector",
                "log2_vector",
                "exp2_vector",
                "exp10_vector",
                "expm1_vector",
                "log1p_vector",
                "sqrt_vector",
                "sign_vector",
                // Trigonometric
                "sin_vector",
                "cos_vector",
                "tan_vector",
                "asin_vector",
                "acos_vector",
                "atan_vector",
                // Hyperbolic
                "sinh_vector",
                "cosh_vector",
                "asinh_vector",
                "acosh_vector",
                "atanh_vector",
                // Additional unary
                "reciprocal_vector",
                "cbrt_vector",
                "log10_vector",
                "negate_vector",
                "floor_vector",
                "ceil_vector",
                "round_vector",
                "trunc_vector",
                // Reductions
                "reduce_sum",
                "reduce_max",
                "reduce_min",
                "sum_axis",
                "bias_add",
                "bias_add_out",
                "conv2d_bias_add",
                // Vectorized (float4) unary
                "relu_vec4",
                "sigmoid_vec4",
                "tanh_activation_vec4",
                "gelu_vec4",
                "swish_vec4",
                "abs_vector_vec4",
                "exp_vector_vec4",
                "log_vector_vec4",
                "sqrt_vector_vec4",
                "negate_vector_vec4",
                "scale_vector_vec4",
                // Vectorized (float4) binary
                "add_vectors_vec4",
                "subtract_vectors_vec4",
                "multiply_vectors_vec4",
                "divide_vectors_vec4",
                "min_vectors_vec4",
                "max_vectors_vec4"
            };
        }
    }
}

