// Copyright (c) CrowdTrainer LLC. All rights reserved.
// HIP kernels for dot product operations. HIP is source-compatible with CUDA device code.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipDotProductKernels
{
    public static string[] GetKernelNames() =>
    [
        "dot_product",
        "dot_product_deterministic",
        "strided_dot_product",
        "strided_dot_product_deterministic",
        "batched_dot_product"
    ];

    public static string GetSource()
    {
        return @"
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

// Block-level sum through shared memory, for a power-of-two blockDim.x <= 256. The previous version shuffled from
// offset 32 and took lanes as threadIdx.x % 64, which is right only for 64-wide waves: on a wave32 RDNA device the
// offset-32 shuffle returns the lane's own value (doubling it) and lanes 32-63 never reach shared memory.
// Every thread returns the block's sum.
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[256];
    shared[threadIdx.x] = val;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float sum = shared[0];
    __syncthreads();
    return sum;
}

extern ""C"" __global__ __launch_bounds__(256) void dot_product(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ result, int size)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }
    sum = blockReduceSum(sum);
    // NON-DETERMINISTIC (issue #382); see dot_product_deterministic.
    if (threadIdx.x == 0) atomicAdd(result, sum);
}

extern ""C"" __global__ __launch_bounds__(256) void dot_product_deterministic(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ result, int size)
{
    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) sum += a[i] * b[i];
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) *result = sum;
}

extern ""C"" __global__ __launch_bounds__(256) void strided_dot_product(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ result, int aSize, int bSize, int bOffset, int bStride)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < aSize; i += blockDim.x * gridDim.x) {
        int bIdx = bOffset + i * bStride;
        if (bIdx >= 0 && bIdx < bSize) {
            sum += a[i] * b[bIdx];
        }
    }
    sum = blockReduceSum(sum);
    // NON-DETERMINISTIC (issue #382); see strided_dot_product_deterministic.
    if (threadIdx.x == 0) atomicAdd(result, sum);
}

extern ""C"" __global__ __launch_bounds__(256) void strided_dot_product_deterministic(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ result, int aSize, int bSize, int bOffset, int bStride)
{
    float sum = 0.0f;
    for (int i = threadIdx.x; i < aSize; i += blockDim.x) {
        int bIdx = bOffset + i * bStride;
        if (bIdx >= 0 && bIdx < bSize) sum += a[i] * b[bIdx];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) *result = sum;
}

extern ""C"" __global__ __launch_bounds__(256) void batched_dot_product(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ result, int batchSize, int vecSize)
{
    int batchIdx = blockIdx.y;
    if (batchIdx >= batchSize) return;

    const float* aRow = a + batchIdx * vecSize;
    const float* bRow = b + batchIdx * vecSize;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < vecSize; i += blockDim.x) {
        sum += aRow[i] * bRow[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) result[batchIdx] = sum;
}
";
    }
}
