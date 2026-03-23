// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for dot product operations. HIP is source-compatible with CUDA device code.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipDotProductKernels
{
    public static string[] GetKernelNames() =>
    [
        "dot_product",
        "strided_dot_product",
        "batched_dot_product"
    ];

    public static string GetSource()
    {
        return @"
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

// Warp-level reduction
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[64];
    int lane = threadIdx.x % 64;
    int wid = threadIdx.x / 64;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 64) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
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
    if (threadIdx.x == 0) atomicAdd(result, sum);
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
    if (threadIdx.x == 0) atomicAdd(result, sum);
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
