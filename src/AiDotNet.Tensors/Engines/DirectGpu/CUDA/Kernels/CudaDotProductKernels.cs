// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for dot product operations including strided variants.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for dot product operations.
/// Includes standard contiguous dot product and strided dot product
/// for reversed/non-contiguous memory access patterns.
/// </summary>
internal static class CudaDotProductKernels
{
    public static string[] GetKernelNames() =>
    [
        "dot_product",
        "strided_dot_product",
        "batched_dot_product",
        "tensor_slice_dot"
    ];

    public static string GetSource()
    {
        return @"
#include <math.h>

// ===========================================================================
// DOT PRODUCT KERNELS
// ===========================================================================

// Warp-level reduction for dot product
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // First warp reduces across all warps
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

// Standard contiguous dot product: result = sum(a[i] * b[i])
// Each block computes a partial sum, atomically accumulated into result.
extern ""C"" __global__ __launch_bounds__(256) void dot_product(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ result, int size)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(result, sum);
    }
}

// Strided dot product: result = sum(a[i] * b[bOffset + i * bStride])
// Supports reversed (stride=-1) and arbitrary stride access.
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

    if (threadIdx.x == 0) {
        atomicAdd(result, sum);
    }
}

// Batched dot product: compute dot products for multiple pairs
// a_batch and b_batch are [batchSize, vecSize], result is [batchSize]
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

    if (threadIdx.x == 0) {
        result[batchIdx] = sum;
    }
}

// Tensor slice dot product: compute dot(tensor[batch, :, dim], vector)
// where tensor is [batchSize, sliceSize, dimSize] stored in row-major.
// This extracts a strided slice along axis 1 and dots with vector.
extern ""C"" __global__ __launch_bounds__(256) void tensor_slice_dot(
    const float* __restrict__ tensor, const float* __restrict__ vector,
    float* __restrict__ result,
    int sliceSize, int innerStride, int tensorOffset)
{
    float sum = 0.0f;
    for (int i = threadIdx.x; i < sliceSize; i += blockDim.x) {
        sum += tensor[tensorOffset + i * innerStride] * vector[i];
    }

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0) {
        result[0] = sum;
    }
}
";
    }
}
