// Copyright (c) AiDotNet. All rights reserved.
// OpenCL C kernels for dot product operations with parallel reduction.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class DotProductKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "dot_product",
        "strided_dot_product",
        "batched_dot_product",
        "reduce_partial_sums"
    };

    public static string GetSource()
    {
        return @"
// Dot product with parallel reduction
// Each workgroup computes a partial sum, stored in partialSums[group_id]
__kernel void dot_product(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict partialSums,
    const int size,
    __local float* shared)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    float sum = 0.0f;
    for (int i = gid; i < size; i += get_global_size(0)) {
        sum += A[i] * B[i];
    }
    shared[lid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partialSums[group_id] = shared[0];
    }
}

// Strided dot product: result = sum(A[i] * B[bOffset + i * bStride])
__kernel void strided_dot_product(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict partialSums,
    const int aSize,
    const int bSize,
    const int bOffset,
    const int bStride,
    __local float* shared)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    float sum = 0.0f;
    for (int i = gid; i < aSize; i += get_global_size(0)) {
        int bIdx = bOffset + i * bStride;
        if (bIdx >= 0 && bIdx < bSize) {
            sum += A[i] * B[bIdx];
        }
    }
    shared[lid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partialSums[group_id] = shared[0];
    }
}

// Batched dot product: result[batch] = dot(A[batch*vecSize:], B[batch*vecSize:])
__kernel void batched_dot_product(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict result,
    const int batchSize,
    const int vecSize,
    __local float* shared)
{
    int batchIdx = get_group_id(1);
    if (batchIdx >= batchSize) return;

    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    __global const float* aRow = A + batchIdx * vecSize;
    __global const float* bRow = B + batchIdx * vecSize;

    float sum = 0.0f;
    for (int i = lid; i < vecSize; i += group_size) {
        sum += aRow[i] * bRow[i];
    }
    shared[lid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        result[batchIdx] = shared[0];
    }
}

// Final reduction of partial sums
__kernel void reduce_partial_sums(
    __global const float* restrict partialSums,
    __global float* restrict result,
    const int count,
    __local float* shared)
{
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    float sum = 0.0f;
    for (int i = lid; i < count; i += group_size) {
        sum += partialSums[i];
    }
    shared[lid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        result[0] = shared[0];
    }
}
";
    }
}
