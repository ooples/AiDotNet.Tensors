// Copyright (c) AiDotNet. All rights reserved.
// Reduction kernels for sum, mean, max operations.
// Works on ALL .NET versions including .NET Framework 4.6.2.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for reduction operations (sum, max, mean, etc.).
    /// </summary>
    internal static class ReductionKernels
    {
        /// <summary>
        /// Gets all reduction kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// REDUCTION KERNELS
// ===========================================================================

#define REDUCTION_BLOCK_SIZE 256

// Parallel sum reduction
// Input is reduced to partial sums, one per workgroup
__kernel void reduce_sum(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const int size)
{
    const int globalIdx = get_global_id(0);
    const int localIdx = get_local_id(0);
    const int groupIdx = get_group_id(0);
    const int localSize = get_local_size(0);

    // Load into local memory
    scratch[localIdx] = (globalIdx < size) ? input[globalIdx] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            scratch[localIdx] += scratch[localIdx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write partial sum
    if (localIdx == 0) {
        output[groupIdx] = scratch[0];
    }
}

// Parallel max reduction
__kernel void reduce_max(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const int size)
{
    const int globalIdx = get_global_id(0);
    const int localIdx = get_local_id(0);
    const int groupIdx = get_group_id(0);
    const int localSize = get_local_size(0);

    // Load into local memory
    scratch[localIdx] = (globalIdx < size) ? input[globalIdx] : -INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            scratch[localIdx] = fmax(scratch[localIdx], scratch[localIdx + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write partial max
    if (localIdx == 0) {
        output[groupIdx] = scratch[0];
    }
}

// Parallel min reduction
__kernel void reduce_min(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const int size)
{
    const int globalIdx = get_global_id(0);
    const int localIdx = get_local_id(0);
    const int groupIdx = get_group_id(0);
    const int localSize = get_local_size(0);

    // Load into local memory
    scratch[localIdx] = (globalIdx < size) ? input[globalIdx] : INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            scratch[localIdx] = fmin(scratch[localIdx], scratch[localIdx + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write partial min
    if (localIdx == 0) {
        output[groupIdx] = scratch[0];
    }
}

// Sum along axis (rows or columns)
// For a 2D tensor of shape [outerSize, reduceSize], sum along the reduce dimension
__kernel void sum_axis(
    __global const float* input,
    __global float* output,
    const int outerSize,
    const int reduceSize)
{
    const int outerIdx = get_global_id(0);
    if (outerIdx >= outerSize) return;

    float sum = 0.0f;
    for (int i = 0; i < reduceSize; i++) {
        sum += input[outerIdx * reduceSize + i];
    }
    output[outerIdx] = sum;
}

// Max along axis
__kernel void max_axis(
    __global const float* input,
    __global float* output,
    const int outerSize,
    const int reduceSize)
{
    const int outerIdx = get_global_id(0);
    if (outerIdx >= outerSize) return;

    float maxVal = -INFINITY;
    for (int i = 0; i < reduceSize; i++) {
        maxVal = fmax(maxVal, input[outerIdx * reduceSize + i]);
    }
    output[outerIdx] = maxVal;
}

// Argmax along axis
__kernel void argmax_axis(
    __global const float* input,
    __global int* output,
    const int outerSize,
    const int reduceSize)
{
    const int outerIdx = get_global_id(0);
    if (outerIdx >= outerSize) return;

    float maxVal = -INFINITY;
    int maxIdx = 0;
    for (int i = 0; i < reduceSize; i++) {
        float val = input[outerIdx * reduceSize + i];
        if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
        }
    }
    output[outerIdx] = maxIdx;
}

// ============================================================================
// Extended reductions: mean, variance, std, product, norm, logsumexp, cumsum
// ============================================================================

__kernel void mean_axis(
    __global const float* input, __global float* output,
    int outerSize, int reduceSize)
{
    int idx = get_global_id(0);
    if (idx >= outerSize) return;
    float sum = 0.0f;
    int base_idx = idx * reduceSize;
    for (int j = 0; j < reduceSize; j++) sum += input[base_idx + j];
    output[idx] = sum / (float)reduceSize;
}

__kernel void variance_axis(
    __global const float* input, __global float* output,
    int outerSize, int reduceSize)
{
    int idx = get_global_id(0);
    if (idx >= outerSize) return;
    int base_idx = idx * reduceSize;
    float sum = 0.0f;
    for (int j = 0; j < reduceSize; j++) sum += input[base_idx + j];
    float mean = sum / (float)reduceSize;
    float var_sum = 0.0f;
    for (int j = 0; j < reduceSize; j++) { float d = input[base_idx + j] - mean; var_sum += d * d; }
    output[idx] = var_sum / (float)reduceSize;
}

__kernel void std_axis(
    __global const float* input, __global float* output,
    int outerSize, int reduceSize)
{
    int idx = get_global_id(0);
    if (idx >= outerSize) return;
    int base_idx = idx * reduceSize;
    float sum = 0.0f;
    for (int j = 0; j < reduceSize; j++) sum += input[base_idx + j];
    float mean = sum / (float)reduceSize;
    float var_sum = 0.0f;
    for (int j = 0; j < reduceSize; j++) { float d = input[base_idx + j] - mean; var_sum += d * d; }
    output[idx] = sqrt(var_sum / (float)reduceSize);
}

__kernel void product_axis(
    __global const float* input, __global float* output,
    int outerSize, int reduceSize)
{
    int idx = get_global_id(0);
    if (idx >= outerSize) return;
    float prod = 1.0f;
    int base_idx = idx * reduceSize;
    for (int j = 0; j < reduceSize; j++) prod *= input[base_idx + j];
    output[idx] = prod;
}

__kernel void norm_axis(
    __global const float* input, __global float* output,
    int outerSize, int reduceSize)
{
    int idx = get_global_id(0);
    if (idx >= outerSize) return;
    float sum_sq = 0.0f;
    int base_idx = idx * reduceSize;
    for (int j = 0; j < reduceSize; j++) { float v = input[base_idx + j]; sum_sq += v * v; }
    output[idx] = sqrt(sum_sq);
}

__kernel void logsumexp_axis(
    __global const float* input, __global float* output,
    int outerSize, int reduceSize)
{
    int idx = get_global_id(0);
    if (idx >= outerSize) return;
    int base_idx = idx * reduceSize;
    float max_val = -INFINITY;
    for (int j = 0; j < reduceSize; j++) max_val = fmax(max_val, input[base_idx + j]);
    float sum_exp = 0.0f;
    for (int j = 0; j < reduceSize; j++) sum_exp += exp(input[base_idx + j] - max_val);
    output[idx] = max_val + log(sum_exp);
}

__kernel void cumsum_axis(
    __global const float* input, __global float* output,
    int outerSize, int innerSize)
{
    int idx = get_global_id(0);
    if (idx >= outerSize) return;
    int base_idx = idx * innerSize;
    float running = 0.0f;
    for (int j = 0; j < innerSize; j++) { running += input[base_idx + j]; output[base_idx + j] = running; }
}

__kernel void scalar_minus_tensor(
    __global const float* input, __global float* output,
    float scalar, int size)
{
    int idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = scalar - input[idx];
}

__kernel void normalize_l2(
    __global const float* input, __global float* output,
    int outerSize, int innerSize)
{
    int row = get_global_id(0);
    if (row >= outerSize) return;
    int base_idx = row * innerSize;
    float sum_sq = 0.0f;
    for (int j = 0; j < innerSize; j++) { float v = input[base_idx + j]; sum_sq += v * v; }
    float inv_norm = 1.0f / (sqrt(sum_sq) + 1e-12f);
    for (int j = 0; j < innerSize; j++) output[base_idx + j] = input[base_idx + j] * inv_norm;
}

__kernel void reduce_sum_backward(
    __global const float* grad_output, __global float* grad_input,
    int outerSize, int reduceSize)
{
    int idx = get_global_id(0);
    if (idx >= outerSize * reduceSize) return;
    grad_input[idx] = grad_output[idx / reduceSize];
}

__kernel void reduce_mean_backward(
    __global const float* grad_output, __global float* grad_input,
    int outerSize, int reduceSize)
{
    int idx = get_global_id(0);
    if (idx >= outerSize * reduceSize) return;
    grad_input[idx] = grad_output[idx / reduceSize] / (float)reduceSize;
}

__kernel void reduce_sum_of_squares(
    __global const float* input, __global float* output,
    int outerSize, int reduceSize)
{
    int idx = get_global_id(0);
    if (idx >= outerSize) return;
    int base_idx = idx * reduceSize;
    float sum_sq = 0.0f;
    for (int j = 0; j < reduceSize; j++) { float v = input[base_idx + j]; sum_sq += v * v; }
    output[idx] = sum_sq;
}
";
        }

        /// <summary>
        /// Gets kernel names for compilation.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new string[]
            {
                "reduce_sum", "reduce_max", "reduce_min",
                "sum_axis", "max_axis", "argmax_axis",
                "mean_axis", "variance_axis", "std_axis",
                "product_axis", "norm_axis", "logsumexp_axis",
                "cumsum_axis", "scalar_minus_tensor", "normalize_l2",
                "reduce_sum_backward", "reduce_mean_backward",
                "reduce_sum_of_squares"
            };
        }
    }
}
