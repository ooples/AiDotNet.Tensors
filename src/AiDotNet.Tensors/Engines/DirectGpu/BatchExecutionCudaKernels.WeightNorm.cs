// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    internal const string WeightNorm = "bx_weight_norm";
    internal const string WeightNormStats = "bx_weight_norm_stats";
    internal const string WeightNormDirectionGrad = "bx_weight_norm_dv";
    internal const string WeightNormGainGrad = "bx_weight_norm_dg";

    /// <summary>The backward kernels, compiled by CUDA without --use_fast_math like the precise forward. Fast math
    /// flushes subnormal directions to zero and approximates 1/sqrt, so with a tiny direction the stats kernel would see a
    /// zero row the precise forward preserved and the gain gradient would collapse to zero.</summary>
    internal static readonly string[] WeightNormBackwardKernelNames = { WeightNormStats, WeightNormDirectionGrad, WeightNormGainGrad };
    internal const string WeightNormBackwardPreciseSource =
        "__device__ __forceinline__ int bx_gid() { return blockIdx.x * blockDim.x + threadIdx.x; }\n" + WeightNormSource;

    // One workgroup owns a row. No weight-sized intermediate is allocated: the
    // backward's only scratch is [rows, 2] (inverse norm and upstream dot product).
    // Keep dV and dG separate: tied targets have an explicit complete-dV-before-dG
    // order, and WebGPU never binds the same storage twice for writing in one call.
    internal const string WeightNormSource = @"
extern ""C"" __global__ __launch_bounds__(256) void bx_weight_norm(const float* direction, const float* gain, float* output, int rows, int columns, float epsilon) {
    int row = (int)blockIdx.x; int lane = (int)threadIdx.x;
    if (row >= rows) return;
    __shared__ float normPart[256];
    float sum = 0.0f;
    for (int j = lane; j < columns; j += (int)blockDim.x) { float v = direction[row * columns + j]; sum += v * v; }
    normPart[lane] = sum; __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s /= 2) {
        if (lane < s) normPart[lane] += normPart[lane + s];
        __syncthreads();
    }
    float scale = gain[row] * (1.0f / sqrtf(normPart[0] + epsilon));
    for (int j = lane; j < columns; j += (int)blockDim.x) output[row * columns + j] = direction[row * columns + j] * scale;
}
extern ""C"" __global__ __launch_bounds__(256) void bx_weight_norm_stats(const float* gradOutput, const float* direction, float* stats, int rows, int columns, float epsilon) {
    int row = (int)blockIdx.x; int lane = (int)threadIdx.x;
    if (row >= rows) return;
    __shared__ float normPart[256]; __shared__ float dotPart[256];
    float sum = 0.0f; float dot = 0.0f;
    for (int j = lane; j < columns; j += (int)blockDim.x) {
        float v = direction[row * columns + j]; sum += v * v; dot += gradOutput[row * columns + j] * v;
    }
    normPart[lane] = sum; dotPart[lane] = dot; __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s /= 2) {
        if (lane < s) { normPart[lane] += normPart[lane + s]; dotPart[lane] += dotPart[lane + s]; }
        __syncthreads();
    }
    if (lane == 0) { stats[2 * row] = 1.0f / sqrtf(normPart[0] + epsilon); stats[2 * row + 1] = dotPart[0]; }
}
extern ""C"" __global__ __launch_bounds__(256) void bx_weight_norm_dv(const float* gradOutput, const float* direction, const float* gain, const float* stats, float* gradDirection, int rows, int columns, int accumulate) {
    int i = bx_gid(); if (i >= rows * columns) return;
    int row = i / columns; float inv = stats[2 * row];
    float value = (gain[row] * inv) * (gradOutput[i] - direction[i] * (stats[2 * row + 1] * inv * inv));
    gradDirection[i] = accumulate != 0 ? gradDirection[i] + value : value;
}
extern ""C"" __global__ __launch_bounds__(256) void bx_weight_norm_dg(const float* stats, float* gradGain, int rows, int accumulate) {
    int row = bx_gid(); if (row >= rows) return;
    float value = stats[2 * row + 1] * stats[2 * row];
    gradGain[row] = accumulate != 0 ? gradGain[row] + value : value;
}
";
}
