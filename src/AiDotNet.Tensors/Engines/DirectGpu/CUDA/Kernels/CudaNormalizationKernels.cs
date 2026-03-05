// Copyright (c) AiDotNet. All rights reserved.
// CUDA normalization kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA kernels for normalization operations.
    /// </summary>
    internal static class CudaNormalizationKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

// ===========================================================================
// NORMALIZATION KERNELS - Parallel Shared Memory Reductions
// Each block handles one normalization unit (channel, batch element, etc.)
// 256 threads cooperate on tree reductions for 10-20x speedup.
// ===========================================================================

// Warp-level sum reduction using shuffle intrinsics (no shared memory needed)
__device__ __forceinline__ float warpReduceSumNorm(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction: shared memory tree for inter-warp, warp shuffle for intra-warp
// ~30% faster than pure shared memory tree for 256 threads
#define BLOCK_REDUCE(sdata, tid) \
    __syncthreads(); \
    for (int s = blockDim.x / 2; s > 32; s >>= 1) { \
        if ((tid) < s) (sdata)[(tid)] += (sdata)[(tid) + s]; \
        __syncthreads(); \
    } \
    if ((tid) < 32) { \
        float wval = (sdata)[(tid)]; \
        if (blockDim.x >= 64) wval += (sdata)[(tid) + 32]; \
        wval = warpReduceSumNorm(wval); \
        (sdata)[(tid)] = wval; \
    } \
    __syncthreads();

// Batch Normalization forward pass
// 1 block per channel, 256 threads parallel reduce across batch*spatial
extern ""C"" __global__ void batchnorm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    float* runningMean, float* runningVar,
    float* saveMean, float* saveInvVar,
    int batch, int channels, int spatialSize,
    float epsilon, float momentum, int training)
{
    extern __shared__ float smem[];
    int c = blockIdx.x;
    if (c >= channels) return;
    int tid = threadIdx.x;
    int batchSpatial = batch * spatialSize;

    float mean, invVar;

    if (training) {
        // Training: compute mean/var from batch data
        float localSum = 0.0f;
        for (int i = tid; i < batchSpatial; i += blockDim.x) {
            int b = i / spatialSize;
            int s = i % spatialSize;
            localSum += input[(b * channels + c) * spatialSize + s];
        }
        smem[tid] = localSum;
        BLOCK_REDUCE(smem, tid);
        mean = smem[0] / (float)batchSpatial;
        __syncthreads();

        float localVar = 0.0f;
        for (int i = tid; i < batchSpatial; i += blockDim.x) {
            int b = i / spatialSize;
            int s = i % spatialSize;
            float diff = input[(b * channels + c) * spatialSize + s] - mean;
            localVar += diff * diff;
        }
        smem[tid] = localVar;
        BLOCK_REDUCE(smem, tid);
        float var = smem[0] / (float)batchSpatial;

        invVar = rsqrtf(var + epsilon);

        if (tid == 0) {
            saveMean[c] = mean;
            saveInvVar[c] = invVar;
            runningMean[c] = (1.0f - momentum) * runningMean[c] + momentum * mean;
            runningVar[c] = (1.0f - momentum) * runningVar[c] + momentum * var;
        }
    } else {
        // Inference: use running statistics
        mean = runningMean[c];
        invVar = rsqrtf(runningVar[c] + epsilon);
    }

    // Parallel normalize + scale + shift
    float g = gamma[c];
    float b_val = beta[c];
    for (int i = tid; i < batchSpatial; i += blockDim.x) {
        int b = i / spatialSize;
        int s = i % spatialSize;
        int idx = (b * channels + c) * spatialSize + s;
        float normalized = (input[idx] - mean) * invVar;
        output[idx] = g * normalized + b_val;
    }
}

// Batch Normalization backward pass
// 1 block per channel, 256 threads parallel reduce
extern ""C"" __global__ void batchnorm_backward(
    const float* gradOutput, const float* input,
    const float* gamma, const float* saveMean, const float* saveInvVar,
    float* gradInput, float* gradGamma, float* gradBeta,
    int batch, int channels, int spatialSize, float epsilon)
{
    extern __shared__ float smem[];
    float* smem2 = smem + blockDim.x;
    float* smem3 = smem2 + blockDim.x;
    int c = blockIdx.x;
    if (c >= channels) return;
    int tid = threadIdx.x;
    int batchSpatial = batch * spatialSize;
    float mean = saveMean[c];
    float invVar = saveInvVar[c];
    float g = gamma[c];

    // Parallel reduction for dGamma, dBeta, and gamma-scaled sums
    float locDGamma = 0.0f, locDBeta = 0.0f, locSumDxhat = 0.0f, locSumDxhatXhat = 0.0f;
    for (int i = tid; i < batchSpatial; i += blockDim.x) {
        int b = i / spatialSize;
        int s = i % spatialSize;
        int idx = (b * channels + c) * spatialSize + s;
        float xhat = (input[idx] - mean) * invVar;
        float dxhat = gradOutput[idx] * g;
        locDGamma += gradOutput[idx] * xhat;
        locDBeta += gradOutput[idx];
        locSumDxhat += dxhat;
        locSumDxhatXhat += dxhat * xhat;
    }
    smem[tid] = locDGamma;
    smem2[tid] = locDBeta;
    smem3[tid] = locSumDxhat;
    BLOCK_REDUCE(smem, tid);
    BLOCK_REDUCE(smem2, tid);
    BLOCK_REDUCE(smem3, tid);

    if (tid == 0) {
        gradGamma[c] = smem[0];
        gradBeta[c] = smem2[0];
    }
    float sumDxhat = smem3[0];
    __syncthreads();

    // Second reduction for sum(dxhat * xhat)
    smem[tid] = locSumDxhatXhat;
    BLOCK_REDUCE(smem, tid);
    float sumDxhatXhat = smem[0];

    // Parallel gradInput computation
    float invN = 1.0f / (float)batchSpatial;
    for (int i = tid; i < batchSpatial; i += blockDim.x) {
        int b = i / spatialSize;
        int s = i % spatialSize;
        int idx = (b * channels + c) * spatialSize + s;
        float xhat = (input[idx] - mean) * invVar;
        float dxhat = gradOutput[idx] * g;
        gradInput[idx] = invVar * (dxhat - invN * (sumDxhat + xhat * sumDxhatXhat));
    }
}

// Layer Normalization forward pass
// 1 block per batch element, 256 threads parallel reduce across normalizedSize
extern ""C"" __global__ void layernorm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    float* saveMean, float* saveInvVar,
    int batchSize, int normalizedSize, float epsilon)
{
    extern __shared__ float smem[];
    int b = blockIdx.x;
    if (b >= batchSize) return;
    int tid = threadIdx.x;
    int base = b * normalizedSize;

    // Parallel mean
    float localSum = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x)
        localSum += input[base + i];
    smem[tid] = localSum;
    BLOCK_REDUCE(smem, tid);
    float mean = smem[0] / (float)normalizedSize;
    __syncthreads();

    // Parallel variance
    float localVar = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        float diff = input[base + i] - mean;
        localVar += diff * diff;
    }
    smem[tid] = localVar;
    BLOCK_REDUCE(smem, tid);
    float invVar = rsqrtf(smem[0] / (float)normalizedSize + epsilon);

    if (tid == 0) {
        saveMean[b] = mean;
        saveInvVar[b] = invVar;
    }

    // Parallel normalize
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        int idx = base + i;
        float normalized = (input[idx] - mean) * invVar;
        output[idx] = gamma[i] * normalized + beta[i];
    }
}

// Layer Normalization backward pass
// 1 block per batch element, 256 threads parallel reduce
extern ""C"" __global__ void layernorm_backward(
    const float* gradOutput, const float* input,
    const float* gamma, const float* saveMean, const float* saveInvVar,
    float* gradInput, float* gradGamma, float* gradBeta,
    int batchSize, int normalizedSize, float epsilon)
{
    extern __shared__ float smem[];
    float* smem2 = smem + blockDim.x;
    int b = blockIdx.x;
    if (b >= batchSize) return;
    int tid = threadIdx.x;
    int base = b * normalizedSize;
    float mean = saveMean[b];
    float invVar = saveInvVar[b];

    // Parallel reduction for sumDy and sumDyXmu
    float locSumDy = 0.0f, locSumDyXmu = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        int idx = base + i;
        float dy = gradOutput[idx] * gamma[i];
        locSumDy += dy;
        locSumDyXmu += dy * (input[idx] - mean);
    }
    smem[tid] = locSumDy;
    smem2[tid] = locSumDyXmu;
    BLOCK_REDUCE(smem, tid);
    BLOCK_REDUCE(smem2, tid);
    float sumDy = smem[0];
    float sumDyXmu = smem2[0];

    // Parallel gradInput
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        int idx = base + i;
        float xmu = input[idx] - mean;
        float dxhat = gradOutput[idx] * gamma[i];
        gradInput[idx] = invVar * (dxhat - (sumDy + xmu * invVar * invVar * sumDyXmu) / (float)normalizedSize);
    }
}

// Layer Normalization gradient accumulation for gamma and beta
// 1 thread per feature, serial over batch (batch is typically small)
extern ""C"" __global__ void layernorm_grad_params(
    const float* gradOutput, const float* input,
    const float* saveMean, const float* saveInvVar,
    float* gradGamma, float* gradBeta,
    int batchSize, int normalizedSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= normalizedSize) return;

    float dGamma = 0.0f;
    float dBeta = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        int idx = b * normalizedSize + i;
        float mean = saveMean[b];
        float invVar = saveInvVar[b];
        float normalized = (input[idx] - mean) * invVar;
        dGamma += gradOutput[idx] * normalized;
        dBeta += gradOutput[idx];
    }
    gradGamma[i] = dGamma;
    gradBeta[i] = dBeta;
}

// Group Normalization forward pass
// 1 block per (batch, group) pair, 256 threads parallel reduce
extern ""C"" __global__ void groupnorm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    float* saveMean, float* saveInvVar,
    int batch, int numGroups, int channels, int spatialSize, float epsilon)
{
    extern __shared__ float smem[];
    int blockId = blockIdx.x;
    int g = blockId % numGroups;
    int b = blockId / numGroups;
    if (b >= batch) return;
    int tid = threadIdx.x;

    int channelsPerGroup = channels / numGroups;
    int groupSize = channelsPerGroup * spatialSize;

    // Parallel mean
    float localSum = 0.0f;
    for (int i = tid; i < groupSize; i += blockDim.x) {
        int c = g * channelsPerGroup + i / spatialSize;
        int s = i % spatialSize;
        localSum += input[(b * channels + c) * spatialSize + s];
    }
    smem[tid] = localSum;
    BLOCK_REDUCE(smem, tid);
    float mean = smem[0] / (float)groupSize;
    __syncthreads();

    // Parallel variance
    float localVar = 0.0f;
    for (int i = tid; i < groupSize; i += blockDim.x) {
        int c = g * channelsPerGroup + i / spatialSize;
        int s = i % spatialSize;
        float diff = input[(b * channels + c) * spatialSize + s] - mean;
        localVar += diff * diff;
    }
    smem[tid] = localVar;
    BLOCK_REDUCE(smem, tid);
    float invVar = rsqrtf(smem[0] / (float)groupSize + epsilon);

    if (tid == 0) {
        int saveIdx = b * numGroups + g;
        saveMean[saveIdx] = mean;
        saveInvVar[saveIdx] = invVar;
    }

    // Parallel normalize
    for (int i = tid; i < groupSize; i += blockDim.x) {
        int c = g * channelsPerGroup + i / spatialSize;
        int s = i % spatialSize;
        int inputIdx = (b * channels + c) * spatialSize + s;
        float normalized = (input[inputIdx] - mean) * invVar;
        output[inputIdx] = gamma[c] * normalized + beta[c];
    }
}

// Instance Normalization forward pass
// 1 block per (batch, channel) pair, 256 threads parallel reduce across spatial
extern ""C"" __global__ void instancenorm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    float* saveMean, float* saveInvVar,
    int batch, int channels, int spatialSize, float epsilon)
{
    extern __shared__ float smem[];
    int blockId = blockIdx.x;
    int c = blockId % channels;
    int b = blockId / channels;
    if (b >= batch) return;
    int tid = threadIdx.x;
    int base = (b * channels + c) * spatialSize;

    // Parallel mean
    float localSum = 0.0f;
    for (int s = tid; s < spatialSize; s += blockDim.x)
        localSum += input[base + s];
    smem[tid] = localSum;
    BLOCK_REDUCE(smem, tid);
    float mean = smem[0] / (float)spatialSize;
    __syncthreads();

    // Parallel variance
    float localVar = 0.0f;
    for (int s = tid; s < spatialSize; s += blockDim.x) {
        float diff = input[base + s] - mean;
        localVar += diff * diff;
    }
    smem[tid] = localVar;
    BLOCK_REDUCE(smem, tid);
    float invVar = rsqrtf(smem[0] / (float)spatialSize + epsilon);

    if (tid == 0) {
        int saveIdx = b * channels + c;
        saveMean[saveIdx] = mean;
        saveInvVar[saveIdx] = invVar;
    }

    // Parallel normalize
    float g = gamma[c];
    float bt = beta[c];
    for (int s = tid; s < spatialSize; s += blockDim.x) {
        int inputIdx = base + s;
        float normalized = (input[inputIdx] - mean) * invVar;
        output[inputIdx] = g * normalized + bt;
    }
}

// RMS Normalization forward pass
// 1 block per batch element, 256 threads parallel reduce
extern ""C"" __global__ void rmsnorm_forward(
    const float* input, float* output,
    const float* gamma, float* saveRms,
    int batchSize, int normalizedSize, float epsilon)
{
    extern __shared__ float smem[];
    int b = blockIdx.x;
    if (b >= batchSize) return;
    int tid = threadIdx.x;
    int base = b * normalizedSize;

    // Parallel mean-squared reduction
    float localMeanSq = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        float x = input[base + i];
        localMeanSq += x * x;
    }
    smem[tid] = localMeanSq;
    BLOCK_REDUCE(smem, tid);
    float rms = sqrtf(smem[0] / (float)normalizedSize + epsilon);
    float invRms = 1.0f / rms;

    if (tid == 0)
        saveRms[b] = rms;

    // Parallel scale
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        int idx = base + i;
        output[idx] = input[idx] * invRms * gamma[i];
    }
}

// RMS Normalization backward pass
// 1 block per batch element, 256 threads parallel reduce
extern ""C"" __global__ void rmsnorm_backward(
    const float* gradOutput, const float* input,
    const float* gamma, const float* saveRms,
    float* gradInput, float* gradGamma,
    int batchSize, int normalizedSize, float epsilon)
{
    extern __shared__ float smem[];
    int b = blockIdx.x;
    if (b >= batchSize) return;
    int tid = threadIdx.x;
    int base = b * normalizedSize;

    float rms = saveRms[b];
    float invRms = 1.0f / rms;
    float invRms3 = invRms * invRms * invRms;

    // Parallel reduction for sumGradGammaX
    float localSum = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        int idx = base + i;
        localSum += gradOutput[idx] * gamma[i] * input[idx];
    }
    smem[tid] = localSum;
    BLOCK_REDUCE(smem, tid);
    float sumGradGammaX = smem[0];

    // Parallel gradInput
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        int idx = base + i;
        float x = input[idx];
        float dy = gradOutput[idx];
        float g = gamma[i];
        gradInput[idx] = g * dy * invRms - x * sumGradGammaX * invRms3 / (float)normalizedSize;
    }
}

// RMS Normalization gradient accumulation for gamma
// 1 thread per feature, serial over batch
extern ""C"" __global__ void rmsnorm_grad_gamma(
    const float* gradOutput, const float* input,
    const float* saveRms, float* gradGamma,
    int batchSize, int normalizedSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= normalizedSize) return;

    float dGamma = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        int idx = b * normalizedSize + i;
        float rms = saveRms[b];
        float invRms = 1.0f / rms;
        dGamma += gradOutput[idx] * input[idx] * invRms;
    }
    gradGamma[i] = dGamma;
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "batchnorm_forward",
                "batchnorm_backward",
                "layernorm_forward",
                "layernorm_backward",
                "layernorm_grad_params",
                "groupnorm_forward",
                "instancenorm_forward",
                "rmsnorm_forward",
                "rmsnorm_backward",
                "rmsnorm_grad_gamma"
            };
        }
    }
}
