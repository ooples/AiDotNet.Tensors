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
extern ""C"" __global__ __launch_bounds__(256) void batchnorm_forward(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ runningMean, float* __restrict__ runningVar,
    float* __restrict__ saveMean, float* __restrict__ saveInvVar,
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
            // Running variance uses the UNBIASED (Bessel-corrected) estimate —
            // paper-faithful (Ioffe & Szegedy 2015 §3.1) and identical to cuDNN's
            // ForwardTraining. The current-batch normalization above still uses
            // the biased var. Guard batchSpatial==1 (unbiased is undefined).
            float runVarUnbiased = batchSpatial > 1
                ? var * (float)batchSpatial / (float)(batchSpatial - 1)
                : var;
            runningVar[c] = (1.0f - momentum) * runningVar[c] + momentum * runVarUnbiased;
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
extern ""C"" __global__ __launch_bounds__(256) void batchnorm_backward(
    const float* __restrict__ gradOutput, const float* __restrict__ input,
    const float* __restrict__ gamma, const float* __restrict__ saveMean, const float* __restrict__ saveInvVar,
    float* __restrict__ gradInput, float* __restrict__ gradGamma, float* __restrict__ gradBeta,
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
extern ""C"" __global__ __launch_bounds__(256) void layernorm_forward(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ saveMean, float* __restrict__ saveInvVar,
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
extern ""C"" __global__ __launch_bounds__(256) void layernorm_backward(
    const float* __restrict__ gradOutput, const float* __restrict__ input,
    const float* __restrict__ gamma, const float* __restrict__ saveMean, const float* __restrict__ saveInvVar,
    float* __restrict__ gradInput, float* __restrict__ gradGamma, float* __restrict__ gradBeta,
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
extern ""C"" __global__ __launch_bounds__(256) void layernorm_grad_params(
    const float* __restrict__ gradOutput, const float* __restrict__ input,
    const float* __restrict__ saveMean, const float* __restrict__ saveInvVar,
    float* __restrict__ gradGamma, float* __restrict__ gradBeta,
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
extern ""C"" __global__ __launch_bounds__(256) void groupnorm_forward(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ saveMean, float* __restrict__ saveInvVar,
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
extern ""C"" __global__ __launch_bounds__(256) void instancenorm_forward(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ saveMean, float* __restrict__ saveInvVar,
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
extern ""C"" __global__ __launch_bounds__(256) void rmsnorm_forward(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ gamma, float* __restrict__ saveRms,
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
extern ""C"" __global__ __launch_bounds__(256) void rmsnorm_backward(
    const float* __restrict__ gradOutput, const float* __restrict__ input,
    const float* __restrict__ gamma, const float* __restrict__ saveRms,
    float* __restrict__ gradInput, float* __restrict__ gradGamma,
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
extern ""C"" __global__ __launch_bounds__(256) void rmsnorm_grad_gamma(
    const float* __restrict__ gradOutput, const float* __restrict__ input,
    const float* __restrict__ saveRms, float* __restrict__ gradGamma,
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

// LayerNorm's inverse std from the saved variance, invStd = 1 / sqrt(var + eps): the same three operations as the
// add_scalar, sqrt_vector and reciprocal_vector kernels (same compile flags), in one launch instead of three.
extern ""C"" __global__ __launch_bounds__(256) void layernorm_inverse_std(
    const float* __restrict__ variance, float* __restrict__ invStd, int rows, float epsilon)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float s = sqrtf(variance[i] + epsilon);
    invStd[i] = 1.0f / s;
}

// LayerNorm's affine gradients over [rows, cols], pass 1 of a deterministic two-pass column reduction. Block (32, 8)
// covers 32 columns of one chunk of rows: each warp reads 32 consecutive floats of a row (coalesced), each of the 8 row
// lanes sums its rows in order, and lane 0 adds the 8 lanes in a fixed order into partials[chunk][col] (gamma) and
// partials[chunks + chunk][col] (beta). The thread-per-column kernel ran on cols / 32 blocks -- 2 for a 64-wide model.
extern ""C"" __global__ __launch_bounds__(256) void layernorm_grad_params_partial(
    const float* __restrict__ gradOutput, const float* __restrict__ input,
    const float* __restrict__ saveMean, const float* __restrict__ saveInvVar,
    float* __restrict__ partials, int rows, int cols, int rowsPerChunk)
{
    __shared__ float sg[8][33];
    __shared__ float sb[8][33];
    int col = blockIdx.x * 32 + threadIdx.x;
    int chunk = blockIdx.y;
    int r0 = chunk * rowsPerChunk;
    int r1 = min(rows, r0 + rowsPerChunk);
    float g = 0.0f, b = 0.0f;
    if (col < cols)
    {
        for (int r = r0 + threadIdx.y; r < r1; r += 8)
        {
            int idx = r * cols + col;
            float dy = gradOutput[idx];
            float normalized = (input[idx] - saveMean[r]) * saveInvVar[r];
            g += dy * normalized;
            b += dy;
        }
    }
    sg[threadIdx.y][threadIdx.x] = g;
    sb[threadIdx.y][threadIdx.x] = b;
    __syncthreads();
    if (threadIdx.y == 0 && col < cols)
    {
        float tg = 0.0f, tb = 0.0f;
        for (int y = 0; y < 8; y++) { tg += sg[y][threadIdx.x]; tb += sb[y][threadIdx.x]; }
        partials[chunk * cols + col] = tg;
        partials[(gridDim.y + chunk) * cols + col] = tb;
    }
}

// Pass 2: each column's chunk partials summed in chunk order, then written to (or added to, for a gradient that
// already holds this step's contributions) gradGamma / gradBeta.
extern ""C"" __global__ __launch_bounds__(256) void layernorm_grad_params_finish(
    const float* __restrict__ partials, float* __restrict__ gradGamma, float* __restrict__ gradBeta,
    int cols, int chunks, int addGamma, int addBeta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float g = 0.0f, b = 0.0f;
    for (int c = 0; c < chunks; c++)
    {
        g += partials[c * cols + col];
        b += partials[(chunks + c) * cols + col];
    }
    gradGamma[col] = addGamma ? gradGamma[col] + g : g;
    gradBeta[col] = addBeta ? gradBeta[col] + b : b;
}

// Column sums of a row-major [rows, cols] matrix (a linear layer's bias gradient), pass 1: the same deterministic
// (column tile x row chunk) blocking as layernorm_grad_params_partial, writing partials[chunk][col]. cuBLAS ran this
// as a ones-GEMV (gemv2N) at ~10 us for 2048 x 64..256.
extern ""C"" __global__ __launch_bounds__(256) void column_sum_partial(
    const float* __restrict__ input, float* __restrict__ partials, int rows, int cols, int rowsPerChunk)
{
    __shared__ float s[8][33];
    int col = blockIdx.x * 32 + threadIdx.x;
    int chunk = blockIdx.y;
    int r0 = chunk * rowsPerChunk;
    int r1 = min(rows, r0 + rowsPerChunk);
    float acc = 0.0f;
    if (col < cols)
        for (int r = r0 + threadIdx.y; r < r1; r += 8)
            acc += input[r * cols + col];
    s[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.y == 0 && col < cols)
    {
        float t = 0.0f;
        for (int y = 0; y < 8; y++) t += s[y][threadIdx.x];
        partials[chunk * cols + col] = t;
    }
}

// Pass 2: each column's partials summed in chunk order, written to (or added to) output.
extern ""C"" __global__ __launch_bounds__(256) void column_sum_finish(
    const float* __restrict__ partials, float* __restrict__ output, int cols, int chunks, int add)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float t = 0.0f;
    for (int c = 0; c < chunks; c++) t += partials[c * cols + col];
    output[col] = add ? output[col] + t : t;
}

// Soft-target softmax cross-entropy (IEngine.TensorSoftmaxCrossEntropy), forward pass 1: one warp per row of [rows, cols]
// writes rowStats[0..rows) = log-sum-exp, [rows..2rows) = sum of targets, [2rows..3rows) = sum target * logSoftmax,
// [3rows..4rows) = 1 when the target row has mass. Lane-strided sums then a fixed xor-shuffle tree: deterministic.
extern ""C"" __global__ __launch_bounds__(256) void softmax_xent_rows(
    const float* __restrict__ logits, const float* __restrict__ target, float* __restrict__ rowStats, int rows, int cols)
{
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* z = logits + (long long)row * cols;
    const float* t = target + (long long)row * cols;
    float m = -INFINITY;
    for (int c = lane; c < cols; c += 32) m = fmaxf(m, z[c]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    float s = 0.0f;
    for (int c = lane; c < cols; c += 32) s += expf(z[c] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    float lse = m + logf(s);
    float sum = 0.0f, dot = 0.0f, mass = 0.0f;
    for (int c = lane; c < cols; c += 32)
    {
        float tv = t[c];
        sum += tv;
        dot += tv * (z[c] - lse);
        mass += fabsf(tv);
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
        dot += __shfl_xor_sync(0xffffffff, dot, o);
        mass += __shfl_xor_sync(0xffffffff, mass, o);
    }
    if (lane == 0)
    {
        rowStats[row] = lse;
        rowStats[rows + row] = sum;
        rowStats[2 * rows + row] = dot;
        rowStats[3 * rows + row] = mass > 0.0f ? 1.0f : 0.0f;
    }
}

// Pass 2, one block of 256: loss[0] = -(sum of row dots) / count and count[0] = max(1, supervised rows), summed in a
// fixed order (strided per thread, then a shared-memory tree).
extern ""C"" __global__ __launch_bounds__(256) void softmax_xent_total(
    const float* __restrict__ rowStats, float* __restrict__ loss, float* __restrict__ count, int rows)
{
    __shared__ float sd[256];
    __shared__ float ss[256];
    int tid = threadIdx.x;
    float d = 0.0f, s = 0.0f;
    for (int r = tid; r < rows; r += 256) { d += rowStats[2 * rows + r]; s += rowStats[3 * rows + r]; }
    sd[tid] = d; ss[tid] = s;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1)
    {
        if (tid < w) { sd[tid] += sd[tid + w]; ss[tid] += ss[tid + w]; }
        __syncthreads();
    }
    if (tid == 0)
    {
        float cnt = fmaxf(ss[0], 1.0f);
        count[0] = cnt;
        loss[0] = -(sd[0] / cnt);
    }
}

// Backward: dLogits = gradOut / count * (softmax * sum of the row's targets - target), written or added.
extern ""C"" __global__ __launch_bounds__(256) void softmax_xent_backward(
    const float* __restrict__ logits, const float* __restrict__ target, const float* __restrict__ rowStats,
    const float* __restrict__ count, const float* __restrict__ gradOut, float* __restrict__ gradLogits,
    int rows, int cols, int add)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = (long long)rows * cols;
    if (i >= n) return;
    int row = (int)(i / cols);
    float g = gradOut[0] / count[0];
    float v = g * (expf(logits[i] - rowStats[row]) * rowStats[rows + row] - target[i]);
    gradLogits[i] = add ? gradLogits[i] + v : v;
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
                "rmsnorm_grad_gamma",
                "layernorm_inverse_std",
                "layernorm_grad_params_partial",
                "layernorm_grad_params_finish",
                "column_sum_partial",
                "column_sum_finish",
                "softmax_xent_rows",
                "softmax_xent_total",
                "softmax_xent_backward"
            };
        }
    }
}
