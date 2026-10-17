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

// ---- Register-resident batch norm (training) for channel slices of at most 2048 float4s (8192 elements) ----
// One block per channel holds the channel's whole slice in registers, so the forward reads its input once (the generic
// kernel reads it three times) and the backward reads x and dY once each (twice each). Statistics are kept as the
// biased variance the engine reports: the forward stores it and the backward derives 1/sqrt(var + eps) itself, which
// removes the separate inverse-std / variance conversion launches. spatial % 4 == 0 (float4 rows).
__device__ __forceinline__ float bn_block_sum(float v, float* red)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();   // red may still be read by the previous reduction
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : 0.0f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

// RES: a residual is added before the ReLU (BatchNorm -> add -> ReLU, a residual block's tail).
template <int K4, bool RELU, bool RES>
__device__ __forceinline__ void bn_fwd_reg(const float* __restrict__ input, const float* __restrict__ residual,
    float* __restrict__ output,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ saveMean, float* __restrict__ saveVar,
    int batch, int channels, int spatial, float epsilon)
{
    __shared__ float red[32];
    const int c = blockIdx.x, tid = threadIdx.x, s4 = spatial >> 2, n4 = batch * s4;
    float4 v[K4];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < K4; k++)
    {
        int q = tid + k * blockDim.x;
        if (q < n4)
        {
            int b = q / s4;
            v[k] = reinterpret_cast<const float4*>(input + ((long long)b * channels + c) * spatial)[q - b * s4];
            sum += (v[k].x + v[k].y) + (v[k].z + v[k].w);
        }
        else v[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float n = (float)(n4 * 4);
    const float mean = bn_block_sum(sum, red) / n;
    float sq = 0.0f;
    #pragma unroll
    for (int k = 0; k < K4; k++)
        if (tid + k * blockDim.x < n4)
        {
            float a = v[k].x - mean, b2 = v[k].y - mean, c2 = v[k].z - mean, d = v[k].w - mean;
            sq += (a * a + b2 * b2) + (c2 * c2 + d * d);
        }
    const float var = bn_block_sum(sq, red) / n;
    const float inv = rsqrtf(var + epsilon);
    if (tid == 0) { saveMean[c] = mean; saveVar[c] = var; }
    const float g = gamma[c], bt = beta[c];
    #pragma unroll
    for (int k = 0; k < K4; k++)
    {
        int q = tid + k * blockDim.x;
        if (q < n4)
        {
            int b = q / s4;
            float4 y;
            y.x = g * ((v[k].x - mean) * inv) + bt;
            y.y = g * ((v[k].y - mean) * inv) + bt;
            y.z = g * ((v[k].z - mean) * inv) + bt;
            y.w = g * ((v[k].w - mean) * inv) + bt;
            if (RES)
            {
                float4 r = reinterpret_cast<const float4*>(residual + ((long long)b * channels + c) * spatial)[q - b * s4];
                y.x += r.x; y.y += r.y; y.z += r.z; y.w += r.w;
            }
            if (RELU) { y.x = fmaxf(y.x, 0.0f); y.y = fmaxf(y.y, 0.0f); y.z = fmaxf(y.z, 0.0f); y.w = fmaxf(y.w, 0.0f); }
            reinterpret_cast<float4*>(output + ((long long)b * channels + c) * spatial)[q - b * s4] = y;
        }
    }
}

// RELU: the forward was BatchNorm followed by ReLU and reluOutput is that ReLU's output; dY is masked where it is not
// positive (ReLU(z) > 0 exactly when z > 0), so the ReLU backward needs no pass of its own.
// RES: the forward added a residual before the ReLU; the residual's gradient is the masked dY, written to gradResidual.
template <int K4, bool RELU, bool RES>
__device__ __forceinline__ void bn_bwd_reg(const float* __restrict__ gradOutput, const float* __restrict__ reluOutput,
    float* __restrict__ gradResidual, const float* __restrict__ input,
    const float* __restrict__ gamma, const float* __restrict__ saveMean, const float* __restrict__ saveVar,
    float* __restrict__ gradInput, float* __restrict__ gradGamma, float* __restrict__ gradBeta,
    int batch, int channels, int spatial, float epsilon)
{
    __shared__ float red[32];
    const int c = blockIdx.x, tid = threadIdx.x, s4 = spatial >> 2, n4 = batch * s4;
    const float mean = saveMean[c], inv = rsqrtf(saveVar[c] + epsilon);
    float4 x[K4], dy[K4];
    float sdy = 0.0f, sdyx = 0.0f;
    #pragma unroll
    for (int k = 0; k < K4; k++)
    {
        int q = tid + k * blockDim.x;
        if (q < n4)
        {
            int b = q / s4;
            long long row = ((long long)b * channels + c) * spatial;
            x[k] = reinterpret_cast<const float4*>(input + row)[q - b * s4];
            dy[k] = reinterpret_cast<const float4*>(gradOutput + row)[q - b * s4];
            if (RELU)
            {
                float4 o = reinterpret_cast<const float4*>(reluOutput + row)[q - b * s4];
                dy[k].x = o.x > 0.0f ? dy[k].x : 0.0f; dy[k].y = o.y > 0.0f ? dy[k].y : 0.0f;
                dy[k].z = o.z > 0.0f ? dy[k].z : 0.0f; dy[k].w = o.w > 0.0f ? dy[k].w : 0.0f;
            }
            if (RES) reinterpret_cast<float4*>(gradResidual + row)[q - b * s4] = dy[k];
            sdy += (dy[k].x + dy[k].y) + (dy[k].z + dy[k].w);
            sdyx += (dy[k].x * ((x[k].x - mean) * inv) + dy[k].y * ((x[k].y - mean) * inv))
                  + (dy[k].z * ((x[k].z - mean) * inv) + dy[k].w * ((x[k].w - mean) * inv));
        }
    }
    const float sumDy = bn_block_sum(sdy, red);
    const float sumDyXhat = bn_block_sum(sdyx, red);
    if (tid == 0) { gradGamma[c] = sumDyXhat; gradBeta[c] = sumDy; }
    const float g = gamma[c], invN = 1.0f / (float)(n4 * 4);
    const float sumDxhat = g * sumDy, sumDxhatXhat = g * sumDyXhat;
    #pragma unroll
    for (int k = 0; k < K4; k++)
    {
        int q = tid + k * blockDim.x;
        if (q < n4)
        {
            int b = q / s4;
            float4 r;
            r.x = inv * (dy[k].x * g - invN * (sumDxhat + ((x[k].x - mean) * inv) * sumDxhatXhat));
            r.y = inv * (dy[k].y * g - invN * (sumDxhat + ((x[k].y - mean) * inv) * sumDxhatXhat));
            r.z = inv * (dy[k].z * g - invN * (sumDxhat + ((x[k].z - mean) * inv) * sumDxhatXhat));
            r.w = inv * (dy[k].w * g - invN * (sumDxhat + ((x[k].w - mean) * inv) * sumDxhatXhat));
            reinterpret_cast<float4*>(gradInput + ((long long)b * channels + c) * spatial)[q - b * s4] = r;
        }
    }
}

#define BN_REG_ENTRY(K4, RELU, RES, SUFFIX) \
extern ""C"" __global__ __launch_bounds__(256) void batchnorm_fwd##SUFFIX##_reg##K4(const float* input, \
    const float* residual, float* output, const float* gamma, const float* beta, float* saveMean, float* saveVar, \
    int batch, int channels, int spatial, float epsilon) \
{ bn_fwd_reg<K4, RELU, RES>(input, residual, output, gamma, beta, saveMean, saveVar, batch, channels, spatial, \
    epsilon); } \
extern ""C"" __global__ __launch_bounds__(256) void batchnorm_bwd##SUFFIX##_reg##K4(const float* gradOutput, \
    const float* reluOutput, float* gradResidual, const float* input, const float* gamma, const float* saveMean, \
    const float* saveVar, float* gradInput, float* gradGamma, float* gradBeta, int batch, int channels, int spatial, \
    float epsilon) \
{ bn_bwd_reg<K4, RELU, RES>(gradOutput, reluOutput, gradResidual, input, gamma, saveMean, saveVar, gradInput, \
    gradGamma, gradBeta, batch, channels, spatial, epsilon); }
BN_REG_ENTRY(1, false, false, )
BN_REG_ENTRY(2, false, false, )
BN_REG_ENTRY(4, false, false, )
BN_REG_ENTRY(8, false, false, )
BN_REG_ENTRY(1, true, false, _relu)
BN_REG_ENTRY(2, true, false, _relu)
BN_REG_ENTRY(4, true, false, _relu)
BN_REG_ENTRY(8, true, false, _relu)
BN_REG_ENTRY(1, true, true, _addrelu)
BN_REG_ENTRY(2, true, true, _addrelu)
BN_REG_ENTRY(4, true, true, _addrelu)
BN_REG_ENTRY(8, true, true, _addrelu)

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

// The inverse: LayerNorm's forward reports true variance, var = 1 / (invStd * invStd) - eps, in place -- the
// multiply_vectors, reciprocal_vector and sub_scalar kernels' arithmetic (same compile flags) in one launch.
extern ""C"" __global__ __launch_bounds__(256) void layernorm_variance_from_inverse_std(
    float* __restrict__ values, int rows, float epsilon)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float inv = values[i];
    float squared = inv * inv;
    float r = 1.0f / squared;
    values[i] = r - epsilon;
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

// A sum over a leading and a trailing block around one kept axis (a convolution's bias gradient): input viewed as
// [L, C, T]. Pass 1: block (c, chunk) sums channel c over rows [chunk * rowsPerChunk, +rowsPerChunk) -- each row a run
// of T contiguous floats -- by a fixed shared-memory tree, into partials[chunk * C + c].
extern ""C"" __global__ __launch_bounds__(256) void sum_around_kept_partial(
    const float* __restrict__ x, float* __restrict__ partials, int L, int C, int T, int rowsPerChunk)
{
    __shared__ float s[256];
    int c = blockIdx.x, chunk = blockIdx.y;
    int l0 = chunk * rowsPerChunk, l1 = min(L, l0 + rowsPerChunk);
    long long span = (long long)max(0, l1 - l0) * T;
    float acc = 0.0f;
    for (long long e = threadIdx.x; e < span; e += blockDim.x)
    {
        int l = l0 + (int)(e / T), t = (int)(e % T);
        acc += x[((long long)l * C + c) * T + t];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1)
    {
        if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[(long long)chunk * C + c] = s[0];
}

// Pass 2: each channel's partials summed in chunk order, times scale.
extern ""C"" __global__ __launch_bounds__(256) void sum_around_kept_finish(
    const float* __restrict__ partials, float* __restrict__ output, int C, int chunks, float scale)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float t = 0.0f;
    for (int k = 0; k < chunks; k++) t += partials[(long long)k * C + c];
    output[c] = scale * t;
}

// LayerNorm's whole backward over [rows, cols <= 32 * CPL] from the saved mean and VARIANCE, in one pass plus the
// layernorm_grad_params_finish reduce: block = 8 warps over one chunk of rows [chunk * rowsPerChunk, +rowsPerChunk),
// each warp taking every 8th row. Per row the warp holds the row in registers (CPL floats per lane), forms
// invStd = 1 / sqrt(var + eps) as layernorm_inverse_std does, writes (or adds) the input gradient
// dx = invStd * (g*dy - mean_c(g*dy) - xhat * mean_c(g*dy*xhat)) as layernorm_input_grad_add does, and accumulates
// gamma's dy*xhat and beta's dy per column. The 8 warps' column sums are added in warp order into
// partials[chunk][col] (gamma) and partials[chunks + chunk][col] (beta). All orders fixed: deterministic. Replaces
// the inverse-std, input-gradient and affine-partial launches (three dependent kernels) with one.
template <int CPL>
__device__ __forceinline__ void layernorm_backward_rows_body(
    const float* __restrict__ dy, const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ mean, const float* __restrict__ variance, float epsilon,
    float* __restrict__ dx, float* __restrict__ partials, int rows, int cols, int rowsPerChunk, int addInput)
{
    __shared__ float sg[8][CPL * 32];
    __shared__ float sb[8][CPL * 32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int chunk = blockIdx.x;
    int r0 = chunk * rowsPerChunk, r1 = min(rows, r0 + rowsPerChunk);
    float gam[CPL], ag[CPL], ab[CPL];
    #pragma unroll
    for (int j = 0; j < CPL; j++)
    {
        int c = lane + 32 * j;
        gam[j] = c < cols ? gamma[c] : 0.0f;
        ag[j] = 0.0f; ab[j] = 0.0f;
    }
    float invCols = 1.0f / cols;
    for (int r = r0 + warp; r < r1; r += 8)
    {
        const float* dyr = dy + (long long)r * cols;
        const float* xr = x + (long long)r * cols;
        float mu = mean[r];
        float is = 1.0f / sqrtf(variance[r] + epsilon);
        float d[CPL], xh[CPL];
        float s1 = 0.0f, s2 = 0.0f;
        #pragma unroll
        for (int j = 0; j < CPL; j++)
        {
            int c = lane + 32 * j;
            d[j] = 0.0f; xh[j] = 0.0f;
            if (c < cols)
            {
                d[j] = dyr[c];
                xh[j] = (xr[c] - mu) * is;
                float gdy = gam[j] * d[j];
                s1 += gdy;
                s2 += gdy * xh[j];
                ag[j] += d[j] * xh[j];
                ab[j] += d[j];
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            s1 += __shfl_xor_sync(0xffffffff, s1, o);
            s2 += __shfl_xor_sync(0xffffffff, s2, o);
        }
        s1 *= invCols;
        s2 *= invCols;
        float* dxr = dx + (long long)r * cols;
        #pragma unroll
        for (int j = 0; j < CPL; j++)
        {
            int c = lane + 32 * j;
            if (c < cols)
            {
                float v = is * (gam[j] * d[j] - s1 - xh[j] * s2);
                dxr[c] = addInput ? dxr[c] + v : v;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < CPL; j++)
    {
        sg[warp][lane + 32 * j] = ag[j];
        sb[warp][lane + 32 * j] = ab[j];
    }
    __syncthreads();
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
    {
        float tg = 0.0f, tb = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) { tg += sg[w][c]; tb += sb[w][c]; }
        partials[(long long)chunk * cols + c] = tg;
        partials[(long long)(gridDim.x + chunk) * cols + c] = tb;
    }
}

#define LN_ROWS_KERNEL(NAME, CPL) \
extern ""C"" __global__ __launch_bounds__(256) void NAME( \
    const float* __restrict__ dy, const float* __restrict__ x, const float* __restrict__ gamma, \
    const float* __restrict__ mean, const float* __restrict__ variance, float epsilon, \
    float* __restrict__ dx, float* __restrict__ partials, int rows, int cols, int rowsPerChunk, int addInput) \
{ layernorm_backward_rows_body<CPL>(dy, x, gamma, mean, variance, epsilon, dx, partials, rows, cols, rowsPerChunk, addInput); }
LN_ROWS_KERNEL(layernorm_backward_rows2, 2)
LN_ROWS_KERNEL(layernorm_backward_rows4, 4)
LN_ROWS_KERNEL(layernorm_backward_rows8, 8)
LN_ROWS_KERNEL(layernorm_backward_rows16, 16)

// LayerNorm's input gradient ADDED into gradInput, one warp per row of [rows, cols]:
// dx += invStd * (g*dy - mean_c(g*dy) - xhat * mean_c(g*dy*xhat)), xhat = (x - mean) * invStd. Lane-strided sums
// reduced by a fixed xor-shuffle tree: deterministic.
extern ""C"" __global__ __launch_bounds__(256) void layernorm_input_grad_add(
    const float* __restrict__ dy, const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ mean, const float* __restrict__ invStd, float* __restrict__ dx, int rows, int cols)
{
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* dyr = dy + (long long)row * cols;
    const float* xr = x + (long long)row * cols;
    float* dxr = dx + (long long)row * cols;
    float mu = mean[row], is = invStd[row];
    float s1 = 0.0f, s2 = 0.0f;
    for (int c = lane; c < cols; c += 32)
    {
        float gdy = gamma[c] * dyr[c];
        s1 += gdy;
        s2 += gdy * ((xr[c] - mu) * is);
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        s1 += __shfl_xor_sync(0xffffffff, s1, o);
        s2 += __shfl_xor_sync(0xffffffff, s2, o);
    }
    float invCols = 1.0f / cols;
    s1 *= invCols;
    s2 *= invCols;
    for (int c = lane; c < cols; c += 32)
    {
        float gdy = gamma[c] * dyr[c];
        float xh = (xr[c] - mu) * is;
        dxr[c] += is * (gdy - s1 - xh * s2);
    }
}

// The gradient of a sum or mean over one middle block of axes: output[o, r, i] = scale * g[o, i] for an input viewed as
// [outer, reduced, inner], written (or added) in one pass -- a float4 per thread when inner is a multiple of four.
extern ""C"" __global__ __launch_bounds__(256) void broadcast_reduced_middle(
    const float* __restrict__ g, float* __restrict__ output, int outer, int reduced, int inner, float scale, int add)
{
    long long total = (long long)outer * reduced * inner;
    long long stride = (long long)gridDim.x * blockDim.x;
    if ((inner & 3) == 0)
    {
        long long quads = total >> 2;
        for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += stride)
        {
            long long e = q << 2;
            int i = (int)(e % inner);
            long long o = e / ((long long)reduced * inner);
            float4 v = *reinterpret_cast<const float4*>(g + o * inner + i);
            float4 r = make_float4(scale * v.x, scale * v.y, scale * v.z, scale * v.w);
            float4* out = reinterpret_cast<float4*>(output + e);
            if (add) { float4 c = *out; r.x += c.x; r.y += c.y; r.z += c.z; r.w += c.w; }
            *out = r;
        }
        return;
    }
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride)
    {
        int i = (int)(e % inner);
        long long o = e / ((long long)reduced * inner);
        float r = scale * g[o * inner + i];
        output[e] = add ? output[e] + r : r;
    }
}

// Column sums of a short matrix (rows <= ColumnSumSmallMaxRows) in one launch: block (32 x 32) sums a 32-column
// tile, each of its 32 row lanes striding over the rows, then lane 0 adds the 32 lane sums in lane order
// (deterministic). Replaces the partial/finish pair, two dependent launches, where the rows are too few to need them.
extern ""C"" __global__ __launch_bounds__(1024) void column_sum_small(
    const float* __restrict__ input, float* __restrict__ output, int rows, int cols, int add)
{
    __shared__ float s[32][33];
    int col = blockIdx.x * 32 + threadIdx.x;
    float acc = 0.0f;
    if (col < cols)
    {
        #pragma unroll 4
        for (int r = threadIdx.y; r < rows; r += 32)
            acc += input[(long long)r * cols + col];
    }
    s[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.y == 0 && col < cols)
    {
        float t = 0.0f;
        for (int y = 0; y < 32; y++) t += s[y][threadIdx.x];
        output[col] = add ? output[col] + t : t;
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
                "batchnorm_fwd_reg1", "batchnorm_fwd_reg2", "batchnorm_fwd_reg4", "batchnorm_fwd_reg8",
                "batchnorm_bwd_reg1", "batchnorm_bwd_reg2", "batchnorm_bwd_reg4", "batchnorm_bwd_reg8",
                "batchnorm_fwd_relu_reg1", "batchnorm_fwd_relu_reg2", "batchnorm_fwd_relu_reg4", "batchnorm_fwd_relu_reg8",
                "batchnorm_bwd_relu_reg1", "batchnorm_bwd_relu_reg2", "batchnorm_bwd_relu_reg4", "batchnorm_bwd_relu_reg8",
                "batchnorm_fwd_addrelu_reg1", "batchnorm_fwd_addrelu_reg2", "batchnorm_fwd_addrelu_reg4", "batchnorm_fwd_addrelu_reg8",
                "batchnorm_bwd_addrelu_reg1", "batchnorm_bwd_addrelu_reg2", "batchnorm_bwd_addrelu_reg4", "batchnorm_bwd_addrelu_reg8",
                "layernorm_forward",
                "layernorm_backward",
                "layernorm_grad_params",
                "groupnorm_forward",
                "instancenorm_forward",
                "rmsnorm_forward",
                "rmsnorm_backward",
                "rmsnorm_grad_gamma",
                "layernorm_inverse_std",
                "layernorm_variance_from_inverse_std",
                "layernorm_backward_rows2",
                "layernorm_backward_rows4",
                "layernorm_backward_rows8",
                "layernorm_backward_rows16",
                "layernorm_grad_params_partial",
                "layernorm_grad_params_finish",
                "column_sum_partial",
                "column_sum_finish",
                "column_sum_small",
                "broadcast_reduced_middle",
                "layernorm_input_grad_add",
                "sum_around_kept_partial",
                "sum_around_kept_finish",
                "softmax_xent_rows",
                "softmax_xent_total",
                "softmax_xent_backward"
            };
        }
    }
}
