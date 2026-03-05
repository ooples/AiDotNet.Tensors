// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for attention operations - FlashAttention, GroupedQueryAttention, and standard attention.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for attention mechanisms used in transformer architectures.
/// Includes ScaledDotProductAttention, FlashAttention v2, and GroupedQueryAttention.
/// </summary>
internal static class HipAttentionKernels
{
    public static string GetSource()
    {
        // Note: hiprtc provides device intrinsics built-in, no includes needed
        return @"
// HIP RTC Compatibility - no includes needed, device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

// Tile sizes for shared-memory tiled attention
#define ATTN_BR 32   // Query tile size (threads per block)
#define ATTN_BC 32   // KV tile size
#define MAX_HEAD_DIM 128

// ===========================================================================
// SCALED DOT-PRODUCT ATTENTION
// Uses online softmax + shared memory for K/V tiles.
// Eliminates float scores[1024] VLA, supports arbitrary seqK.
// Grid: (ceil(seqQ/ATTN_BR), batch*numHeads), Block: (ATTN_BR, 1)
// Shared mem: 2 * ATTN_BC * headDim floats
// ===========================================================================

extern ""C"" __global__ void scaled_dot_product_attention(
    const float* query,      // [batch * heads * seqQ * headDim]
    const float* key,        // [batch * heads * seqK * headDim]
    const float* value,      // [batch * heads * seqK * headDim]
    float* output,           // [batch * heads * seqQ * headDim]
    float* attentionWeights, // [batch * heads * seqQ * seqK] (optional)
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    int storeWeights)
{
    int qBase = blockIdx.x * ATTN_BR;
    int qi = qBase + threadIdx.x;
    int bh = blockIdx.y;

    if (bh >= batch * numHeads) return;
    if (headDim > MAX_HEAD_DIM) return;

    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = smem + ATTN_BC * headDim;

    int kBase = bh * seqK * headDim;
    int vBase = bh * seqK * headDim;

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outAcc[MAX_HEAD_DIM];
    for (int d = 0; d < headDim; d++) outAcc[d] = 0.0f;

    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        if (isCausal && kvStart > qBase + ATTN_BR - 1) break;

        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Ks[row * headDim + col] = key[kBase + (kvStart + row) * headDim + col];
        }
        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Vs[row * headDim + col] = value[vBase + (kvStart + row) * headDim + col];
        }
        __syncthreads();

        if (qi < seqQ) {
            int qOffset = bh * seqQ * headDim + qi * headDim;

            for (int t = 0; t < tileSize; t++) {
                int ki = kvStart + t;
                if (isCausal && ki > qi) continue;

                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += query[qOffset + d] * Ks[t * headDim + d];
                }
                score *= scale;

                float newMax = fmaxf(rowMax, score);
                float rescale = expf(rowMax - newMax);
                float expScore = expf(score - newMax);
                rowSum = rowSum * rescale + expScore;

                for (int d = 0; d < headDim; d++) {
                    outAcc[d] = outAcc[d] * rescale + expScore * Vs[t * headDim + d];
                }
                rowMax = newMax;
            }
        }
        __syncthreads();
    }

    if (qi < seqQ) {
        int oOffset = bh * seqQ * headDim + qi * headDim;
        float invSum = (rowSum > 0.0f) ? (1.0f / rowSum) : 0.0f;
        for (int d = 0; d < headDim; d++) {
            output[oOffset + d] = outAcc[d] * invSum;
        }

        if (storeWeights) {
            float logsumexp = rowMax + logf(rowSum);
            int wOffset = bh * seqQ * seqK + qi * seqK;
            int qOffset = bh * seqQ * headDim + qi * headDim;
            int kOff = bh * seqK * headDim;
            for (int ki = 0; ki < seqK; ki++) {
                if (isCausal && ki > qi) {
                    attentionWeights[wOffset + ki] = 0.0f;
                    continue;
                }
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += query[qOffset + d] * key[kOff + ki * headDim + d];
                }
                score *= scale;
                attentionWeights[wOffset + ki] = expf(score - logsumexp);
            }
        }
    }
}

// ===========================================================================
// FLASH ATTENTION V2
// Tiled algorithm with shared memory for K/V. Online softmax eliminates VLA.
// Grid: (ceil(seqQ/ATTN_BR), batch*numHeads), Block: (ATTN_BR, 1)
// Shared mem: 2 * ATTN_BC * headDim floats
// ===========================================================================

extern ""C"" __global__ void flash_attention_v2(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* softmaxStats,
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    const float* attentionBias,
    int hasBias,
    int biasBatchStride)
{
    int qBase = blockIdx.x * ATTN_BR;
    int qi = qBase + threadIdx.x;
    int bh = blockIdx.y;

    if (bh >= batch * numHeads) return;
    if (headDim > MAX_HEAD_DIM) return;

    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = smem + ATTN_BC * headDim;

    int kBase = bh * seqK * headDim;
    int vBase = bh * seqK * headDim;

    int biasBase = 0;
    if (hasBias && qi < seqQ) {
        int b = bh / numHeads;
        int h = bh % numHeads;
        biasBase = b * biasBatchStride + h * seqQ * seqK + qi * seqK;
    }

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outAcc[MAX_HEAD_DIM];
    for (int d = 0; d < headDim; d++) outAcc[d] = 0.0f;

    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        if (isCausal && kvStart > qBase + ATTN_BR - 1) break;

        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Ks[row * headDim + col] = key[kBase + (kvStart + row) * headDim + col];
        }
        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Vs[row * headDim + col] = value[vBase + (kvStart + row) * headDim + col];
        }
        __syncthreads();

        if (qi < seqQ) {
            int qOffset = bh * seqQ * headDim + qi * headDim;

            for (int t = 0; t < tileSize; t++) {
                int ki = kvStart + t;
                if (isCausal && ki > qi) continue;

                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += query[qOffset + d] * Ks[t * headDim + d];
                }
                score *= scale;
                if (hasBias) score += attentionBias[biasBase + ki];

                float newMax = fmaxf(rowMax, score);
                float rescale = expf(rowMax - newMax);
                float expScore = expf(score - newMax);
                rowSum = rowSum * rescale + expScore;

                for (int d = 0; d < headDim; d++) {
                    outAcc[d] = outAcc[d] * rescale + expScore * Vs[t * headDim + d];
                }
                rowMax = newMax;
            }
        }
        __syncthreads();
    }

    if (qi < seqQ) {
        int oOffset = bh * seqQ * headDim + qi * headDim;
        int sOffset = bh * seqQ + qi;

        float invSum = (rowSum > 0.0f) ? (1.0f / rowSum) : 0.0f;
        for (int d = 0; d < headDim; d++) {
            output[oOffset + d] = outAcc[d] * invSum;
        }
        softmaxStats[sOffset] = rowMax + logf(fmaxf(rowSum, 1e-20f));
    }
}

// ===========================================================================
// FLASH ATTENTION BACKWARD
// Uses shared memory for K/V tiles to reduce global memory reads.
// Grid: (ceil(seqQ/ATTN_BR), batch*numHeads), Block: (ATTN_BR, 1)
// Shared mem: 2 * ATTN_BC * headDim floats
// ===========================================================================

extern ""C"" __global__ void flash_attention_backward(
    const float* gradOutput,
    const float* query,
    const float* key,
    const float* value,
    const float* output,
    const float* softmaxStats,
    float* gradQuery,
    float* gradKey,
    float* gradValue,
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    const float* attentionBias,
    int hasBias,
    int biasBatchStride)
{
    int qBase = blockIdx.x * ATTN_BR;
    int qi = qBase + threadIdx.x;
    int bh = blockIdx.y;

    if (bh >= batch * numHeads) return;
    if (headDim > MAX_HEAD_DIM) return;

    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = smem + ATTN_BC * headDim;

    int kBase = bh * seqK * headDim;
    int vBase = bh * seqK * headDim;

    float logsumexp = 0.0f;
    float doO = 0.0f;
    int qOffset = 0, gOffset = 0;
    int biasBase = 0;

    if (qi < seqQ) {
        qOffset = bh * seqQ * headDim + qi * headDim;
        gOffset = bh * seqQ * headDim + qi * headDim;
        int sOffset = bh * seqQ + qi;
        logsumexp = softmaxStats[sOffset];

        if (hasBias) {
            int b = bh / numHeads;
            int h = bh % numHeads;
            biasBase = b * biasBatchStride + h * seqQ * seqK + qi * seqK;
        }

        for (int d = 0; d < headDim; d++) {
            doO += gradOutput[gOffset + d] * output[qOffset + d];
        }
    }

    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        if (isCausal && kvStart > qBase + ATTN_BR - 1) break;

        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Ks[row * headDim + col] = key[kBase + (kvStart + row) * headDim + col];
        }
        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Vs[row * headDim + col] = value[vBase + (kvStart + row) * headDim + col];
        }
        __syncthreads();

        if (qi < seqQ) {
            for (int t = 0; t < tileSize; t++) {
                int ki = kvStart + t;
                if (isCausal && ki > qi) continue;

                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += query[qOffset + d] * Ks[t * headDim + d];
                }
                score *= scale;
                if (hasBias) score += attentionBias[biasBase + ki];

                float attnWeight = expf(score - logsumexp);

                for (int d = 0; d < headDim; d++) {
                    atomicAdd(&gradValue[vBase + ki * headDim + d],
                              attnWeight * gradOutput[gOffset + d]);
                }

                float doV = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    doV += gradOutput[gOffset + d] * Vs[t * headDim + d];
                }

                float dS = attnWeight * (doV - doO) * scale;

                for (int d = 0; d < headDim; d++) {
                    gradQuery[qOffset + d] += dS * Ks[t * headDim + d];
                }

                for (int d = 0; d < headDim; d++) {
                    atomicAdd(&gradKey[kBase + ki * headDim + d],
                              dS * query[qOffset + d]);
                }
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// GROUPED QUERY ATTENTION (GQA)
// Uses online softmax + shared memory. Eliminates float scores[1024] VLA.
// Grid: (ceil(seqQ/ATTN_BR), batch*numQHeads), Block: (ATTN_BR, 1)
// Shared mem: 2 * ATTN_BC * headDim floats
// ===========================================================================

extern ""C"" __global__ void grouped_query_attention(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* attentionWeights,
    int batch,
    int numQHeads,
    int numKVHeads,
    int queriesPerKV,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    int storeWeights)
{
    int qBase = blockIdx.x * ATTN_BR;
    int qi = qBase + threadIdx.x;
    int bqh = blockIdx.y;

    if (bqh >= batch * numQHeads) return;
    if (headDim > MAX_HEAD_DIM) return;

    int b = bqh / numQHeads;
    int qh = bqh % numQHeads;
    int kvh = qh / queriesPerKV;

    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = smem + ATTN_BC * headDim;

    int kBase = (b * numKVHeads + kvh) * seqK * headDim;
    int vBase = (b * numKVHeads + kvh) * seqK * headDim;

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outAcc[MAX_HEAD_DIM];
    for (int d = 0; d < headDim; d++) outAcc[d] = 0.0f;

    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        if (isCausal && kvStart > qBase + ATTN_BR - 1) break;

        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Ks[row * headDim + col] = key[kBase + (kvStart + row) * headDim + col];
        }
        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Vs[row * headDim + col] = value[vBase + (kvStart + row) * headDim + col];
        }
        __syncthreads();

        if (qi < seqQ) {
            int qOffset = bqh * seqQ * headDim + qi * headDim;

            for (int t = 0; t < tileSize; t++) {
                int ki = kvStart + t;
                if (isCausal && ki > qi) continue;

                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += query[qOffset + d] * Ks[t * headDim + d];
                }
                score *= scale;

                float newMax = fmaxf(rowMax, score);
                float rescale = expf(rowMax - newMax);
                float expScore = expf(score - newMax);
                rowSum = rowSum * rescale + expScore;

                for (int d = 0; d < headDim; d++) {
                    outAcc[d] = outAcc[d] * rescale + expScore * Vs[t * headDim + d];
                }
                rowMax = newMax;
            }
        }
        __syncthreads();
    }

    if (qi < seqQ) {
        int oOffset = bqh * seqQ * headDim + qi * headDim;
        float invSum = (rowSum > 0.0f) ? (1.0f / rowSum) : 0.0f;
        for (int d = 0; d < headDim; d++) {
            output[oOffset + d] = outAcc[d] * invSum;
        }

        if (storeWeights) {
            float logsumexp = rowMax + logf(fmaxf(rowSum, 1e-20f));
            int wOffset = bqh * seqQ * seqK + qi * seqK;
            int qOffset = bqh * seqQ * headDim + qi * headDim;
            for (int ki = 0; ki < seqK; ki++) {
                if (isCausal && ki > qi) {
                    attentionWeights[wOffset + ki] = 0.0f;
                    continue;
                }
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += query[qOffset + d] * key[kBase + ki * headDim + d];
                }
                score *= scale;
                attentionWeights[wOffset + ki] = expf(score - logsumexp);
            }
        }
    }
}

// ===========================================================================
// GQA BACKWARD
// Uses shared memory for K/V tiles. Eliminates float gradWeights[1024] VLA.
// Grid: (ceil(seqQ/ATTN_BR), batch*numQHeads), Block: (ATTN_BR, 1)
// Shared mem: 2 * ATTN_BC * headDim floats
// ===========================================================================

extern ""C"" __global__ void grouped_query_attention_backward(
    const float* gradOutput,
    const float* query,
    const float* key,
    const float* value,
    const float* attentionWeights,
    float* gradQuery,
    float* gradKey,
    float* gradValue,
    int batch,
    int numQHeads,
    int numKVHeads,
    int queriesPerKV,
    int seqQ,
    int seqK,
    int headDim,
    float scale)
{
    int qBase = blockIdx.x * ATTN_BR;
    int qi = qBase + threadIdx.x;
    int bqh = blockIdx.y;

    if (bqh >= batch * numQHeads) return;
    if (headDim > MAX_HEAD_DIM) return;

    int b_idx = bqh / numQHeads;
    int qh = bqh % numQHeads;
    int kvh = qh / queriesPerKV;

    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = smem + ATTN_BC * headDim;

    int kBase = (b_idx * numKVHeads + kvh) * seqK * headDim;
    int vBase = (b_idx * numKVHeads + kvh) * seqK * headDim;

    float dotWgW = 0.0f;
    int qOffset = 0, gOffset = 0, wOffset = 0;

    if (qi < seqQ) {
        qOffset = bqh * seqQ * headDim + qi * headDim;
        gOffset = bqh * seqQ * headDim + qi * headDim;
        wOffset = bqh * seqQ * seqK + qi * seqK;
    }

    // First pass: compute dot(weights, gradWeights) using tiles
    // All threads participate in cooperative loads and barriers
    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Vs[row * headDim + col] = value[vBase + (kvStart + row) * headDim + col];
        }
        __syncthreads();

        if (qi < seqQ) {
            for (int t = 0; t < tileSize; t++) {
                int ki = kvStart + t;
                float weight = attentionWeights[wOffset + ki];
                float gw = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    gw += gradOutput[gOffset + d] * Vs[t * headDim + d];
                }
                dotWgW += weight * gw;
            }
        }
        __syncthreads();
    }

    // Second pass: compute actual gradients using tiles
    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Ks[row * headDim + col] = key[kBase + (kvStart + row) * headDim + col];
        }
        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Vs[row * headDim + col] = value[vBase + (kvStart + row) * headDim + col];
        }
        __syncthreads();

        if (qi < seqQ) {
            for (int t = 0; t < tileSize; t++) {
                int ki = kvStart + t;
                float weight = attentionWeights[wOffset + ki];

                float gw = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    gw += gradOutput[gOffset + d] * Vs[t * headDim + d];
                }

                float gradScore = weight * (gw - dotWgW) * scale;

                for (int d = 0; d < headDim; d++) {
                    atomicAdd(&gradValue[vBase + ki * headDim + d],
                              weight * gradOutput[gOffset + d]);
                }

                for (int d = 0; d < headDim; d++) {
                    gradQuery[qOffset + d] += gradScore * Ks[t * headDim + d];
                }

                for (int d = 0; d < headDim; d++) {
                    atomicAdd(&gradKey[kBase + ki * headDim + d],
                              gradScore * query[qOffset + d]);
                }
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// FLASH ATTENTION FORWARD (Compatibility version without stats)
// Same tiled shared-memory approach. Fixed double expf computation.
// Grid: (ceil(seqLen/ATTN_BR), batch*numHeads), Block: (ATTN_BR, 1)
// Shared mem: 2 * ATTN_BC * headDim floats
// ===========================================================================

extern ""C"" __global__ void flash_attention_forward(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int batch,
    int numHeads,
    int seqLen,
    int headDim,
    float scale,
    int isCausal)
{
    int qBase = blockIdx.x * ATTN_BR;
    int qi = qBase + threadIdx.x;
    int bh = blockIdx.y;

    if (bh >= batch * numHeads) return;
    if (headDim > MAX_HEAD_DIM) return;

    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = smem + ATTN_BC * headDim;

    int kBase = bh * seqLen * headDim;
    int vBase = bh * seqLen * headDim;

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outAcc[MAX_HEAD_DIM];
    for (int d = 0; d < headDim; d++) outAcc[d] = 0.0f;

    for (int kvStart = 0; kvStart < seqLen; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqLen - kvStart);

        if (isCausal && kvStart > qBase + ATTN_BR - 1) break;

        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Ks[row * headDim + col] = key[kBase + (kvStart + row) * headDim + col];
        }
        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Vs[row * headDim + col] = value[vBase + (kvStart + row) * headDim + col];
        }
        __syncthreads();

        if (qi < seqLen) {
            int qOffset = bh * seqLen * headDim + qi * headDim;

            for (int t = 0; t < tileSize; t++) {
                int ki = kvStart + t;
                if (isCausal && ki > qi) continue;

                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += query[qOffset + d] * Ks[t * headDim + d];
                }
                score *= scale;

                float newMax = fmaxf(rowMax, score);
                float rescale = expf(rowMax - newMax);
                float expScore = expf(score - newMax);
                rowSum = rowSum * rescale + expScore;

                for (int d = 0; d < headDim; d++) {
                    outAcc[d] = outAcc[d] * rescale + expScore * Vs[t * headDim + d];
                }
                rowMax = newMax;
            }
        }
        __syncthreads();
    }

    if (qi < seqLen) {
        int oOffset = bh * seqLen * headDim + qi * headDim;
        float invSum = (rowSum > 0.0f) ? (1.0f / rowSum) : 0.0f;
        for (int d = 0; d < headDim; d++) {
            output[oOffset + d] = outAcc[d] * invSum;
        }
    }
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "scaled_dot_product_attention",
            "flash_attention_v2",
            "flash_attention_backward",
            "grouped_query_attention",
            "grouped_query_attention_backward",
            "flash_attention_forward"
        };
    }
}
