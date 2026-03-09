// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for attention operations - FlashAttention, GroupedQueryAttention, and standard attention.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA kernels for attention mechanisms used in transformer architectures.
    /// Includes ScaledDotProductAttention, FlashAttention v2, and GroupedQueryAttention.
    /// </summary>
    internal static class CudaAttentionKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

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
    if (headDim > MAX_HEAD_DIM) return; // Prevent silent truncation for unsupported head dims

    extern __shared__ float smem[];
    float* Ks = smem;                        // [ATTN_BC * headDim]
    float* Vs = smem + ATTN_BC * headDim;    // [ATTN_BC * headDim]

    int kBase = bh * seqK * headDim;
    int vBase = bh * seqK * headDim;

    // Per-thread online softmax accumulators
    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outAcc[MAX_HEAD_DIM];
    for (int d = 0; d < headDim; d++) outAcc[d] = 0.0f;

    // Cache Q row in registers — avoids repeated global memory reads in KV-tile loop
    float qRow[MAX_HEAD_DIM];
    int qOffset = bh * seqQ * headDim + qi * headDim;
    if (qi < seqQ) {
        for (int d = 0; d < headDim; d++) qRow[d] = query[qOffset + d];
    }

    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        // Early exit for causal: all keys in tile are after all queries in block
        if (isCausal && kvStart > qBase + ATTN_BR - 1) break;

        // Cooperative load K tile into shared memory
        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Ks[row * headDim + col] = key[kBase + (kvStart + row) * headDim + col];
        }
        // Cooperative load V tile
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

                // Q dot K — Q from registers, K from shared memory
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += qRow[d] * Ks[t * headDim + d];
                }
                score *= scale;

                // Online softmax update
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

    // Write output
    if (qi < seqQ) {
        int oOffset = bh * seqQ * headDim + qi * headDim;
        float invSum = (rowSum > 0.0f) ? (1.0f / rowSum) : 0.0f;
        for (int d = 0; d < headDim; d++) {
            output[oOffset + d] = outAcc[d] * invSum;
        }

        // Store attention weights if requested (requires recomputation with known logsumexp)
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
// Single expf per score (no double computation).
// Grid: (ceil(seqQ/ATTN_BR), batch*numHeads), Block: (ATTN_BR, 1)
// Shared mem: 2 * ATTN_BC * headDim floats
// ===========================================================================

extern ""C"" __global__ void flash_attention_v2(
    const float* query,          // [batch * heads * seqQ * headDim]
    const float* key,            // [batch * heads * seqK * headDim]
    const float* value,          // [batch * heads * seqK * headDim]
    float* output,               // [batch * heads * seqQ * headDim]
    float* softmaxStats,         // [batch * heads * seqQ] (log-sum-exp)
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    const float* attentionBias,  // optional, NULL if unused
    int hasBias,
    int biasBatchStride)
{
    int qBase = blockIdx.x * ATTN_BR;
    int qi = qBase + threadIdx.x;
    int bh = blockIdx.y;

    if (bh >= batch * numHeads) return;
    if (headDim > MAX_HEAD_DIM) return;

    extern __shared__ float smem[];
    float* Ks = smem;                        // [ATTN_BC * headDim]
    float* Vs = smem + ATTN_BC * headDim;    // [ATTN_BC * headDim]

    int kBase = bh * seqK * headDim;
    int vBase = bh * seqK * headDim;

    // Bias offset computation
    int biasBase = 0;
    if (hasBias && qi < seqQ) {
        int b = bh / numHeads;
        int h = bh % numHeads;
        biasBase = b * biasBatchStride + h * seqQ * seqK + qi * seqK;
    }

    // Per-thread online softmax accumulators
    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outAcc[MAX_HEAD_DIM];
    for (int d = 0; d < headDim; d++) outAcc[d] = 0.0f;

    // Cache Q row in registers — avoids repeated global memory reads in KV-tile loop
    float qRow[MAX_HEAD_DIM];
    int qOffset = bh * seqQ * headDim + qi * headDim;
    if (qi < seqQ) {
        for (int d = 0; d < headDim; d++) qRow[d] = query[qOffset + d];
    }

    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        // Early exit for causal
        if (isCausal && kvStart > qBase + ATTN_BR - 1) break;

        // Cooperative load K tile
        for (int i = threadIdx.x; i < tileSize * headDim; i += ATTN_BR) {
            int row = i / headDim;
            int col = i % headDim;
            Ks[row * headDim + col] = key[kBase + (kvStart + row) * headDim + col];
        }
        // Cooperative load V tile
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

                // Q dot K — Q from registers, K from shared memory
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += qRow[d] * Ks[t * headDim + d];
                }
                score *= scale;
                if (hasBias) score += attentionBias[biasBase + ki];

                // Online softmax: single expf per score
                float newMax = fmaxf(rowMax, score);
                float rescale = expf(rowMax - newMax);
                float expScore = expf(score - newMax);
                rowSum = rowSum * rescale + expScore;

                // Accumulate weighted V (V from shared memory)
                for (int d = 0; d < headDim; d++) {
                    outAcc[d] = outAcc[d] * rescale + expScore * Vs[t * headDim + d];
                }
                rowMax = newMax;
            }
        }
        __syncthreads();
    }

    // Final normalization and write
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
// Recomputes attention weights during backward pass.
// Uses shared memory for K/V tiles to reduce global memory reads.
// Grid: (ceil(seqQ/ATTN_BR), batch*numHeads), Block: (ATTN_BR, 1)
// Shared mem: 2 * ATTN_BC * headDim floats
// ===========================================================================

extern ""C"" __global__ void flash_attention_backward(
    const float* gradOutput,     // [batch * heads * seqQ * headDim]
    const float* query,          // [batch * heads * seqQ * headDim]
    const float* key,            // [batch * heads * seqK * headDim]
    const float* value,          // [batch * heads * seqK * headDim]
    const float* output,         // [batch * heads * seqQ * headDim]
    const float* softmaxStats,   // [batch * heads * seqQ]
    float* gradQuery,            // [batch * heads * seqQ * headDim]
    float* gradKey,              // [batch * heads * seqK * headDim]
    float* gradValue,            // [batch * heads * seqK * headDim]
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    const float* attentionBias,  // optional bias (NULL if unused)
    int hasBias,
    int biasBatchStride)
{
    int qBase = blockIdx.x * ATTN_BR;
    int qi = qBase + threadIdx.x;
    int bh = blockIdx.y;

    if (bh >= batch * numHeads) return;

    extern __shared__ float smem[];
    float* Ks = smem;                        // [ATTN_BC * headDim]
    float* Vs = smem + ATTN_BC * headDim;    // [ATTN_BC * headDim]

    int kBase = bh * seqK * headDim;
    int vBase = bh * seqK * headDim;

    // Pre-compute per-thread constants
    float logsumexp = 0.0f;
    float doO = 0.0f;
    int qOffset = 0, gOffset = 0;
    int biasBase = 0;

    // Cache Q row and gradOutput row in registers to avoid repeated global reads
    float qRow[MAX_HEAD_DIM];
    float goRow[MAX_HEAD_DIM];

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

        // Load Q and gradOutput into registers, compute dO dot O
        for (int d = 0; d < headDim; d++) {
            qRow[d] = query[qOffset + d];
            goRow[d] = gradOutput[gOffset + d];
            doO += goRow[d] * output[qOffset + d];
        }
    }

    // Process KV in tiles
    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        if (isCausal && kvStart > qBase + ATTN_BR - 1) break;

        // Cooperative load K, V tiles
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

                // Recompute attention score: Q from registers, K from shared memory
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += qRow[d] * Ks[t * headDim + d];
                }
                score *= scale;
                if (hasBias) score += attentionBias[biasBase + ki];

                float attnWeight = expf(score - logsumexp);

                // Gradient w.r.t. V: gradOutput from registers
                for (int d = 0; d < headDim; d++) {
                    atomicAdd(&gradValue[vBase + ki * headDim + d],
                              attnWeight * goRow[d]);
                }

                // Compute dO dot V from shared memory, gradOutput from registers
                float doV = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    doV += goRow[d] * Vs[t * headDim + d];
                }

                float dS = attnWeight * (doV - doO) * scale;

                // Gradient w.r.t. Q (from shared memory K)
                for (int d = 0; d < headDim; d++) {
                    gradQuery[qOffset + d] += dS * Ks[t * headDim + d];
                }

                // Gradient w.r.t. K: Q from registers
                for (int d = 0; d < headDim; d++) {
                    atomicAdd(&gradKey[kBase + ki * headDim + d],
                              dS * qRow[d]);
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
    const float* query,          // [batch * numQHeads * seqQ * headDim]
    const float* key,            // [batch * numKVHeads * seqK * headDim]
    const float* value,          // [batch * numKVHeads * seqK * headDim]
    float* output,               // [batch * numQHeads * seqQ * headDim]
    float* attentionWeights,     // [batch * numQHeads * seqQ * seqK] (optional)
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
    if (queriesPerKV <= 0 || numKVHeads <= 0) return;

    int b = bqh / numQHeads;
    int qh = bqh % numQHeads;
    int kvh = qh / queriesPerKV;
    if (kvh >= numKVHeads) return;

    extern __shared__ float smem[];
    float* Ks = smem;                        // [ATTN_BC * headDim]
    float* Vs = smem + ATTN_BC * headDim;    // [ATTN_BC * headDim]

    int kBase = (b * numKVHeads + kvh) * seqK * headDim;
    int vBase = (b * numKVHeads + kvh) * seqK * headDim;

    // Per-thread online softmax
    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outAcc[MAX_HEAD_DIM];
    for (int d = 0; d < headDim; d++) outAcc[d] = 0.0f;

    // Cache Q row in registers — avoids repeated global memory reads
    float qRow[MAX_HEAD_DIM];
    int qOffset = bqh * seqQ * headDim + qi * headDim;
    if (qi < seqQ) {
        for (int d = 0; d < headDim; d++) qRow[d] = query[qOffset + d];
    }

    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        if (isCausal && kvStart > qBase + ATTN_BR - 1) break;

        // Cooperative load K, V tiles
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

                // Q dot K — Q from registers, K from shared memory
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += qRow[d] * Ks[t * headDim + d];
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
// Computes gradWeights on-the-fly per KV tile.
// Grid: (ceil(seqQ/ATTN_BR), batch*numQHeads), Block: (ATTN_BR, 1)
// Shared mem: 2 * ATTN_BC * headDim floats
// ===========================================================================

extern ""C"" __global__ void grouped_query_attention_backward(
    const float* gradOutput,       // [batch * numQHeads * seqQ * headDim]
    const float* query,            // [batch * numQHeads * seqQ * headDim]
    const float* key,              // [batch * numKVHeads * seqK * headDim]
    const float* value,            // [batch * numKVHeads * seqK * headDim]
    const float* attentionWeights, // [batch * numQHeads * seqQ * seqK]
    float* gradQuery,              // [batch * numQHeads * seqQ * headDim]
    float* gradKey,                // [batch * numKVHeads * seqK * headDim]
    float* gradValue,              // [batch * numKVHeads * seqK * headDim]
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
    if (queriesPerKV <= 0 || numKVHeads <= 0) return;

    int b_idx = bqh / numQHeads;
    int qh = bqh % numQHeads;
    int kvh = qh / queriesPerKV;
    if (kvh >= numKVHeads) return;

    extern __shared__ float smem[];
    float* Ks = smem;                        // [ATTN_BC * headDim]
    float* Vs = smem + ATTN_BC * headDim;    // [ATTN_BC * headDim]

    int kBase = (b_idx * numKVHeads + kvh) * seqK * headDim;
    int vBase = (b_idx * numKVHeads + kvh) * seqK * headDim;

    // Pre-compute per-thread dot(weights, gradWeights) over ALL seqK
    float dotWgW = 0.0f;
    int qOffset = 0, gOffset = 0, wOffset = 0;

    // Cache Q row and gradOutput row in registers
    float qRow[MAX_HEAD_DIM];
    float goRow[MAX_HEAD_DIM];

    if (qi < seqQ) {
        qOffset = bqh * seqQ * headDim + qi * headDim;
        gOffset = bqh * seqQ * headDim + qi * headDim;
        wOffset = bqh * seqQ * seqK + qi * seqK;
        for (int d = 0; d < headDim; d++) {
            qRow[d] = query[qOffset + d];
            goRow[d] = gradOutput[gOffset + d];
        }
    }

    // First pass: compute dot(weights, gradWeights) using tiles
    // All threads participate in cooperative loads and barriers
    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        // Load V tile for gradWeight computation (all threads cooperate)
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
                    gw += goRow[d] * Vs[t * headDim + d];
                }
                dotWgW += weight * gw;
            }
        }
        __syncthreads();
    }

    // Second pass: compute actual gradients using tiles
    for (int kvStart = 0; kvStart < seqK; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqK - kvStart);

        // Load K, V tiles
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

                // Compute gradWeight: gradOutput from registers, V from shared memory
                float gw = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    gw += goRow[d] * Vs[t * headDim + d];
                }

                float gradScore = weight * (gw - dotWgW) * scale;

                // Gradient w.r.t. V: gradOutput from registers
                for (int d = 0; d < headDim; d++) {
                    atomicAdd(&gradValue[vBase + ki * headDim + d],
                              weight * goRow[d]);
                }

                // Gradient w.r.t. Q (K from shared memory)
                for (int d = 0; d < headDim; d++) {
                    gradQuery[qOffset + d] += gradScore * Ks[t * headDim + d];
                }

                // Gradient w.r.t. K: Q from registers
                for (int d = 0; d < headDim; d++) {
                    atomicAdd(&gradKey[kBase + ki * headDim + d],
                              gradScore * qRow[d]);
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
    float* Ks = smem;                        // [ATTN_BC * headDim]
    float* Vs = smem + ATTN_BC * headDim;    // [ATTN_BC * headDim]

    int kBase = bh * seqLen * headDim;
    int vBase = bh * seqLen * headDim;

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outAcc[MAX_HEAD_DIM];
    for (int d = 0; d < headDim; d++) outAcc[d] = 0.0f;

    // Cache Q row in registers — avoids repeated global memory reads
    float qRow[MAX_HEAD_DIM];
    int qOffset = bh * seqLen * headDim + qi * headDim;
    if (qi < seqLen) {
        for (int d = 0; d < headDim; d++) qRow[d] = query[qOffset + d];
    }

    for (int kvStart = 0; kvStart < seqLen; kvStart += ATTN_BC) {
        int tileSize = min(ATTN_BC, seqLen - kvStart);

        if (isCausal && kvStart > qBase + ATTN_BR - 1) break;

        // Cooperative load K, V tiles
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
            for (int t = 0; t < tileSize; t++) {
                int ki = kvStart + t;
                if (isCausal && ki > qi) continue;

                // Q dot K — Q from registers, K from shared memory
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += qRow[d] * Ks[t * headDim + d];
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
}
