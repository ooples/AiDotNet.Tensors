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
// One invocation owns one query row. Scores are recomputed in fixed key order,
// avoiding sequence-sized local arrays and supporting arbitrary sequence/head sizes.
// ===========================================================================

extern ""C"" __global__ __launch_bounds__(256) void scaled_dot_product_attention(
    const float* __restrict__ query,      // [batch * heads * seqQ * headDim]
    const float* __restrict__ key,        // [batch * heads * seqK * headDim]
    const float* __restrict__ value,      // [batch * heads * seqK * headDim]
    float* __restrict__ output,           // [batch * heads * seqQ * headDim]
    float* __restrict__ attentionWeights, // [batch * heads * seqQ * seqK] (optional)
    const float* __restrict__ mask,       // shared [seqQ * seqK] or full [batch * heads * seqQ * seqK]
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    int maskMode,
    int storeWeights,
    float softcap,
    int numKVHeads)  // Grouped-Query Attention: <numHeads shares each KV head; == numHeads for MHA.
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int totalRows = batch * numHeads * seqQ;
    if (row >= totalRows) return;

    int qi = row % seqQ;
    int bh = row / seqQ;
    int qOffset = row * headDim;
    // GQA: map query head -> shared KV head. bh = b*numHeads + h; kvHead = h / (numHeads/numKVHeads);
    // the K/V slab for this row lives at (b*numKVHeads + kvHead). numKVHeads==numHeads collapses to bh.
    int b = bh / numHeads;
    int h = bh - b * numHeads;
    int kvGroup = numHeads / numKVHeads;
    int bkvh = b * numKVHeads + (h / kvGroup);
    int kOffset = bkvh * seqK * headDim;
    int vOffset = kOffset;
    int wOffset = row * seqK;
    int maskOffset = (maskMode == 2 ? bh * seqQ * seqK : 0) + qi * seqK;
    // Causal with a KV-cache offset: query row qi is at absolute position qi + (seqK - seqQ) (seqQ==seqK
    // collapses to ki > qi; a decode step seqQ=1<seqK attends to the whole cached prefix, not just key 0).
    int qPos = qi + (seqK - seqQ);

    // Attention-logit soft-cap (Gemma-2): softcap * tanh((score*scale)/softcap); softcap<=0 disables.
    #define ATTN_LOGIT(sc) (softcap > 0.0f ? softcap * tanhf(((sc) * scale) / softcap) : (sc) * scale)

    float rowMax = -INFINITY;
    for (int ki = 0; ki < seqK; ki++) {
        if ((isCausal && ki > qPos) || (maskMode != 0 && mask[maskOffset + ki] == 0.0f)) continue;
        float score = 0.0f;
        for (int d = 0; d < headDim; d++) score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        rowMax = fmaxf(rowMax, ATTN_LOGIT(score));
    }

    float denominator = 0.0f;
    for (int ki = 0; ki < seqK; ki++) {
        if ((isCausal && ki > qPos) || (maskMode != 0 && mask[maskOffset + ki] == 0.0f)) continue;
        float score = 0.0f;
        for (int d = 0; d < headDim; d++) score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        denominator += expf(ATTN_LOGIT(score) - rowMax);
    }

    float inverseDenominator = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    if (storeWeights) {
        for (int ki = 0; ki < seqK; ki++) {
            float weight = 0.0f;
            if (!((isCausal && ki > qPos) || (maskMode != 0 && mask[maskOffset + ki] == 0.0f))) {
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) score += query[qOffset + d] * key[kOffset + ki * headDim + d];
                weight = expf(ATTN_LOGIT(score) - rowMax) * inverseDenominator;
            }
            attentionWeights[wOffset + ki] = weight;
        }
    }

    for (int d = 0; d < headDim; d++) {
        float sum = 0.0f;
        for (int ki = 0; ki < seqK; ki++) {
            if ((isCausal && ki > qPos) || (maskMode != 0 && mask[maskOffset + ki] == 0.0f)) continue;
            float score = 0.0f;
            for (int inner = 0; inner < headDim; inner++) score += query[qOffset + inner] * key[kOffset + ki * headDim + inner];
            float weight = expf(ATTN_LOGIT(score) - rowMax) * inverseDenominator;
            sum += weight * value[vOffset + ki * headDim + d];
        }
        output[qOffset + d] = sum;
    }
    #undef ATTN_LOGIT
}

// ===========================================================================
// FLASH ATTENTION V2
// Tiled algorithm with shared memory for K/V. Online softmax eliminates VLA.
// Single expf per score (no double computation).
// Grid: (ceil(seqQ/ATTN_BR), batch*numHeads), Block: (ATTN_BR, 1)
// Shared mem: 2 * ATTN_BC * headDim floats
// ===========================================================================

extern ""C"" __global__ __launch_bounds__(256) void flash_attention_v2(
    const float* __restrict__ query,          // [batch * heads * seqQ * headDim]
    const float* __restrict__ key,            // [batch * heads * seqK * headDim]
    const float* __restrict__ value,          // [batch * heads * seqK * headDim]
    float* __restrict__ output,               // [batch * heads * seqQ * headDim]
    float* __restrict__ softmaxStats,         // [batch * heads * seqQ] (log-sum-exp)
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    const float* __restrict__ attentionBias,  // optional, NULL if unused
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

// FlashAttention backward — bit-deterministic variants (issue #382).
// The tiled flash_attention_backward uses atomicAdd for gradKey/gradValue
// (scheduler-dependent ordering -> FP-non-deterministic). The two kernels below
// split the work so each output cell is owned by exactly one thread:
//   gradq: per (bh, qi) -> writes gradQuery only, no atomics
//   gradkv: per (bh, ki, d) -> writes gradKey[ki, d] and gradValue[ki, d]
//          by iterating qi in fixed ascending order
extern ""C"" __global__ __launch_bounds__(256) void flash_attention_backward_gradq_deterministic(
    const float* __restrict__ gradOutput,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ output,
    const float* __restrict__ softmaxStats,
    float* __restrict__ gradQuery,
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    const float* __restrict__ attentionBias,
    int hasBias,
    int biasBatchStride)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    int bh = blockIdx.y;
    if (qi >= seqQ || bh >= batch * numHeads) return;

    int qOffset = bh * seqQ * headDim + qi * headDim;
    int kOffset = bh * seqK * headDim;
    int vOffset = bh * seqK * headDim;
    int gOffset = bh * seqQ * headDim + qi * headDim;
    int sOffset = bh * seqQ + qi;

    int biasBase = 0;
    if (hasBias) {
        int b = bh / numHeads;
        int h = bh % numHeads;
        biasBase = b * biasBatchStride + h * seqQ * seqK + qi * seqK;
    }

    float logsumexp = softmaxStats[sOffset];

    float doO = 0.0f;
    for (int d = 0; d < headDim; d++) {
        doO += gradOutput[gOffset + d] * output[qOffset + d];
        gradQuery[qOffset + d] = 0.0f;
    }

    for (int ki = 0; ki < seqK; ki++) {
        if (isCausal && ki > qi) continue;

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;
        if (hasBias) score += attentionBias[biasBase + ki];

        float attnWeight = expf(score - logsumexp);

        float doV = 0.0f;
        for (int d = 0; d < headDim; d++) {
            doV += gradOutput[gOffset + d] * value[vOffset + ki * headDim + d];
        }

        float dS = attnWeight * (doV - doO) * scale;

        for (int d = 0; d < headDim; d++) {
            gradQuery[qOffset + d] += dS * key[kOffset + ki * headDim + d];
        }
    }
}

extern ""C"" __global__ __launch_bounds__(256) void flash_attention_backward_gradkv_deterministic(
    const float* __restrict__ gradOutput,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ output,
    const float* __restrict__ softmaxStats,
    float* __restrict__ gradKey,
    float* __restrict__ gradValue,
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    const float* __restrict__ attentionBias,
    int hasBias,
    int biasBatchStride)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int ki = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z;
    if (d >= headDim || ki >= seqK || bh >= batch * numHeads) return;

    int kOffset = bh * seqK * headDim;
    int vOffset = bh * seqK * headDim;
    int qBase = bh * seqQ * headDim;
    int gBase = bh * seqQ * headDim;
    int sBase = bh * seqQ;

    float accV = 0.0f;
    float accK = 0.0f;

    int b = bh / numHeads;
    int h = bh % numHeads;
    int biasHeadBase = hasBias ? (b * biasBatchStride + h * seqQ * seqK) : 0;

    for (int qi = 0; qi < seqQ; qi++) {
        if (isCausal && ki > qi) continue;

        int qOffset = qBase + qi * headDim;
        int gOffset = gBase + qi * headDim;

        float logsumexp = softmaxStats[sBase + qi];
        float score = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            score += query[qOffset + dd] * key[kOffset + ki * headDim + dd];
        }
        score *= scale;
        if (hasBias) score += attentionBias[biasHeadBase + qi * seqK + ki];

        float attnWeight = expf(score - logsumexp);

        float doO = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            doO += gradOutput[gOffset + dd] * output[qOffset + dd];
        }
        float doV = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            doV += gradOutput[gOffset + dd] * value[vOffset + ki * headDim + dd];
        }

        accV += attnWeight * gradOutput[gOffset + d];
        float dS = attnWeight * (doV - doO) * scale;
        accK += dS * query[qOffset + d];
    }

    gradValue[vOffset + ki * headDim + d] = accV;
    gradKey[kOffset + ki * headDim + d] = accK;
}

extern ""C"" __global__ __launch_bounds__(256) void flash_attention_backward(
    const float* __restrict__ gradOutput,     // [batch * heads * seqQ * headDim]
    const float* __restrict__ query,          // [batch * heads * seqQ * headDim]
    const float* __restrict__ key,            // [batch * heads * seqK * headDim]
    const float* __restrict__ value,          // [batch * heads * seqK * headDim]
    const float* __restrict__ output,         // [batch * heads * seqQ * headDim]
    const float* __restrict__ softmaxStats,   // [batch * heads * seqQ]
    float* __restrict__ gradQuery,            // [batch * heads * seqQ * headDim]
    float* __restrict__ gradKey,              // [batch * heads * seqK * headDim]
    float* __restrict__ gradValue,            // [batch * heads * seqK * headDim]
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    const float* __restrict__ attentionBias,  // optional bias (NULL if unused)
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

extern ""C"" __global__ __launch_bounds__(256) void grouped_query_attention(
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

    // Causal with a KV-cache offset: query row qi is at absolute position qi + (seqK - seqQ),
    // matching scaled_dot_product_attention above and ScaledDotProductAttentionGqa on the CPU
    // engine. seqQ == seqK makes the offset zero and prefill is unchanged; a decode step
    // (seqQ = 1 < seqK) attends to the whole cached prefix rather than to key 0 alone.
    int qPos = qi + (seqK - seqQ);

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

        // The last query of this block sits at absolute position qBase + ATTN_BR - 1 + (seqK - seqQ),
        // so no later tile can be visible to any thread. Written from block-uniform terms rather than
        // from the per-thread qPos: this break precedes __syncthreads() below and must not diverge.
        if (isCausal && kvStart > qBase + ATTN_BR - 1 + (seqK - seqQ)) break;

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
                if (isCausal && ki > qPos) continue;

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
                if (isCausal && ki > qPos) {
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

// GQA backward — bit-deterministic variants (issue #382). Same split as Flash:
//   gradq: per (bqh, qi) writes gradQuery only (no atomics)
//   gradkv: per (b, kvh, ki, d) writes gradKey/gradValue with fixed (qh, qi) order
extern ""C"" __global__ __launch_bounds__(256) void grouped_query_attention_backward_gradq_deterministic(
    const float* gradOutput,
    const float* query,
    const float* key,
    const float* value,
    const float* attentionWeights,
    float* gradQuery,
    int batch,
    int numQHeads,
    int numKVHeads,
    int queriesPerKV,
    int seqQ,
    int seqK,
    int headDim,
    float scale)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    int bqh = blockIdx.y;
    if (qi >= seqQ || bqh >= batch * numQHeads) return;

    int b = bqh / numQHeads;
    int qh = bqh % numQHeads;
    int kvh = qh / queriesPerKV;

    int qOffset = bqh * seqQ * headDim + qi * headDim;
    int kOffset = (b * numKVHeads + kvh) * seqK * headDim;
    int vOffset = (b * numKVHeads + kvh) * seqK * headDim;
    int gOffset = bqh * seqQ * headDim + qi * headDim;
    int wOffset = bqh * seqQ * seqK + qi * seqK;

    float dotProduct = 0.0f;
    for (int dd = 0; dd < headDim; dd++) gradQuery[qOffset + dd] = 0.0f;
    for (int ki = 0; ki < seqK; ki++) {
        float w = attentionWeights[wOffset + ki];
        float gw = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            gw += gradOutput[gOffset + dd] * value[vOffset + ki * headDim + dd];
        }
        dotProduct += w * gw;
    }

    for (int ki = 0; ki < seqK; ki++) {
        float w = attentionWeights[wOffset + ki];
        float gw = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            gw += gradOutput[gOffset + dd] * value[vOffset + ki * headDim + dd];
        }
        float gradScore = w * (gw - dotProduct) * scale;
        for (int dd = 0; dd < headDim; dd++) {
            gradQuery[qOffset + dd] += gradScore * key[kOffset + ki * headDim + dd];
        }
    }
}

extern ""C"" __global__ __launch_bounds__(256) void grouped_query_attention_backward_gradkv_deterministic(
    const float* gradOutput,
    const float* query,
    const float* key,
    const float* value,
    const float* attentionWeights,
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
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int ki = blockIdx.y * blockDim.y + threadIdx.y;
    int bkvh = blockIdx.z;
    if (d >= headDim || ki >= seqK || bkvh >= batch * numKVHeads) return;

    int b = bkvh / numKVHeads;
    int kvh = bkvh % numKVHeads;
    int kvOffsetBase = bkvh * seqK * headDim;

    float accV = 0.0f;
    float accK = 0.0f;

    for (int qhOff = 0; qhOff < queriesPerKV; qhOff++) {
        int qh = kvh * queriesPerKV + qhOff;
        // BUGFIX: bound qh to the real query heads. For an inconsistent GQA config where
        // numKVHeads*queriesPerKV > numQHeads (e.g. numQHeads=3, numKVHeads=3, queriesPerKV=2 → kvh=1
        // would gather qh∈{2,3}, but qh=3 does not exist), gathering the phantom head reads out-of-bounds
        // query/gradOutput/weights and corrupts gradKey/gradValue for that kvh. The CPU reference maps
        // kvh = qh/queriesPerKV over the REAL heads only, so a kvh with no (or fewer) mapped heads must
        // accumulate only the heads that exist. qhOff increases, so break once past the last real head.
        if (qh >= numQHeads) break;
        int bqh = b * numQHeads + qh;
        int qheadBase = bqh * seqQ * headDim;
        int wheadBase = bqh * seqQ * seqK;

        for (int qi = 0; qi < seqQ; qi++) {
            int qOffset = qheadBase + qi * headDim;
            int gOffset = qOffset;
            int wOffset = wheadBase + qi * seqK;

            float weight = attentionWeights[wOffset + ki];

            float dotProduct = 0.0f;
            for (int kj = 0; kj < seqK; kj++) {
                float wj = attentionWeights[wOffset + kj];
                float gwj = 0.0f;
                for (int dd = 0; dd < headDim; dd++) {
                    gwj += gradOutput[gOffset + dd] * value[kvOffsetBase + kj * headDim + dd];
                }
                dotProduct += wj * gwj;
            }
            float gw = 0.0f;
            for (int dd = 0; dd < headDim; dd++) {
                gw += gradOutput[gOffset + dd] * value[kvOffsetBase + ki * headDim + dd];
            }
            float gradScore = weight * (gw - dotProduct) * scale;

            accV += weight * gradOutput[gOffset + d];
            accK += gradScore * query[qOffset + d];
        }
    }

    gradValue[kvOffsetBase + ki * headDim + d] = accV;
    gradKey[kvOffsetBase + ki * headDim + d] = accK;
}

extern ""C"" __global__ __launch_bounds__(256) void grouped_query_attention_backward(
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

extern ""C"" __global__ __launch_bounds__(256) void flash_attention_forward(
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

// ===========================================================================
// FUSED ROTARY POSITION EMBEDDING (RoPE) - interleaved (GPT-NeoX / LLaMA / GGML)
// One thread per (row, pair). cos/sin are [maxSeq, headDim/2], indexed by absolute position.
// ===========================================================================
extern ""C"" __global__ void rope_interleaved(
    const float* __restrict__ input,      // [rows * headDim]
    const float* __restrict__ cosCache,   // [maxSeq * halfDim]
    const float* __restrict__ sinCache,   // [maxSeq * halfDim]
    float* __restrict__ output,           // [rows * headDim]
    int rows,
    int headDim,
    int seqLen,
    int startPosition)
{
    int halfDim = headDim / 2;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= rows * halfDim) return;

    int i = g % halfDim;
    int row = g / halfDim;
    int s = row % seqLen;
    int pos = startPosition + s;
    int baseIdx = row * headDim;
    int cacheIdx = pos * halfDim + i;

    float c = cosCache[cacheIdx];
    float sn = sinCache[cacheIdx];
    float xEven = input[baseIdx + 2 * i];
    float xOdd = input[baseIdx + 2 * i + 1];
    output[baseIdx + 2 * i] = xEven * c - xOdd * sn;
    output[baseIdx + 2 * i + 1] = xEven * sn + xOdd * c;
}
" + TiledBackwardSource(32) + TiledBackwardSource(16);
        }

        /// <summary>The tile edge of the tiled backward for a head dimension: 32 up to 64, 16 up to 128, else none.</summary>
        internal static int TiledBackwardTile(int headDim) => headDim <= 64 ? 32 : headDim <= 128 ? 16 : 0;

        /// <summary>Dynamic shared memory of a tiled backward kernel: four padded [T, headDim + 1] tiles, two [T, T]
        /// tiles and two T-vectors.</summary>
        internal static int TiledBackwardSharedBytes(int tile, int headDim) =>
            (4 * tile * (headDim + 1) + 2 * tile * tile + 2 * tile) * sizeof(float);

        // The FlashAttention backward as two tiled kernels (#1528), atomic-free and deterministic: every output cell is
        // owned by one thread and accumulated in a fixed order. A block holds T keys (dK/dV) or T queries (dQ) and
        // walks the other side tile by tile, recomputing the scores and dS = P * (dP - D) * scale for the tile pair in
        // shared memory; D = dO . O per query row is reduced in-kernel with warp shuffles, so no workspace is needed.
        // Rows are padded to headDim + 1 floats so the dot products across rows do not share banks.
        private static string TiledBackwardSource(int t) => @"
extern ""C"" __global__ __launch_bounds__(256) void flash_attention_backward_kv_t" + t + @"(
    const float* __restrict__ gradOutput, const float* __restrict__ query, const float* __restrict__ key,
    const float* __restrict__ value, const float* __restrict__ output, const float* __restrict__ softmaxStats,
    float* __restrict__ gradKey, float* __restrict__ gradValue,
    int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, int isCausal,
    const float* __restrict__ attentionBias, int hasBias, int biasBatchStride)
{
    const int T = " + t + @";
    const int PER = 256 / T;
    extern __shared__ float sm[];
    const int ld = headDim + 1;
    float* Ks = sm;
    float* Vs = Ks + T * ld;
    float* Qs = Vs + T * ld;
    float* Gs = Qs + T * ld;
    float* Ps = Gs + T * ld;
    float* Ss = Ps + T * T;
    float* Ls = Ss + T * T;
    float* Ds = Ls + T;
    const int bh = blockIdx.y;
    const int k0 = blockIdx.x * T;
    const int tid = threadIdx.x;
    if (bh >= batch * numHeads || k0 >= seqK) return;
    const long kBase = (long)bh * seqK * headDim, qBase = (long)bh * seqQ * headDim;
    const long biasHead = hasBias ? (long)(bh / numHeads) * biasBatchStride + (long)(bh % numHeads) * seqQ * seqK : 0;
    const int kt = min(T, seqK - k0);
    for (int i = tid; i < T * headDim; i += 256) {
        int r = i / headDim, c = i - r * headDim;
        bool ok = r < kt;
        Ks[r * ld + c] = ok ? key[kBase + (long)(k0 + r) * headDim + c] : 0.0f;
        Vs[r * ld + c] = ok ? value[kBase + (long)(k0 + r) * headDim + c] : 0.0f;
    }
    float accK[8], accV[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { accK[j] = 0.0f; accV[j] = 0.0f; }
    // Causal (key <= query): a query before k0 sees no key of this tile.
    for (int q0 = isCausal ? k0 : 0; q0 < seqQ; q0 += T) {
        const int qt = min(T, seqQ - q0);
        __syncthreads();
        for (int i = tid; i < T * headDim; i += 256) {
            int r = i / headDim, c = i - r * headDim;
            bool ok = r < qt;
            Qs[r * ld + c] = ok ? query[qBase + (long)(q0 + r) * headDim + c] : 0.0f;
            Gs[r * ld + c] = ok ? gradOutput[qBase + (long)(q0 + r) * headDim + c] : 0.0f;
        }
        {
            int row = tid / PER, lane = tid - row * PER;
            float sum = 0.0f;
            if (row < qt)
                for (int c = lane; c < headDim; c += PER)
                    sum += gradOutput[qBase + (long)(q0 + row) * headDim + c] * output[qBase + (long)(q0 + row) * headDim + c];
            for (int off = PER / 2; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off, PER);
            if (lane == 0) { Ds[row] = sum; Ls[row] = row < qt ? softmaxStats[(long)bh * seqQ + q0 + row] : 0.0f; }
        }
        __syncthreads();
        for (int i = tid; i < T * T; i += 256) {
            int qq = i / T, kk = i - qq * T;
            float pw = 0.0f, ds = 0.0f;
            if (qq < qt && kk < kt && !(isCausal && k0 + kk > q0 + qq)) {
                float sc = 0.0f, dp = 0.0f;
                for (int c = 0; c < headDim; c++) {
                    sc += Qs[qq * ld + c] * Ks[kk * ld + c];
                    dp += Gs[qq * ld + c] * Vs[kk * ld + c];
                }
                sc *= scale;
                if (hasBias) sc += attentionBias[biasHead + (long)(q0 + qq) * seqK + k0 + kk];
                pw = expf(sc - Ls[qq]);
                ds = pw * (dp - Ds[qq]) * scale;
            }
            Ps[i] = pw;
            Ss[i] = ds;
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int i = tid + j * 256;
            if (i < T * headDim) {
                int kk = i / headDim, c = i - kk * headDim;
                float av = 0.0f, ak = 0.0f;
                for (int qq = 0; qq < qt; qq++) {
                    av += Ps[qq * T + kk] * Gs[qq * ld + c];
                    ak += Ss[qq * T + kk] * Qs[qq * ld + c];
                }
                accV[j] += av;
                accK[j] += ak;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int i = tid + j * 256;
        if (i < T * headDim) {
            int kk = i / headDim, c = i - kk * headDim;
            if (kk < kt) {
                gradValue[kBase + (long)(k0 + kk) * headDim + c] = accV[j];
                gradKey[kBase + (long)(k0 + kk) * headDim + c] = accK[j];
            }
        }
    }
}

extern ""C"" __global__ __launch_bounds__(256) void flash_attention_backward_q_t" + t + @"(
    const float* __restrict__ gradOutput, const float* __restrict__ query, const float* __restrict__ key,
    const float* __restrict__ value, const float* __restrict__ output, const float* __restrict__ softmaxStats,
    float* __restrict__ gradQuery,
    int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, int isCausal,
    const float* __restrict__ attentionBias, int hasBias, int biasBatchStride)
{
    const int T = " + t + @";
    const int PER = 256 / T;
    extern __shared__ float sm[];
    const int ld = headDim + 1;
    float* Ks = sm;
    float* Vs = Ks + T * ld;
    float* Qs = Vs + T * ld;
    float* Gs = Qs + T * ld;
    float* Ss = Gs + T * ld;
    float* Ls = Ss + 2 * T * T;
    float* Ds = Ls + T;
    const int bh = blockIdx.y;
    const int q0 = blockIdx.x * T;
    const int tid = threadIdx.x;
    if (bh >= batch * numHeads || q0 >= seqQ) return;
    const long kBase = (long)bh * seqK * headDim, qBase = (long)bh * seqQ * headDim;
    const long biasHead = hasBias ? (long)(bh / numHeads) * biasBatchStride + (long)(bh % numHeads) * seqQ * seqK : 0;
    const int qt = min(T, seqQ - q0);
    for (int i = tid; i < T * headDim; i += 256) {
        int r = i / headDim, c = i - r * headDim;
        bool ok = r < qt;
        Qs[r * ld + c] = ok ? query[qBase + (long)(q0 + r) * headDim + c] : 0.0f;
        Gs[r * ld + c] = ok ? gradOutput[qBase + (long)(q0 + r) * headDim + c] : 0.0f;
    }
    {
        int row = tid / PER, lane = tid - row * PER;
        float sum = 0.0f;
        if (row < qt)
            for (int c = lane; c < headDim; c += PER)
                sum += gradOutput[qBase + (long)(q0 + row) * headDim + c] * output[qBase + (long)(q0 + row) * headDim + c];
        for (int off = PER / 2; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off, PER);
        if (lane == 0) { Ds[row] = sum; Ls[row] = row < qt ? softmaxStats[(long)bh * seqQ + q0 + row] : 0.0f; }
    }
    float accQ[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) accQ[j] = 0.0f;
    // Causal (key <= query): no key past the tile's last query.
    const int kEnd = isCausal ? min(seqK, q0 + qt) : seqK;
    for (int k0 = 0; k0 < kEnd; k0 += T) {
        const int kt = min(T, seqK - k0);
        __syncthreads();
        for (int i = tid; i < T * headDim; i += 256) {
            int r = i / headDim, c = i - r * headDim;
            bool ok = r < kt;
            Ks[r * ld + c] = ok ? key[kBase + (long)(k0 + r) * headDim + c] : 0.0f;
            Vs[r * ld + c] = ok ? value[kBase + (long)(k0 + r) * headDim + c] : 0.0f;
        }
        __syncthreads();
        for (int i = tid; i < T * T; i += 256) {
            int qq = i / T, kk = i - qq * T;
            float ds = 0.0f;
            if (qq < qt && kk < kt && !(isCausal && k0 + kk > q0 + qq)) {
                float sc = 0.0f, dp = 0.0f;
                for (int c = 0; c < headDim; c++) {
                    sc += Qs[qq * ld + c] * Ks[kk * ld + c];
                    dp += Gs[qq * ld + c] * Vs[kk * ld + c];
                }
                sc *= scale;
                if (hasBias) sc += attentionBias[biasHead + (long)(q0 + qq) * seqK + k0 + kk];
                float pw = expf(sc - Ls[qq]);
                ds = pw * (dp - Ds[qq]) * scale;
            }
            Ss[i] = ds;
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int i = tid + j * 256;
            if (i < T * headDim) {
                int qq = i / headDim, c = i - qq * headDim;
                float aq = 0.0f;
                for (int kk = 0; kk < kt; kk++) aq += Ss[qq * T + kk] * Ks[kk * ld + c];
                accQ[j] += aq;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int i = tid + j * 256;
        if (i < T * headDim) {
            int qq = i / headDim, c = i - qq * headDim;
            if (qq < qt) gradQuery[qBase + (long)(q0 + qq) * headDim + c] = accQ[j];
        }
    }
}
";

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "scaled_dot_product_attention",
                "flash_attention_v2",
                "flash_attention_backward",
                "flash_attention_backward_gradq_deterministic",
                "flash_attention_backward_gradkv_deterministic",
                "flash_attention_backward_kv_t32",
                "flash_attention_backward_q_t32",
                "flash_attention_backward_kv_t16",
                "flash_attention_backward_q_t16",
                "grouped_query_attention",
                "grouped_query_attention_backward",
                "grouped_query_attention_backward_gradq_deterministic",
                "grouped_query_attention_backward_gradkv_deterministic",
                "flash_attention_forward",
                "rope_interleaved"
            };
        }
    }
}
