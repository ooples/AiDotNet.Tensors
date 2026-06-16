// Copyright (c) AiDotNet. All rights reserved.
// GPU kernels for attention operations including standard attention, FlashAttention, and GQA.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for attention mechanisms used in transformer architectures.
    /// Includes ScaledDotProductAttention, FlashAttention, and GroupedQueryAttention.
    /// </summary>
    internal static class AttentionKernels
    {
        /// <summary>
        /// Gets all attention kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// COMPATIBILITY DEFINITIONS
// OpenCL compatibility - define NEGATIVE_INFINITY as explicit float constant
// (avoids issues with -INFINITY macro on some OpenCL drivers)
// ===========================================================================
#define NEGATIVE_INFINITY (-3.402823466e+38f)

// Atomic add for float (not natively supported in OpenCL, uses CAS loop)
inline void atomic_add_float(__global float* addr, float val) {
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr,
                                      expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

// ===========================================================================
// SCALED DOT-PRODUCT ATTENTION
// ===========================================================================

// Standard scaled dot-product attention
// Output: attention(Q, K, V) = softmax(Q @ K^T / scale) @ V
__kernel void scaled_dot_product_attention(
    __global const float* query,      // [batch * heads * seqQ * headDim]
    __global const float* key,        // [batch * heads * seqK * headDim]
    __global const float* value,      // [batch * heads * seqK * headDim]
    __global float* output,           // [batch * heads * seqQ * headDim]
    __global float* attentionWeights, // [batch * heads * seqQ * seqK] (optional)
    __global const int* mask,         // [seqQ * seqK] (optional, 0=masked, 1=valid)
    const int batch,
    const int numHeads,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale,
    const int isCausal,
    const int hasMask,
    const int storeWeights)
{
    const int bh = get_global_id(1);  // batch * head index
    const int qi = get_global_id(0);  // query position

    if (bh >= batch * numHeads || qi >= seqQ) return;

    const int b = bh / numHeads;
    const int h = bh % numHeads;

    // Offsets
    const int qOffset = bh * seqQ * headDim + qi * headDim;
    const int kOffset = bh * seqK * headDim;
    const int vOffset = bh * seqK * headDim;
    const int oOffset = bh * seqQ * headDim + qi * headDim;
    const int wOffset = bh * seqQ * seqK + qi * seqK;

    // Compute attention scores and find max for numerical stability
    float maxScore = NEGATIVE_INFINITY;
    for (int ki = 0; ki < seqK; ki++) {
        // Causal mask
        if (isCausal && ki > qi) continue;

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;
        maxScore = fmax(maxScore, score);
    }

    // Compute softmax
    float sumExp = 0.0f;
    float scores[1024]; // Temporary storage (adjust size as needed)
    for (int ki = 0; ki < seqK; ki++) {
        if (isCausal && ki > qi) {
            scores[ki] = 0.0f;
            continue;
        }

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;

        float expScore = exp(score - maxScore);
        scores[ki] = expScore;
        sumExp += expScore;
    }

    // Normalize and compute output
    for (int d = 0; d < headDim; d++) {
        float val = 0.0f;
        for (int ki = 0; ki < seqK; ki++) {
            float weight = scores[ki] / sumExp;
            val += weight * value[vOffset + ki * headDim + d];
        }
        output[oOffset + d] = val;
    }

    // Store attention weights if requested
    if (storeWeights) {
        for (int ki = 0; ki < seqK; ki++) {
            attentionWeights[wOffset + ki] = scores[ki] / sumExp;
        }
    }
}

// ===========================================================================
// FLASH ATTENTION V2
// Memory-efficient attention using online softmax and tiling
// ===========================================================================

// Block size for tiling (adjust based on GPU shared memory)
#define FLASH_BLOCK_SIZE 64

__kernel void flash_attention_v2(
    __global const float* query,          // [batch * heads * seqQ * headDim]
    __global const float* key,            // [batch * heads * seqK * headDim]
    __global const float* value,          // [batch * heads * seqK * headDim]
    __global float* output,               // [batch * heads * seqQ * headDim]
    __global float* softmaxStats,         // [batch * heads * seqQ] (log-sum-exp)
    const int batch,
    const int numHeads,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale,
    const int isCausal,
    __global const float* attentionBias,  // optional bias (pass dummy buffer if unused)
    const int hasBias,
    const int biasBatchStride)
{
    const int bh = get_global_id(1);  // batch * head index
    const int qi = get_global_id(0);  // query position

    if (bh >= batch * numHeads || qi >= seqQ) return;

    // Offsets
    const int qOffset = bh * seqQ * headDim + qi * headDim;
    const int kOffset = bh * seqK * headDim;
    const int vOffset = bh * seqK * headDim;
    const int oOffset = bh * seqQ * headDim + qi * headDim;
    const int sOffset = bh * seqQ + qi;

    // Bias offset base
    int biasBase = 0;
    if (hasBias) {
        const int b = bh / numHeads;
        const int h = bh % numHeads;
        biasBase = b * biasBatchStride + h * seqQ * seqK + qi * seqK;
    }

    // Initialize accumulators for online softmax
    float rowMax = NEGATIVE_INFINITY;
    float rowSum = 0.0f;

    // Initialize output to zero
    float outAcc[128]; // Max headDim supported
    for (int d = 0; d < headDim; d++) {
        outAcc[d] = 0.0f;
    }

    // Process key-value pairs in blocks
    for (int kvBlockStart = 0; kvBlockStart < seqK; kvBlockStart += FLASH_BLOCK_SIZE) {
        int kvBlockEnd = min(kvBlockStart + FLASH_BLOCK_SIZE, seqK);

        // Skip block if causal and all keys are after query
        if (isCausal && kvBlockStart > qi) continue;

        // Find max in this block
        float blockMax = NEGATIVE_INFINITY;
        for (int ki = kvBlockStart; ki < kvBlockEnd; ki++) {
            if (isCausal && ki > qi) continue;

            float score = 0.0f;
            for (int d = 0; d < headDim; d++) {
                score += query[qOffset + d] * key[kOffset + ki * headDim + d];
            }
            score *= scale;
            if (hasBias) score += attentionBias[biasBase + ki];
            blockMax = fmax(blockMax, score);
        }

        // New global max
        float newMax = fmax(rowMax, blockMax);

        // Rescale factor for previous accumulator
        float rescale = exp(rowMax - newMax);
        float newSum = rowSum * rescale;

        // Rescale output accumulator
        for (int d = 0; d < headDim; d++) {
            outAcc[d] *= rescale;
        }

        // Add contributions from this block
        for (int ki = kvBlockStart; ki < kvBlockEnd; ki++) {
            if (isCausal && ki > qi) continue;

            float score = 0.0f;
            for (int d = 0; d < headDim; d++) {
                score += query[qOffset + d] * key[kOffset + ki * headDim + d];
            }
            score *= scale;
            if (hasBias) score += attentionBias[biasBase + ki];

            float expScore = exp(score - newMax);
            newSum += expScore;

            // Accumulate weighted value
            for (int d = 0; d < headDim; d++) {
                outAcc[d] += expScore * value[vOffset + ki * headDim + d];
            }
        }

        rowMax = newMax;
        rowSum = newSum;
    }

    // Final normalization and write output
    float invSum = 1.0f / rowSum;
    for (int d = 0; d < headDim; d++) {
        output[oOffset + d] = outAcc[d] * invSum;
    }

    // Store log-sum-exp for backward pass
    softmaxStats[sOffset] = rowMax + log(rowSum);
}

// FlashAttention backward — bit-deterministic variants (issue #382).
// The original flash_attention_backward parallelizes one work-item per (bh, qi) and
// uses atomic_add_float to accumulate into gradKey[ki] / gradValue[ki] across the
// many qi threads that touch the same ki. That is FP-non-deterministic across runs.
//
// Deterministic split into two atomic-free kernels:
//   gradq: parallelize per (bh, qi) — writes gradQuery (each thread's own qi).
//          Identical work to the atomic kernel but DOES NOT write gradKey/gradValue.
//   gradkv: parallelize per (bh, ki, d) — each work-item owns one (ki, d) cell of
//           gradKey AND gradValue, scans qi in fixed ascending order, and
//           accumulates contributions. No atomics.
// Both kernels recompute the forward scores from the saved logsumexp; total work
// doubles in deterministic mode — documented DeterministicMode trade-off.
__kernel void flash_attention_backward_gradq_deterministic(
    __global const float* gradOutput,
    __global const float* query,
    __global const float* key,
    __global const float* value,
    __global const float* output,
    __global const float* softmaxStats,
    __global float* gradQuery,
    const int batch,
    const int numHeads,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale,
    const int isCausal,
    __global const float* attentionBias,
    const int hasBias,
    const int biasBatchStride)
{
    const int bh = get_global_id(1);
    const int qi = get_global_id(0);

    if (bh >= batch * numHeads || qi >= seqQ) return;

    const int qOffset = bh * seqQ * headDim + qi * headDim;
    const int kOffset = bh * seqK * headDim;
    const int vOffset = bh * seqK * headDim;
    const int gOffset = bh * seqQ * headDim + qi * headDim;
    const int sOffset = bh * seqQ + qi;

    int biasBase = 0;
    if (hasBias) {
        const int b = bh / numHeads;
        const int h = bh % numHeads;
        biasBase = b * biasBatchStride + h * seqQ * seqK + qi * seqK;
    }

    float logsumexp = softmaxStats[sOffset];

    float doO = 0.0f;
    for (int d = 0; d < headDim; d++) {
        doO += gradOutput[gOffset + d] * output[qOffset + d];
    }

    for (int ki = 0; ki < seqK; ki++) {
        if (isCausal && ki > qi) continue;

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;
        if (hasBias) score += attentionBias[biasBase + ki];

        float attnWeight = exp(score - logsumexp);

        float doV = 0.0f;
        for (int d = 0; d < headDim; d++) {
            doV += gradOutput[gOffset + d] * value[vOffset + ki * headDim + d];
        }

        float dS = attnWeight * (doV - doO) * scale;

        // Only gradQuery is written here — gradKey/gradValue are produced by the
        // companion gradkv kernel.
        for (int d = 0; d < headDim; d++) {
            gradQuery[qOffset + d] += dS * key[kOffset + ki * headDim + d];
        }
    }
}

__kernel void flash_attention_backward_gradkv_deterministic(
    __global const float* gradOutput,
    __global const float* query,
    __global const float* key,
    __global const float* value,
    __global const float* output,
    __global const float* softmaxStats,
    __global float* gradKey,
    __global float* gradValue,
    const int batch,
    const int numHeads,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale,
    const int isCausal,
    __global const float* attentionBias,
    const int hasBias,
    const int biasBatchStride)
{
    const int bh = get_global_id(2);
    const int ki = get_global_id(1);
    const int d = get_global_id(0);
    if (bh >= batch * numHeads || ki >= seqK || d >= headDim) return;

    const int kOffset = bh * seqK * headDim;
    const int vOffset = bh * seqK * headDim;
    const int qBase = bh * seqQ * headDim;
    const int gBase = bh * seqQ * headDim;
    const int sBase = bh * seqQ;

    float accV = 0.0f;
    float accK = 0.0f;

    int b = bh / numHeads;
    int h = bh % numHeads;
    int biasHeadBase = hasBias ? (b * biasBatchStride + h * seqQ * seqK) : 0;

    for (int qi = 0; qi < seqQ; qi++) {
        if (isCausal && ki > qi) continue;

        const int qOffset = qBase + qi * headDim;
        const int gOffset = gBase + qi * headDim;

        float logsumexp = softmaxStats[sBase + qi];
        float score = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            score += query[qOffset + dd] * key[kOffset + ki * headDim + dd];
        }
        score *= scale;
        if (hasBias) score += attentionBias[biasHeadBase + qi * seqK + ki];

        float attnWeight = exp(score - logsumexp);

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

    gradValue[vOffset + ki * headDim + d] += accV;
    gradKey[kOffset + ki * headDim + d] += accK;
}

// FlashAttention backward pass with recomputation
__kernel void flash_attention_backward(
    __global const float* gradOutput,     // [batch * heads * seqQ * headDim]
    __global const float* query,          // [batch * heads * seqQ * headDim]
    __global const float* key,            // [batch * heads * seqK * headDim]
    __global const float* value,          // [batch * heads * seqK * headDim]
    __global const float* output,         // [batch * heads * seqQ * headDim]
    __global const float* softmaxStats,   // [batch * heads * seqQ]
    __global float* gradQuery,            // [batch * heads * seqQ * headDim]
    __global float* gradKey,              // [batch * heads * seqK * headDim]
    __global float* gradValue,            // [batch * heads * seqK * headDim]
    const int batch,
    const int numHeads,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale,
    const int isCausal,
    __global const float* attentionBias,
    const int hasBias,
    const int biasBatchStride)
{
    const int bh = get_global_id(1);
    const int qi = get_global_id(0);

    if (bh >= batch * numHeads || qi >= seqQ) return;

    // Offsets
    const int qOffset = bh * seqQ * headDim + qi * headDim;
    const int kOffset = bh * seqK * headDim;
    const int vOffset = bh * seqK * headDim;
    const int gOffset = bh * seqQ * headDim + qi * headDim;
    const int sOffset = bh * seqQ + qi;

    int biasBase = 0;
    if (hasBias) {
        const int b = bh / numHeads;
        const int h = bh % numHeads;
        biasBase = b * biasBatchStride + h * seqQ * seqK + qi * seqK;
    }

    float logsumexp = softmaxStats[sOffset];

    // Compute dO @ O (for softmax backward)
    float doO = 0.0f;
    for (int d = 0; d < headDim; d++) {
        doO += gradOutput[gOffset + d] * output[qOffset + d];
    }

    // Process each key position
    for (int ki = 0; ki < seqK; ki++) {
        if (isCausal && ki > qi) continue;

        // Recompute attention score (must match forward pass exactly)
        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;
        if (hasBias) score += attentionBias[biasBase + ki];

        // Recompute attention weight
        float attnWeight = exp(score - logsumexp);

        // Gradient w.r.t. V: attnWeight * gradOutput
        for (int d = 0; d < headDim; d++) {
            atomic_add_float(&gradValue[vOffset + ki * headDim + d],
                             attnWeight * gradOutput[gOffset + d]);
        }

        // Compute dO @ v
        float doV = 0.0f;
        for (int d = 0; d < headDim; d++) {
            doV += gradOutput[gOffset + d] * value[vOffset + ki * headDim + d];
        }

        // dS = attnWeight * (doV - doO) * scale
        float dS = attnWeight * (doV - doO) * scale;

        // Gradient w.r.t. Q: dS * K
        for (int d = 0; d < headDim; d++) {
            gradQuery[qOffset + d] += dS * key[kOffset + ki * headDim + d];
        }

        // Gradient w.r.t. K: dS * Q
        for (int d = 0; d < headDim; d++) {
            atomic_add_float(&gradKey[kOffset + ki * headDim + d],
                             dS * query[qOffset + d]);
        }
    }
}

// ===========================================================================
// GROUPED QUERY ATTENTION (GQA)
// Multiple query heads share the same key-value head
// ===========================================================================

__kernel void grouped_query_attention(
    __global const float* query,          // [batch * numQHeads * seqQ * headDim]
    __global const float* key,            // [batch * numKVHeads * seqK * headDim]
    __global const float* value,          // [batch * numKVHeads * seqK * headDim]
    __global float* output,               // [batch * numQHeads * seqQ * headDim]
    __global float* attentionWeights,     // [batch * numQHeads * seqQ * seqK] (optional)
    const int batch,
    const int numQHeads,
    const int numKVHeads,
    const int queriesPerKV,               // numQHeads / numKVHeads
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale,
    const int isCausal,
    const int storeWeights)
{
    const int d = get_global_id(0);       // head dimension
    const int qi = get_global_id(1);      // query position
    const int bqh = get_global_id(2);     // batch * query head

    if (d >= headDim || qi >= seqQ || bqh >= batch * numQHeads) return;

    const int b = bqh / numQHeads;
    const int qh = bqh % numQHeads;
    const int kvh = qh / queriesPerKV;    // Which KV head this query uses

    // Offsets
    const int qOffset = bqh * seqQ * headDim + qi * headDim;
    const int kOffset = (b * numKVHeads + kvh) * seqK * headDim;
    const int vOffset = (b * numKVHeads + kvh) * seqK * headDim;
    const int oOffset = bqh * seqQ * headDim + qi * headDim;
    const int wOffset = bqh * seqQ * seqK + qi * seqK;

    // Compute attention scores and softmax (only thread d=0 does this)
    if (d == 0) {
        float maxScore = NEGATIVE_INFINITY;
        float scores[1024];

        // Compute scores and find max
        for (int ki = 0; ki < seqK; ki++) {
            if (isCausal && ki > qi) {
                scores[ki] = NEGATIVE_INFINITY;
                continue;
            }

            float score = 0.0f;
            for (int dd = 0; dd < headDim; dd++) {
                score += query[qOffset + dd] * key[kOffset + ki * headDim + dd];
            }
            score *= scale;
            scores[ki] = score;
            maxScore = fmax(maxScore, score);
        }

        // Softmax
        float sumExp = 0.0f;
        for (int ki = 0; ki < seqK; ki++) {
            float expScore = exp(scores[ki] - maxScore);
            scores[ki] = expScore;
            sumExp += expScore;
        }

        // Compute output and optionally store weights
        for (int dd = 0; dd < headDim; dd++) {
            float val = 0.0f;
            for (int ki = 0; ki < seqK; ki++) {
                float weight = scores[ki] / sumExp;
                val += weight * value[vOffset + ki * headDim + dd];
                if (storeWeights && dd == 0) {
                    attentionWeights[wOffset + ki] = weight;
                }
            }
            output[oOffset + dd] = val;
        }
    }
}

// GQA backward — bit-deterministic variants (issue #382).
// Same split strategy as flash_attention_backward_*_deterministic above:
//   gradq: per (bqh, qi) — writes gradQuery only, no atomics
//   gradkv: per (b, kvh, ki, d) — iterates the queriesPerKV query heads + all qi,
//           accumulates gradKey and gradValue with fixed (qh, qi) traversal order
__kernel void grouped_query_attention_backward_gradq_deterministic(
    __global const float* gradOutput,
    __global const float* query,
    __global const float* key,
    __global const float* value,
    __global const float* attentionWeights,
    __global float* gradQuery,
    const int batch,
    const int numQHeads,
    const int numKVHeads,
    const int queriesPerKV,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale)
{
    const int d = get_global_id(0);
    const int qi = get_global_id(1);
    const int bqh = get_global_id(2);

    if (d >= headDim || qi >= seqQ || bqh >= batch * numQHeads) return;

    const int b = bqh / numQHeads;
    const int qh = bqh % numQHeads;
    const int kvh = qh / queriesPerKV;

    const int qOffset = bqh * seqQ * headDim + qi * headDim;
    const int kOffset = (b * numKVHeads + kvh) * seqK * headDim;
    const int vOffset = (b * numKVHeads + kvh) * seqK * headDim;
    const int gOffset = bqh * seqQ * headDim + qi * headDim;
    const int wOffset = bqh * seqQ * seqK + qi * seqK;

    // Each work-item owns one output cell gradQuery[qOffset + d].
    // dotProduct depends only on (qi, bqh) and is recomputed locally per
    // work-item — this trades headDim-x extra arithmetic for true
    // parallelism over d (the prior `if (d == 0)` gating wasted
    // headDim - 1 threads doing nothing). Writes are bit-deterministic:
    // each cell has a unique owner and uses `=`, not `+=`.
    float dotProduct = 0.0f;
    for (int ki = 0; ki < seqK; ki++) {
        float w = attentionWeights[wOffset + ki];
        float gw = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            gw += gradOutput[gOffset + dd] * value[vOffset + ki * headDim + dd];
        }
        dotProduct += w * gw;
    }

    float acc = 0.0f;
    for (int ki = 0; ki < seqK; ki++) {
        float w = attentionWeights[wOffset + ki];
        float gw = 0.0f;
        for (int dd = 0; dd < headDim; dd++) {
            gw += gradOutput[gOffset + dd] * value[vOffset + ki * headDim + dd];
        }
        float gradScore = w * (gw - dotProduct) * scale;
        acc += gradScore * key[kOffset + ki * headDim + d];
    }
    gradQuery[qOffset + d] = acc;
}

__kernel void grouped_query_attention_backward_gradkv_deterministic(
    __global const float* gradOutput,
    __global const float* query,
    __global const float* key,
    __global const float* value,
    __global const float* attentionWeights,
    __global float* gradKey,
    __global float* gradValue,
    const int batch,
    const int numQHeads,
    const int numKVHeads,
    const int queriesPerKV,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale)
{
    // One work-item per (b, kvh, ki, d) cell of gradKey/gradValue
    const int d = get_global_id(0);
    const int ki = get_global_id(1);
    const int bkvh = get_global_id(2);
    if (d >= headDim || ki >= seqK || bkvh >= batch * numKVHeads) return;

    const int b = bkvh / numKVHeads;
    const int kvh = bkvh % numKVHeads;
    const int kvOffsetBase = bkvh * seqK * headDim;

    float accV = 0.0f;
    float accK = 0.0f;

    for (int qhOff = 0; qhOff < queriesPerKV; qhOff++) {
        int qh = kvh * queriesPerKV + qhOff;
        // The inverse mapping qh = kvh*queriesPerKV + qhOff overshoots numQHeads when the config is
        // not perfectly divisible (the parity harness drives Q=K=3 heads with queriesPerKV=2). The CPU
        // maps each EXISTING qh forward via kvh = qh/queriesPerKV, so a qh past numQHeads simply does
        // not exist — skip it (qh increases monotonically, so break). Without this the kernel read
        // out-of-bounds query/gradOutput and produced garbage gradients (#628).
        if (qh >= numQHeads) break;
        int bqh = b * numQHeads + qh;
        int qheadBase = bqh * seqQ * headDim;
        int wheadBase = bqh * seqQ * seqK;

        for (int qi = 0; qi < seqQ; qi++) {
            int qOffset = qheadBase + qi * headDim;
            int gOffset = qOffset;
            int wOffset = wheadBase + qi * seqK;

            float weight = attentionWeights[wOffset + ki];

            // Reconstruct gradScore as in the atomic kernel.
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

    // Each (b, kvh, ki, d) cell has exactly one owning work-item that computed the FULL accumulation
    // locally, so write with `=` (not `+=`) — `+=` read the un-zeroed output buffer and added garbage.
    gradValue[kvOffsetBase + ki * headDim + d] = accV;
    gradKey[kvOffsetBase + ki * headDim + d] = accK;
}

__kernel void grouped_query_attention_backward(
    __global const float* gradOutput,     // [batch * numQHeads * seqQ * headDim]
    __global const float* query,          // [batch * numQHeads * seqQ * headDim]
    __global const float* key,            // [batch * numKVHeads * seqK * headDim]
    __global const float* value,          // [batch * numKVHeads * seqK * headDim]
    __global const float* attentionWeights, // [batch * numQHeads * seqQ * seqK]
    __global float* gradQuery,            // [batch * numQHeads * seqQ * headDim]
    __global float* gradKey,              // [batch * numKVHeads * seqK * headDim]
    __global float* gradValue,            // [batch * numKVHeads * seqK * headDim]
    const int batch,
    const int numQHeads,
    const int numKVHeads,
    const int queriesPerKV,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale)
{
    const int d = get_global_id(0);
    const int qi = get_global_id(1);
    const int bqh = get_global_id(2);

    if (d >= headDim || qi >= seqQ || bqh >= batch * numQHeads) return;

    const int b = bqh / numQHeads;
    const int qh = bqh % numQHeads;
    const int kvh = qh / queriesPerKV;

    // Offsets
    const int qOffset = bqh * seqQ * headDim + qi * headDim;
    const int kOffset = (b * numKVHeads + kvh) * seqK * headDim;
    const int vOffset = (b * numKVHeads + kvh) * seqK * headDim;
    const int gOffset = bqh * seqQ * headDim + qi * headDim;
    const int wOffset = bqh * seqQ * seqK + qi * seqK;

    // Only thread d=0 computes gradients for this position
    if (d == 0) {
        // Compute gradients w.r.t. attention weights: gradOutput @ V^T
        float gradWeights[1024];
        for (int ki = 0; ki < seqK; ki++) {
            float sum = 0.0f;
            for (int dd = 0; dd < headDim; dd++) {
                sum += gradOutput[gOffset + dd] * value[vOffset + ki * headDim + dd];
            }
            gradWeights[ki] = sum;
        }

        // Softmax backward: gradScores = weights * (gradWeights - dot(weights, gradWeights))
        float dotProduct = 0.0f;
        for (int ki = 0; ki < seqK; ki++) {
            dotProduct += attentionWeights[wOffset + ki] * gradWeights[ki];
        }

        // Gradient w.r.t. V and compute gradScores
        for (int ki = 0; ki < seqK; ki++) {
            float weight = attentionWeights[wOffset + ki];
            float gradScore = weight * (gradWeights[ki] - dotProduct) * scale;

            // Gradient w.r.t. V (accumulated across query heads)
            for (int dd = 0; dd < headDim; dd++) {
                atomic_add_float(&gradValue[vOffset + ki * headDim + dd],
                                 weight * gradOutput[gOffset + dd]);
            }

            // Gradient w.r.t. Q
            for (int dd = 0; dd < headDim; dd++) {
                gradQuery[qOffset + dd] += gradScore * key[kOffset + ki * headDim + dd];
            }

            // Gradient w.r.t. K (accumulated across query heads)
            for (int dd = 0; dd < headDim; dd++) {
                atomic_add_float(&gradKey[kOffset + ki * headDim + dd],
                                 gradScore * query[qOffset + dd]);
            }
        }
    }
}

// ===========================================================================
// FLASH ATTENTION FORWARD (Original tiled version for compatibility)
// ===========================================================================

__kernel void flash_attention_forward(
    __global const float* query,
    __global const float* key,
    __global const float* value,
    __global float* output,
    __global const int* mask,
    const int batch,
    const int numHeads,
    const int seqLen,
    const int headDim,
    const float scale,
    const int isCausal,
    const int hasMask)
{
    const int bh = get_global_id(1);
    const int qi = get_global_id(0);

    if (bh >= batch * numHeads || qi >= seqLen) return;

    // Same as flash_attention_v2 but without storing stats
    const int qOffset = bh * seqLen * headDim + qi * headDim;
    const int kOffset = bh * seqLen * headDim;
    const int vOffset = bh * seqLen * headDim;
    const int oOffset = bh * seqLen * headDim + qi * headDim;

    float rowMax = NEGATIVE_INFINITY;
    float rowSum = 0.0f;
    float outAcc[128];
    for (int d = 0; d < headDim; d++) outAcc[d] = 0.0f;

    for (int ki = 0; ki < seqLen; ki++) {
        if (isCausal && ki > qi) continue;

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;

        float newMax = fmax(rowMax, score);
        float rescale = exp(rowMax - newMax);
        rowSum = rowSum * rescale + exp(score - newMax);

        for (int d = 0; d < headDim; d++) {
            outAcc[d] = outAcc[d] * rescale + exp(score - newMax) * value[vOffset + ki * headDim + d];
        }
        rowMax = newMax;
    }

    float invSum = 1.0f / rowSum;
    for (int d = 0; d < headDim; d++) {
        output[oOffset + d] = outAcc[d] * invSum;
    }
}
";
        }

        /// <summary>
        /// Gets the names of all kernels defined in this source.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new[]
            {
                "scaled_dot_product_attention",
                "flash_attention_v2",
                "flash_attention_backward",
                "flash_attention_backward_gradq_deterministic",
                "flash_attention_backward_gradkv_deterministic",
                "grouped_query_attention",
                "grouped_query_attention_backward",
                "grouped_query_attention_backward_gradq_deterministic",
                "grouped_query_attention_backward_gradkv_deterministic",
                "flash_attention_forward"
            };
        }
    }
}
