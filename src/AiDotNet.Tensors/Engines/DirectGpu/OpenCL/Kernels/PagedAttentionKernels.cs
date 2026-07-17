// Copyright (c) AiDotNet. All rights reserved.
// GPU paged-attention kernels (P1): attention that gathers K/V through a block table (vLLM-style),
// so the KV cache need not be contiguous. Works on ALL .NET versions incl. .NET Framework 4.6.2.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// Paged-attention decode kernel. K/V live in a physical block pool laid out as
    /// <c>[maxBlocks, blockSize, heads, headDim]</c> (matching the CPU <c>PagedKVCache</c>); a per-sequence
    /// block table maps logical block index -&gt; physical block id. For a single query token this computes
    /// <c>out[h] = softmax(scale · Q[h]·K[.,h]) · V[.,h]</c> over the sequence, reading K/V via the block
    /// table — the core mechanism that lets vLLM pack many sequences without KV fragmentation.
    /// </summary>
    /// <remarks>
    /// Online-softmax (FlashAttention-style) single pass; one work-item per head; correctness-first
    /// baseline (headDim ≤ 256). Validated against a CPU attention oracle on the materialized K/V.
    /// </remarks>
    internal static class PagedAttentionKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "paged_attention_decode", "paged_attention_prefill",
            "paged_attention_decode_gqa", "paged_attention_prefill_gqa",
        };

        public static string GetSource()
        {
            return @"
#define PA_MAX_HEAD_DIM 256
// Single-query paged attention. Block pool: [maxBlocks, blockSize, heads, headDim].
__kernel void paged_attention_decode(
    __global const float* q,          // [heads*headDim]
    __global const float* kcache,     // [maxBlocks*blockSize*heads*headDim]
    __global const float* vcache,     // same layout as kcache
    __global const int*   blockTable, // [ceil(seqLen/blockSize)] physical block ids
    __global float*       outbuf,     // [heads*headDim]
    const int heads, const int headDim, const int blockSize, const int seqLen, const float scale)
{
    int h = get_global_id(0);
    if (h >= heads) return;

    float acc[PA_MAX_HEAD_DIM];
    for (int d = 0; d < headDim; ++d) acc[d] = 0.0f;

    float m = -INFINITY;   // running max logit
    float l = 0.0f;        // running softmax denominator

    for (int t = 0; t < seqLen; ++t) {
        int blk = blockTable[t / blockSize];
        int pos = t % blockSize;
        long base = (((long)blk * blockSize + pos) * heads + h) * headDim;

        float dot = 0.0f;
        for (int d = 0; d < headDim; ++d) dot += q[h * headDim + d] * kcache[base + d];
        float logit = dot * scale;

        float new_m = fmax(m, logit);
        float corr = exp(m - new_m);
        float p = exp(logit - new_m);
        l = l * corr + p;
        for (int d = 0; d < headDim; ++d) acc[d] = acc[d] * corr + p * vcache[base + d];
        m = new_m;
    }

    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = 0; d < headDim; ++d) outbuf[h * headDim + d] = acc[d] * inv;
}

// Prefill / multi-query paged attention with causal masking. Query qi (logical position
// startPos+qi) attends to key positions 0..(startPos+qi). One work-item per (query, head).
// q/out are [numQueries, heads, headDim]; K/V pool + block table cover positions 0..startPos+numQueries-1.
__kernel void paged_attention_prefill(
    __global const float* q,          // [numQueries*heads*headDim]
    __global const float* kcache,     // [maxBlocks*blockSize*heads*headDim]
    __global const float* vcache,     // same layout
    __global const int*   blockTable, // physical block ids
    __global float*       outbuf,     // [numQueries*heads*headDim]
    const int heads, const int headDim, const int blockSize,
    const int numQueries, const int startPos, const float scale)
{
    int gid = get_global_id(0);
    int total = numQueries * heads;
    if (gid >= total) return;
    int qi = gid / heads;
    int h  = gid % heads;
    int keyLen = startPos + qi + 1; // causal: attend to keys 0..(startPos+qi)
    int qbase = (qi * heads + h) * headDim;

    float acc[PA_MAX_HEAD_DIM];
    for (int d = 0; d < headDim; ++d) acc[d] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int t = 0; t < keyLen; ++t) {
        int blk = blockTable[t / blockSize];
        int pos = t % blockSize;
        long kbase = (((long)blk * blockSize + pos) * heads + h) * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; ++d) dot += q[qbase + d] * kcache[kbase + d];
        float logit = dot * scale;
        float new_m = fmax(m, logit);
        float corr = exp(m - new_m);
        float p = exp(logit - new_m);
        l = l * corr + p;
        for (int d = 0; d < headDim; ++d) acc[d] = acc[d] * corr + p * vcache[kbase + d];
        m = new_m;
    }

    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = 0; d < headDim; ++d) outbuf[qbase + d] = acc[d] * inv;
}

// GQA decode: query head h shares KV head kvHead = h / (heads/kvHeads). K/V pool laid out with
// kvHeads: [maxBlocks, blockSize, kvHeads, headDim]. Q/out use full heads. (kvHeads==heads => MHA.)
__kernel void paged_attention_decode_gqa(
    __global const float* q, __global const float* kcache, __global const float* vcache,
    __global const int* blockTable, __global float* outbuf,
    const int heads, const int kvHeads, const int headDim, const int blockSize, const int seqLen, const float scale)
{
    int h = get_global_id(0);
    if (h >= heads) return;
    int kvHead = h / (heads / kvHeads);
    float acc[PA_MAX_HEAD_DIM];
    for (int d = 0; d < headDim; ++d) acc[d] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    for (int t = 0; t < seqLen; ++t) {
        int blk = blockTable[t / blockSize];
        int pos = t % blockSize;
        long kbase = (((long)blk * blockSize + pos) * kvHeads + kvHead) * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; ++d) dot += q[h * headDim + d] * kcache[kbase + d];
        float logit = dot * scale;
        float new_m = fmax(m, logit);
        float corr = exp(m - new_m);
        float p = exp(logit - new_m);
        l = l * corr + p;
        for (int d = 0; d < headDim; ++d) acc[d] = acc[d] * corr + p * vcache[kbase + d];
        m = new_m;
    }
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = 0; d < headDim; ++d) outbuf[h * headDim + d] = acc[d] * inv;
}

// GQA prefill: causal multi-query with grouped KV heads.
__kernel void paged_attention_prefill_gqa(
    __global const float* q, __global const float* kcache, __global const float* vcache,
    __global const int* blockTable, __global float* outbuf,
    const int heads, const int kvHeads, const int headDim, const int blockSize, const int numQueries, const int startPos, const float scale)
{
    int gid = get_global_id(0);
    int total = numQueries * heads;
    if (gid >= total) return;
    int qi = gid / heads;
    int h  = gid % heads;
    int kvHead = h / (heads / kvHeads);
    int keyLen = startPos + qi + 1;
    int qbase = (qi * heads + h) * headDim;
    float acc[PA_MAX_HEAD_DIM];
    for (int d = 0; d < headDim; ++d) acc[d] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    for (int t = 0; t < keyLen; ++t) {
        int blk = blockTable[t / blockSize];
        int pos = t % blockSize;
        long kbase = (((long)blk * blockSize + pos) * kvHeads + kvHead) * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; ++d) dot += q[qbase + d] * kcache[kbase + d];
        float logit = dot * scale;
        float new_m = fmax(m, logit);
        float corr = exp(m - new_m);
        float p = exp(logit - new_m);
        l = l * corr + p;
        for (int d = 0; d < headDim; ++d) acc[d] = acc[d] * corr + p * vcache[kbase + d];
        m = new_m;
    }
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = 0; d < headDim; ++d) outbuf[qbase + d] = acc[d] * inv;
}
";
        }
    }
}
