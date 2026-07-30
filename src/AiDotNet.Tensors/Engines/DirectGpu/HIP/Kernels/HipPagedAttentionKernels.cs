// Copyright (c) AiDotNet. All rights reserved.
// GPU paged-attention kernels (P1) for HIP (hiprtc).

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels
{
    /// <summary>
    /// HIP paged-attention decode kernel. K/V live in a physical block pool
    /// <c>[maxBlocks, blockSize, heads, headDim]</c>; a block table maps logical -&gt; physical block id.
    /// Computes single-query <c>out[h] = softmax(scale·Q[h]·K[.,h])·V[.,h]</c> reading K/V through the
    /// block table. Online-softmax single pass, one thread per head, headDim ≤ 256.
    /// </summary>
    internal static class HipPagedAttentionKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "paged_attention_decode", "paged_attention_prefill",
            "paged_attention_decode_gqa", "paged_attention_prefill_gqa",
        };

        public static string GetSource() => @"
// hiprtc: device intrinsics built-in.
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif
#define PA_MAX_HEAD_DIM 256
extern ""C"" __global__ void paged_attention_decode(
    const float* q, const float* kcache, const float* vcache, const int* blockTable, float* outbuf,
    int heads, int headDim, int blockSize, int seqLen, float scale)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads) return;

    float acc[PA_MAX_HEAD_DIM];
    for (int d = 0; d < headDim; ++d) acc[d] = 0.0f;

    float m = -INFINITY;
    float l = 0.0f;

    for (int t = 0; t < seqLen; ++t) {
        int blk = blockTable[t / blockSize];
        int pos = t % blockSize;
        long base = (((long)blk * blockSize + pos) * heads + h) * headDim;

        float dot = 0.0f;
        for (int d = 0; d < headDim; ++d) dot += q[h * headDim + d] * kcache[base + d];
        float logit = dot * scale;

        float new_m = fmaxf(m, logit);
        float corr = expf(m - new_m);
        float p = expf(logit - new_m);
        l = l * corr + p;
        for (int d = 0; d < headDim; ++d) acc[d] = acc[d] * corr + p * vcache[base + d];
        m = new_m;
    }

    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = 0; d < headDim; ++d) outbuf[h * headDim + d] = acc[d] * inv;
}

// Prefill / multi-query paged attention with causal masking. One thread per (query, head).
extern ""C"" __global__ void paged_attention_prefill(
    const float* q, const float* kcache, const float* vcache, const int* blockTable, float* outbuf,
    int heads, int headDim, int blockSize, int numQueries, int startPos, float scale)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numQueries * heads;
    if (gid >= total) return;
    int qi = gid / heads;
    int h  = gid % heads;
    int keyLen = startPos + qi + 1;
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
        float new_m = fmaxf(m, logit);
        float corr = expf(m - new_m);
        float p = expf(logit - new_m);
        l = l * corr + p;
        for (int d = 0; d < headDim; ++d) acc[d] = acc[d] * corr + p * vcache[kbase + d];
        m = new_m;
    }

    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = 0; d < headDim; ++d) outbuf[qbase + d] = acc[d] * inv;
}

// GQA decode: query head h shares KV head h/(heads/kvHeads). K/V pool [maxBlocks,blockSize,kvHeads,headDim].
extern ""C"" __global__ void paged_attention_decode_gqa(
    const float* q, const float* kcache, const float* vcache, const int* blockTable, float* outbuf,
    int heads, int kvHeads, int headDim, int blockSize, int seqLen, float scale)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
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
        float new_m = fmaxf(m, logit);
        float corr = expf(m - new_m);
        float p = expf(logit - new_m);
        l = l * corr + p;
        for (int d = 0; d < headDim; ++d) acc[d] = acc[d] * corr + p * vcache[kbase + d];
        m = new_m;
    }
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = 0; d < headDim; ++d) outbuf[h * headDim + d] = acc[d] * inv;
}

// GQA prefill: causal multi-query with grouped KV heads.
extern ""C"" __global__ void paged_attention_prefill_gqa(
    const float* q, const float* kcache, const float* vcache, const int* blockTable, float* outbuf,
    int heads, int kvHeads, int headDim, int blockSize, int numQueries, int startPos, float scale)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
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
        float new_m = fmaxf(m, logit);
        float corr = expf(m - new_m);
        float p = expf(logit - new_m);
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
