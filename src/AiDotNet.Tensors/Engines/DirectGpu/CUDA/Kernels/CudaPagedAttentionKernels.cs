// Copyright (c) AiDotNet. All rights reserved.
// GPU paged-attention kernels (P1) for CUDA (nvRTC).

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA paged-attention decode kernel. K/V live in a physical block pool
    /// <c>[maxBlocks, blockSize, heads, headDim]</c> (matching the CPU <c>PagedKVCache</c>); a block table
    /// maps logical block index -&gt; physical block id. Computes single-query
    /// <c>out[h] = softmax(scale·Q[h]·K[.,h])·V[.,h]</c> reading K/V through the block table.
    /// Online-softmax single pass, one thread per head, headDim ≤ 256 (correctness-first baseline).
    /// </summary>
    internal static class CudaPagedAttentionKernels
    {
        public static string[] GetKernelNames() => new[] { "paged_attention_decode" };

        public static string GetSource() => @"
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
";
    }
}
