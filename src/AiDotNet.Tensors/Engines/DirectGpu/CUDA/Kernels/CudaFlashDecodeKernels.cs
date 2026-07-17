// Copyright (c) AiDotNet. All rights reserved.
// GPU fused decode-attention kernels (P2, FlashDecoding) for CUDA (nvRTC). Single-query attention over
// contiguous K/V, sequence split across threads and merged by an online-softmax reduction.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA fused decode attention (FlashDecoding). For one query token per head over contiguous K/V
    /// <c>[seqLen, kvHeads, headDim]</c>: pass 1 computes per-(head, split) online-softmax partials,
    /// pass 2 merges them per head. GQA via <c>kvHead = h/(heads/kvHeads)</c> (kvHeads==heads => MHA).
    /// Correctness-first baseline (headDim ≤ 256); matches a standard-attention CPU oracle.
    /// </summary>
    internal static class CudaFlashDecodeKernels
    {
        public static string[] GetKernelNames() => new[] { "flash_decode_partial", "flash_decode_reduce" };

        public static string GetSource() => @"
#define FD_MAX_HEAD_DIM 256
extern ""C"" __global__ void flash_decode_partial(
    const float* q, const float* k, const float* v,
    float* partialM, float* partialL, float* partialAcc,
    int heads, int kvHeads, int headDim, int seqLen, int splits, int splitLen, float scale)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * splits;
    if (gid >= total) return;
    int h = gid / splits;
    int s = gid % splits;
    int kvHead = h / (heads / kvHeads);
    int start = s * splitLen;
    int end = start + splitLen;
    if (end > seqLen) end = seqLen;

    float acc[FD_MAX_HEAD_DIM];
    for (int d = 0; d < headDim; ++d) acc[d] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    for (int t = start; t < end; ++t) {
        long kb = ((long)t * kvHeads + kvHead) * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; ++d) dot += q[h * headDim + d] * k[kb + d];
        float logit = dot * scale;
        float new_m = fmaxf(m, logit);
        float corr = expf(m - new_m);
        float p = expf(logit - new_m);
        l = l * corr + p;
        for (int d = 0; d < headDim; ++d) acc[d] = acc[d] * corr + p * v[kb + d];
        m = new_m;
    }
    partialM[gid] = m;
    partialL[gid] = l;
    long accBase = (long)gid * headDim;
    for (int d = 0; d < headDim; ++d) partialAcc[accBase + d] = acc[d];
}

extern ""C"" __global__ void flash_decode_reduce(
    const float* partialM, const float* partialL, const float* partialAcc,
    float* outbuf, int heads, int headDim, int splits)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads) return;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        float ms = partialM[h * splits + s];
        if (ms > m) m = ms;
    }
    float acc[FD_MAX_HEAD_DIM];
    for (int d = 0; d < headDim; ++d) acc[d] = 0.0f;
    float l = 0.0f;
    for (int s = 0; s < splits; ++s) {
        int idx = h * splits + s;
        float ls = partialL[idx];
        if (ls <= 0.0f) continue;
        float w = expf(partialM[idx] - m);
        l += ls * w;
        long accBase = (long)idx * headDim;
        for (int d = 0; d < headDim; ++d) acc[d] += partialAcc[accBase + d] * w;
    }
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = 0; d < headDim; ++d) outbuf[h * headDim + d] = acc[d] * inv;
}
";
    }
}
