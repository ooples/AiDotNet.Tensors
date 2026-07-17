// Copyright (c) AiDotNet. All rights reserved.
// GPU fused decode-attention kernels (P2): FlashDecoding-style single-query attention that splits the
// KV sequence across work-items and combines the partials with an online-softmax reduction. This is the
// autoregressive-decode hot path — one query token attending to the whole cached sequence — where the
// P1 one-thread-per-head decode leaves the GPU idle. Works on ALL .NET versions incl. .NET Framework.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// Fused decode attention (FlashDecoding). For a single query token per head it computes
    /// <c>out[h] = softmax(scale · Q[h]·K[.,h]) · V[.,h]</c> over contiguous K/V of shape
    /// <c>[seqLen, kvHeads, headDim]</c>. The sequence is partitioned into <c>splits</c> chunks; each
    /// (head, split) work-item streams its chunk with an online-softmax pass into partial
    /// <c>(m, l, acc)</c> stats, then a per-head reduction merges the partials. GQA is supported via
    /// <c>kvHead = h / (heads/kvHeads)</c> (pass <c>kvHeads == heads</c> for standard MHA).
    /// </summary>
    /// <remarks>
    /// Correctness-first baseline (headDim ≤ 256): parallelism is over the KV sequence, not tensor cores.
    /// Validated against a standard-attention CPU oracle. Tensor-core (WMMA/MFMA) acceleration is a
    /// backend-specific perf follow-up that does not change the result.
    /// </remarks>
    internal static class FlashDecodeKernels
    {
        public static string[] GetKernelNames() => new[] { "flash_decode_partial", "flash_decode_reduce" };

        public static string GetSource()
        {
            return @"
#define FD_MAX_HEAD_DIM 256
// Pass 1: each work-item handles one (head, split), streaming its key range into partial stats.
__kernel void flash_decode_partial(
    __global const float* q,          // [heads*headDim]
    __global const float* k,          // [seqLen*kvHeads*headDim]
    __global const float* v,          // same layout as k
    __global float*       partialM,   // [heads*splits] running max per (head,split)
    __global float*       partialL,   // [heads*splits] softmax denom per (head,split)
    __global float*       partialAcc, // [heads*splits*headDim] weighted V per (head,split)
    const int heads, const int kvHeads, const int headDim,
    const int seqLen, const int splits, const int splitLen, const float scale)
{
    int gid = get_global_id(0);
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
        float new_m = fmax(m, logit);
        float corr = exp(m - new_m);
        float p = exp(logit - new_m);
        l = l * corr + p;
        for (int d = 0; d < headDim; ++d) acc[d] = acc[d] * corr + p * v[kb + d];
        m = new_m;
    }

    partialM[gid] = m;
    partialL[gid] = l;
    long accBase = (long)gid * headDim;
    for (int d = 0; d < headDim; ++d) partialAcc[accBase + d] = acc[d];
}

// Pass 2: per head, merge the `splits` partial stats with the online-softmax combine rule.
__kernel void flash_decode_reduce(
    __global const float* partialM,   // [heads*splits]
    __global const float* partialL,   // [heads*splits]
    __global const float* partialAcc, // [heads*splits*headDim]
    __global float*       outbuf,     // [heads*headDim]
    const int heads, const int headDim, const int splits)
{
    int h = get_global_id(0);
    if (h >= heads) return;

    // Global running max across this head's splits (empty splits carry m = -INFINITY, l = 0).
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
        if (ls <= 0.0f) continue; // empty split contributes nothing
        float w = exp(partialM[idx] - m);
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
}
