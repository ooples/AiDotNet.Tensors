// Copyright (c) AiDotNet. All rights reserved.
// Shared host-boundary validation for the DirectGpu attention / quantized-GEMM entrypoints.
// The kernels assume these preconditions (fixed-size per-head accumulators, in-kernel integer
// divides, contiguous buffer reads); violating them corrupts device memory or divides by zero
// inside the kernel rather than throwing, so every public dispatch validates up front.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    internal static class GpuKernelGuards
    {
        /// <summary>Kernel-side fixed accumulator size (float acc[256]); headDim must not exceed it.</summary>
        public const int MaxHeadDim = 256;

        /// <summary>Validates paged-attention / decode shape params (non-GQA).</summary>
        public static void Attention(int heads, int headDim, int blockSize, int seqOrQueries, string op)
        {
            if (heads <= 0)
                throw new ArgumentOutOfRangeException(nameof(heads), $"{op}: heads must be > 0 (got {heads}).");
            if (headDim <= 0 || headDim > MaxHeadDim)
                throw new ArgumentOutOfRangeException(nameof(headDim), $"{op}: headDim must be in [1, {MaxHeadDim}] (got {headDim}).");
            if (blockSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(blockSize), $"{op}: blockSize must be > 0 (got {blockSize}).");
            if (seqOrQueries <= 0)
                throw new ArgumentOutOfRangeException(nameof(seqOrQueries), $"{op}: sequence/query count must be > 0 (got {seqOrQueries}).");
        }

        /// <summary>Validates the grouped-query ratio: 1 &lt;= kvHeads &lt;= heads and heads % kvHeads == 0.
        /// The kernels compute <c>kvHead = h / (heads / kvHeads)</c>, so a bad ratio divides by zero or
        /// reads K/V out of bounds.</summary>
        public static void Gqa(int heads, int kvHeads, string op)
        {
            if (kvHeads <= 0 || kvHeads > heads || heads % kvHeads != 0)
                throw new ArgumentOutOfRangeException(nameof(kvHeads),
                    $"{op}: kvHeads must satisfy 1 <= kvHeads <= heads and heads % kvHeads == 0 (heads={heads}, kvHeads={kvHeads}).");
        }

        /// <summary>Validates fused decode-attention shape params (contiguous K/V, GQA-aware).</summary>
        public static void FlashDecode(int heads, int kvHeads, int headDim, int seqLen, string op)
        {
            if (heads <= 0)
                throw new ArgumentOutOfRangeException(nameof(heads), $"{op}: heads must be > 0 (got {heads}).");
            if (headDim <= 0 || headDim > MaxHeadDim)
                throw new ArgumentOutOfRangeException(nameof(headDim), $"{op}: headDim must be in [1, {MaxHeadDim}] (got {headDim}).");
            if (seqLen <= 0)
                throw new ArgumentOutOfRangeException(nameof(seqLen), $"{op}: seqLen must be > 0 (got {seqLen}).");
            Gqa(heads, kvHeads, op);
        }

        /// <summary>Validates weight-only dequant-GEMM shape/quant params.</summary>
        public static void DequantGemm(int M, int K, int N, int groupSize, int scaleCount, string op)
        {
            if (M <= 0 || K <= 0 || N <= 0)
                throw new ArgumentOutOfRangeException(nameof(M), $"{op}: M, K, N must all be > 0 (got M={M}, K={K}, N={N}).");
            if (groupSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(groupSize), $"{op}: groupSize must be > 0 (kernel divides by it) (got {groupSize}).");
            if (scaleCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(scaleCount), $"{op}: scaleCount must be > 0 (got {scaleCount}).");
        }

        /// <summary>Rejects a device buffer that is too small for the elements a kernel will read.</summary>
        public static void Capacity(IGpuBuffer buffer, long requiredElements, string bufferName, string op)
        {
            if (buffer is null)
                throw new ArgumentNullException(bufferName, $"{op}: {bufferName} must not be null.");
            if (buffer.Size < requiredElements)
                throw new ArgumentException(
                    $"{op}: {bufferName} holds {buffer.Size} elements but the kernel needs {requiredElements}.", bufferName);
        }
    }
}
