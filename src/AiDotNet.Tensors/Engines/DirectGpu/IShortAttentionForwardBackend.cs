// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: the forward of head-interleaved attention (query, key, value and output all
/// [batch, seq, heads * headDim]) for short sequences, with nothing saved for a backward. The fused attention kernels
/// (<see cref="IFusedAttentionBackend"/>) are tiled for long sequences and lose to the composed form below 64M score
/// elements; the composed form permutes every operand into [batch, heads, seq, dim], runs two batched GEMMs around a
/// separate scale and softmax, and permutes the result back - six passes over memory for a ViT block's 65-token
/// attention, whose arithmetic is a few microseconds.
/// </summary>
[BackendSpecificRoute("OpenCL short-sequence attention forward; every backend composes it from permutes, batched GEMMs and a softmax",
    nameof(IDirectGpuBackend.Permute), nameof(IDirectGpuBackend.BatchedGemm), nameof(IDirectGpuBackend.Softmax))]
internal interface IShortAttentionForwardBackend
{
    /// <summary>Whether <see cref="TryShortAttentionForward"/> takes this head width.</summary>
    bool SupportsShortAttention(int headDim);

    /// <summary>output = softmax(scale * Q K^T [+ causal mask]) V per batch item and head, reading and writing the
    /// interleaved layout in place. Causal masking hides key j from query i when j > i + causalOffset. False (nothing
    /// launched) when unavailable.</summary>
    bool TryShortAttentionForward(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value, IGpuBuffer output,
        int batch, int heads, int seqQ, int seqK, int headDim, float scale, bool causal, int causalOffset);
}