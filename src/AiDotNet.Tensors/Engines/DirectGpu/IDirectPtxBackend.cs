// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Hand-written PTX kernels (#1154): NVIDIA-only by construction. Every route is a Try that a backend without them
/// answers false (and the Enabled flags false), so the engine takes its portable kernel instead. The routes take FP16
/// operands, hence the FP16 contract underneath.
/// </summary>
internal interface IDirectPtxBackend : IFp16ActivationBackend
{
    /// <summary>The direct-PTX IsDirectPtxConvolutionEnabled route.</summary>
    bool IsDirectPtxConvolutionEnabled { get; }

    /// <summary>The direct-PTX IsDirectPtxAttentionEnabled route.</summary>
    bool IsDirectPtxAttentionEnabled { get; }

    /// <summary>The direct-PTX DirectPtxLastError route.</summary>
    string? DirectPtxLastError { get; }

    /// <summary>The direct-PTX TryDirectPtxFusedConv2DBiasRelu route.</summary>
    bool TryDirectPtxFusedConv2DBiasRelu(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, IGpuBuffer output, DirectPtxConvolutionShape shape);

    /// <summary>The direct-PTX TryDirectPtxRegBlockedConv2DBiasRelu route.</summary>
    bool TryDirectPtxRegBlockedConv2DBiasRelu(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, IGpuBuffer output, int batch, int inChannels, int height, int width, int outChannels);

    /// <summary>The direct-PTX TryDirectPtxDepthwiseConv2D3x3 route.</summary>
    bool TryDirectPtxDepthwiseConv2D3x3(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output, DirectPtxConvolutionShape shape);

    /// <summary>The direct-PTX TryDirectPtxOnlineAttentionFamily route.</summary>
    bool TryDirectPtxOnlineAttentionFamily(IGpuBuffer queryHalf, IGpuBuffer keyHalf, IGpuBuffer valueHalf, IGpuBuffer outputFloat, IGpuBuffer? softmaxStatsFloat, int batch, int queryHeads, int keyValueHeads, int querySequence, int keyValueSequence, float scale, bool isCausal, bool emitSoftmaxStats = true, int causalQueryOffset = 0);

    /// <summary>
    /// The FP16 flash-attention forward (#1549): any sequence lengths, head dimension 64 or 128, causal or not,
    /// grouped-query heads; FP32 output and the per-row log-sum-exp (natural log) of the scaled scores. Runs under the
    /// direct-PTX attention opt-in, or without it when <paramref name="reducedPrecisionRequested"/> (the caller already
    /// chose FP16: an autocast scope or FP16-resident activations).
    /// </summary>
    bool TryDirectPtxFlashAttention(IGpuBuffer queryHalf, IGpuBuffer keyHalf, IGpuBuffer valueHalf, IGpuBuffer outputFloat, IGpuBuffer softmaxStatsFloat, int batch, int queryHeads, int keyValueHeads, int querySequence, int keyValueSequence, int headDimension, float scale, bool isCausal, int causalQueryOffset = 0, bool reducedPrecisionRequested = false);

    /// <summary>The direct-PTX TryDirectPtxGroupNormSwishUnit64 route.</summary>
    bool TryDirectPtxGroupNormSwishUnit64(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta, int batch, int groups, int channels, int spatialSize, float epsilon);

    /// <summary>The direct-PTX TryDirectPtxAddGroupNormUnit64 route.</summary>
    bool TryDirectPtxAddGroupNormUnit64(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta, int batch, int groups, int channels, int spatialSize, float epsilon);

    /// <summary>
    /// GroupNorm backward through its declared program (#890 phase 4a), at any shape. <c>rstd</c> is
    /// 1/sqrt(var + eps), the slot the declared forward writes, not the variance. False declines, and the caller
    /// composes the gradient from elementwise kernels.
    /// </summary>
    bool TryDeclaredGroupNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer rstd, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int batch, int groups, int channels, int spatialSize);

    /// <summary>The direct-PTX TryDirectPtxGroupNormBackwardUnit64 route.</summary>
    bool TryDirectPtxGroupNormBackwardUnit64(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer variance, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int batch, int groups, int channels, int spatialSize, float epsilon);
}
