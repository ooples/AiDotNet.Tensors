// Copyright (c) AiDotNet. All rights reserved.
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// No PTX on this backend: every direct-PTX route declines, so the engine runs its portable kernels.
public sealed partial class HipBackend : IDirectPtxBackend
{
    bool IDirectPtxBackend.IsDirectPtxConvolutionEnabled => false;

    bool IDirectPtxBackend.IsDirectPtxAttentionEnabled => false;

    string? IDirectPtxBackend.DirectPtxLastError => null;

    bool IDirectPtxBackend.TryDirectPtxFusedConv2DBiasRelu(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, IGpuBuffer output, DirectPtxConvolutionShape shape)
        => false;

    bool IDirectPtxBackend.TryDirectPtxRegBlockedConv2DBiasRelu(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, IGpuBuffer output, int batch, int inChannels, int height, int width, int outChannels)
        => false;

    bool IDirectPtxBackend.TryDirectPtxDepthwiseConv2D3x3(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output, DirectPtxConvolutionShape shape)
        => false;

    bool IDirectPtxBackend.TryDirectPtxOnlineAttentionFamily(IGpuBuffer queryHalf, IGpuBuffer keyHalf, IGpuBuffer valueHalf, IGpuBuffer outputFloat, IGpuBuffer? softmaxStatsFloat, int batch, int queryHeads, int keyValueHeads, int querySequence, int keyValueSequence, float scale, bool isCausal, bool emitSoftmaxStats, int causalQueryOffset)
        => false;

    bool IDirectPtxBackend.TryDirectPtxFlashAttention(IGpuBuffer queryHalf, IGpuBuffer keyHalf, IGpuBuffer valueHalf, IGpuBuffer outputFloat, IGpuBuffer softmaxStatsFloat, int batch, int queryHeads, int keyValueHeads, int querySequence, int keyValueSequence, int headDimension, float scale, bool isCausal, int causalQueryOffset, bool reducedPrecisionRequested)
        => false;

    bool IDirectPtxBackend.TryDirectPtxGroupNormSwishUnit64(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta, int batch, int groups, int channels, int spatialSize, float epsilon)
        => false;

    bool IDirectPtxBackend.TryDirectPtxAddGroupNormUnit64(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta, int batch, int groups, int channels, int spatialSize, float epsilon)
        => false;

    bool IDirectPtxBackend.TryDirectPtxGroupNormBackwardUnit64(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer variance, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int batch, int groups, int channels, int spatialSize, float epsilon)
        => false;

    bool IDirectPtxBackend.TryDeclaredGroupNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer rstd, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int batch, int groups, int channels, int spatialSize)
        => false;
}
