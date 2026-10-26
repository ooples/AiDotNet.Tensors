// Copyright (c) AiDotNet. All rights reserved.
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The direct-PTX kernels live in CudaBackend.DirectPtx*.cs; this maps the contract onto them.
public sealed partial class CudaBackend : IDirectPtxBackend
{
    bool IDirectPtxBackend.IsDirectPtxConvolutionEnabled => IsDirectPtxConvolutionEnabled;

    bool IDirectPtxBackend.IsDirectPtxAttentionEnabled => IsDirectPtxAttentionEnabled;

    string? IDirectPtxBackend.DirectPtxLastError => DirectPtxLastError;

    bool IDirectPtxBackend.TryDirectPtxFusedConv2DBiasRelu(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, IGpuBuffer output, DirectPtxConvolutionShape shape)
        => TryDirectPtxFusedConv2DBiasRelu(input, weights, bias, output, shape);

    bool IDirectPtxBackend.TryDirectPtxRegBlockedConv2DBiasRelu(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, IGpuBuffer output, int batch, int inChannels, int height, int width, int outChannels)
        => TryDirectPtxRegBlockedConv2DBiasRelu(input, weights, bias, output, batch, inChannels, height, width, outChannels);

    bool IDirectPtxBackend.TryDirectPtxDepthwiseConv2D3x3(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output, DirectPtxConvolutionShape shape)
        => TryDirectPtxDepthwiseConv2D3x3(input, weights, output, shape);

    bool IDirectPtxBackend.TryDirectPtxOnlineAttentionFamily(IGpuBuffer queryHalf, IGpuBuffer keyHalf, IGpuBuffer valueHalf, IGpuBuffer outputFloat, IGpuBuffer? softmaxStatsFloat, int batch, int queryHeads, int keyValueHeads, int querySequence, int keyValueSequence, float scale, bool isCausal, bool emitSoftmaxStats, int causalQueryOffset)
        => TryDirectPtxOnlineAttentionFamily(queryHalf, keyHalf, valueHalf, outputFloat, softmaxStatsFloat, batch, queryHeads, keyValueHeads, querySequence, keyValueSequence, scale, isCausal, emitSoftmaxStats, causalQueryOffset);

    bool IDirectPtxBackend.TryDirectPtxFlashAttention(IGpuBuffer queryHalf, IGpuBuffer keyHalf, IGpuBuffer valueHalf, IGpuBuffer outputFloat, IGpuBuffer softmaxStatsFloat, int batch, int queryHeads, int keyValueHeads, int querySequence, int keyValueSequence, int headDimension, float scale, bool isCausal, int causalQueryOffset)
        => TryDirectPtxFlashAttention(queryHalf, keyHalf, valueHalf, outputFloat, softmaxStatsFloat, batch, queryHeads, keyValueHeads, querySequence, keyValueSequence, headDimension, scale, isCausal, causalQueryOffset);

    bool IDirectPtxBackend.TryDirectPtxGroupNormSwishUnit64(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta, int batch, int groups, int channels, int spatialSize, float epsilon)
        => TryDirectPtxGroupNormSwishUnit64(input, output, gamma, beta, batch, groups, channels, spatialSize, epsilon);

    bool IDirectPtxBackend.TryDirectPtxAddGroupNormUnit64(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta, int batch, int groups, int channels, int spatialSize, float epsilon)
        => TryDirectPtxAddGroupNormUnit64(left, right, output, gamma, beta, batch, groups, channels, spatialSize, epsilon);

    bool IDirectPtxBackend.TryDeclaredGroupNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer rstd, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int batch, int groups, int channels, int spatialSize)
        => TryDeclaredGroupNormBackward(gradOutput, input, gamma, mean, rstd, gradInput, gradGamma, gradBeta, batch, groups, channels, spatialSize);

    bool IDirectPtxBackend.TryDirectPtxGroupNormBackwardUnit64(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer variance, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int batch, int groups, int channels, int spatialSize, float epsilon)
        => TryDirectPtxGroupNormBackwardUnit64(gradOutput, input, gamma, mean, variance, gradInput, gradGamma, gradBeta, batch, groups, channels, spatialSize, epsilon);
}
