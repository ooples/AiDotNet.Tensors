// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Channels-last activations (IChannelsLastBackend) through PortableChannelsLast: the layout change is the channel
// epilogue module's kernel; convolutions and BatchNorm run this backend's NCHW kernels between layout changes.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IChannelsLastBackend
{
    bool IChannelsLastBackend.PrefersChannelsLast => false;   // NCHW convolutions: the layout would only add passes

    bool IChannelsLastBackend.AcceptsChannelsLast(int channels, int outChannels)
        => channels > 0 && outChannels > 0 && ChannelEpilogueReady;

    bool IChannelsLastBackend.TryConvolutionChannelsLast(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch,
        int channels, int height, int width, int outChannels, int kernelH, int kernelW, int outHeight, int outWidth,
        int padH, int padW, int strideH, int strideW, int dilationH, int dilationW, bool inputNhwc, bool outputNhwc)
        => ChannelEpilogueReady && PortableChannelsLast.TryConvolution(this, LaunchChannelEpilogue, pass, a, b, c, batch,
            channels, height, width, outChannels, kernelH, kernelW, outHeight, outWidth, padH, padW, strideH, strideW,
            dilationH, dilationW, inputNhwc, outputNhwc);

    bool IChannelsLastBackend.TryBatchNormChannelsLast(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma,
        IGpuBuffer beta, IGpuBuffer saveMean, IGpuBuffer saveVariance, int rows, int channels, float epsilon, int act,
        float slope)
        => ChannelEpilogueReady && PortableChannelsLast.TryBatchNorm(this, LaunchChannelEpilogue, input, output, gamma, beta,
            saveMean, saveVariance, rows, channels, epsilon, act, slope);

    bool IChannelsLastBackend.TryBatchNormChannelsLastBackward(IGpuBuffer gradOutput, IGpuBuffer? maskSource,
        IGpuBuffer? gradOutput2, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveMean, IGpuBuffer saveVariance,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int rows, int channels, float epsilon,
        int maskMode, int act, float slope, bool accumulate)
        => ChannelEpilogueReady && PortableChannelsLast.TryBatchNormBackward(this, LaunchChannelEpilogue, gradOutput,
            maskSource, gradOutput2, input, gamma, saveMean, saveVariance, gradInput, gradGamma, gradBeta, rows, channels,
            epsilon, maskMode, act, slope, accumulate);

    bool IChannelsLastBackend.TryChangeLayout(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int spatial,
        bool toChannelsLast)
        => ChannelEpilogueReady && PortableChannelEpilogue.TryChangeLayout(LaunchChannelEpilogue, input, output, batch,
            channels, spatial, toChannelsLast);
}
