// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IChannelsLastBackend"/> on HIP, Metal, OpenCL, Vulkan and WebGPU: the layout change is
/// <see cref="ChannelEpilogueKernelTemplate.Layout"/>; the convolution passes and the BatchNorm change their channels-last
/// operands to NCHW, run the backend's own NCHW kernels (the same values) and change the results back. These backends'
/// convolutions are NCHW, so the layout only adds passes there (<see cref="IChannelsLastBackend.PrefersChannelsLast"/> is
/// false and a plan keeps NCHW); the entries exist so a channels-last tensor is never a dead end on them.
/// </summary>
internal static class PortableChannelsLast
{
    // An NCHW copy of a channels-last operand (owned: disposed by the caller), or the operand itself.
    private static IGpuBuffer ToNchw(IDirectGpuBackend backend, PortableChannelEpilogue.Launcher launch, IGpuBuffer source,
        bool channelsLast, int batch, int channels, int spatial, out bool owned)
    {
        owned = channelsLast;
        if (!channelsLast) return source;
        var copy = backend.AllocateBuffer(batch * channels * spatial);
        try
        {
            if (!PortableChannelEpilogue.TryChangeLayout(launch, source, copy, batch, channels, spatial, toChannelsLast: false))
                throw new System.InvalidOperationException("The channels-last layout change did not run.");
            return copy;
        }
        catch
        {
            copy.Dispose();
            throw;
        }
    }

    /// <summary>As <see cref="IChannelsLastBackend.TryConvolutionChannelsLast"/>.</summary>
    internal static bool TryConvolution(IDirectGpuBackend backend, PortableChannelEpilogue.Launcher launch, int pass,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int channels, int height, int width, int outChannels,
        int kernelH, int kernelW, int outHeight, int outWidth, int padH, int padW, int strideH, int strideW, int dilationH,
        int dilationW, bool inputNhwc, bool outputNhwc)
    {
        int inSpatial = height * width, outSpatial = outHeight * outWidth;
        if (pass is < 0 or > 2 || batch <= 0 || channels <= 0 || outChannels <= 0 || inSpatial <= 0 || outSpatial <= 0)
            return false;
        IGpuBuffer? ownedA = null, ownedB = null, staged = null;
        try
        {
            switch (pass)
            {
                case 0:
                {
                    var x = ToNchw(backend, launch, a, inputNhwc, batch, channels, inSpatial, out bool ox);
                    if (ox) ownedA = x;
                    var y = outputNhwc ? staged = backend.AllocateBuffer(batch * outChannels * outSpatial) : c;
                    backend.Conv2D(x, b, y, batch, channels, height, width, outChannels, outHeight, outWidth, kernelH,
                        kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
                    return !outputNhwc || PortableChannelEpilogue.TryChangeLayout(launch, y, c, batch, outChannels,
                        outSpatial, toChannelsLast: true);
                }
                case 1:
                {
                    var dy = ToNchw(backend, launch, b, outputNhwc, batch, outChannels, outSpatial, out bool ody);
                    if (ody) ownedB = dy;
                    var dx = inputNhwc ? staged = backend.AllocateBuffer(batch * channels * inSpatial) : c;
                    backend.Conv2DBackwardInput(dy, a, dx, batch, channels, height, width, outChannels, outHeight,
                        outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
                    return !inputNhwc || PortableChannelEpilogue.TryChangeLayout(launch, dx, c, batch, channels, inSpatial,
                        toChannelsLast: true);
                }
                default:
                {
                    var x = ToNchw(backend, launch, a, inputNhwc, batch, channels, inSpatial, out bool ox);
                    if (ox) ownedA = x;
                    var dy = ToNchw(backend, launch, b, outputNhwc, batch, outChannels, outSpatial, out bool ody);
                    if (ody) ownedB = dy;
                    backend.Conv2DBackwardKernel(x, dy, c, batch, channels, height, width, outChannels, outHeight,
                        outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
                    return true;
                }
            }
        }
        finally
        {
            ownedA?.Dispose();
            ownedB?.Dispose();
            staged?.Dispose();
        }
    }

    /// <summary>As <see cref="IChannelsLastBackend.TryBatchNormChannelsLast"/>: the [rows, C] activation as one NCHW
    /// sample of C channels over <paramref name="rows"/> positions (the statistics run over the rows either way).</summary>
    internal static bool TryBatchNorm(IDirectGpuBackend backend, PortableChannelEpilogue.Launcher launch,
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer saveMean,
        IGpuBuffer saveVariance, int rows, int channels, float epsilon, int act, float slope)
    {
        if (rows <= 0 || channels <= 0 || act is < 0 or > 2 || (act == 2 && !ChannelEpilogueActivation.IsSlope(slope))
            || backend is not IBatchNormVarianceBackend variance || !variance.AcceptsTrainingShape(1, channels, rows))
            return false;
        using var x = backend.AllocateBuffer(rows * channels);
        using var y = backend.AllocateBuffer(rows * channels);
        if (!PortableChannelEpilogue.TryChangeLayout(launch, input, x, 1, channels, rows, toChannelsLast: false))
            return false;
        bool ran = act == 2
            ? backend is IBatchNormLeakyReluBackend leaky
              && leaky.TryBatchNormLeakyReluTraining(x, y, gamma, beta, saveMean, saveVariance, 1, channels, rows, epsilon,
                  slope)
            : variance.TryBatchNormTrainingWithVariance(x, y, gamma, beta, saveMean, saveVariance, 1, channels, rows,
                epsilon, relu: act == 1);
        return ran && PortableChannelEpilogue.TryChangeLayout(launch, y, output, 1, channels, rows, toChannelsLast: true);
    }

    /// <summary>As <see cref="IChannelsLastBackend.TryBatchNormChannelsLastBackward"/>; the mask from the output
    /// (<paramref name="maskMode"/> 1) or none (0). Rebuilding it from the input (2) is refused here.</summary>
    internal static bool TryBatchNormBackward(IDirectGpuBackend backend, PortableChannelEpilogue.Launcher launch,
        IGpuBuffer gradOutput, IGpuBuffer? maskSource, IGpuBuffer? gradOutput2, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveVariance, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int rows, int channels, float epsilon, int maskMode, int act, float slope, bool accumulate)
    {
        if (rows <= 0 || channels <= 0 || maskMode is < 0 or > 1 || (maskMode == 1) != (act != 0)
            || (maskMode == 1 && maskSource is null) || (act == 2 && !ChannelEpilogueActivation.IsSlope(slope))
            || backend is not IBatchNormVarianceBackend variance || !variance.AcceptsTrainingShape(1, channels, rows))
            return false;
        int n = rows * channels;
        using var dy = backend.AllocateBuffer(n);
        using var x = backend.AllocateBuffer(n);
        using var dx = backend.AllocateBuffer(n);
        using var mask = maskMode == 1 ? backend.AllocateBuffer(n) : null;
        using var dy2 = gradOutput2 is not null ? backend.AllocateBuffer(n) : null;
        if (!PortableChannelEpilogue.TryChangeLayout(launch, gradOutput, dy, 1, channels, rows, false)
            || !PortableChannelEpilogue.TryChangeLayout(launch, input, x, 1, channels, rows, false)
            || (mask is not null && !PortableChannelEpilogue.TryChangeLayout(launch, maskSource!, mask, 1, channels, rows, false))
            || (dy2 is not null && !PortableChannelEpilogue.TryChangeLayout(launch, gradOutput2!, dy2, 1, channels, rows, false)))
            return false;
        bool ran = act == 2
            ? backend is IBatchNormLeakyReluBackend leaky
              && leaky.TryBatchNormLeakyReluBackward(dy, mask!, x, gamma, null, saveMean, saveVariance, dx, gradGamma,
                  gradBeta, 1, channels, rows, epsilon, slope, dy2, accumulateGradInput: false)
            : variance.TryBatchNormBackwardFromVariance(dy, x, gamma, saveMean, saveVariance, dx, gradGamma, gradBeta, 1,
                channels, rows, epsilon, reluOutput: mask, gradOutput2: dy2);
        if (!ran) return false;
        if (!accumulate)
            return PortableChannelEpilogue.TryChangeLayout(launch, dx, gradInput, 1, channels, rows, toChannelsLast: true);
        using var back = backend.AllocateBuffer(n);
        if (!PortableChannelEpilogue.TryChangeLayout(launch, dx, back, 1, channels, rows, toChannelsLast: true)) return false;
        backend.Add(gradInput, back, gradInput, n);
        return true;
    }
}
