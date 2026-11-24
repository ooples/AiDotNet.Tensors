// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Capability every GPU backend implements: the operations a compiled plan needs to keep a convolution stack's
/// activations channels-last (NHWC, [batch, height, width, channels]) between its layers -- the convolution passes with
/// a layout per tensor, training BatchNorm (with its activation) over a channels-last activation, and the layout changes
/// at the stack's edges. A backend whose convolutions run channels-last natively (cuDNN's tensor-core kernels) then
/// skips the layout passes it otherwise runs around every convolution; <see cref="PrefersChannelsLast"/> says whether
/// it does, and a plan keeps NCHW where it does not.
/// </summary>
/// <remarks>
/// <para>A channels-last activation of logical shape [n, c, h, w] holds element (n, c, h, w) at
/// <c>((n * h + y) * w + x) * c + channel</c>: a [rows = n * h * w, c] matrix. Filters stay [k, c, r, s] either way.
/// Every method returns false -- nothing launched -- when the backend cannot run the shape; a plan checks
/// <see cref="AcceptsChannelsLast"/> before it chooses the layout, so it never meets that mid-step.</para>
/// <para>Convolution passes (<c>pass</c>): 0 forward (a = x, b = w, c = y); 1 backward data (a = w, b = dy, c = dx);
/// 2 backward filter (a = x, b = dy, c = dw). <c>inputNhwc</c> is x's / dx's layout, <c>outputNhwc</c> y's / dy's.
/// A transposed convolution is the backward data pass of the convolution it transposes (and its gradients the forward and
/// backward filter passes), so the same entry serves it.</para>
/// </remarks>
internal interface IChannelsLastBackend
{
    /// <summary>Whether this backend's convolutions run faster on channels-last activations than on NCHW ones.</summary>
    bool PrefersChannelsLast { get; }

    /// <summary>Whether <see cref="TryConvolutionChannelsLast"/> and the channels-last BatchNorm take this geometry and
    /// channel count (a query; nothing is launched).</summary>
    bool AcceptsChannelsLast(int channels, int outChannels);

    /// <summary>One convolution pass with a layout per activation tensor (see the remarks for the operand roles).</summary>
    bool TryConvolutionChannelsLast(int pass, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int channels,
        int height, int width, int outChannels, int kernelH, int kernelW, int outHeight, int outWidth, int padH, int padW,
        int strideH, int strideW, int dilationH, int dilationW, bool inputNhwc, bool outputNhwc);

    /// <summary>
    /// Training BatchNorm over a channels-last [rows, channels] activation, then the activation (<paramref name="act"/>:
    /// 0 none, 1 ReLU, 2 LeakyReLU of <paramref name="slope"/>): writes the output and the batch mean and biased variance
    /// per channel, the values the NCHW training BatchNorm computes.
    /// </summary>
    bool TryBatchNormChannelsLast(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveVariance, int rows, int channels, float epsilon, int act, float slope);

    /// <summary>
    /// Its backward: gradInput (accumulated into when <paramref name="accumulate"/>), gradGamma and gradBeta.
    /// <paramref name="maskMode"/> 0: no activation; 1: the mask from the activation output in
    /// <paramref name="maskSource"/>; 2: rebuilt from the input (maskSource is beta). <paramref name="gradOutput2"/>, when
    /// given, is added to the output gradient first.
    /// </summary>
    bool TryBatchNormChannelsLastBackward(IGpuBuffer gradOutput, IGpuBuffer? maskSource, IGpuBuffer? gradOutput2,
        IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveMean, IGpuBuffer saveVariance, IGpuBuffer gradInput,
        IGpuBuffer gradGamma, IGpuBuffer gradBeta, int rows, int channels, float epsilon, int maskMode, int act,
        float slope, bool accumulate);

    /// <summary>The layout change [n, c, spatial] to [n, spatial, c] (<paramref name="toChannelsLast"/>) or back.</summary>
    bool TryChangeLayout(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int spatial, bool toChannelsLast);
}
