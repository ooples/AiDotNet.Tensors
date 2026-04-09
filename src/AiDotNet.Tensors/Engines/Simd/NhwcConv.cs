using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// NHWC (channels-last) convolution kernels.
///
/// In NHWC layout, all channels at a spatial position are contiguous,
/// enabling SIMD vectorization over the channel dimension. Particularly
/// effective for 1x1 pointwise convolutions (pure GEMM, no im2col).
///
/// Layout: [batch, height, width, channels]
/// vs NCHW: [batch, channels, height, width]
/// </summary>
internal static class NhwcConv
{
    /// <summary>
    /// Converts NCHW [batch, channels, height, width] to NHWC [batch, height, width, channels].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void NchwToNhwc(ReadOnlySpan<float> nchw, Span<float> nhwc,
        int batch, int channels, int height, int width)
    {
        int spatialSize = height * width;
        int batchSize = channels * spatialSize;

        for (int b = 0; b < batch; b++)
        {
            int bOff = b * batchSize;
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    int spatialIdx = h * width + w;
                    int nhwcOff = bOff + spatialIdx * channels;
                    for (int c = 0; c < channels; c++)
                        nhwc[nhwcOff + c] = nchw[bOff + c * spatialSize + spatialIdx];
                }
        }
    }

    /// <summary>
    /// Converts NHWC [batch, height, width, channels] to NCHW [batch, channels, height, width].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void NhwcToNchw(ReadOnlySpan<float> nhwc, Span<float> nchw,
        int batch, int channels, int height, int width)
    {
        int spatialSize = height * width;
        int batchSize = channels * spatialSize;

        for (int b = 0; b < batch; b++)
        {
            int bOff = b * batchSize;
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    int spatialIdx = h * width + w;
                    int nhwcOff = bOff + spatialIdx * channels;
                    for (int c = 0; c < channels; c++)
                        nchw[bOff + c * spatialSize + spatialIdx] = nhwc[nhwcOff + c];
                }
        }
    }

    /// <summary>
    /// General Conv2D in NHWC layout using im2col + GEMM.
    /// Kernel stored as [outChannels, inChannels * kernelH * kernelW] (row-major).
    /// </summary>
    internal static void Conv2DNhwc(
        float[] inputNhwc, float[] kernelFlat, float[] outputNhwc,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int outH, int outW)
    {
        int patchLen = inChannels * kernelH * kernelW;
        int spatialOut = outH * outW;
        var workspace = new float[spatialOut * patchLen];

        for (int b = 0; b < batch; b++)
        {
            int inputOff = b * height * width * inChannels;
            int outputOff = b * spatialOut * outChannels;

            // im2col: extract patches with channels contiguous
            Im2ColNhwc(inputNhwc, inputOff, workspace,
                inChannels, height, width,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                dilationH, dilationW,
                outH, outW);

            // GEMM: output[spatial, outC] = workspace[spatial, patchLen] @ kernel[outC, patchLen]^T
            if (!BlasProvider.TryGemmEx(spatialOut, outChannels, patchLen,
                workspace, 0, patchLen, false,
                kernelFlat, 0, patchLen, true,
                outputNhwc, outputOff, outChannels))
            {
                // Scalar fallback: C[i,j] = sum_k A[i,k] * B[j,k]
                for (int i = 0; i < spatialOut; i++)
                    for (int j = 0; j < outChannels; j++)
                    {
                        float sum = 0f;
                        for (int k = 0; k < patchLen; k++)
                            sum += workspace[i * patchLen + k] * kernelFlat[j * patchLen + k];
                        outputNhwc[outputOff + i * outChannels + j] = sum;
                    }
            }
        }
    }

    /// <summary>
    /// im2col for NHWC layout — channels contiguous per patch position.
    /// Output: [outH * outW, kernelH * kernelW * inChannels]
    /// </summary>
    private static void Im2ColNhwc(
        float[] input, int inputOffset, float[] output,
        int channels, int height, int width,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int outH, int outW)
    {
        int patchLen = channels * kernelH * kernelW;

        for (int oh = 0; oh < outH; oh++)
            for (int ow = 0; ow < outW; ow++)
            {
                int dstOff = (oh * outW + ow) * patchLen;
                int idx = 0;

                for (int kh = 0; kh < kernelH; kh++)
                {
                    int ih = oh * strideH - padH + kh * dilationH;
                    for (int kw = 0; kw < kernelW; kw++)
                    {
                        int iw = ow * strideW - padW + kw * dilationW;

                        if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                        {
                            int srcOff = inputOffset + (ih * width + iw) * channels;
                            for (int c = 0; c < channels; c++)
                                output[dstOff + idx++] = input[srcOff + c];
                        }
                        else
                        {
                            for (int c = 0; c < channels; c++)
                                output[dstOff + idx++] = 0f;
                        }
                    }
                }
            }
    }
}
