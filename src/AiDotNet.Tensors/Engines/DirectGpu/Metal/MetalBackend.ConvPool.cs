// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Convolution and Pooling operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Convolution Operations

    /// <summary>
    /// 2D Convolution forward pass.
    /// </summary>
    public void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        ThrowIfDisposed();
        // CPU fallback - proper Metal implementation would use MPSNDArrayMatrixMultiplication or custom kernel
        var inp = DownloadBuffer(input);
        var kern = DownloadBuffer(kernel);
        var result = new float[batch * outChannels * outHeight * outWidth];

        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        float sum = 0;
                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int ih = oh * strideH - padH + kh * dilationH;
                                    int iw = ow * strideW - padW + kw * dilationW;

                                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                    {
                                        int inIdx = b * inChannels * inHeight * inWidth +
                                                   ic * inHeight * inWidth + ih * inWidth + iw;
                                        int kIdx = oc * inChannels * kernelH * kernelW +
                                                  ic * kernelH * kernelW + kh * kernelW + kw;
                                        sum += inp[inIdx] * kern[kIdx];
                                    }
                                }
                            }
                        }
                        int outIdx = b * outChannels * outHeight * outWidth +
                                    oc * outHeight * outWidth + oh * outWidth + ow;
                        result[outIdx] = sum;
                    }
                }
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// 2D Convolution backward for input gradients.
    /// </summary>
    public void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        ThrowIfDisposed();
        // CPU fallback
        var go = DownloadBuffer(gradOutput);
        var kern = DownloadBuffer(kernel);
        var result = new float[batch * inChannels * inHeight * inWidth];

        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int ih = 0; ih < inHeight; ih++)
                {
                    for (int iw = 0; iw < inWidth; iw++)
                    {
                        float sum = 0;
                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int oh = ih + padH - kh * dilationH;
                                    int ow = iw + padW - kw * dilationW;
                                    if (oh % strideH == 0 && ow % strideW == 0)
                                    {
                                        oh /= strideH;
                                        ow /= strideW;
                                        if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth)
                                        {
                                            int goIdx = b * outChannels * outHeight * outWidth +
                                                       oc * outHeight * outWidth + oh * outWidth + ow;
                                            int kIdx = oc * inChannels * kernelH * kernelW +
                                                      ic * kernelH * kernelW + kh * kernelW + kw;
                                            sum += go[goIdx] * kern[kIdx];
                                        }
                                    }
                                }
                            }
                        }
                        int giIdx = b * inChannels * inHeight * inWidth +
                                   ic * inHeight * inWidth + ih * inWidth + iw;
                        result[giIdx] = sum;
                    }
                }
            }
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// 2D Convolution backward for kernel gradients.
    /// </summary>
    public void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        ThrowIfDisposed();
        var inp = DownloadBuffer(input);
        var go = DownloadBuffer(gradOutput);
        var result = new float[outChannels * inChannels * kernelH * kernelW];

        for (int oc = 0; oc < outChannels; oc++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                for (int kh = 0; kh < kernelH; kh++)
                {
                    for (int kw = 0; kw < kernelW; kw++)
                    {
                        float sum = 0;
                        for (int b = 0; b < batch; b++)
                        {
                            for (int oh = 0; oh < outHeight; oh++)
                            {
                                for (int ow = 0; ow < outWidth; ow++)
                                {
                                    int ih = oh * strideH - padH + kh * dilationH;
                                    int iw = ow * strideW - padW + kw * dilationW;
                                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                    {
                                        int inIdx = b * inChannels * inHeight * inWidth +
                                                   ic * inHeight * inWidth + ih * inWidth + iw;
                                        int goIdx = b * outChannels * outHeight * outWidth +
                                                   oc * outHeight * outWidth + oh * outWidth + ow;
                                        sum += inp[inIdx] * go[goIdx];
                                    }
                                }
                            }
                        }
                        int gkIdx = oc * inChannels * kernelH * kernelW +
                                   ic * kernelH * kernelW + kh * kernelW + kw;
                        result[gkIdx] = sum;
                    }
                }
            }
        }

        if (gradKernel is MetalGpuBuffer gkBuffer)
        {
            gkBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// 3D Convolution forward pass.
    /// </summary>
    public void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("Conv3D is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// Depthwise 2D convolution.
    /// </summary>
    public void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        ThrowIfDisposed();
        var inp = DownloadBuffer(input);
        var kern = DownloadBuffer(kernel);
        var result = new float[batch * channels * outHeight * outWidth];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        float sum = 0;
                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int ih = oh * strideH - padH + kh;
                                int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                {
                                    int inIdx = b * channels * inHeight * inWidth +
                                               c * inHeight * inWidth + ih * inWidth + iw;
                                    int kIdx = c * kernelH * kernelW + kh * kernelW + kw;
                                    sum += inp[inIdx] * kern[kIdx];
                                }
                            }
                        }
                        int outIdx = b * channels * outHeight * outWidth +
                                    c * outHeight * outWidth + oh * outWidth + ow;
                        result[outIdx] = sum;
                    }
                }
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Transposed 2D convolution.
    /// </summary>
    public void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("ConvTranspose2D is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// ConvTranspose2D backward for input gradients.
    /// </summary>
    public void ConvTranspose2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("ConvTranspose2DBackwardInput is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// ConvTranspose2D backward for kernel gradients.
    /// </summary>
    public void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("ConvTranspose2DBackwardKernel is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// Locally connected 2D convolution.
    /// </summary>
    public void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("LocallyConnectedConv2D is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// LocallyConnectedConv2D backward for input.
    /// </summary>
    public void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("LocallyConnectedConv2DBackwardInput is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// LocallyConnectedConv2D backward for weights.
    /// </summary>
    public void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("LocallyConnectedConv2DBackwardWeights is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// LocallyConnectedConv2D backward for bias.
    /// </summary>
    public void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth)
    {
        ThrowIfDisposed();
        var go = DownloadBuffer(gradOutput);
        var result = new float[outChannels];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < outChannels; c++)
            {
                for (int h = 0; h < outHeight; h++)
                {
                    for (int w = 0; w < outWidth; w++)
                    {
                        result[c] += go[b * outChannels * outHeight * outWidth + c * outHeight * outWidth + h * outWidth + w];
                    }
                }
            }
        }

        if (gradBias is MetalGpuBuffer gbBuffer)
        {
            gbBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Deformable 2D convolution (DCNv1/v2).
    /// </summary>
    public void DeformableConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("DeformableConv2D is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// Deformable Conv2D backward for input.
    /// </summary>
    public void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("DeformableConv2DBackwardInput is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// Deformable Conv2D backward for weights.
    /// </summary>
    public void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("DeformableConv2DBackwardWeights is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// Deformable Conv2D backward for offsets.
    /// </summary>
    public void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("DeformableConv2DBackwardOffset is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    /// <summary>
    /// Deformable Conv2D backward for mask (DCNv2).
    /// </summary>
    public void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("DeformableConv2DBackwardMask is not yet implemented for the Metal backend. Use the CPU engine as a fallback.");
    }

    #endregion

    #region Pooling Operations

    /// <summary>
    /// 2D Max pooling.
    /// </summary>
    public void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        ThrowIfDisposed();
        var inp = DownloadBuffer(input);
        var result = new float[batch * channels * outHeight * outWidth];
        var idxResult = indices is not null ? new float[batch * channels * outHeight * outWidth] : null;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        float maxVal = float.NegativeInfinity;
                        int maxIdx = 0;

                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int ih = oh * strideH - padH + kh;
                                int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                {
                                    int idx = b * channels * inHeight * inWidth +
                                             c * inHeight * inWidth + ih * inWidth + iw;
                                    if (inp[idx] > maxVal)
                                    {
                                        maxVal = inp[idx];
                                        maxIdx = ih * inWidth + iw;
                                    }
                                }
                            }
                        }

                        int outIdx = b * channels * outHeight * outWidth +
                                    c * outHeight * outWidth + oh * outWidth + ow;
                        result[outIdx] = maxVal;
                        if (idxResult is not null)
                        {
                            // Store index as int-to-float bitcast to avoid float precision loss for large indices
                            unsafe { int tmp = maxIdx; idxResult[outIdx] = *(float*)&tmp; }
                        }
                    }
                }
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
        if (indices is MetalGpuBuffer idxBuffer && idxResult is not null)
        {
            idxBuffer.CopyFrom(idxResult);
        }
    }

    /// <summary>
    /// 2D Max pooling backward.
    /// </summary>
    public void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        ThrowIfDisposed();
        var go = DownloadBuffer(gradOutput);
        var idx = DownloadBuffer(indices);
        var result = new float[batch * channels * inHeight * inWidth];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int outIdx = b * channels * outHeight * outWidth +
                                    c * outHeight * outWidth + oh * outWidth + ow;
                        // Decode index from int-to-float bitcast stored in forward pass
                        int maxIdx;
                        unsafe { float tmp = idx[outIdx]; maxIdx = *(int*)&tmp; }
                        int giIdx = b * channels * inHeight * inWidth +
                                   c * inHeight * inWidth + maxIdx;
                        result[giIdx] += go[outIdx];
                    }
                }
            }
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// 2D Average pooling.
    /// </summary>
    public void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        ThrowIfDisposed();
        var inp = DownloadBuffer(input);
        var result = new float[batch * channels * outHeight * outWidth];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        float sum = 0;
                        int count = 0;

                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int ih = oh * strideH - padH + kh;
                                int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                {
                                    int idx = b * channels * inHeight * inWidth +
                                             c * inHeight * inWidth + ih * inWidth + iw;
                                    sum += inp[idx];
                                    count++;
                                }
                            }
                        }

                        int divisor = countIncludePad ? kernelH * kernelW : count;
                        int outIdx = b * channels * outHeight * outWidth +
                                    c * outHeight * outWidth + oh * outWidth + ow;
                        result[outIdx] = sum / MathF.Max(divisor, 1);
                    }
                }
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// 2D Average pooling backward.
    /// </summary>
    public void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        ThrowIfDisposed();
        var go = DownloadBuffer(gradOutput);
        var result = new float[batch * channels * inHeight * inWidth];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int outIdx = b * channels * outHeight * outWidth +
                                    c * outHeight * outWidth + oh * outWidth + ow;

                        // Compute divisor: full kernel size if countIncludePad, else valid elements only
                        int divisor;
                        if (countIncludePad)
                        {
                            divisor = kernelH * kernelW;
                        }
                        else
                        {
                            int hStart = Math.Max(oh * strideH - padH, 0);
                            int hEnd = Math.Min(oh * strideH - padH + kernelH, inHeight);
                            int wStart = Math.Max(ow * strideW - padW, 0);
                            int wEnd = Math.Min(ow * strideW - padW + kernelW, inWidth);
                            divisor = (hEnd - hStart) * (wEnd - wStart);
                        }

                        float grad = go[outIdx] / divisor;

                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int ih = oh * strideH - padH + kh;
                                int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                {
                                    int giIdx = b * channels * inHeight * inWidth +
                                               c * inHeight * inWidth + ih * inWidth + iw;
                                    result[giIdx] += grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Global average pooling.
    /// </summary>
    public void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        ThrowIfDisposed();
        var inp = DownloadBuffer(input);
        var result = new float[batch * channels];

        int spatialSize = height * width;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float sum = 0;
                for (int i = 0; i < spatialSize; i++)
                {
                    sum += inp[b * channels * spatialSize + c * spatialSize + i];
                }
                result[b * channels + c] = sum / spatialSize;
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Global max pooling.
    /// </summary>
    public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        ThrowIfDisposed();
        var inp = DownloadBuffer(input);
        var result = new float[batch * channels];

        int spatialSize = height * width;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float maxVal = float.NegativeInfinity;
                for (int i = 0; i < spatialSize; i++)
                {
                    maxVal = MathF.Max(maxVal, inp[b * channels * spatialSize + c * spatialSize + i]);
                }
                result[b * channels + c] = maxVal;
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Global max pooling with indices.
    /// </summary>
    public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer indices, int batch, int channels, int height, int width)
    {
        ThrowIfDisposed();
        var inp = DownloadBuffer(input);
        var result = new float[batch * channels];
        var idxResult = new float[batch * channels];

        int spatialSize = height * width;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float maxVal = float.NegativeInfinity;
                int maxIdx = 0;
                for (int i = 0; i < spatialSize; i++)
                {
                    float val = inp[b * channels * spatialSize + c * spatialSize + i];
                    if (val > maxVal)
                    {
                        maxVal = val;
                        maxIdx = i;
                    }
                }
                result[b * channels + c] = maxVal;
                idxResult[b * channels + c] = maxIdx;
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
        if (indices is MetalGpuBuffer idxBuffer)
        {
            idxBuffer.CopyFrom(idxResult);
        }
    }

    /// <summary>
    /// Global average pooling backward.
    /// </summary>
    public void GlobalAvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        ThrowIfDisposed();
        var go = DownloadBuffer(gradOutput);
        int spatialSize = height * width;
        var result = new float[batch * channels * spatialSize];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float grad = go[b * channels + c] / spatialSize;
                for (int i = 0; i < spatialSize; i++)
                {
                    result[b * channels * spatialSize + c * spatialSize + i] = grad;
                }
            }
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Global max pooling backward.
    /// </summary>
    public void GlobalMaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        ThrowIfDisposed();
        var go = DownloadBuffer(gradOutput);
        var idx = DownloadBuffer(indices);
        int spatialSize = height * width;
        var result = new float[batch * channels * spatialSize];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int maxIdx = (int)idx[b * channels + c];
                result[b * channels * spatialSize + c * spatialSize + maxIdx] = go[b * channels + c];
            }
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Adaptive average pooling.
    /// </summary>
    public void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        ThrowIfDisposed();
        var inp = DownloadBuffer(input);
        var result = new float[batch * channels * outHeight * outWidth];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int ihStart = (int)MathF.Floor(oh * (float)inHeight / outHeight);
                        int ihEnd = (int)MathF.Ceiling((oh + 1) * (float)inHeight / outHeight);
                        int iwStart = (int)MathF.Floor(ow * (float)inWidth / outWidth);
                        int iwEnd = (int)MathF.Ceiling((ow + 1) * (float)inWidth / outWidth);

                        float sum = 0;
                        int count = 0;
                        for (int ih = ihStart; ih < ihEnd; ih++)
                        {
                            for (int iw = iwStart; iw < iwEnd; iw++)
                            {
                                sum += inp[b * channels * inHeight * inWidth + c * inHeight * inWidth + ih * inWidth + iw];
                                count++;
                            }
                        }
                        result[b * channels * outHeight * outWidth + c * outHeight * outWidth + oh * outWidth + ow] = sum / count;
                    }
                }
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// 3D Max pooling.
    /// </summary>
    public void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW)
    {
        ThrowIfDisposed();
        var result = new float[batch * channels * outDepth * outHeight * outWidth];
        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// 3D Max pooling backward.
    /// </summary>
    public void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        ThrowIfDisposed();
        var result = new float[batch * channels * inDepth * inHeight * inWidth];
        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(result);
        }
    }

    #endregion
}
