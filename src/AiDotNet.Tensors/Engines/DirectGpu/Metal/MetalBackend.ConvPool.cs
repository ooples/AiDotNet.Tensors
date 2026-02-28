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
        var inputData = DownloadBuffer(input);
        var weightsData = DownloadBuffer(weights);
        var offsetsData = DownloadBuffer(offsets);
        float[]? maskData = mask is not null ? DownloadBuffer(mask) : null;
        var result = new float[batch * outChannels * outHeight * outWidth];
        int inChannelsPerGroup = inChannels / groups;
        int kernelSize = kernelH * kernelW;

        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                int dg = oc / (outChannels / deformGroups);
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        float sum = 0;
                        int g = oc / (outChannels / groups);
                        for (int ic = 0; ic < inChannelsPerGroup; ic++)
                        {
                            int actualIC = g * inChannelsPerGroup + ic;
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int kIdx = kh * kernelW + kw;
                                    int offBase = ((b * deformGroups + dg) * 2 * kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
                                    int offBaseY = ((b * deformGroups + dg) * 2 * kernelSize + kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
                                    float offH = offsetsData[offBase];
                                    float offW = offsetsData[offBaseY];

                                    float h = oh * strideH - padH + kh * dilationH + offH;
                                    float w = ow * strideW - padW + kw * dilationW + offW;

                                    float val = BilinearSample(inputData, b, actualIC, h, w, inHeight, inWidth, inChannels);

                                    if (maskData is not null)
                                    {
                                        int mIdx = ((b * deformGroups + dg) * kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
                                        val *= maskData[mIdx];
                                    }

                                    int wIdx = ((oc * inChannelsPerGroup + ic) * kernelH + kh) * kernelW + kw;
                                    sum += val * weightsData[wIdx];
                                }
                            }
                        }
                        result[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
                    }
                }
            }
        }
        UploadToBuffer(output, result);
    }

    /// <summary>
    /// Deformable Conv2D backward for input gradients.
    /// Each input pixel accumulates gradients from all output positions that sampled it
    /// via bilinear interpolation with learned offsets.
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
        var gradOutputData = DownloadBuffer(gradOutput);
        var weightsData = DownloadBuffer(weights);
        var offsetsData = DownloadBuffer(offsets);
        float[]? maskData = mask is not null ? DownloadBuffer(mask) : null;
        var gradInputData = new float[batch * inChannels * inHeight * inWidth];
        int inChannelsPerGroup = inChannels / groups;
        int outChannelsPerGroup = outChannels / groups;
        int kernelSize = kernelH * kernelW;

        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                int g = ic / inChannelsPerGroup;
                int icLocal = ic - g * inChannelsPerGroup;
                for (int ih = 0; ih < inHeight; ih++)
                {
                    for (int iw = 0; iw < inWidth; iw++)
                    {
                        float sumGrad = 0;
                        for (int ocOff = 0; ocOff < outChannelsPerGroup; ocOff++)
                        {
                            int oc = g * outChannelsPerGroup + ocOff;
                            int dg = oc / (outChannels / deformGroups);
                            for (int oh = 0; oh < outHeight; oh++)
                            {
                                for (int ow = 0; ow < outWidth; ow++)
                                {
                                    for (int kh = 0; kh < kernelH; kh++)
                                    {
                                        for (int kw = 0; kw < kernelW; kw++)
                                        {
                                            int kIdx = kh * kernelW + kw;
                                            int offIdx = ((b * deformGroups + dg) * 2 * kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
                                            int offIdxY = ((b * deformGroups + dg) * 2 * kernelSize + kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
                                            float h = oh * strideH - padH + kh * dilationH + offsetsData[offIdx];
                                            float w = ow * strideW - padW + kw * dilationW + offsetsData[offIdxY];

                                            int hLow = (int)MathF.Floor(h);
                                            int wLow = (int)MathF.Floor(w);
                                            float lh = h - hLow;
                                            float lw = w - wLow;

                                            float wc = 0;
                                            if (ih == hLow && iw == wLow) wc = (1 - lh) * (1 - lw);
                                            else if (ih == hLow && iw == wLow + 1) wc = (1 - lh) * lw;
                                            else if (ih == hLow + 1 && iw == wLow) wc = lh * (1 - lw);
                                            else if (ih == hLow + 1 && iw == wLow + 1) wc = lh * lw;
                                            else continue;

                                            float goVal = gradOutputData[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                                            int wIdx = ((oc * inChannelsPerGroup + icLocal) * kernelH + kh) * kernelW + kw;
                                            float contrib = goVal * weightsData[wIdx] * wc;

                                            if (maskData is not null)
                                            {
                                                int mIdx = ((b * deformGroups + dg) * kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
                                                contrib *= maskData[mIdx];
                                            }
                                            sumGrad += contrib;
                                        }
                                    }
                                }
                            }
                        }
                        int idx = ((b * inChannels + ic) * inHeight + ih) * inWidth + iw;
                        gradInputData[idx] = sumGrad;
                    }
                }
            }
        }
        UploadToBuffer(gradInput, gradInputData);
    }

    /// <summary>
    /// Deformable Conv2D backward for weight gradients.
    /// Each weight element accumulates gradients from all batch and spatial positions
    /// using bilinear-interpolated input samples.
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
        var gradOutputData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var offsetsData = DownloadBuffer(offsets);
        float[]? maskData = mask is not null ? DownloadBuffer(mask) : null;
        int inChannelsPerGroup = inChannels / groups;
        var gradWeightsData = new float[outChannels * inChannelsPerGroup * kernelH * kernelW];
        int kernelSize = kernelH * kernelW;

        for (int oc = 0; oc < outChannels; oc++)
        {
            int g = oc / (outChannels / groups);
            int dg = oc / (outChannels / deformGroups);
            for (int icLocal = 0; icLocal < inChannelsPerGroup; icLocal++)
            {
                int ic = g * inChannelsPerGroup + icLocal;
                for (int kh = 0; kh < kernelH; kh++)
                {
                    for (int kw = 0; kw < kernelW; kw++)
                    {
                        int kIdx = kh * kernelW + kw;
                        float sumGrad = 0;
                        for (int b = 0; b < batch; b++)
                        {
                            for (int oh = 0; oh < outHeight; oh++)
                            {
                                for (int ow = 0; ow < outWidth; ow++)
                                {
                                    int offIdx = ((b * deformGroups + dg) * 2 * kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
                                    int offIdxY = ((b * deformGroups + dg) * 2 * kernelSize + kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
                                    float h = oh * strideH - padH + kh * dilationH + offsetsData[offIdx];
                                    float w = ow * strideW - padW + kw * dilationW + offsetsData[offIdxY];

                                    float inputVal = BilinearSample(inputData, b, ic, h, w, inHeight, inWidth, inChannels);

                                    if (maskData is not null)
                                    {
                                        int mIdx = ((b * deformGroups + dg) * kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
                                        inputVal *= maskData[mIdx];
                                    }

                                    float goVal = gradOutputData[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                                    sumGrad += goVal * inputVal;
                                }
                            }
                        }
                        int wIdx = ((oc * inChannelsPerGroup + icLocal) * kernelH + kh) * kernelW + kw;
                        gradWeightsData[wIdx] = sumGrad;
                    }
                }
            }
        }
        UploadToBuffer(gradWeights, gradWeightsData);
    }

    /// <summary>
    /// Deformable Conv2D backward for offset gradients.
    /// Computes gradients through the bilinear interpolation with respect to the spatial offsets.
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
        var gradOutputData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var weightsData = DownloadBuffer(weights);
        var offsetsData = DownloadBuffer(offsets);
        float[]? maskData = mask is not null ? DownloadBuffer(mask) : null;
        int inChannelsPerGroup = inChannels / groups;
        int outChannelsPerGroup = outChannels / groups;
        int kernelSize = kernelH * kernelW;
        int totalOffsets = batch * deformGroups * 2 * kernelSize * outHeight * outWidth;
        var gradOffsetsData = new float[totalOffsets];

        for (int idx = 0; idx < totalOffsets; idx++)
        {
            int ow = idx % outWidth;
            int oh = (idx / outWidth) % outHeight;
            int offsetComp = (idx / (outWidth * outHeight)) % (2 * kernelSize);
            int dg = (idx / (outWidth * outHeight * 2 * kernelSize)) % deformGroups;
            int b = idx / (outWidth * outHeight * 2 * kernelSize * deformGroups);

            int isYOffset = offsetComp >= kernelSize ? 1 : 0;
            int kIdx = offsetComp % kernelSize;
            int kh = kIdx / kernelW;
            int kw = kIdx % kernelW;

            int offIdxX = ((b * deformGroups + dg) * 2 * kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
            int offIdxY = ((b * deformGroups + dg) * 2 * kernelSize + kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
            float offsetH = offsetsData[offIdxX];
            float offsetW = offsetsData[offIdxY];

            float h = oh * strideH - padH + kh * dilationH + offsetH;
            float w = ow * strideW - padW + kw * dilationW + offsetW;

            int hLow = (int)MathF.Floor(h);
            int wLow = (int)MathF.Floor(w);
            int hHigh = hLow + 1;
            int wHigh = wLow + 1;
            float lh = h - hLow;
            float lw = w - wLow;

            float sumGrad = 0;
            int outChPerDg = outChannels / deformGroups;

            for (int ocOff = 0; ocOff < outChPerDg; ocOff++)
            {
                int oc = dg * outChPerDg + ocOff;
                int g = oc / outChannelsPerGroup;

                float goVal = gradOutputData[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];

                if (maskData is not null)
                {
                    int mIdx = ((b * deformGroups + dg) * kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
                    goVal *= maskData[mIdx];
                }

                for (int icOff = 0; icOff < inChannelsPerGroup; icOff++)
                {
                    int ic = g * inChannelsPerGroup + icOff;
                    int wIdx = ((oc * inChannelsPerGroup + icOff) * kernelH + kh) * kernelW + kw;
                    float weightVal = weightsData[wIdx];

                    int inBase = (b * inChannels + ic) * inHeight * inWidth;
                    float v00 = 0, v01 = 0, v10 = 0, v11 = 0;
                    if (hLow >= 0 && hLow < inHeight && wLow >= 0 && wLow < inWidth)
                        v00 = inputData[inBase + hLow * inWidth + wLow];
                    if (hLow >= 0 && hLow < inHeight && wHigh >= 0 && wHigh < inWidth)
                        v01 = inputData[inBase + hLow * inWidth + wHigh];
                    if (hHigh >= 0 && hHigh < inHeight && wLow >= 0 && wLow < inWidth)
                        v10 = inputData[inBase + hHigh * inWidth + wLow];
                    if (hHigh >= 0 && hHigh < inHeight && wHigh >= 0 && wHigh < inWidth)
                        v11 = inputData[inBase + hHigh * inWidth + wHigh];

                    if (isYOffset == 0)
                    {
                        float gradH = (1 - lw) * (v10 - v00) + lw * (v11 - v01);
                        sumGrad += goVal * weightVal * gradH;
                    }
                    else
                    {
                        float gradW = (1 - lh) * (v01 - v00) + lh * (v11 - v10);
                        sumGrad += goVal * weightVal * gradW;
                    }
                }
            }
            gradOffsetsData[idx] = sumGrad;
        }
        UploadToBuffer(gradOffsets, gradOffsetsData);
    }

    /// <summary>
    /// Deformable Conv2D backward for mask gradients (DCNv2).
    /// Computes how each modulation mask value affects the output loss.
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
        var gradOutputData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var weightsData = DownloadBuffer(weights);
        var offsetsData = DownloadBuffer(offsets);
        int inChannelsPerGroup = inChannels / groups;
        int outChannelsPerGroup = outChannels / groups;
        int kernelSize = kernelH * kernelW;
        int totalMask = batch * deformGroups * kernelSize * outHeight * outWidth;
        var gradMaskData = new float[totalMask];

        for (int idx = 0; idx < totalMask; idx++)
        {
            int ow = idx % outWidth;
            int oh = (idx / outWidth) % outHeight;
            int kIdx = (idx / (outWidth * outHeight)) % kernelSize;
            int dg = (idx / (outWidth * outHeight * kernelSize)) % deformGroups;
            int b = idx / (outWidth * outHeight * kernelSize * deformGroups);

            int kh = kIdx / kernelW;
            int kw = kIdx % kernelW;

            int offIdxX = ((b * deformGroups + dg) * 2 * kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
            int offIdxY = ((b * deformGroups + dg) * 2 * kernelSize + kernelSize + kIdx) * outHeight * outWidth + oh * outWidth + ow;
            float h = oh * strideH - padH + kh * dilationH + offsetsData[offIdxX];
            float w = ow * strideW - padW + kw * dilationW + offsetsData[offIdxY];

            float sumGrad = 0;
            int outChPerDg = outChannels / deformGroups;

            for (int ocOff = 0; ocOff < outChPerDg; ocOff++)
            {
                int oc = dg * outChPerDg + ocOff;
                int g = oc / outChannelsPerGroup;
                float goVal = gradOutputData[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];

                for (int icOff = 0; icOff < inChannelsPerGroup; icOff++)
                {
                    int ic = g * inChannelsPerGroup + icOff;
                    int wIdx = ((oc * inChannelsPerGroup + icOff) * kernelH + kh) * kernelW + kw;
                    float weightVal = weightsData[wIdx];
                    float inputVal = BilinearSample(inputData, b, ic, h, w, inHeight, inWidth, inChannels);
                    sumGrad += goVal * weightVal * inputVal;
                }
            }
            gradMaskData[idx] = sumGrad;
        }
        UploadToBuffer(gradMask, gradMaskData);
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
                // Store index as int-to-float bitcast to avoid float precision loss for large indices
                unsafe { int tmp = maxIdx; idxResult[b * channels + c] = *(float*)&tmp; }
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
                // Decode index from int-to-float bitcast stored in forward pass
                int maxIdx;
                unsafe { float tmp = idx[b * channels + c]; maxIdx = *(int*)&tmp; }
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
        throw new NotSupportedException("MaxPool3D is not yet implemented in the Metal backend. Use MaxPool2D or implement a CPU fallback.");
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
        throw new NotSupportedException("MaxPool3DBackward is not yet implemented in the Metal backend. Use MaxPool2DBackward or implement a CPU fallback.");
    }

    #endregion

    #region Bilinear Sampling Helper

    private static float BilinearSample(float[] input, int b, int c, float h, float w,
        int inHeight, int inWidth, int inChannels)
    {
        int hLow = (int)MathF.Floor(h);
        int wLow = (int)MathF.Floor(w);
        int hHigh = hLow + 1;
        int wHigh = wLow + 1;

        float lh = h - hLow;
        float lw = w - wLow;
        float hh = 1.0f - lh;
        float hw = 1.0f - lw;

        float v1 = 0, v2 = 0, v3 = 0, v4 = 0;
        int baseIdx = (b * inChannels + c) * inHeight * inWidth;

        if (hLow >= 0 && hLow < inHeight && wLow >= 0 && wLow < inWidth)
            v1 = input[baseIdx + hLow * inWidth + wLow];
        if (hLow >= 0 && hLow < inHeight && wHigh >= 0 && wHigh < inWidth)
            v2 = input[baseIdx + hLow * inWidth + wHigh];
        if (hHigh >= 0 && hHigh < inHeight && wLow >= 0 && wLow < inWidth)
            v3 = input[baseIdx + hHigh * inWidth + wLow];
        if (hHigh >= 0 && hHigh < inHeight && wHigh >= 0 && wHigh < inWidth)
            v4 = input[baseIdx + hHigh * inWidth + wHigh];

        return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
    }

    #endregion
}
