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
        var inp = DownloadBuffer(input);
        var kern = DownloadBuffer(kernel);
        var result = new float[batch * outChannels * outDepth * outHeight * outWidth];

        for (int b = 0; b < batch; b++)
        for (int oc = 0; oc < outChannels; oc++)
        for (int od = 0; od < outDepth; od++)
        for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++)
        {
            float sum = 0;
            for (int ic = 0; ic < inChannels; ic++)
            for (int kd = 0; kd < kernelD; kd++)
            for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++)
            {
                int id = od * strideD - padD + kd * dilationD;
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;
                if (id >= 0 && id < inDepth && ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                {
                    int inIdx = ((b * inChannels + ic) * inDepth + id) * inHeight * inWidth + ih * inWidth + iw;
                    int kIdx = ((oc * inChannels + ic) * kernelD + kd) * kernelH * kernelW + kh * kernelW + kw;
                    sum += inp[inIdx] * kern[kIdx];
                }
            }
            result[((b * outChannels + oc) * outDepth + od) * outHeight * outWidth + oh * outWidth + ow] = sum;
        }

        UploadToBuffer(output, result);
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
        var inp = DownloadBuffer(input);
        var kern = DownloadBuffer(kernel);
        var result = new float[batch * outChannels * outHeight * outWidth];

        // kernel layout: [inChannels, outChannels, kH, kW]
        for (int b = 0; b < batch; b++)
        for (int ic = 0; ic < inChannels; ic++)
        for (int ih = 0; ih < inHeight; ih++)
        for (int iw = 0; iw < inWidth; iw++)
        {
            float inVal = inp[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
            for (int oc = 0; oc < outChannels; oc++)
            for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++)
            {
                int oh = ih * strideH - padH + kh;
                int ow = iw * strideW - padW + kw;
                if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth)
                {
                    int kIdx = ((ic * outChannels + oc) * kernelH + kh) * kernelW + kw;
                    result[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] += inVal * kern[kIdx];
                }
            }
        }

        UploadToBuffer(output, result);
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
        var go = DownloadBuffer(gradOutput);
        var kern = DownloadBuffer(kernel);
        var result = new float[batch * inChannels * inHeight * inWidth];

        // gradInput is a standard conv of gradOutput with kernel
        for (int b = 0; b < batch; b++)
        for (int ic = 0; ic < inChannels; ic++)
        for (int ih = 0; ih < inHeight; ih++)
        for (int iw = 0; iw < inWidth; iw++)
        {
            float sum = 0;
            for (int oc = 0; oc < outChannels; oc++)
            for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++)
            {
                int oh = ih * strideH - padH + kh;
                int ow = iw * strideW - padW + kw;
                if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth)
                {
                    int kIdx = ((ic * outChannels + oc) * kernelH + kh) * kernelW + kw;
                    sum += go[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] * kern[kIdx];
                }
            }
            result[((b * inChannels + ic) * inHeight + ih) * inWidth + iw] = sum;
        }

        UploadToBuffer(gradInput, result);
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
        var inp = DownloadBuffer(input);
        var go = DownloadBuffer(gradOutput);
        var result = new float[inChannels * outChannels * kernelH * kernelW];

        for (int b = 0; b < batch; b++)
        for (int ic = 0; ic < inChannels; ic++)
        for (int ih = 0; ih < inHeight; ih++)
        for (int iw = 0; iw < inWidth; iw++)
        {
            float inVal = inp[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
            for (int oc = 0; oc < outChannels; oc++)
            for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++)
            {
                int oh = ih * strideH - padH + kh;
                int ow = iw * strideW - padW + kw;
                if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth)
                {
                    int kIdx = ((ic * outChannels + oc) * kernelH + kh) * kernelW + kw;
                    result[kIdx] += inVal * go[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                }
            }
        }

        UploadToBuffer(gradKernel, result);
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
        var inp = DownloadBuffer(input);
        var w = DownloadBuffer(weights);
        float[]? b = bias is not null ? DownloadBuffer(bias) : null;
        var result = new float[batch * outChannels * outHeight * outWidth];

        // weights: [outH, outW, outC, inC, kH, kW]
        int wStride = inChannels * kernelH * kernelW;
        for (int ba = 0; ba < batch; ba++)
        for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++)
        for (int oc = 0; oc < outChannels; oc++)
        {
            float sum = b is not null ? b[oc] : 0f;
            int wBase = ((oh * outWidth + ow) * outChannels + oc) * wStride;
            for (int ic = 0; ic < inChannels; ic++)
            for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++)
            {
                int ih = oh * strideH + kh;
                int iw = ow * strideW + kw;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                {
                    int inIdx = ((ba * inChannels + ic) * inHeight + ih) * inWidth + iw;
                    int wIdx = wBase + (ic * kernelH + kh) * kernelW + kw;
                    sum += inp[inIdx] * w[wIdx];
                }
            }
            result[((ba * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
        }

        UploadToBuffer(output, result);
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
        var go = DownloadBuffer(gradOutput);
        var w = DownloadBuffer(weights);
        var result = new float[batch * inChannels * inHeight * inWidth];

        int wStride = inChannels * kernelH * kernelW;
        for (int ba = 0; ba < batch; ba++)
        for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++)
        for (int oc = 0; oc < outChannels; oc++)
        {
            float goVal = go[((ba * outChannels + oc) * outHeight + oh) * outWidth + ow];
            int wBase = ((oh * outWidth + ow) * outChannels + oc) * wStride;
            for (int ic = 0; ic < inChannels; ic++)
            for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++)
            {
                int ih = oh * strideH + kh;
                int iw = ow * strideW + kw;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                {
                    int wIdx = wBase + (ic * kernelH + kh) * kernelW + kw;
                    result[((ba * inChannels + ic) * inHeight + ih) * inWidth + iw] += goVal * w[wIdx];
                }
            }
        }

        UploadToBuffer(gradInput, result);
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
        var go = DownloadBuffer(gradOutput);
        var inp = DownloadBuffer(input);
        var result = new float[outHeight * outWidth * outChannels * inChannels * kernelH * kernelW];

        int wStride = inChannels * kernelH * kernelW;
        for (int ba = 0; ba < batch; ba++)
        for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++)
        for (int oc = 0; oc < outChannels; oc++)
        {
            float goVal = go[((ba * outChannels + oc) * outHeight + oh) * outWidth + ow];
            int wBase = ((oh * outWidth + ow) * outChannels + oc) * wStride;
            for (int ic = 0; ic < inChannels; ic++)
            for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++)
            {
                int ih = oh * strideH + kh;
                int iw = ow * strideW + kw;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                {
                    int wIdx = wBase + (ic * kernelH + kh) * kernelW + kw;
                    int inIdx = ((ba * inChannels + ic) * inHeight + ih) * inWidth + iw;
                    result[wIdx] += goVal * inp[inIdx];
                }
            }
        }

        UploadToBuffer(gradWeights, result);
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
        var inp = DownloadBuffer(input);
        var w = DownloadBuffer(weights);
        var off = DownloadBuffer(offsets);
        float[]? m = mask is not null ? DownloadBuffer(mask) : null;
        var result = new float[batch * outChannels * outHeight * outWidth];

        int inChPerG = inChannels / groups;
        int ks = kernelH * kernelW;

        for (int b = 0; b < batch; b++)
        for (int oc = 0; oc < outChannels; oc++)
        for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++)
        {
            int g = oc / (outChannels / groups);
            int dg = oc / (outChannels / deformGroups);
            float sum = 0;
            for (int ic = 0; ic < inChPerG; ic++)
            {
                int actualIC = g * inChPerG + ic;
                for (int kh = 0; kh < kernelH; kh++)
                for (int kw = 0; kw < kernelW; kw++)
                {
                    int ki = kh * kernelW + kw;
                    int offBase = (b * deformGroups + dg) * 2 * ks * outHeight * outWidth;
                    int offSp = ki * outHeight * outWidth + oh * outWidth + ow;
                    float offH = off[offBase + offSp];
                    float offW = off[offBase + ks * outHeight * outWidth + offSp];
                    float h = oh * strideH - padH + kh * dilationH + offH;
                    float hw = ow * strideW - padW + kw * dilationW + offW;
                    float val = BilinearSample(inp, b, actualIC, h, hw, inHeight, inWidth, inChannels);
                    if (m is not null)
                    {
                        int mBase = (b * deformGroups + dg) * ks * outHeight * outWidth;
                        val *= m[mBase + ki * outHeight * outWidth + oh * outWidth + ow];
                    }
                    int wIdx = ((oc * inChPerG + ic) * kernelH + kh) * kernelW + kw;
                    sum += val * w[wIdx];
                }
            }
            result[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
        }

        UploadToBuffer(output, result);
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
        var go = DownloadBuffer(gradOutput);
        var w = DownloadBuffer(weights);
        var off = DownloadBuffer(offsets);
        float[]? m = mask is not null ? DownloadBuffer(mask) : null;
        var result = new float[batch * inChannels * inHeight * inWidth];

        int inChPerG = inChannels / groups;
        int outChPerG = outChannels / groups;
        int ks = kernelH * kernelW;

        for (int b = 0; b < batch; b++)
        for (int ic = 0; ic < inChannels; ic++)
        for (int ih = 0; ih < inHeight; ih++)
        for (int iw = 0; iw < inWidth; iw++)
        {
            int g = ic / inChPerG;
            int icLocal = ic - g * inChPerG;
            float sumGrad = 0;
            for (int oc = g * outChPerG; oc < (g + 1) * outChPerG; oc++)
            {
                int dg = oc / (outChannels / deformGroups);
                for (int oh = 0; oh < outHeight; oh++)
                for (int ow = 0; ow < outWidth; ow++)
                for (int kh = 0; kh < kernelH; kh++)
                for (int kw = 0; kw < kernelW; kw++)
                {
                    int ki = kh * kernelW + kw;
                    int offBase = (b * deformGroups + dg) * 2 * ks * outHeight * outWidth;
                    int offSp = ki * outHeight * outWidth + oh * outWidth + ow;
                    float offH = off[offBase + offSp];
                    float offW = off[offBase + ks * outHeight * outWidth + offSp];
                    float h = oh * strideH - padH + kh * dilationH + offH;
                    float hw = ow * strideW - padW + kw * dilationW + offW;
                    int h0 = (int)MathF.Floor(h); int w0 = (int)MathF.Floor(hw);
                    int h1 = h0 + 1; int w1 = w0 + 1;
                    float lh = h - h0; float lw = hw - w0;
                    float hh = 1f - lh; float hhw = 1f - lw;
                    float wc = 0;
                    if (ih == h0 && iw == w0) wc = hh * hhw;
                    else if (ih == h0 && iw == w1) wc = hh * lw;
                    else if (ih == h1 && iw == w0) wc = lh * hhw;
                    else if (ih == h1 && iw == w1) wc = lh * lw;
                    else continue;
                    float goVal = go[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                    int wIdx = ((oc * inChPerG + icLocal) * kernelH + kh) * kernelW + kw;
                    float contrib = goVal * w[wIdx] * wc;
                    if (m is not null)
                    {
                        int mBase = (b * deformGroups + dg) * ks * outHeight * outWidth;
                        contrib *= m[mBase + ki * outHeight * outWidth + oh * outWidth + ow];
                    }
                    sumGrad += contrib;
                }
            }
            result[((b * inChannels + ic) * inHeight + ih) * inWidth + iw] = sumGrad;
        }

        UploadToBuffer(gradInput, result);
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
        var go = DownloadBuffer(gradOutput);
        var inp = DownloadBuffer(input);
        var off = DownloadBuffer(offsets);
        float[]? m = mask is not null ? DownloadBuffer(mask) : null;
        int inChPerG = inChannels / groups;
        var result = new float[outChannels * inChPerG * kernelH * kernelW];
        int ks = kernelH * kernelW;

        for (int oc = 0; oc < outChannels; oc++)
        {
            int g = oc / (outChannels / groups);
            int dg = oc / (outChannels / deformGroups);
            for (int icL = 0; icL < inChPerG; icL++)
            {
                int ic = g * inChPerG + icL;
                for (int kh = 0; kh < kernelH; kh++)
                for (int kw = 0; kw < kernelW; kw++)
                {
                    int ki = kh * kernelW + kw;
                    float sumGrad = 0;
                    for (int b = 0; b < batch; b++)
                    for (int oh = 0; oh < outHeight; oh++)
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int offBase = (b * deformGroups + dg) * 2 * ks * outHeight * outWidth;
                        int offSp = ki * outHeight * outWidth + oh * outWidth + ow;
                        float offH = off[offBase + offSp];
                        float offW = off[offBase + ks * outHeight * outWidth + offSp];
                        float h = oh * strideH - padH + kh * dilationH + offH;
                        float hw = ow * strideW - padW + kw * dilationW + offW;
                        float inputVal = BilinearSample(inp, b, ic, h, hw, inHeight, inWidth, inChannels);
                        if (m is not null)
                        {
                            int mBase = (b * deformGroups + dg) * ks * outHeight * outWidth;
                            inputVal *= m[mBase + ki * outHeight * outWidth + oh * outWidth + ow];
                        }
                        float goVal = go[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                        sumGrad += goVal * inputVal;
                    }
                    result[((oc * inChPerG + icL) * kernelH + kh) * kernelW + kw] = sumGrad;
                }
            }
        }

        UploadToBuffer(gradWeights, result);
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
        var go = DownloadBuffer(gradOutput);
        var inp = DownloadBuffer(input);
        var w = DownloadBuffer(weights);
        var off = DownloadBuffer(offsets);
        float[]? m = mask is not null ? DownloadBuffer(mask) : null;
        int ks = kernelH * kernelW;
        int inChPerG = inChannels / groups;
        int outChPerDg = outChannels / deformGroups;
        var result = new float[batch * deformGroups * 2 * ks * outHeight * outWidth];

        for (int b = 0; b < batch; b++)
        for (int dg = 0; dg < deformGroups; dg++)
        for (int comp = 0; comp < 2 * ks; comp++)
        for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++)
        {
            int isY = comp >= ks ? 1 : 0;
            int ki = comp % ks;
            int kh = ki / kernelW; int kw = ki % kernelW;
            int offBase = (b * deformGroups + dg) * 2 * ks * outHeight * outWidth;
            int offSp = ki * outHeight * outWidth + oh * outWidth + ow;
            float offH = off[offBase + offSp];
            float offW = off[offBase + ks * outHeight * outWidth + offSp];
            float h = oh * strideH - padH + kh * dilationH + offH;
            float hw = ow * strideW - padW + kw * dilationW + offW;
            int h0 = (int)MathF.Floor(h); int w0 = (int)MathF.Floor(hw);
            int h1 = h0 + 1; int w1 = w0 + 1;
            float lh = h - h0; float lw = hw - w0;
            float sumGrad = 0;
            for (int ocOff = 0; ocOff < outChPerDg; ocOff++)
            {
                int oc = dg * outChPerDg + ocOff;
                int g = oc / (outChannels / groups);
                float goVal = go[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                if (m is not null)
                {
                    int mBase = (b * deformGroups + dg) * ks * outHeight * outWidth;
                    goVal *= m[mBase + ki * outHeight * outWidth + oh * outWidth + ow];
                }
                for (int ic = g * inChPerG; ic < (g + 1) * inChPerG; ic++)
                {
                    int icLocal = ic - g * inChPerG;
                    int wIdx = ((oc * inChPerG + icLocal) * kernelH + kh) * kernelW + kw;
                    float wv = w[wIdx];
                    float v1 = 0, v2 = 0, v3 = 0, v4 = 0;
                    int ib = (b * inChannels + ic) * inHeight * inWidth;
                    if (h0 >= 0 && h0 < inHeight && w0 >= 0 && w0 < inWidth) v1 = inp[ib + h0 * inWidth + w0];
                    if (h0 >= 0 && h0 < inHeight && w1 >= 0 && w1 < inWidth) v2 = inp[ib + h0 * inWidth + w1];
                    if (h1 >= 0 && h1 < inHeight && w0 >= 0 && w0 < inWidth) v3 = inp[ib + h1 * inWidth + w0];
                    if (h1 >= 0 && h1 < inHeight && w1 >= 0 && w1 < inWidth) v4 = inp[ib + h1 * inWidth + w1];
                    if (isY == 0)
                        sumGrad += goVal * wv * ((1f - lw) * (v3 - v1) + lw * (v4 - v2));
                    else
                        sumGrad += goVal * wv * ((1f - lh) * (v2 - v1) + lh * (v4 - v3));
                }
            }
            int rIdx = ((b * deformGroups + dg) * 2 * ks + comp) * outHeight * outWidth + oh * outWidth + ow;
            result[rIdx] = sumGrad;
        }

        UploadToBuffer(gradOffsets, result);
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
        var go = DownloadBuffer(gradOutput);
        var inp = DownloadBuffer(input);
        var w = DownloadBuffer(weights);
        var off = DownloadBuffer(offsets);
        int ks = kernelH * kernelW;
        int inChPerG = inChannels / groups;
        int outChPerDg = outChannels / deformGroups;
        var result = new float[batch * deformGroups * ks * outHeight * outWidth];

        for (int b = 0; b < batch; b++)
        for (int dg = 0; dg < deformGroups; dg++)
        for (int ki = 0; ki < ks; ki++)
        for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++)
        {
            int kh = ki / kernelW; int kw = ki % kernelW;
            int offBase = (b * deformGroups + dg) * 2 * ks * outHeight * outWidth;
            int offSp = ki * outHeight * outWidth + oh * outWidth + ow;
            float offH = off[offBase + offSp];
            float offW = off[offBase + ks * outHeight * outWidth + offSp];
            float h = oh * strideH - padH + kh * dilationH + offH;
            float hw = ow * strideW - padW + kw * dilationW + offW;
            float sumGrad = 0;
            for (int ocOff = 0; ocOff < outChPerDg; ocOff++)
            {
                int oc = dg * outChPerDg + ocOff;
                int g = oc / (outChannels / groups);
                float goVal = go[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                for (int ic = g * inChPerG; ic < (g + 1) * inChPerG; ic++)
                {
                    int icLocal = ic - g * inChPerG;
                    int wIdx = ((oc * inChPerG + icLocal) * kernelH + kh) * kernelW + kw;
                    float wv = w[wIdx];
                    float iv = BilinearSample(inp, b, ic, h, hw, inHeight, inWidth, inChannels);
                    sumGrad += goVal * wv * iv;
                }
            }
            result[((b * deformGroups + dg) * ks + ki) * outHeight * outWidth + oh * outWidth + ow] = sumGrad;
        }

        UploadToBuffer(gradMask, result);
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
        var inp = DownloadBuffer(input);
        var result = new float[batch * channels * outDepth * outHeight * outWidth];
        var idxResult = indices is not null ? new float[batch * channels * outDepth * outHeight * outWidth] : null;

        for (int b = 0; b < batch; b++)
        for (int c = 0; c < channels; c++)
        for (int od = 0; od < outDepth; od++)
        for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++)
        {
            float maxVal = float.NegativeInfinity;
            int maxIdx = 0;

            for (int kd = 0; kd < kernelD; kd++)
            for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++)
            {
                int id = od * strideD + kd;
                int ih = oh * strideH + kh;
                int iw = ow * strideW + kw;
                if (id >= 0 && id < inDepth && ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                {
                    int idx = ((b * channels + c) * inDepth + id) * inHeight * inWidth +
                              ih * inWidth + iw;
                    if (inp[idx] > maxVal)
                    {
                        maxVal = inp[idx];
                        maxIdx = id * inHeight * inWidth + ih * inWidth + iw;
                    }
                }
            }

            int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth +
                         oh * outWidth + ow;
            result[outIdx] = maxVal;
            if (idxResult is not null)
            {
                unsafe { int tmp = maxIdx; idxResult[outIdx] = *(float*)&tmp; }
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
    /// 3D Max pooling backward.
    /// </summary>
    public void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        ThrowIfDisposed();
        var go = DownloadBuffer(gradOutput);
        var idx = DownloadBuffer(indices);
        var result = new float[batch * channels * inDepth * inHeight * inWidth];

        for (int b = 0; b < batch; b++)
        for (int c = 0; c < channels; c++)
        for (int od = 0; od < outDepth; od++)
        for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++)
        {
            int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth +
                         oh * outWidth + ow;
            int maxIdx;
            unsafe { float tmp = idx[outIdx]; maxIdx = *(int*)&tmp; }
            int giIdx = (b * channels + c) * inDepth * inHeight * inWidth + maxIdx;
            result[giIdx] += go[outIdx];
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Bilinear sample from NCHW buffer at fractional (h, w) position.
    /// </summary>
    private static float BilinearSample(float[] data, int b, int c, float h, float w,
        int height, int width, int channels)
    {
        int h0 = (int)MathF.Floor(h);
        int w0 = (int)MathF.Floor(w);
        int h1 = h0 + 1;
        int w1 = w0 + 1;
        float lh = h - h0;
        float lw = w - w0;

        float v00 = (h0 >= 0 && h0 < height && w0 >= 0 && w0 < width)
            ? data[((b * channels + c) * height + h0) * width + w0] : 0f;
        float v01 = (h0 >= 0 && h0 < height && w1 >= 0 && w1 < width)
            ? data[((b * channels + c) * height + h0) * width + w1] : 0f;
        float v10 = (h1 >= 0 && h1 < height && w0 >= 0 && w0 < width)
            ? data[((b * channels + c) * height + h1) * width + w0] : 0f;
        float v11 = (h1 >= 0 && h1 < height && w1 >= 0 && w1 < width)
            ? data[((b * channels + c) * height + h1) * width + w1] : 0f;

        return (1 - lh) * (1 - lw) * v00 + (1 - lh) * lw * v01 +
               lh * (1 - lw) * v10 + lh * lw * v11;
    }

    #endregion
}
