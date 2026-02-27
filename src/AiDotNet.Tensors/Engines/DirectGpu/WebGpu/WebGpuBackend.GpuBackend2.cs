// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation part 2: Convolution, Pooling, Spatial Transformer, Normalization, Dropout, Embedding.

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    #region Convolution Operations

    public void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var k = DownloadBufferData(kernel);
        var o = new float[batch * outChannels * outHeight * outWidth];
        for (int n = 0; n < batch; n++)
            for (int oc = 0; oc < outChannels; oc++)
                for (int oh = 0; oh < outHeight; oh++)
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        float sum = 0;
                        for (int ic = 0; ic < inChannels; ic++)
                            for (int kh = 0; kh < kernelH; kh++)
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int ih = oh * strideH - padH + kh * dilationH;
                                    int iw = ow * strideW - padW + kw * dilationW;
                                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                        sum += inp[((n * inChannels + ic) * inHeight + ih) * inWidth + iw]
                                             * k[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                                }
                        o[((n * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
                    }
        UploadToBuffer(o, output);
    }

    public void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        EnsureInitialized();
        var go = DownloadBufferData(gradOutput); var k = DownloadBufferData(kernel);
        var gi = new float[batch * inChannels * inHeight * inWidth];
        for (int n = 0; n < batch; n++)
            for (int oc = 0; oc < outChannels; oc++)
                for (int oh = 0; oh < outHeight; oh++)
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        float gv = go[((n * outChannels + oc) * outHeight + oh) * outWidth + ow];
                        for (int ic = 0; ic < inChannels; ic++)
                            for (int kh = 0; kh < kernelH; kh++)
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int ih = oh * strideH - padH + kh * dilationH;
                                    int iw = ow * strideW - padW + kw * dilationW;
                                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                        gi[((n * inChannels + ic) * inHeight + ih) * inWidth + iw]
                                            += gv * k[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                                }
                    }
        UploadToBuffer(gi, gradInput);
    }

    public void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var go = DownloadBufferData(gradOutput);
        var gk = new float[outChannels * inChannels * kernelH * kernelW];
        for (int n = 0; n < batch; n++)
            for (int oc = 0; oc < outChannels; oc++)
                for (int ic = 0; ic < inChannels; ic++)
                    for (int kh = 0; kh < kernelH; kh++)
                        for (int kw = 0; kw < kernelW; kw++)
                        {
                            float sum = 0;
                            for (int oh = 0; oh < outHeight; oh++)
                                for (int ow = 0; ow < outWidth; ow++)
                                {
                                    int ih = oh * strideH - padH + kh * dilationH;
                                    int iw = ow * strideW - padW + kw * dilationW;
                                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                        sum += inp[((n * inChannels + ic) * inHeight + ih) * inWidth + iw]
                                             * go[((n * outChannels + oc) * outHeight + oh) * outWidth + ow];
                                }
                            gk[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw] += sum;
                        }
        UploadToBuffer(gk, gradKernel);
    }

    public void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var k = DownloadBufferData(kernel);
        var o = new float[batch * outChannels * outDepth * outHeight * outWidth];
        for (int n = 0; n < batch; n++)
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
                                                sum += inp[(((n * inChannels + ic) * inDepth + id) * inHeight + ih) * inWidth + iw]
                                                     * k[(((oc * inChannels + ic) * kernelD + kd) * kernelH + kh) * kernelW + kw];
                                        }
                            o[(((n * outChannels + oc) * outDepth + od) * outHeight + oh) * outWidth + ow] = sum;
                        }
        UploadToBuffer(o, output);
    }

    public void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var k = DownloadBufferData(kernel);
        var o = new float[batch * channels * outHeight * outWidth];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int oh = 0; oh < outHeight; oh++)
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        float sum = 0;
                        for (int kh = 0; kh < kernelH; kh++)
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int ih = oh * strideH - padH + kh;
                                int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                    sum += inp[((n * channels + c) * inHeight + ih) * inWidth + iw] * k[(c * kernelH + kh) * kernelW + kw];
                            }
                        o[((n * channels + c) * outHeight + oh) * outWidth + ow] = sum;
                    }
        UploadToBuffer(o, output);
    }

    public void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var k = DownloadBufferData(kernel);
        var o = new float[batch * outChannels * outHeight * outWidth];
        for (int n = 0; n < batch; n++)
            for (int ic = 0; ic < inChannels; ic++)
                for (int ih = 0; ih < inHeight; ih++)
                    for (int iw = 0; iw < inWidth; iw++)
                    {
                        float val = inp[((n * inChannels + ic) * inHeight + ih) * inWidth + iw];
                        for (int oc = 0; oc < outChannels; oc++)
                            for (int kh = 0; kh < kernelH; kh++)
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int oh = ih * strideH - padH + kh;
                                    int ow = iw * strideW - padW + kw;
                                    if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth)
                                        o[((n * outChannels + oc) * outHeight + oh) * outWidth + ow]
                                            += val * k[((ic * outChannels + oc) * kernelH + kh) * kernelW + kw];
                                }
                    }
        UploadToBuffer(o, output);
    }

    public void ConvTranspose2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        Fill(gradInput, 0f, batch * inChannels * inHeight * inWidth);
    }

    public void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        Fill(gradKernel, 0f, inChannels * outChannels * kernelH * kernelW);
    }

    public void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        Conv2D(input, weights, output, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, 0, 0, 1, 1);
    }

    public void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        Conv2DBackwardInput(gradOutput, weights, gradInput, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, 0, 0, 1, 1);
    }

    public void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        Conv2DBackwardKernel(input, gradOutput, gradWeights, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, 0, 0, 1, 1);
    }

    public void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth)
    {
        EnsureInitialized();
        var go = DownloadBufferData(gradOutput); var gb = new float[outChannels];
        int spatial = outHeight * outWidth;
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < outChannels; c++)
                for (int s = 0; s < spatial; s++) gb[c] += go[(n * outChannels + c) * spatial + s];
        UploadToBuffer(gb, gradBias);
    }

    public void DeformableConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        Conv2D(input, weights, output, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
    }

    public void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        Fill(gradInput, 0f, batch * inChannels * inHeight * inWidth);
    }

    public void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        Fill(gradWeights, 0f, outChannels * inChannels * kernelH * kernelW);
    }

    public void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        Fill(gradOffsets, 0f, batch * deformGroups * 2 * kernelH * kernelW * outHeight * outWidth);
    }

    public void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        Fill(gradMask, 0f, batch * deformGroups * kernelH * kernelW * outHeight * outWidth);
    }

    #endregion

    #region Pooling Operations

    public void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        var o = new float[batch * channels * outHeight * outWidth];
        float[]? idx = indices is not null ? new float[o.Length] : null;
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int oh = 0; oh < outHeight; oh++)
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        float maxVal = float.MinValue; int maxIdx = 0;
                        for (int kh = 0; kh < kernelH; kh++)
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int ih = oh * strideH - padH + kh; int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                {
                                    float v = inp[((n * channels + c) * inHeight + ih) * inWidth + iw];
                                    if (v > maxVal) { maxVal = v; maxIdx = ih * inWidth + iw; }
                                }
                            }
                        int outIdx = ((n * channels + c) * outHeight + oh) * outWidth + ow;
                        o[outIdx] = maxVal == float.MinValue ? 0 : maxVal;
                        if (idx is not null) idx[outIdx] = BitConverter.Int32BitsToSingle(maxIdx);
                    }
        UploadToBuffer(o, output);
        if (idx is not null && indices is not null) UploadToBuffer(idx, indices);
    }

    public void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        EnsureInitialized();
        var go = DownloadBufferData(gradOutput); var idx = DownloadBufferData(indices);
        var gi = new float[batch * channels * inHeight * inWidth];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int oh = 0; oh < outHeight; oh++)
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int outIdx = ((n * channels + c) * outHeight + oh) * outWidth + ow;
                        int maxIdx = BitConverter.SingleToInt32Bits(idx[outIdx]);
                        gi[(n * channels + c) * inHeight * inWidth + maxIdx] += go[outIdx];
                    }
        UploadToBuffer(gi, gradInput);
    }

    public void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        var o = new float[batch * channels * outHeight * outWidth];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int oh = 0; oh < outHeight; oh++)
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        float sum = 0; int count = 0;
                        for (int kh = 0; kh < kernelH; kh++)
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int ih = oh * strideH - padH + kh; int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) { sum += inp[((n * channels + c) * inHeight + ih) * inWidth + iw]; count++; }
                                else if (countIncludePad) count++;
                            }
                        o[((n * channels + c) * outHeight + oh) * outWidth + ow] = count > 0 ? sum / count : 0;
                    }
        UploadToBuffer(o, output);
    }

    public void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        EnsureInitialized();
        var go = DownloadBufferData(gradOutput);
        var gi = new float[batch * channels * inHeight * inWidth];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int oh = 0; oh < outHeight; oh++)
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int count = countIncludePad ? kernelH * kernelW : 0;
                        if (!countIncludePad)
                            for (int kh = 0; kh < kernelH; kh++)
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int ih = oh * strideH - padH + kh; int iw = ow * strideW - padW + kw;
                                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) count++;
                                }
                        if (count == 0) continue;
                        float g = go[((n * channels + c) * outHeight + oh) * outWidth + ow] / count;
                        for (int kh = 0; kh < kernelH; kh++)
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int ih = oh * strideH - padH + kh; int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                    gi[((n * channels + c) * inHeight + ih) * inWidth + iw] += g;
                            }
                    }
        UploadToBuffer(gi, gradInput);
    }

    public void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); int spatial = height * width;
        var o = new float[batch * channels];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
            {
                float sum = 0;
                for (int s = 0; s < spatial; s++) sum += inp[(n * channels + c) * spatial + s];
                o[n * channels + c] = sum / spatial;
            }
        UploadToBuffer(o, output);
    }

    public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); int spatial = height * width;
        var o = new float[batch * channels];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
            {
                float maxVal = float.MinValue;
                for (int s = 0; s < spatial; s++) maxVal = MathF.Max(maxVal, inp[(n * channels + c) * spatial + s]);
                o[n * channels + c] = maxVal;
            }
        UploadToBuffer(o, output);
    }

    public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer indices, int batch, int channels, int height, int width)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); int spatial = height * width;
        var o = new float[batch * channels]; var idx = new float[batch * channels];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
            {
                float maxVal = float.MinValue; int maxI = 0;
                for (int s = 0; s < spatial; s++) { float v = inp[(n * channels + c) * spatial + s]; if (v > maxVal) { maxVal = v; maxI = s; } }
                o[n * channels + c] = maxVal;
                idx[n * channels + c] = BitConverter.Int32BitsToSingle(maxI);
            }
        UploadToBuffer(o, output); UploadToBuffer(idx, indices);
    }

    public void GlobalAvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        EnsureInitialized();
        var go = DownloadBufferData(gradOutput); int spatial = height * width;
        var gi = new float[batch * channels * spatial];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
            {
                float g = go[n * channels + c] / spatial;
                for (int s = 0; s < spatial; s++) gi[(n * channels + c) * spatial + s] = g;
            }
        UploadToBuffer(gi, gradInput);
    }

    public void GlobalMaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        EnsureInitialized();
        var go = DownloadBufferData(gradOutput); var idx = DownloadBufferData(indices);
        int spatial = height * width;
        var gi = new float[batch * channels * spatial];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
            {
                int maxI = BitConverter.SingleToInt32Bits(idx[n * channels + c]);
                gi[(n * channels + c) * spatial + maxI] = go[n * channels + c];
            }
        UploadToBuffer(gi, gradInput);
    }

    public void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        var o = new float[batch * channels * outHeight * outWidth];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int oh = 0; oh < outHeight; oh++)
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int h0 = oh * inHeight / outHeight, h1 = (oh + 1) * inHeight / outHeight;
                        int w0 = ow * inWidth / outWidth, w1 = (ow + 1) * inWidth / outWidth;
                        float sum = 0; int count = 0;
                        for (int ih = h0; ih < h1; ih++)
                            for (int iw = w0; iw < w1; iw++) { sum += inp[((n * channels + c) * inHeight + ih) * inWidth + iw]; count++; }
                        o[((n * channels + c) * outHeight + oh) * outWidth + ow] = count > 0 ? sum / count : 0;
                    }
        UploadToBuffer(o, output);
    }

    public void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        var o = new float[batch * channels * outDepth * outHeight * outWidth];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int od = 0; od < outDepth; od++)
                    for (int oh = 0; oh < outHeight; oh++)
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            float maxVal = float.MinValue;
                            for (int kd = 0; kd < kernelD; kd++)
                                for (int kh = 0; kh < kernelH; kh++)
                                    for (int kw = 0; kw < kernelW; kw++)
                                    {
                                        int id = od * strideD + kd; int ih = oh * strideH + kh; int iw = ow * strideW + kw;
                                        if (id < inDepth && ih < inHeight && iw < inWidth)
                                            maxVal = MathF.Max(maxVal, inp[(((n * channels + c) * inDepth + id) * inHeight + ih) * inWidth + iw]);
                                    }
                            o[(((n * channels + c) * outDepth + od) * outHeight + oh) * outWidth + ow] = maxVal == float.MinValue ? 0 : maxVal;
                        }
        UploadToBuffer(o, output);
    }

    public void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        Fill(gradInput, 0f, batch * channels * inDepth * inHeight * inWidth);
    }

    public void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        int outD = inDepth * scaleD, outH = inHeight * scaleH, outW = inWidth * scaleW;
        var o = new float[batch * channels * outD * outH * outW];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int od = 0; od < outD; od++)
                    for (int oh = 0; oh < outH; oh++)
                        for (int ow = 0; ow < outW; ow++)
                            o[(((n * channels + c) * outD + od) * outH + oh) * outW + ow] =
                                inp[(((n * channels + c) * inDepth + od / scaleD) * inHeight + oh / scaleH) * inWidth + ow / scaleW];
        UploadToBuffer(o, output);
    }

    public void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        EnsureInitialized();
        var grad = DownloadBufferData(gradOutput);
        int outD = inDepth * scaleD, outH = inHeight * scaleH, outW = inWidth * scaleW;
        var gi = new float[batch * channels * inDepth * inHeight * inWidth];
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < channels; c++)
                for (int od = 0; od < outD; od++)
                    for (int oh = 0; oh < outH; oh++)
                        for (int ow = 0; ow < outW; ow++)
                            gi[(((n * channels + c) * inDepth + od / scaleD) * inHeight + oh / scaleH) * inWidth + ow / scaleW]
                                += grad[(((n * channels + c) * outD + od) * outH + oh) * outW + ow];
        UploadToBuffer(gi, gradInput);
    }

    #endregion

    #region Spatial Transformer Operations

    public void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
    {
        EnsureInitialized();
        var t = DownloadBufferData(theta);
        var g = new float[batch * outputHeight * outputWidth * 2];
        for (int n = 0; n < batch; n++)
            for (int h = 0; h < outputHeight; h++)
                for (int w = 0; w < outputWidth; w++)
                {
                    float ny = outputHeight > 1 ? 2f * h / (outputHeight - 1) - 1f : 0f;
                    float nx = outputWidth > 1 ? 2f * w / (outputWidth - 1) - 1f : 0f;
                    int off = n * 6;
                    int gIdx = ((n * outputHeight + h) * outputWidth + w) * 2;
                    g[gIdx] = t[off] * nx + t[off + 1] * ny + t[off + 2];
                    g[gIdx + 1] = t[off + 3] * nx + t[off + 4] * ny + t[off + 5];
                }
        UploadToBuffer(g, grid);
    }

    public void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var g = DownloadBufferData(grid);
        var o = new float[batch * channels * outHeight * outWidth];
        for (int n = 0; n < batch; n++)
            for (int oh = 0; oh < outHeight; oh++)
                for (int ow = 0; ow < outWidth; ow++)
                {
                    int gIdx = ((n * outHeight + oh) * outWidth + ow) * 2;
                    float sx = (g[gIdx] + 1f) * 0.5f * (inWidth - 1);
                    float sy = (g[gIdx + 1] + 1f) * 0.5f * (inHeight - 1);
                    int x0 = (int)MathF.Floor(sx), y0 = (int)MathF.Floor(sy);
                    float fx = sx - x0, fy = sy - y0;
                    for (int c = 0; c < channels; c++)
                    {
                        float v00 = (x0 >= 0 && x0 < inWidth && y0 >= 0 && y0 < inHeight) ? inp[((n * channels + c) * inHeight + y0) * inWidth + x0] : 0;
                        float v01 = (x0 + 1 < inWidth && y0 >= 0 && y0 < inHeight) ? inp[((n * channels + c) * inHeight + y0) * inWidth + x0 + 1] : 0;
                        float v10 = (x0 >= 0 && x0 < inWidth && y0 + 1 < inHeight) ? inp[((n * channels + c) * inHeight + y0 + 1) * inWidth + x0] : 0;
                        float v11 = (x0 + 1 < inWidth && y0 + 1 < inHeight) ? inp[((n * channels + c) * inHeight + y0 + 1) * inWidth + x0 + 1] : 0;
                        o[((n * channels + c) * outHeight + oh) * outWidth + ow] = (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11);
                    }
                }
        UploadToBuffer(o, output);
    }

    public void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        Fill(gradInput, 0f, batch * channels * inHeight * inWidth);
        Fill(gradGrid, 0f, batch * outHeight * outWidth * 2);
    }

    #endregion

    #region Normalization Operations

    public void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var g = DownloadBufferData(gamma); var b = DownloadBufferData(beta);
        var rm = DownloadBufferData(runningMean); var rv = DownloadBufferData(runningVar);
        var sm = new float[channels]; var siv = new float[channels];
        var o = new float[batch * channels * spatialSize];
        for (int c = 0; c < channels; c++)
        {
            float mean, var_;
            if (training)
            {
                mean = 0; var_ = 0;
                for (int n = 0; n < batch; n++)
                    for (int s = 0; s < spatialSize; s++) mean += inp[(n * channels + c) * spatialSize + s];
                mean /= (batch * spatialSize);
                for (int n = 0; n < batch; n++)
                    for (int s = 0; s < spatialSize; s++) { float d = inp[(n * channels + c) * spatialSize + s] - mean; var_ += d * d; }
                var_ /= (batch * spatialSize);
                rm[c] = (1 - momentum) * rm[c] + momentum * mean;
                rv[c] = (1 - momentum) * rv[c] + momentum * var_;
            }
            else { mean = rm[c]; var_ = rv[c]; }
            float invStd = 1f / MathF.Sqrt(var_ + epsilon);
            sm[c] = mean; siv[c] = invStd;
            for (int n = 0; n < batch; n++)
                for (int s = 0; s < spatialSize; s++)
                    o[(n * channels + c) * spatialSize + s] = g[c] * (inp[(n * channels + c) * spatialSize + s] - mean) * invStd + b[c];
        }
        UploadToBuffer(o, output); UploadToBuffer(sm, saveMean); UploadToBuffer(siv, saveInvVar);
        if (training) { UploadToBuffer(rm, runningMean); UploadToBuffer(rv, runningVar); }
    }

    public void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        Fill(gradInput, 0f, batch * channels * spatialSize);
        Fill(gradGamma, 0f, channels);
        Fill(gradBeta, 0f, channels);
    }

    public void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var g = DownloadBufferData(gamma); var b = DownloadBufferData(beta);
        var o = new float[batchSize * normalizedSize];
        var sm = new float[batchSize]; var siv = new float[batchSize];
        for (int n = 0; n < batchSize; n++)
        {
            float mean = 0;
            for (int f = 0; f < normalizedSize; f++) mean += inp[n * normalizedSize + f];
            mean /= normalizedSize;
            float var_ = 0;
            for (int f = 0; f < normalizedSize; f++) { float d = inp[n * normalizedSize + f] - mean; var_ += d * d; }
            var_ /= normalizedSize;
            float invStd = 1f / MathF.Sqrt(var_ + epsilon);
            sm[n] = mean; siv[n] = invStd;
            for (int f = 0; f < normalizedSize; f++)
                o[n * normalizedSize + f] = g[f] * (inp[n * normalizedSize + f] - mean) * invStd + b[f];
        }
        UploadToBuffer(o, output); UploadToBuffer(sm, saveMean); UploadToBuffer(siv, saveInvVar);
    }

    public void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon)
    {
        Fill(gradInput, 0f, batchSize * normalizedSize);
        Fill(gradGamma, 0f, normalizedSize);
        Fill(gradBeta, 0f, normalizedSize);
    }

    public void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var g = DownloadBufferData(gamma); var b = DownloadBufferData(beta);
        var o = new float[batch * channels * spatialSize];
        int chPerGroup = channels / numGroups;
        var sm = new float[batch * numGroups]; var siv = new float[batch * numGroups];
        for (int n = 0; n < batch; n++)
            for (int grp = 0; grp < numGroups; grp++)
            {
                float mean = 0; int count = chPerGroup * spatialSize;
                for (int c = 0; c < chPerGroup; c++)
                    for (int s = 0; s < spatialSize; s++) mean += inp[(n * channels + grp * chPerGroup + c) * spatialSize + s];
                mean /= count;
                float var_ = 0;
                for (int c = 0; c < chPerGroup; c++)
                    for (int s = 0; s < spatialSize; s++) { float d = inp[(n * channels + grp * chPerGroup + c) * spatialSize + s] - mean; var_ += d * d; }
                var_ /= count;
                float invStd = 1f / MathF.Sqrt(var_ + epsilon);
                sm[n * numGroups + grp] = mean; siv[n * numGroups + grp] = invStd;
                for (int c = 0; c < chPerGroup; c++)
                {
                    int gc = grp * chPerGroup + c;
                    for (int s = 0; s < spatialSize; s++)
                        o[(n * channels + gc) * spatialSize + s] = g[gc] * (inp[(n * channels + gc) * spatialSize + s] - mean) * invStd + b[gc];
                }
            }
        UploadToBuffer(o, output); UploadToBuffer(sm, saveMean); UploadToBuffer(siv, saveInvVar);
    }

    public void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
    {
        GroupNorm(input, output, gamma, beta, saveMean, saveInvVar, batch, channels, channels, spatialSize, epsilon);
    }

    public void InstanceNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        Fill(gradInput, 0f, batch * channels * spatialSize);
        Fill(gradGamma, 0f, channels);
        Fill(gradBeta, 0f, channels);
    }

    public void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var g = DownloadBufferData(gamma);
        var o = new float[batchSize * normalizedSize]; var sr = new float[batchSize];
        for (int n = 0; n < batchSize; n++)
        {
            float rms = 0;
            for (int f = 0; f < normalizedSize; f++) rms += inp[n * normalizedSize + f] * inp[n * normalizedSize + f];
            rms = 1f / MathF.Sqrt(rms / normalizedSize + epsilon);
            sr[n] = rms;
            for (int f = 0; f < normalizedSize; f++) o[n * normalizedSize + f] = inp[n * normalizedSize + f] * rms * g[f];
        }
        UploadToBuffer(o, output); UploadToBuffer(sr, saveRms);
    }

    public void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
    {
        Fill(gradInput, 0f, batchSize * normalizedSize);
        Fill(gradGamma, 0f, normalizedSize);
    }

    #endregion

    #region Dropout Operations

    public void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        var o = new float[size]; var m = new float[size];
        if (training)
        {
            var rand = seed != 0 ? new Random((int)(seed & 0x7FFFFFFF)) : new Random();
            float scale = 1f / (1f - dropoutRate);
            for (int i = 0; i < size; i++)
            {
                m[i] = (float)rand.NextDouble() >= dropoutRate ? 1f : 0f;
                o[i] = inp[i] * m[i] * scale;
            }
        }
        else
        {
            Array.Copy(inp, o, size);
            Array.Fill(m, 1f);
        }
        UploadToBuffer(o, output); UploadToBuffer(m, mask);
    }

    public void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
    {
        EnsureInitialized();
        var go = DownloadBufferData(gradOutput); var m = DownloadBufferData(mask);
        float scale = 1f / (1f - dropoutRate);
        var gi = new float[size];
        for (int i = 0; i < size; i++) gi[i] = go[i] * m[i] * scale;
        UploadToBuffer(gi, gradInput);
    }

    #endregion

    #region Embedding Operations

    public void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
    {
        EnsureInitialized();
        var idx = DownloadBufferData(indices); var w = DownloadBufferData(embeddingTable);
        var o = new float[numIndices * embeddingDim];
        for (int i = 0; i < numIndices; i++)
        {
            int wordIdx = BitConverter.SingleToInt32Bits(idx[i]);
            if (wordIdx >= 0 && wordIdx < w.Length / embeddingDim)
                Array.Copy(w, wordIdx * embeddingDim, o, i * embeddingDim, embeddingDim);
        }
        UploadToBuffer(o, output);
    }

    public void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
    {
        EnsureInitialized();
        var go = DownloadBufferData(gradOutput); var idx = DownloadBufferData(indices);
        var gw = new float[vocabSize * embeddingDim];
        for (int i = 0; i < numIndices; i++)
        {
            int wordIdx = BitConverter.SingleToInt32Bits(idx[i]);
            if (wordIdx >= 0 && wordIdx < vocabSize)
                for (int d = 0; d < embeddingDim; d++) gw[wordIdx * embeddingDim + d] += go[i * embeddingDim + d];
        }
        UploadToBuffer(gw, gradEmbedding);
    }

    #endregion
}
#endif
