// Copyright (c) AiDotNet. All rights reserved.
// Convolution kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for convolution operations.
    /// Implements im2col + GEMM approach for efficiency.
    /// </summary>
    internal static class ConvolutionKernels
    {
        /// <summary>
        /// Gets all convolution kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// CONVOLUTION KERNELS
// ===========================================================================

// Im2Col transformation for efficient convolution via GEMM
// Transforms input patches into columns for matrix multiplication
__kernel void im2col(
    __global const float* input,
    __global float* output,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int outH,
    const int outW)
{
    const int idx = get_global_id(0);
    const int totalPatches = batch * outH * outW;
    if (idx >= totalPatches) return;

    const int b = idx / (outH * outW);
    const int rem = idx % (outH * outW);
    const int oh = rem / outW;
    const int ow = rem % outW;

    const int patchSize = channels * kernelH * kernelW;
    __global float* outPtr = output + idx * patchSize;

    for (int c = 0; c < channels; c++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                float val = 0.0f;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    val = input[((b * channels + c) * height + ih) * width + iw];
                }

                int outIdx = (c * kernelH + kh) * kernelW + kw;
                outPtr[outIdx] = val;
            }
        }
    }
}

// Col2Im transformation for convolution backward pass
// Accumulates gradients from columns back to input gradient
__kernel void col2im(
    __global const float* input,
    __global float* output,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int outH,
    const int outW)
{
    const int idx = get_global_id(0);
    const int totalSize = batch * channels * height * width;
    if (idx >= totalSize) return;

    const int b = idx / (channels * height * width);
    const int rem1 = idx % (channels * height * width);
    const int c = rem1 / (height * width);
    const int rem2 = rem1 % (height * width);
    const int ih = rem2 / width;
    const int iw = rem2 % width;

    float sum = 0.0f;
    const int patchSize = channels * kernelH * kernelW;

    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int oh_base = ih + padH - kh * dilationH;
            int ow_base = iw + padW - kw * dilationW;

            if (oh_base % strideH == 0 && ow_base % strideW == 0) {
                int oh = oh_base / strideH;
                int ow = ow_base / strideW;

                if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                    int patchIdx = (b * outH + oh) * outW + ow;
                    int colIdx = (c * kernelH + kh) * kernelW + kw;
                    sum += input[patchIdx * patchSize + colIdx];
                }
            }
        }
    }

    output[idx] = sum;
}

// Direct Conv2D kernel for small kernels (3x3, 5x5)
// Uses shared memory for input tile caching
__kernel void conv2d_direct(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW)
{
    const int ow = get_global_id(0);
    const int oh = get_global_id(1);
    const int idx2 = get_global_id(2);
    const int oc = idx2 % outChannels;
    const int b = idx2 / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float wVal = weights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * wVal;
                }
            }
        }
    }

    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}

// Conv2D backward pass for input gradients
__kernel void conv2d_backward_input(
    __global const float* gradOutput,
    __global const float* weights,
    __global float* gradInput,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW)
{
    const int iw = get_global_id(0);
    const int ih = get_global_id(1);
    const int idx2 = get_global_id(2);
    const int ic = idx2 % inChannels;
    const int b = idx2 / inChannels;

    if (iw >= inWidth || ih >= inHeight || b >= batch) return;

    float sum = 0.0f;

    for (int oc = 0; oc < outChannels; oc++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int oh_base = ih + padH - kh * dilationH;
                int ow_base = iw + padW - kw * dilationW;

                if (oh_base % strideH == 0 && ow_base % strideW == 0) {
                    int oh = oh_base / strideH;
                    int ow = ow_base / strideW;

                    if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth) {
                        float gradVal = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                        float wVal = weights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                        sum += gradVal * wVal;
                    }
                }
            }
        }
    }

    gradInput[((b * inChannels + ic) * inHeight + ih) * inWidth + iw] = sum;
}

// Conv2D backward pass for weight gradients
__kernel void conv2d_backward_weights(
    __global const float* input,
    __global const float* gradOutput,
    __global float* gradKernel,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW)
{
    const int kw = get_global_id(0);
    const int kh = get_global_id(1);
    const int idx2 = get_global_id(2);
    const int ic = idx2 % inChannels;
    const int oc = idx2 / inChannels;

    if (kw >= kernelW || kh >= kernelH || oc >= outChannels) return;

    float sum = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < outHeight; oh++) {
            for (int ow = 0; ow < outWidth; ow++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float gradVal = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                    sum += inVal * gradVal;
                }
            }
        }
    }

    gradKernel[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw] = sum;
}

// Depthwise Conv2D - each channel is convolved independently
__kernel void depthwise_conv2d(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    const int batch,
    const int channels,
    const int inHeight,
    const int inWidth,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW)
{
    const int ow = get_global_id(0);
    const int oh = get_global_id(1);
    const int idx2 = get_global_id(2);
    const int c = idx2 % channels;
    const int b = idx2 / channels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;

            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                float inVal = input[((b * channels + c) * inHeight + ih) * inWidth + iw];
                float wVal = weights[(c * kernelH + kh) * kernelW + kw];
                sum += inVal * wVal;
            }
        }
    }

    output[((b * channels + c) * outHeight + oh) * outWidth + ow] = sum;
}

// Transposed Conv2D (deconvolution) with output padding support
__kernel void conv_transpose2d(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int outputPadH,
    const int outputPadW)
{
    const int ow = get_global_id(0);
    const int oh = get_global_id(1);
    const int idx2 = get_global_id(2);
    const int oc = idx2 % outChannels;
    const int b = idx2 / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih_base = oh + padH - kh;
                int iw_base = ow + padW - kw;

                if (ih_base % strideH == 0 && iw_base % strideW == 0) {
                    int ih = ih_base / strideH;
                    int iw = iw_base / strideW;

                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                        float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                        // Note: weights layout is [inChannels, outChannels, kH, kW]
                        float wVal = weights[((ic * outChannels + oc) * kernelH + kh) * kernelW + kw];
                        sum += inVal * wVal;
                    }
                }
            }
        }
    }

    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}

// Transposed Conv2D backward pass for input gradients
__kernel void conv_transpose2d_backward_input(
    __global const float* gradOutput,
    __global const float* weights,
    __global float* gradInput,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int outputPadH,
    const int outputPadW,
    const int totalInput)
{
    const int idx = get_global_id(0);
    if (idx >= totalInput) return;

    const int iw = idx % inWidth;
    const int ih = (idx / inWidth) % inHeight;
    const int ic = (idx / (inWidth * inHeight)) % inChannels;
    const int b = idx / (inWidth * inHeight * inChannels);

    // Effective output dimensions excluding output padding
    const int outHeight_eff = outHeight - outputPadH;
    const int outWidth_eff = outWidth - outputPadW;

    float sum = 0.0f;

    for (int oc = 0; oc < outChannels; oc++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int oh = ih * strideH - padH + kh;
                int ow = iw * strideW - padW + kw;

                // Only consider positions within effective output (excluding output padding region)
                if (oh >= 0 && oh < outHeight_eff && ow >= 0 && ow < outWidth_eff) {
                    int goIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
                    int kIdx = ((ic * outChannels + oc) * kernelH + kh) * kernelW + kw;
                    sum += gradOutput[goIdx] * weights[kIdx];
                }
            }
        }
    }

    gradInput[idx] = sum;
}

// Transposed Conv2D backward pass for kernel gradients
__kernel void conv_transpose2d_backward_weights(
    __global const float* input,
    __global const float* gradOutput,
    __global float* gradWeights,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int outputPadH,
    const int outputPadW,
    const int totalKernel)
{
    const int idx = get_global_id(0);
    if (idx >= totalKernel) return;

    const int kw = idx % kernelW;
    const int kh = (idx / kernelW) % kernelH;
    const int oc = (idx / (kernelW * kernelH)) % outChannels;
    const int ic = idx / (kernelW * kernelH * outChannels);

    // Effective output dimensions excluding output padding
    const int outHeight_eff = outHeight - outputPadH;
    const int outWidth_eff = outWidth - outputPadW;

    float sum = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int ih = 0; ih < inHeight; ih++) {
            for (int iw = 0; iw < inWidth; iw++) {
                int oh = ih * strideH - padH + kh;
                int ow = iw * strideW - padW + kw;

                // Only consider positions within effective output (excluding output padding region)
                if (oh >= 0 && oh < outHeight_eff && ow >= 0 && ow < outWidth_eff) {
                    int inIdx = ((b * inChannels + ic) * inHeight + ih) * inWidth + iw;
                    int goIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
                    sum += input[inIdx] * gradOutput[goIdx];
                }
            }
        }
    }

    gradWeights[idx] = sum;
}

// ===========================================================================
// TILED CONV2D WITH SHARED MEMORY
// 16x16 output tile, cooperative input tile loading with halo
// ===========================================================================

#define TILE_OUT 16

__kernel void conv2d_tiled(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int kernelH,
    const int kernelW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    __local float* sharedInput)
{
    // Each workgroup handles one output tile for one (batch, outChannel) pair
    const int tileX = get_group_id(0);  // output tile column
    const int tileY = get_group_id(1);  // output tile row
    const int boc = get_group_id(2);    // batch * outChannels + oc
    const int oc = boc % outChannels;
    const int b = boc / outChannels;

    const int lidX = get_local_id(0);
    const int lidY = get_local_id(1);
    const int outX = tileX * TILE_OUT + lidX;
    const int outY = tileY * TILE_OUT + lidY;

    if (b >= batch) return;

    float sum = 0.0f;

    // Process one input channel at a time through shared memory
    for (int ic = 0; ic < inChannels; ic++) {
        // Compute shared memory tile dimensions (input tile = output tile * stride + kernel halo)
        int tileInH = TILE_OUT * strideH + (kernelH - 1) * dilationH;
        int tileInW = TILE_OUT * strideW + (kernelW - 1) * dilationW;
        int tileInStartH = tileY * TILE_OUT * strideH - padH;
        int tileInStartW = tileX * TILE_OUT * strideW - padW;

        // Cooperative loading of input tile into shared memory
        int totalElements = tileInH * tileInW;
        int threadsPerGroup = TILE_OUT * TILE_OUT;
        for (int i = lidY * TILE_OUT + lidX; i < totalElements; i += threadsPerGroup) {
            int localH = i / tileInW;
            int localW = i % tileInW;
            int globalH = tileInStartH + localH;
            int globalW = tileInStartW + localW;
            float val = 0.0f;
            if (globalH >= 0 && globalH < inHeight && globalW >= 0 && globalW < inWidth) {
                val = input[((b * inChannels + ic) * inHeight + globalH) * inWidth + globalW];
            }
            sharedInput[localH * tileInW + localW] = val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute convolution for this output element using shared memory
        if (outX < outWidth && outY < outHeight) {
            for (int kh = 0; kh < kernelH; kh++) {
                for (int kw = 0; kw < kernelW; kw++) {
                    int localH = lidY * strideH + kh * dilationH;
                    int localW = lidX * strideW + kw * dilationW;
                    float inVal = sharedInput[localH * tileInW + localW];
                    float wVal = weights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * wVal;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (outX < outWidth && outY < outHeight) {
        output[((b * outChannels + oc) * outHeight + outY) * outWidth + outX] = sum;
    }
}

// ===========================================================================
// WINOGRAD F(2x2, 3x3) CONVOLUTION
// Reduces multiplications from 36 to 16 per 2x2 output tile for 3x3 kernels
// stride=1, dilation=1 only
// ===========================================================================

__kernel void conv2d_winograd_f2x2_3x3(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    const int batch,
    const int inChannels,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outHeight,
    const int outWidth,
    const int padH,
    const int padW)
{
    // Each work-item computes one 2x2 output tile
    const int tileX = get_global_id(0);  // tile column index
    const int tileY = get_global_id(1);  // tile row index
    const int boc = get_global_id(2);    // batch * outChannels + oc
    const int oc = boc % outChannels;
    const int b = boc / outChannels;

    int numTilesX = (outWidth + 1) / 2;
    int numTilesY = (outHeight + 1) / 2;
    if (tileX >= numTilesX || tileY >= numTilesY || b >= batch) return;

    int outBaseY = tileY * 2;
    int outBaseX = tileX * 2;

    // Accumulate across input channels in transform domain
    float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
    float m4 = 0.0f, m5 = 0.0f, m6 = 0.0f, m7 = 0.0f;
    float m8 = 0.0f, m9 = 0.0f, m10 = 0.0f, m11 = 0.0f;
    float m12 = 0.0f, m13 = 0.0f, m14 = 0.0f, m15 = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        // Load 4x4 input tile
        float d[4][4];
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                int ih = outBaseY + r - padH;
                int iw = outBaseX + c - padW;
                d[r][c] = (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                    ? input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw]
                    : 0.0f;
            }
        }

        // B^T * d * B (input transform)
        float bt0 = d[0][0] - d[2][0];
        float bt1 = d[1][0] + d[2][0];
        float bt2 = d[2][0] - d[1][0];
        float bt3 = d[1][0] - d[3][0];
        float bt4 = d[0][1] - d[2][1];
        float bt5 = d[1][1] + d[2][1];
        float bt6 = d[2][1] - d[1][1];
        float bt7 = d[1][1] - d[3][1];
        float bt8 = d[0][2] - d[2][2];
        float bt9 = d[1][2] + d[2][2];
        float bt10 = d[2][2] - d[1][2];
        float bt11 = d[1][2] - d[3][2];
        float bt12 = d[0][3] - d[2][3];
        float bt13 = d[1][3] + d[2][3];
        float bt14 = d[2][3] - d[1][3];
        float bt15 = d[1][3] - d[3][3];

        float v0 = bt0 - bt8;
        float v1 = bt4 + bt8;
        float v2 = bt8 - bt4;
        float v3 = bt4 - bt12;
        float v4 = bt1 - bt9;
        float v5 = bt5 + bt9;
        float v6 = bt9 - bt5;
        float v7 = bt5 - bt13;
        float v8 = bt2 - bt10;
        float v9 = bt6 + bt10;
        float v10 = bt10 - bt6;
        float v11 = bt6 - bt14;
        float v12 = bt3 - bt11;
        float v13 = bt7 + bt11;
        float v14 = bt11 - bt7;
        float v15 = bt7 - bt15;

        // Load 3x3 filter and compute G * g * G^T (filter transform)
        float g[3][3];
        for (int r = 0; r < 3; r++) {
            for (int c2 = 0; c2 < 3; c2++) {
                g[r][c2] = weights[((oc * inChannels + ic) * 3 + r) * 3 + c2];
            }
        }

        float u0 = g[0][0];
        float u1 = 0.5f * (g[0][0] + g[0][1] + g[0][2]);
        float u2 = 0.5f * (g[0][0] - g[0][1] + g[0][2]);
        float u3 = g[0][2];
        float u4 = 0.5f * (g[0][0] + g[1][0] + g[2][0]);
        float u5 = 0.25f * (g[0][0]+g[0][1]+g[0][2]+g[1][0]+g[1][1]+g[1][2]+g[2][0]+g[2][1]+g[2][2]);
        float u6 = 0.25f * (g[0][0]-g[0][1]+g[0][2]+g[1][0]-g[1][1]+g[1][2]+g[2][0]-g[2][1]+g[2][2]);
        float u7 = 0.5f * (g[0][2] + g[1][2] + g[2][2]);
        float u8 = 0.5f * (g[0][0] - g[1][0] + g[2][0]);
        float u9 = 0.25f * (g[0][0]+g[0][1]+g[0][2]-g[1][0]-g[1][1]-g[1][2]+g[2][0]+g[2][1]+g[2][2]);
        float u10 = 0.25f * (g[0][0]-g[0][1]+g[0][2]-g[1][0]+g[1][1]-g[1][2]+g[2][0]-g[2][1]+g[2][2]);
        float u11 = 0.5f * (g[0][2] - g[1][2] + g[2][2]);
        float u12 = g[2][0];
        float u13 = 0.5f * (g[2][0] + g[2][1] + g[2][2]);
        float u14 = 0.5f * (g[2][0] - g[2][1] + g[2][2]);
        float u15 = g[2][2];

        // Element-wise multiply (accumulate)
        m0 += v0 * u0;   m1 += v1 * u1;   m2 += v2 * u2;   m3 += v3 * u3;
        m4 += v4 * u4;   m5 += v5 * u5;   m6 += v6 * u6;   m7 += v7 * u7;
        m8 += v8 * u8;   m9 += v9 * u9;   m10 += v10 * u10; m11 += v11 * u11;
        m12 += v12 * u12; m13 += v13 * u13; m14 += v14 * u14; m15 += v15 * u15;
    }

    // A^T * M * A (output transform)
    float t0 = m0 + m4 + m8;
    float t1 = m1 + m5 + m9;
    float t2 = m2 + m6 + m10;
    float t3 = m3 + m7 + m11;
    float t4 = m4 - m8 - m12;
    float t5 = m5 - m9 - m13;
    float t6 = m6 - m10 - m14;
    float t7 = m7 - m11 - m15;

    float y00 = t0 + t1 + t2;
    float y01 = t1 - t2 - t3;
    float y10 = t4 + t5 + t6;
    float y11 = t5 - t6 - t7;

    // Write output (handle boundary)
    if (outBaseY < outHeight && outBaseX < outWidth)
        output[((b * outChannels + oc) * outHeight + outBaseY) * outWidth + outBaseX] = y00;
    if (outBaseY < outHeight && outBaseX + 1 < outWidth)
        output[((b * outChannels + oc) * outHeight + outBaseY) * outWidth + outBaseX + 1] = y01;
    if (outBaseY + 1 < outHeight && outBaseX < outWidth)
        output[((b * outChannels + oc) * outHeight + (outBaseY + 1)) * outWidth + outBaseX] = y10;
    if (outBaseY + 1 < outHeight && outBaseX + 1 < outWidth)
        output[((b * outChannels + oc) * outHeight + (outBaseY + 1)) * outWidth + outBaseX + 1] = y11;
}

// Conv3D for volumetric data
__kernel void conv3d_direct(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    const int batch,
    const int inChannels,
    const int inDepth,
    const int inHeight,
    const int inWidth,
    const int outChannels,
    const int outDepth,
    const int outHeight,
    const int outWidth,
    const int kernelD,
    const int kernelH,
    const int kernelW,
    const int strideD,
    const int strideH,
    const int strideW,
    const int padD,
    const int padH,
    const int padW,
    const int dilationD,
    const int dilationH,
    const int dilationW)
{
    const int ow = get_global_id(0);
    const int idx1 = get_global_id(1);
    const int oh = idx1 % outHeight;
    const int od = idx1 / outHeight;
    const int idx2 = get_global_id(2);
    const int oc = idx2 % outChannels;
    const int b = idx2 / outChannels;

    if (ow >= outWidth || od >= outDepth || b >= batch) return;

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kd = 0; kd < kernelD; kd++) {
            for (int kh = 0; kh < kernelH; kh++) {
                for (int kw = 0; kw < kernelW; kw++) {
                    int id = od * strideD - padD + kd * dilationD;
                    int ih = oh * strideH - padH + kh * dilationH;
                    int iw = ow * strideW - padW + kw * dilationW;

                    if (id >= 0 && id < inDepth && ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                        float inVal = input[(((b * inChannels + ic) * inDepth + id) * inHeight + ih) * inWidth + iw];
                        float wVal = weights[(((oc * inChannels + ic) * kernelD + kd) * kernelH + kh) * kernelW + kw];
                        sum += inVal * wVal;
                    }
                }
            }
        }
    }

    output[(((b * outChannels + oc) * outDepth + od) * outHeight + oh) * outWidth + ow] = sum;
}
";
        }

        /// <summary>
        /// Gets kernel names for compilation.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new string[]
            {
                "im2col",
                "col2im",
                "conv2d_direct",
                "conv2d_backward_input",
                "conv2d_backward_weights",
                "depthwise_conv2d",
                "conv_transpose2d",
                "conv_transpose2d_backward_input",
                "conv_transpose2d_backward_weights",
                "conv2d_tiled",
                "conv2d_winograd_f2x2_3x3",
                "conv3d_direct"
            };
        }
    }
}
