// Copyright (c) AiDotNet. All rights reserved.
// HIP convolution kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipConvolutionKernels
{
    public static string GetSource()
    {
        // Note: hiprtc provides device intrinsics built-in, no includes needed
        return @"
// HIP RTC Compatibility - no includes needed, device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

extern ""C"" __global__ void im2col(
    const float* input, float* output,
    int batch, int channels, int height, int width,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPatches = batch * outH * outW;
    if (idx >= totalPatches) return;

    int b = idx / (outH * outW);
    int rem = idx % (outH * outW);
    int oh = rem / outW;
    int ow = rem % outW;

    int patchSize = channels * kernelH * kernelW;
    float* outPtr = output + idx * patchSize;

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

extern ""C"" __global__ void col2im(
    const float* input, float* output,
    int batch, int channels, int height, int width,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batch * channels * height * width;
    if (idx >= totalSize) return;

    int b = idx / (channels * height * width);
    int rem1 = idx % (channels * height * width);
    int c = rem1 / (height * width);
    int rem2 = rem1 % (height * width);
    int ih = rem2 / width;
    int iw = rem2 % width;

    float sum = 0.0f;
    int patchSize = channels * kernelH * kernelW;

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

extern ""C"" __global__ void conv2d_direct(
    const float* input, const float* kernel, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % outChannels;
    int b = blockIdx.z / outChannels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;
    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    float inVal = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                    float kernelVal = kernel[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * kernelVal;
                }
            }
        }
    }
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}

extern ""C"" __global__ void conv2d_backward_input(
    const float* gradOutput, const float* kernel, float* gradInput,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = blockIdx.z % inChannels;
    int b = blockIdx.z / inChannels;

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
                        float kernelVal = kernel[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                        sum += gradVal * kernelVal;
                    }
                }
            }
        }
    }
    gradInput[((b * inChannels + ic) * inHeight + ih) * inWidth + iw] = sum;
}

extern ""C"" __global__ void conv2d_backward_kernel(
    const float* input, const float* gradOutput, float* gradKernel,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int kw = blockIdx.x * blockDim.x + threadIdx.x;
    int kh = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = blockIdx.z % inChannels;
    int oc = blockIdx.z / inChannels;

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

extern ""C"" __global__ void depthwise_conv2d(
    const float* input, const float* kernel, float* output,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int kernelH, int kernelW,
    int strideH, int strideW, int padH, int padW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    if (ow >= outWidth || oh >= outHeight || b >= batch) return;

    float sum = 0.0f;
    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;
            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                float inVal = input[((b * channels + c) * inHeight + ih) * inWidth + iw];
                float kernelVal = kernel[(c * kernelH + kh) * kernelW + kw];
                sum += inVal * kernelVal;
            }
        }
    }
    output[((b * channels + c) * outHeight + oh) * outWidth + ow] = sum;
}

extern ""C"" __global__ void conv_transpose2d(
    const float* input, const float* kernel, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
    int outputPadH, int outputPadW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % outChannels;
    int b = blockIdx.z / outChannels;

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
                        float kernelVal = kernel[((ic * outChannels + oc) * kernelH + kh) * kernelW + kw];
                        sum += inVal * kernelVal;
                    }
                }
            }
        }
    }
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}

// Transposed Conv2D backward pass for input gradients
extern ""C"" __global__ void conv_transpose2d_backward_input(
    const float* gradOutput, const float* kernel, float* gradInput,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
    int outputPadH, int outputPadW, int totalInput)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalInput) return;

    int iw = idx % inWidth;
    int ih = (idx / inWidth) % inHeight;
    int ic = (idx / (inWidth * inHeight)) % inChannels;
    int b = idx / (inWidth * inHeight * inChannels);

    // Effective output dimensions excluding output padding
    int outHeight_eff = outHeight - outputPadH;
    int outWidth_eff = outWidth - outputPadW;

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
                    sum += gradOutput[goIdx] * kernel[kIdx];
                }
            }
        }
    }
    gradInput[idx] = sum;
}

// Transposed Conv2D backward pass for kernel gradients
extern ""C"" __global__ void conv_transpose2d_backward_kernel(
    const float* input, const float* gradOutput, float* gradKernel,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
    int outputPadH, int outputPadW, int totalKernel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalKernel) return;

    int kw = idx % kernelW;
    int kh = (idx / kernelW) % kernelH;
    int oc = (idx / (kernelW * kernelH)) % outChannels;
    int ic = idx / (kernelW * kernelH * outChannels);

    // Effective output dimensions excluding output padding
    int outHeight_eff = outHeight - outputPadH;
    int outWidth_eff = outWidth - outputPadW;

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
    gradKernel[idx] = sum;
}

// ===========================================================================
// SHARED-MEMORY TILED CONV2D
// ===========================================================================
#define TILE_OUT 16

extern ""C"" __global__ void conv2d_tiled(
    const float* input, const float* weight, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int tileRow = blockIdx.y * TILE_OUT;
    int tileCol = blockIdx.x * TILE_OUT;
    int oc = blockIdx.z % outChannels;
    int b  = blockIdx.z / outChannels;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int oh = tileRow + ty;
    int ow = tileCol + tx;

    int effKH = (kernelH - 1) * dilationH + 1;
    int effKW = (kernelW - 1) * dilationW + 1;
    int tileInH = TILE_OUT * strideH + effKH - strideH;
    int tileInW = TILE_OUT * strideW + effKW - strideW;

    extern __shared__ float smem[];

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        int inRowStart = tileRow * strideH - padH;
        int inColStart = tileCol * strideW - padW;

        int tilesPerThread = (tileInH * tileInW + TILE_OUT * TILE_OUT - 1) / (TILE_OUT * TILE_OUT);
        int tid = ty * TILE_OUT + tx;

        for (int t = 0; t < tilesPerThread; t++) {
            int idx = tid + t * TILE_OUT * TILE_OUT;
            if (idx < tileInH * tileInW) {
                int sy = idx / tileInW;
                int sx = idx % tileInW;
                int iy = inRowStart + sy;
                int ix = inColStart + sx;
                float val = 0.0f;
                if (iy >= 0 && iy < inHeight && ix >= 0 && ix < inWidth) {
                    val = input[((b * inChannels + ic) * inHeight + iy) * inWidth + ix];
                }
                smem[idx] = val;
            }
        }
        __syncthreads();

        if (oh < outHeight && ow < outWidth) {
            for (int kh = 0; kh < kernelH; kh++) {
                for (int kw = 0; kw < kernelW; kw++) {
                    int sy = ty * strideH + kh * dilationH;
                    int sx = tx * strideW + kw * dilationW;
                    float inVal = smem[sy * tileInW + sx];
                    float wVal = weight[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                    sum += inVal * wVal;
                }
            }
        }
        __syncthreads();
    }

    if (oh < outHeight && ow < outWidth) {
        output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
    }
}

// ===========================================================================
// WINOGRAD F(2x2, 3x3) CONVOLUTION
// ===========================================================================

__device__ __forceinline__ void winograd_input_transform(const float tile[4][4], float d[4][4])
{
    float tmp[4][4];
    for (int j = 0; j < 4; j++) {
        tmp[0][j] = tile[0][j] - tile[2][j];
        tmp[1][j] = tile[1][j] + tile[2][j];
        tmp[2][j] = -tile[1][j] + tile[2][j];
        tmp[3][j] = tile[1][j] - tile[3][j];
    }
    for (int i = 0; i < 4; i++) {
        d[i][0] = tmp[i][0] - tmp[i][2];
        d[i][1] = tmp[i][1] + tmp[i][2];
        d[i][2] = -tmp[i][1] + tmp[i][2];
        d[i][3] = tmp[i][1] - tmp[i][3];
    }
}

__device__ __forceinline__ void winograd_filter_transform(const float g[3][3], float u[4][4])
{
    float tmp[4][3];
    for (int j = 0; j < 3; j++) {
        tmp[0][j] = g[0][j];
        tmp[1][j] = 0.5f * (g[0][j] + g[1][j] + g[2][j]);
        tmp[2][j] = 0.5f * (g[0][j] - g[1][j] + g[2][j]);
        tmp[3][j] = g[2][j];
    }
    for (int i = 0; i < 4; i++) {
        u[i][0] = tmp[i][0];
        u[i][1] = 0.5f * (tmp[i][0] + tmp[i][1] + tmp[i][2]);
        u[i][2] = 0.5f * (tmp[i][0] - tmp[i][1] + tmp[i][2]);
        u[i][3] = tmp[i][2];
    }
}

__device__ __forceinline__ void winograd_output_transform(const float m[4][4], float out2x2[2][2])
{
    float tmp[2][4];
    for (int j = 0; j < 4; j++) {
        tmp[0][j] = m[0][j] + m[1][j] + m[2][j];
        tmp[1][j] = m[1][j] - m[2][j] - m[3][j];
    }
    out2x2[0][0] = tmp[0][0] + tmp[0][1] + tmp[0][2];
    out2x2[0][1] = tmp[0][1] - tmp[0][2] - tmp[0][3];
    out2x2[1][0] = tmp[1][0] + tmp[1][1] + tmp[1][2];
    out2x2[1][1] = tmp[1][1] - tmp[1][2] - tmp[1][3];
}

extern ""C"" __global__ void conv2d_winograd_f2x2_3x3(
    const float* input, const float* weight, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int padH, int padW)
{
    int tilesH = (outHeight + 1) / 2;
    int tilesW = (outWidth + 1) / 2;

    int tileIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalTiles = batch * outChannels * tilesH * tilesW;
    if (tileIdx >= totalTiles) return;

    int tw = tileIdx % tilesW;
    int th = (tileIdx / tilesW) % tilesH;
    int oc = (tileIdx / (tilesW * tilesH)) % outChannels;
    int b  = tileIdx / (tilesW * tilesH * outChannels);

    int outRow = th * 2;
    int outCol = tw * 2;

    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        float tile[4][4];
        int inRow = outRow - padH;
        int inCol = outCol - padW;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int r = inRow + i;
                int c = inCol + j;
                tile[i][j] = (r >= 0 && r < inHeight && c >= 0 && c < inWidth)
                    ? input[((b * inChannels + ic) * inHeight + r) * inWidth + c]
                    : 0.0f;
            }
        }

        float g[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                g[i][j] = weight[((oc * inChannels + ic) * 3 + i) * 3 + j];

        float d[4][4], u[4][4];
        winograd_input_transform(tile, d);
        winograd_filter_transform(g, u);

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                acc[i][j] += d[i][j] * u[i][j];
    }

    float out2x2[2][2];
    winograd_output_transform(acc, out2x2);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int r = outRow + i;
            int c = outCol + j;
            if (r < outHeight && c < outWidth) {
                output[((b * outChannels + oc) * outHeight + r) * outWidth + c] = out2x2[i][j];
            }
        }
    }
}

extern ""C"" __global__ void conv3d_direct(
    const float* input, const float* kernel, float* output,
    int batch, int inChannels, int inDepth, int inHeight, int inWidth,
    int outChannels, int outDepth, int outHeight, int outWidth,
    int kernelD, int kernelH, int kernelW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int dilationD, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int od = blockIdx.z % outDepth;
    int oc = (blockIdx.z / outDepth) % outChannels;
    int b = blockIdx.z / (outDepth * outChannels);

    if (ow >= outWidth || oh >= outHeight || od >= outDepth || b >= batch) return;

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
                        float kernelVal = kernel[(((oc * inChannels + ic) * kernelD + kd) * kernelH + kh) * kernelW + kw];
                        sum += inVal * kernelVal;
                    }
                }
            }
        }
    }
    output[(((b * outChannels + oc) * outDepth + od) * outHeight + oh) * outWidth + ow] = sum;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "im2col", "col2im", "conv2d_direct", "conv2d_backward_input",
            "conv2d_backward_kernel", "depthwise_conv2d", "conv_transpose2d",
            "conv_transpose2d_backward_input", "conv_transpose2d_backward_kernel",
            "conv2d_tiled", "conv2d_winograd_f2x2_3x3", "conv3d_direct"
        };
    }
}
