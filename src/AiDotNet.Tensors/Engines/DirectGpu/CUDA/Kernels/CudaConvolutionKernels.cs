// Copyright (c) AiDotNet. All rights reserved.
// CUDA convolution kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA kernels for convolution operations.
    /// </summary>
    internal static class CudaConvolutionKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

// ===========================================================================
// CONVOLUTION KERNELS
// ===========================================================================

// Im2Col transformation for efficient convolution via GEMM
extern ""C"" __global__ __launch_bounds__(256) void im2col(
    const float* __restrict__ input, float* __restrict__ output,
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

// Col2Im transformation for convolution backward pass
extern ""C"" __global__ __launch_bounds__(256) void col2im(
    const float* __restrict__ input, float* __restrict__ output,
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

// Direct Conv2D kernel
extern ""C"" __global__ __launch_bounds__(256) void conv2d_direct(
    const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output,
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

// Conv2D backward pass for input gradients
extern ""C"" __global__ __launch_bounds__(256) void conv2d_backward_input(
    const float* __restrict__ gradOutput, const float* __restrict__ kernel, float* __restrict__ gradInput,
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

// Conv2D backward pass for kernel gradients
extern ""C"" __global__ __launch_bounds__(256) void conv2d_backward_kernel(
    const float* __restrict__ input, const float* __restrict__ gradOutput, float* __restrict__ gradKernel,
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

// Depthwise Conv2D
extern ""C"" __global__ __launch_bounds__(256) void depthwise_conv2d(
    const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output,
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

// Transposed Conv2D (deconvolution) with output padding support
extern ""C"" __global__ __launch_bounds__(256) void conv_transpose2d(
    const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output,
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

    // Output padding shifts which output pixels receive contributions
    // Effective output position accounting for output padding
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
// dL/dX = conv2d(dL/dY, W) - essentially a regular convolution
extern ""C"" __global__ __launch_bounds__(256) void conv_transpose2d_backward_input(
    const float* __restrict__ gradOutput, const float* __restrict__ kernel, float* __restrict__ gradInput,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
    int outputPadH, int outputPadW, int totalInput)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalInput) return;

    // Decompose linear index into (b, ic, ih, iw)
    int iw = idx % inWidth;
    int ih = (idx / inWidth) % inHeight;
    int ic = (idx / (inWidth * inHeight)) % inChannels;
    int b = idx / (inWidth * inHeight * inChannels);

    // Effective output dimensions excluding output padding
    // Output padding adds zeros at the end that don't receive any contributions from input
    int outHeight_eff = outHeight - outputPadH;
    int outWidth_eff = outWidth - outputPadW;

    float sum = 0.0f;

    // For transposed conv backward w.r.t input, we convolve gradOutput with kernel
    for (int oc = 0; oc < outChannels; oc++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                // Output position that this input element contributed to
                int oh = ih * strideH - padH + kh;
                int ow = iw * strideW - padW + kw;

                // Only consider positions within effective output (excluding output padding region)
                if (oh >= 0 && oh < outHeight_eff && ow >= 0 && ow < outWidth_eff) {
                    int goIdx = ((b * outChannels + oc) * outHeight + oh) * outWidth + ow;
                    // Kernel layout: [inChannels, outChannels, kernelH, kernelW]
                    int kIdx = ((ic * outChannels + oc) * kernelH + kh) * kernelW + kw;
                    sum += gradOutput[goIdx] * kernel[kIdx];
                }
            }
        }
    }

    gradInput[idx] = sum;
}

// Transposed Conv2D backward pass for kernel gradients
// dL/dW[ic,oc,kh,kw] = sum over batch and positions of input[b,ic,ih,iw] * gradOutput[b,oc,oh,ow]
extern ""C"" __global__ __launch_bounds__(256) void conv_transpose2d_backward_kernel(
    const float* __restrict__ input, const float* __restrict__ gradOutput, float* __restrict__ gradKernel,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
    int outputPadH, int outputPadW, int totalKernel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalKernel) return;

    // Decompose linear index into (ic, oc, kh, kw)
    int kw = idx % kernelW;
    int kh = (idx / kernelW) % kernelH;
    int oc = (idx / (kernelW * kernelH)) % outChannels;
    int ic = idx / (kernelW * kernelH * outChannels);

    // Effective output dimensions excluding output padding
    // Output padding adds zeros at the end that don't receive any contributions from input
    int outHeight_eff = outHeight - outputPadH;
    int outWidth_eff = outWidth - outputPadW;

    float sum = 0.0f;

    // Accumulate gradient over batch and spatial positions
    for (int b = 0; b < batch; b++) {
        for (int ih = 0; ih < inHeight; ih++) {
            for (int iw = 0; iw < inWidth; iw++) {
                // Output position that this input+kernel contributed to
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
// Each block computes a TILE_OUT x TILE_OUT tile of output for one (batch, outChannel).
// Input tile (including halo) is loaded into shared memory for reuse across threads.
// ===========================================================================
#define TILE_OUT 16

extern ""C"" __global__ __launch_bounds__(256) void conv2d_tiled(
    const float* input, const float* weight, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    // Block handles one (batch, outChannel) pair for a TILE_OUT x TILE_OUT output region
    int tileRow = blockIdx.y * TILE_OUT;
    int tileCol = blockIdx.x * TILE_OUT;
    int oc = blockIdx.z % outChannels;
    int b  = blockIdx.z / outChannels;

    int tx = threadIdx.x; // 0..TILE_OUT-1
    int ty = threadIdx.y; // 0..TILE_OUT-1

    int oh = tileRow + ty;
    int ow = tileCol + tx;

    // Effective kernel footprint with dilation
    int effKH = (kernelH - 1) * dilationH + 1;
    int effKW = (kernelW - 1) * dilationW + 1;

    // Input tile dimensions (output tile * stride + kernel footprint - stride)
    int tileInH = TILE_OUT * strideH + effKH - strideH;
    int tileInW = TILE_OUT * strideW + effKW - strideW;

    // Dynamic shared memory for one input channel tile
    extern __shared__ float smem[];

    float sum = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        // Load input tile into shared memory cooperatively
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

        // Compute convolution from shared memory
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
// For 3x3 kernels with stride=1. Each 4x4 input tile produces a 2x2 output tile.
// 16 multiplies per tile instead of 36 = 2.25x compute reduction.
// ===========================================================================

// Winograd input transform: d = B^T * tile * B (4x4 -> 4x4)
// B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
__device__ __forceinline__ void winograd_input_transform(const float tile[4][4], float d[4][4])
{
    // temp = B^T * tile (rows)
    float tmp[4][4];
    for (int j = 0; j < 4; j++) {
        tmp[0][j] = tile[0][j] - tile[2][j];
        tmp[1][j] = tile[1][j] + tile[2][j];
        tmp[2][j] = -tile[1][j] + tile[2][j];
        tmp[3][j] = tile[1][j] - tile[3][j];
    }
    // d = tmp * B (cols)
    for (int i = 0; i < 4; i++) {
        d[i][0] = tmp[i][0] - tmp[i][2];
        d[i][1] = tmp[i][1] + tmp[i][2];
        d[i][2] = -tmp[i][1] + tmp[i][2];
        d[i][3] = tmp[i][1] - tmp[i][3];
    }
}

// Winograd filter transform: G * g * G^T (3x3 -> 4x4)
// G = [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]
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

// Winograd output transform: A^T * m * A (4x4 -> 2x2)
// A^T = [[1,1,1,0],[0,1,-1,-1]]
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

// Winograd F(2x2, 3x3) convolution kernel
// Each thread handles one 2x2 output tile for one (batch, outChannel) pair.
extern ""C"" __global__ __launch_bounds__(256) void conv2d_winograd_f2x2_3x3(
    const float* input, const float* weight, float* output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int padH, int padW)
{
    // Number of 2x2 output tiles
    int tilesH = (outHeight + 1) / 2;
    int tilesW = (outWidth + 1) / 2;

    int tileIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalTiles = batch * outChannels * tilesH * tilesW;
    if (tileIdx >= totalTiles) return;

    // Decode tile index
    int tw = tileIdx % tilesW;
    int th = (tileIdx / tilesW) % tilesH;
    int oc = (tileIdx / (tilesW * tilesH)) % outChannels;
    int b  = tileIdx / (tilesW * tilesH * outChannels);

    // Output position (top-left of 2x2 tile)
    int outRow = th * 2;
    int outCol = tw * 2;

    // Accumulate Winograd products across input channels
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        // Load 4x4 input tile
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

        // Load 3x3 filter
        float g[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                g[i][j] = weight[((oc * inChannels + ic) * 3 + i) * 3 + j];

        // Transform
        float d[4][4], u[4][4];
        winograd_input_transform(tile, d);
        winograd_filter_transform(g, u);

        // Elementwise multiply-accumulate
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                acc[i][j] += d[i][j] * u[i][j];
    }

    // Output transform
    float out2x2[2][2];
    winograd_output_transform(acc, out2x2);

    // Write 2x2 output tile
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

// Conv3D for volumetric data
extern ""C"" __global__ __launch_bounds__(256) void conv3d_direct(
    const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output,
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

// #775: Trilinear interpolation of a [D,H,W,C] grid at [P,3] positions -> [P,C]. One thread per output
// element (n,c). CUDA mirror of the OpenCL trilinear_interpolate kernel.
extern ""C"" __global__ __launch_bounds__(256) void trilinear_interpolate(
    const float* __restrict__ grid,
    const float* __restrict__ positions,
    float* __restrict__ output,
    int D, int H, int W, int C,
    int P, float upperEps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P * C) return;
    int c = idx % C;
    int n = idx / C;
    float z = fmaxf(0.0f, fminf((float)(D - 1) - upperEps, positions[n * 3 + 0]));
    float y = fmaxf(0.0f, fminf((float)(H - 1) - upperEps, positions[n * 3 + 1]));
    float x = fmaxf(0.0f, fminf((float)(W - 1) - upperEps, positions[n * 3 + 2]));
    int z0 = (int)floorf(z), y0 = (int)floorf(y), x0 = (int)floorf(x);
    int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
    float fz = z - z0, fy = y - y0, fx = x - x0;
    float w000 = (1 - fz) * (1 - fy) * (1 - fx), w001 = (1 - fz) * (1 - fy) * fx;
    float w010 = (1 - fz) * fy * (1 - fx),       w011 = (1 - fz) * fy * fx;
    float w100 = fz * (1 - fy) * (1 - fx),       w101 = fz * (1 - fy) * fx;
    float w110 = fz * fy * (1 - fx),             w111 = fz * fy * fx;
    output[n * C + c] =
        w000 * grid[(((z0 * H + y0) * W + x0) * C) + c] + w001 * grid[(((z0 * H + y0) * W + x1) * C) + c] +
        w010 * grid[(((z0 * H + y1) * W + x0) * C) + c] + w011 * grid[(((z0 * H + y1) * W + x1) * C) + c] +
        w100 * grid[(((z1 * H + y0) * W + x0) * C) + c] + w101 * grid[(((z1 * H + y0) * W + x1) * C) + c] +
        w110 * grid[(((z1 * H + y1) * W + x0) * C) + c] + w111 * grid[(((z1 * H + y1) * W + x1) * C) + c];
}

// #775: Trilinear-interpolate backward w.r.t. the grid. GATHER form (one thread per grid cell,
// deterministic, no atomics): the 8-corner weight factorizes per axis. CUDA mirror of OpenCL.
extern ""C"" __global__ __launch_bounds__(256) void trilinear_interpolate_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ positions,
    float* __restrict__ gradGrid,
    int D, int H, int W, int C,
    int P, float upperEps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= D * H * W * C) return;
    int c = idx % C;
    int gx = (idx / C) % W;
    int gy = (idx / (C * W)) % H;
    int gz = idx / (C * W * H);
    float sum = 0.0f;
    for (int n = 0; n < P; n++) {
        float z = fmaxf(0.0f, fminf((float)(D - 1) - upperEps, positions[n * 3 + 0]));
        float y = fmaxf(0.0f, fminf((float)(H - 1) - upperEps, positions[n * 3 + 1]));
        float x = fmaxf(0.0f, fminf((float)(W - 1) - upperEps, positions[n * 3 + 2]));
        int z0 = (int)floorf(z), y0 = (int)floorf(y), x0 = (int)floorf(x);
        int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float fz = z - z0, fy = y - y0, fx = x - x0;
        float wz = (gz == z0 ? (1.0f - fz) : 0.0f) + (gz == z1 ? fz : 0.0f);
        if (wz == 0.0f) continue;
        float wy = (gy == y0 ? (1.0f - fy) : 0.0f) + (gy == y1 ? fy : 0.0f);
        if (wy == 0.0f) continue;
        float wx = (gx == x0 ? (1.0f - fx) : 0.0f) + (gx == x1 ? fx : 0.0f);
        sum += wz * wy * wx * gradOutput[n * C + c];
    }
    gradGrid[idx] = sum;
}

// #775: ConvTranspose3D forward (NCDHW), weights [inC,outC,kD,kH,kW]. GATHER over output elements.
// CUDA mirror of the OpenCL conv_transpose3d kernel.
extern ""C"" __global__ __launch_bounds__(256) void conv_transpose3d(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int N, int inC, int iD, int iH, int iW,
    int outC, int outD, int outH, int outW,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * outC * outD * outH * outW) return;
    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int od = (idx / (outW * outH)) % outD;
    int oc = (idx / (outW * outH * outD)) % outC;
    int n = idx / (outW * outH * outD * outC);
    float sum = 0.0f;
    for (int kd = 0; kd < kD; kd++) {
        int td = od + padD - kd;
        if (td < 0 || (td % strideD) != 0) continue;
        int id = td / strideD;
        if (id < 0 || id >= iD) continue;
        for (int kh = 0; kh < kH; kh++) {
            int th = oh + padH - kh;
            if (th < 0 || (th % strideH) != 0) continue;
            int ih = th / strideH;
            if (ih < 0 || ih >= iH) continue;
            for (int kw = 0; kw < kW; kw++) {
                int tw = ow + padW - kw;
                if (tw < 0 || (tw % strideW) != 0) continue;
                int iw = tw / strideW;
                if (iw < 0 || iw >= iW) continue;
                for (int ic = 0; ic < inC; ic++) {
                    sum += input[(((n * inC + ic) * iD + id) * iH + ih) * iW + iw]
                         * weights[((((ic * outC + oc) * kD + kd) * kH + kh) * kW + kw)];
                }
            }
        }
    }
    output[idx] = sum;
}

// #775: ConvTranspose3D backward w.r.t. input. GATHER over gradInput. CUDA mirror of OpenCL.
extern ""C"" __global__ __launch_bounds__(256) void conv_transpose3d_backward_input(
    const float* __restrict__ gradOutput,
    const float* __restrict__ weights,
    float* __restrict__ gradInput,
    int N, int inC, int iD, int iH, int iW,
    int outC, int outD, int outH, int outW,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * inC * iD * iH * iW) return;
    int iw = idx % iW;
    int ih = (idx / iW) % iH;
    int id = (idx / (iW * iH)) % iD;
    int ic = (idx / (iW * iH * iD)) % inC;
    int n = idx / (iW * iH * iD * inC);
    float sum = 0.0f;
    for (int kd = 0; kd < kD; kd++) {
        int od = id * strideD - padD + kd;
        if (od < 0 || od >= outD) continue;
        for (int kh = 0; kh < kH; kh++) {
            int oh = ih * strideH - padH + kh;
            if (oh < 0 || oh >= outH) continue;
            for (int kw = 0; kw < kW; kw++) {
                int ow = iw * strideW - padW + kw;
                if (ow < 0 || ow >= outW) continue;
                for (int oc = 0; oc < outC; oc++) {
                    sum += gradOutput[(((n * outC + oc) * outD + od) * outH + oh) * outW + ow]
                         * weights[((((ic * outC + oc) * kD + kd) * kH + kh) * kW + kw)];
                }
            }
        }
    }
    gradInput[idx] = sum;
}

// #775: ConvTranspose3D backward w.r.t. weights. GATHER over gradWeights [inC,outC,kD,kH,kW]. CUDA mirror.
extern ""C"" __global__ __launch_bounds__(256) void conv_transpose3d_backward_weights(
    const float* __restrict__ gradOutput,
    const float* __restrict__ input,
    float* __restrict__ gradWeights,
    int N, int inC, int iD, int iH, int iW,
    int outC, int outD, int outH, int outW,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inC * outC * kD * kH * kW) return;
    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int kd = (idx / (kW * kH)) % kD;
    int oc = (idx / (kW * kH * kD)) % outC;
    int ic = idx / (kW * kH * kD * outC);
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int id = 0; id < iD; id++) {
            int od = id * strideD - padD + kd;
            if (od < 0 || od >= outD) continue;
            for (int ih = 0; ih < iH; ih++) {
                int oh = ih * strideH - padH + kh;
                if (oh < 0 || oh >= outH) continue;
                for (int iw = 0; iw < iW; iw++) {
                    int ow = iw * strideW - padW + kw;
                    if (ow < 0 || ow >= outW) continue;
                    sum += input[(((n * inC + ic) * iD + id) * iH + ih) * iW + iw]
                         * gradOutput[(((n * outC + oc) * outD + od) * outH + oh) * outW + ow];
                }
            }
        }
    }
    gradWeights[idx] = sum;
}

// #775: SpiralConv (mesh conv). Per (vertex, out-channel): gather spiralLength neighbour features and
// matmul with weights [outC, inC*spiralLength] + bias. Invalid neighbour -> 0. CUDA mirror of OpenCL.
extern ""C"" __global__ __launch_bounds__(256) void spiral_conv(
    const float* __restrict__ vertexFeatures,
    const int* __restrict__ spiralIndices,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ output,
    int V, int inC, int spiralLength, int outC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V * outC) return;
    int oc = idx % outC;
    int v = idx / outC;
    int gatheredSize = inC * spiralLength;
    float sum = biases[oc];
    for (int s = 0; s < spiralLength; s++) {
        int neighborIdx = spiralIndices[v * spiralLength + s];
        if (neighborIdx < 0 || neighborIdx >= V) continue;
        int gatherOffset = s * inC;
        for (int c = 0; c < inC; c++) {
            sum += vertexFeatures[neighborIdx * inC + c] * weights[oc * gatheredSize + gatherOffset + c];
        }
    }
    output[idx] = sum;
}

// #775: SpiralConv backward w.r.t. vertex features. GATHER over gradVertexFeatures [V,inC]. CUDA mirror.
extern ""C"" __global__ __launch_bounds__(256) void spiral_conv_backward_input(
    const float* __restrict__ gradOutput,
    const int* __restrict__ spiralIndices,
    const float* __restrict__ weights,
    float* __restrict__ gradVertexFeatures,
    int V, int inC, int spiralLength, int outC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V * inC) return;
    int ic = idx % inC;
    int nbr = idx / inC;
    int gatheredSize = inC * spiralLength;
    float sum = 0.0f;
    for (int v = 0; v < V; v++) {
        for (int s = 0; s < spiralLength; s++) {
            if (spiralIndices[v * spiralLength + s] != nbr) continue;
            for (int oc = 0; oc < outC; oc++) {
                sum += gradOutput[v * outC + oc] * weights[oc * gatheredSize + s * inC + ic];
            }
        }
    }
    gradVertexFeatures[idx] = sum;
}

// #775: SpiralConv backward w.r.t. weights. GATHER over gradWeights [outC, inC*spiralLength]. CUDA mirror.
extern ""C"" __global__ __launch_bounds__(256) void spiral_conv_backward_weights(
    const float* __restrict__ gradOutput,
    const float* __restrict__ vertexFeatures,
    const int* __restrict__ spiralIndices,
    float* __restrict__ gradWeights,
    int V, int inC, int spiralLength, int outC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gatheredSize = inC * spiralLength;
    if (idx >= outC * gatheredSize) return;
    int g = idx % gatheredSize;
    int oc = idx / gatheredSize;
    int s = g / inC;
    int ic = g % inC;
    float sum = 0.0f;
    for (int v = 0; v < V; v++) {
        int neighborIdx = spiralIndices[v * spiralLength + s];
        if (neighborIdx < 0 || neighborIdx >= V) continue;
        sum += gradOutput[v * outC + oc] * vertexFeatures[neighborIdx * inC + ic];
    }
    gradWeights[idx] = sum;
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "im2col",
                "col2im",
                "conv2d_direct",
                "conv2d_backward_input",
                "conv2d_backward_kernel",
                "depthwise_conv2d",
                "conv_transpose2d",
                "conv_transpose2d_backward_input",
                "conv_transpose2d_backward_kernel",
                "conv2d_tiled",
                "conv2d_winograd_f2x2_3x3",
                "conv3d_direct",
                "trilinear_interpolate",
                "trilinear_interpolate_backward",
                "conv_transpose3d",
                "conv_transpose3d_backward_input",
                "conv_transpose3d_backward_weights",
                "spiral_conv",
                "spiral_conv_backward_input",
                "spiral_conv_backward_weights"
            };
        }
    }
}
