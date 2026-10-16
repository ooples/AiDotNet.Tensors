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

    // Output is [batch, C*kH*kW, L] (L = outH*outW), matching CpuEngine.Unfold. This kernel wrote
    //     float* outPtr = output + idx * patchSize;  outPtr[(c*kernelH+kh)*kernelW+kw] = val;
    // which is the TRANSPOSED [batch, L, C*kH*kW] layout, so GPU columns came out permuted vs the CPU
    // convention. OpenCL fixed this; CUDA never got the fix (same #775 drift class as the SELU >=,
    // nll_loss int*, and cross_entropy_backward misses in this branch). col2im below reads this layout
    // back and carried the identical transposition, so the two were self-consistent and only diverged
    // from CPU -- which is why Unfold/Fold looked plausible on GPU while disagreeing with CpuEngine.
    int L = outH * outW;
    int s = oh * outW + ow;

    for (int c = 0; c < channels; c++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;

                float val = 0.0f;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    val = input[((b * channels + c) * height + ih) * width + iw];
                }

                int colRow = (c * kernelH + kh) * kernelW + kw;
                output[(b * patchSize + colRow) * L + s] = val;
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
                    // Input is [batch, C*kH*kW, L] (L = outH*outW patches), matching CpuEngine.Fold:
                    // flat = b*patchSize*L + colIdx*L + s. The old index (patchIdx*patchSize + colIdx)
                    // assumed the transposed [batch, L, C*kH*kW] layout im2col used to write above.
                    int L = outH * outW;
                    int s = oh * outW + ow;
                    int colIdx = (c * kernelH + kh) * kernelW + kw;
                    sum += input[(b * patchSize + colIdx) * L + s];
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
    // One thread per gradKernel element over a flat 1D grid. Mapping outChannels * inChannels onto
    // gridDim.z failed to launch once the product passed 65,535 (any 256x256 or wider conv layer), and
    // 16x16 blocks over a 3x3 kernel left 247 of every 256 threads idle.
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)outChannels * inChannels * kernelH * kernelW;
    if (idx >= total) return;
    int kw = (int)(idx % kernelW);
    int kh = (int)((idx / kernelW) % kernelH);
    int ic = (int)((idx / ((long)kernelW * kernelH)) % inChannels);
    int oc = (int)(idx / ((long)kernelW * kernelH * inChannels));

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

    gradKernel[idx] = sum;
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

// #775: Depthwise Conv2D backward w.r.t. input (NCHW, oc = ic*M + m). GATHER over gradInput. CUDA mirror.
extern ""C"" __global__ __launch_bounds__(256) void depthwise_conv2d_backward_input(
    const float* __restrict__ gradOutput,
    const float* __restrict__ weights,
    float* __restrict__ gradInput,
    int N, int inC, int H, int W,
    int M, int outH, int outW,
    int kH, int kW,
    int strideH, int strideW, int padH, int padW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * inC * H * W;
    if (idx >= total) return;
    int iw = idx % W;
    int ih = (idx / W) % H;
    int ic = (idx / (W * H)) % inC;
    int b = idx / (W * H * inC);
    int outC = inC * M;
    float sum = 0.0f;
    for (int m = 0; m < M; m++) {
        int oc = ic * M + m;
        for (int kh = 0; kh < kH; kh++) {
            int t = ih + padH - kh;
            if (t < 0 || (t % strideH) != 0) continue;
            int oh = t / strideH;
            if (oh < 0 || oh >= outH) continue;
            for (int kw = 0; kw < kW; kw++) {
                int tw = iw + padW - kw;
                if (tw < 0 || (tw % strideW) != 0) continue;
                int ow = tw / strideW;
                if (ow < 0 || ow >= outW) continue;
                sum += weights[(oc * kH + kh) * kW + kw]
                     * gradOutput[((b * outC + oc) * outH + oh) * outW + ow];
            }
        }
    }
    gradInput[idx] = sum;
}

// #775: Depthwise Conv2D backward w.r.t. weights (NCHW). Gathers per gradKernel element (no race). CUDA mirror.
extern ""C"" __global__ __launch_bounds__(256) void depthwise_conv2d_backward_weights(
    const float* __restrict__ gradOutput,
    const float* __restrict__ input,
    float* __restrict__ gradKernel,
    int N, int inC, int H, int W,
    int M, int outH, int outW,
    int kH, int kW,
    int strideH, int strideW, int padH, int padW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outC = inC * M;
    int total = outC * kH * kW;
    if (idx >= total) return;
    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int oc = idx / (kW * kH);
    int ic = oc / M;
    float sum = 0.0f;
    for (int b = 0; b < N; b++) {
        for (int oh = 0; oh < outH; oh++) {
            int ih = oh * strideH - padH + kh;
            if (ih < 0 || ih >= H) continue;
            for (int ow = 0; ow < outW; ow++) {
                int iw = ow * strideW - padW + kw;
                if (iw < 0 || iw >= W) continue;
                sum += input[((b * inC + ic) * H + ih) * W + iw]
                     * gradOutput[((b * outC + oc) * outH + oh) * outW + ow];
            }
        }
    }
    gradKernel[idx] = sum;
}

// #1313: Conv2D weight gradient as an implicit GEMM, without an im2col buffer.
//   dW[co, p] = sum over k = (b, oh, ow) of dY[b, co, oh, ow] * X[b, ci, oh*sH - pH + kh*dH, ow*sW - pW + kw*dW]
// with p = (ci, kh, kw) and X zero outside the input: C[M = Cout, N = P] = A[M, K] * B[K, N], K = batch * outH * outW.
// A block owns a 64 x 64 tile of C (4 x 4 per thread) over the K range [z * kPerSplit, (z + 1) * kPerSplit); both
// operand tiles go through padded shared memory, and the B tile is gathered from X with the convolution's index
// arithmetic, consecutive lanes on consecutive output positions so the loads coalesce. Each split writes its own
// slice of partials ([split][Cout][P]); conv2d_wgrad_splitk_reduce adds the slices in split order, so the result
// does not depend on scheduling. With one split the block writes dW directly.
#define WG_BM 64
#define WG_BN 64
#define WG_BK 16
extern ""C"" __global__ __launch_bounds__(256) void conv2d_wgrad_implicit_gemm(
    const float* __restrict__ input, const float* __restrict__ gradOutput, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit)
{
    // Two buffers each: the next K tile's global loads go to registers while this tile's FMAs run.
    __shared__ __align__(16) float As[2][WG_BK][WG_BM + 4];
    __shared__ __align__(16) float Bs[2][WG_BK][WG_BN + 4];
    const int L = outH * outW, P = inC * kH * kW, K = batch * L, khw = kH * kW;
    const int m0 = blockIdx.y * WG_BM, n0 = blockIdx.x * WG_BN;
    const int kBegin = blockIdx.z * kPerSplit;
    const int kEnd = min(K, kBegin + kPerSplit);
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;

    // Loader lanes: element e = tid + 256 * r of a 16 x 64 tile, k = e % 16 (fastest), m or n = e / 16.
    const int lk = tid & 15, lc = tid >> 4;   // lc in [0, 16); the four loads cover columns lc, lc + 16, lc + 32, lc + 48
    int bci[4], bkh[4], bkw[4];
    bool bValid[4];
    #pragma unroll
    for (int r = 0; r < 4; r++)
    {
        int p = n0 + lc + 16 * r;
        bValid[r] = p < P;
        int pp = bValid[r] ? p : 0;
        bci[r] = pp / khw;
        int rem = pp - bci[r] * khw;
        bkh[r] = rem / kW;
        bkw[r] = rem - bkh[r] * kW;
    }

    float ra[4], rb[4];
    auto load = [&](int k0)
    {
        int k = k0 + lk;
        bool kIn = k < kEnd;
        int b = 0, oh = 0, ow = 0;
        if (kIn) { b = k / L; int l = k - b * L; oh = l / outW; ow = l - oh * outW; }
        const float* dyRow = gradOutput + (long long)b * outC * L + oh * outW + ow;
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            int m = m0 + lc + 16 * r;
            ra[r] = (kIn && m < outC) ? dyRow[(long long)m * L] : 0.0f;
            float x = 0.0f;
            if (kIn && bValid[r])
            {
                int ih = oh * sH - pH + bkh[r] * dH;
                int iw = ow * sW - pW + bkw[r] * dW;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    x = input[(((long long)b * inC + bci[r]) * H + ih) * W + iw];
            }
            rb[r] = x;
        }
    };
    auto store = [&](int buf)
    {
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            As[buf][lk][lc + 16 * r] = ra[r];
            Bs[buf][lk][lc + 16 * r] = rb[r];
        }
    };

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    int buf = 0;
    if (kBegin < kEnd)
    {
        load(kBegin);
        store(0);
        __syncthreads();
    }
    for (int k0 = kBegin; k0 < kEnd; k0 += WG_BK)
    {
        bool more = k0 + WG_BK < kEnd;
        if (more) load(k0 + WG_BK);
        #pragma unroll
        for (int kk = 0; kk < WG_BK; kk++)
        {
            float4 a = *reinterpret_cast<const float4*>(&As[buf][kk][ty * 4]);
            float4 bv = *reinterpret_cast<const float4*>(&Bs[buf][kk][tx * 4]);
            float av[4] = { a.x, a.y, a.z, a.w }, bw[4] = { bv.x, bv.y, bv.z, bv.w };
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += av[i] * bw[j];
        }
        if (more)
        {
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }

    float* slice = out + (long long)blockIdx.z * outC * P;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int m = m0 + ty * 4 + i;
        if (m >= outC) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int n = n0 + tx * 4 + j;
            if (n < P) slice[(long long)m * P + n] = acc[i][j];
        }
    }
}

// Adds the split-K slices of conv2d_wgrad_implicit_gemm in split order: dW[i] = sum over z of partials[z][i].
extern ""C"" __global__ __launch_bounds__(256) void conv2d_wgrad_splitk_reduce(
    const float* __restrict__ partials, float* __restrict__ gradKernel, int count, int splits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float sum = 0.0f;
    for (int z = 0; z < splits; z++) sum += partials[(long long)z * count + i];
    gradKernel[i] = sum;
}

// The dgrad engine's weight layout: wt[tap][co][ci] = w[co][ci][tap] (tap = kh * kW + kw).
extern ""C"" __global__ __launch_bounds__(256) void conv2d_dgrad_weight_tapmajor(
    const float* __restrict__ w, float* __restrict__ wt, int outC, int inC, int khw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outC * inC * khw) return;
    int tap = i % khw, ci = (i / khw) % inC, co = i / (khw * inC);
    wt[((long long)tap * outC + co) * inC + ci] = w[i];
}

// Sums the phase input gradient's split-K slices in split order: pixel (ih, iw) belongs to phase
// ((ih + pH) mod sH, (iw + pW) mod sW), which has ceil(outC * taps / kPerSplit) slices (see conv2d_dgrad_ce_body).
extern ""C"" __global__ __launch_bounds__(256) void conv2d_dgrad_phase_reduce(
    const float* __restrict__ partials, float* __restrict__ gradInput, int count, int H, int W,
    int outC, int kH, int kW, int sH, int sW, int pH, int pW, int kPerSplit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int iw = i % W, ih = (i / W) % H;
    int rh = (ih + pH) % sH, rw = (iw + pW) % sW;
    int units = (outC * ((kH - rh + sH - 1) / sH) * ((kW - rw + sW - 1) / sW) + kPerSplit - 1) / kPerSplit;
    float sum = 0.0f;
    for (int z = 0; z < units; z++) sum += partials[(long long)z * count + i];
    gradInput[i] = sum;
}

// #1313: Conv2D input gradient as an implicit GEMM, without the col buffer and col2im scatter.
//   dX[b, ci, ih, iw] = sum over k = (co, kh, kw) of W[co, ci, kh, kw] * dY[b, co, oh, ow],
//   oh = (ih + pH - kh*dH) / sH and ow = (iw + pW - kw*dW) / sW when both divide exactly and land inside dY.
// C[M = Cin, N = batch * H * W] = A[M, K] * B[K, N], K = Cout * kH * kW. Same tiling as conv2d_wgrad_implicit_gemm
// (64 x 64 per block, 4 x 4 per thread, BK = 16); the B loader puts consecutive lanes on consecutive input pixels, so
// each lane's (b, ih, iw) is fixed for the whole K loop and dY reads coalesce along a row. Split-K slices are written
// in dX's layout to out + z * batch * Cin * H * W and summed in order by conv2d_wgrad_splitk_reduce.
extern ""C"" __global__ __launch_bounds__(256) void conv2d_dgrad_implicit_gemm(
    const float* __restrict__ gradOutput, const float* __restrict__ kernel, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit)
{
    __shared__ __align__(16) float As[WG_BK][WG_BM + 4];
    __shared__ __align__(16) float Bs[WG_BK][WG_BN + 4];
    const int HW = H * W, N = batch * HW, khw = kH * kW, K = outC * khw, P = inC * khw, L = outH * outW;
    const int m0 = blockIdx.y * WG_BM, n0 = blockIdx.x * WG_BN;
    const int kBegin = blockIdx.z * kPerSplit;
    const int kEnd = min(K, kBegin + kPerSplit);
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;

    // A loader: k = k0 + (tid & 15), rows m = m0 + (tid >> 4) + 16 * r.
    const int lk = tid & 15, lc = tid >> 4;
    // B loader: one input pixel n per lane, rows k = k0 + (tid >> 6) + 4 * r.
    const int bn = tid & 63, bkRow = tid >> 6;
    const int n = n0 + bn;
    const bool nIn = n < N;
    int nb = 0, ihp = 0, iwp = 0;
    if (nIn) { nb = n / HW; int hw = n - nb * HW; int ih = hw / W; ihp = ih + pH; iwp = hw - ih * W + pW; }
    const float* dyBatch = gradOutput + (long long)nb * outC * L;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    for (int k0 = kBegin; k0 < kEnd; k0 += WG_BK)
    {
        int ka = k0 + lk;
        bool kaIn = ka < kEnd;
        int aco = 0, at = 0;
        if (kaIn) { aco = ka / khw; at = ka - aco * khw; }
        const float* wRow = kernel + (long long)aco * P + at;
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            int m = m0 + lc + 16 * r;
            As[lk][lc + 16 * r] = (kaIn && m < inC) ? wRow[(long long)m * khw] : 0.0f;
        }
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            int kr = bkRow + 4 * r, kb = k0 + kr;
            float v = 0.0f;
            if (nIn && kb < kEnd)
            {
                int co = kb / khw, t = kb - co * khw;
                int kh = t / kW, kw = t - kh * kW;
                int th = ihp - kh * dH, tw = iwp - kw * dW;
                if (th >= 0 && tw >= 0)
                {
                    int oh = th / sH, ow = tw / sW;
                    if (oh * sH == th && ow * sW == tw && oh < outH && ow < outW)
                        v = dyBatch[(long long)co * L + oh * outW + ow];
                }
            }
            Bs[kr][bn] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < WG_BK; kk++)
        {
            float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
            float4 bv = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
            float av[4] = { a.x, a.y, a.z, a.w }, bw[4] = { bv.x, bv.y, bv.z, bv.w };
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += av[i] * bw[j];
        }
        __syncthreads();
    }

    float* slice = out + (long long)blockIdx.z * batch * inC * HW;
    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        int nn = n0 + tx * 4 + j;
        if (nn >= N) continue;
        int b = nn / HW, hw = nn - b * HW;
        float* col = slice + (long long)b * inC * HW + hw;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int m = m0 + ty * 4 + i;
            if (m < inC) col[(long long)m * HW] = acc[i][j];
        }
    }
}

// #1313: conv2d_dgrad_implicit_gemm for Cout % 16 == 0 (every ResNet-style layer), with K ordered (kh, kw, co):
// k = t * Cout + co, t = kh * kW + kw. A 16-wide K tile then shares one kernel tap, so each lane checks its pixel's
// source position once per tile instead of once per element, and (t, co0) advance by addition: the K loop has no
// integer division. Unit stride is its own instantiation (no remainder test). Output columns go out as float4 when
// H * W is a multiple of 4. Split-K slices as in the generic kernel.
template <bool UNIT>
__device__ __forceinline__ void conv2d_dgrad_c16_body(float (*As)[WG_BK][WG_BM + 4], float (*Bs)[WG_BK][WG_BN + 4],
    const float* __restrict__ gradOutput, const float* __restrict__ kernel, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit)
{
    const int HW = H * W, N = batch * HW, khw = kH * kW, K = outC * khw, P = inC * khw, L = outH * outW;
    const int m0 = blockIdx.y * WG_BM, n0 = blockIdx.x * WG_BN;
    const int kBegin = blockIdx.z * kPerSplit;
    const int kEnd = min(K, kBegin + kPerSplit);
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int lk = tid & 15, lc = tid >> 4;
    const int bn = tid & 63, bkRow = tid >> 6;
    const int n = n0 + bn;
    const bool nIn = n < N;
    int nb = 0, ihp = 0, iwp = 0;
    if (nIn) { nb = n / HW; int hw = n - nb * HW; int ih = hw / W; ihp = ih + pH; iwp = hw - ih * W + pW; }
    const float* dyBatch = gradOutput + (long long)nb * outC * L;

    // Tile state: tap t = (kh, kw) and first output channel co0 of the tile at k0 (kBegin is a multiple of 16).
    int t = kBegin / outC, co0 = kBegin - t * outC;
    int kh = t / kW, kw = t - kh * kW;

    // Global -> register loads of one K tile; the stores to shared memory happen after the next compute, so the loads
    // of tile i + 1 are in flight while tile i's FMAs run (two shared buffers, one barrier per tile).
    float ra[4], rb[4];
    auto load = [&]()
    {
        const float* wRow = kernel + (long long)(co0 + lk) * P + t;
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            int m = m0 + lc + 16 * r;
            ra[r] = m < inC ? wRow[(long long)m * khw] : 0.0f;
        }
        int th = ihp - kh * dH, tw = iwp - kw * dW;
        int oh = th, ow = tw;
        bool valid = nIn && th >= 0 && tw >= 0;
        if (!UNIT && valid)
        {
            oh = th / sH; ow = tw / sW;
            valid = oh * sH == th && ow * sW == tw;
        }
        valid = valid && oh < outH && ow < outW;
        const float* src = dyBatch + (long long)(co0 + bkRow) * L + (valid ? oh * outW + ow : 0);
        #pragma unroll
        for (int r = 0; r < 4; r++)
            rb[r] = valid ? src[(long long)(4 * r) * L] : 0.0f;
    };
    auto store = [&](int buf)
    {
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            As[buf][lk][lc + 16 * r] = ra[r];
            Bs[buf][bkRow + 4 * r][bn] = rb[r];
        }
    };
    auto advance = [&]()
    {
        co0 += WG_BK;
        if (co0 == outC) { co0 = 0; t++; if (++kw == kW) { kw = 0; kh++; } }
    };

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    int buf = 0;
    if (kBegin < kEnd)
    {
        load();
        store(0);
        __syncthreads();
    }
    for (int k0 = kBegin; k0 < kEnd; k0 += WG_BK)
    {
        bool more = k0 + WG_BK < kEnd;
        if (more) { advance(); load(); }
        #pragma unroll
        for (int kk = 0; kk < WG_BK; kk++)
        {
            float4 a = *reinterpret_cast<const float4*>(&As[buf][kk][ty * 4]);
            float4 bv = *reinterpret_cast<const float4*>(&Bs[buf][kk][tx * 4]);
            float av[4] = { a.x, a.y, a.z, a.w }, bw[4] = { bv.x, bv.y, bv.z, bv.w };
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += av[i] * bw[j];
        }
        if (more)
        {
            // The other buffer was last read in the previous iteration, before its closing barrier.
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }

    float* slice = out + (long long)blockIdx.z * batch * inC * HW;
    int nn = n0 + tx * 4;
    if ((HW & 3) == 0)
    {
        // Four consecutive pixels of one image plane, 16-byte aligned.
        if (nn >= N) return;
        int b = nn / HW, hw = nn - b * HW;
        float* col = slice + (long long)b * inC * HW + hw;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int m = m0 + ty * 4 + i;
            if (m < inC)
                *reinterpret_cast<float4*>(col + (long long)m * HW) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        }
        return;
    }
    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        if (nn + j >= N) continue;
        int b = (nn + j) / HW, hw = nn + j - b * HW;
        float* col = slice + (long long)b * inC * HW + hw;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int m = m0 + ty * 4 + i;
            if (m < inC) col[(long long)m * HW] = acc[i][j];
        }
    }
}

extern ""C"" __global__ __launch_bounds__(256) void conv2d_dgrad_implicit_gemm_c16(
    const float* __restrict__ gradOutput, const float* __restrict__ kernel, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit)
{
    // Declared here, not in the body, so the two instantiations share one allocation. Two buffers each (prefetch).
    __shared__ __align__(16) float As[2][WG_BK][WG_BM + 4];
    __shared__ __align__(16) float Bs[2][WG_BK][WG_BN + 4];
    if (sH == 1 && sW == 1)
        conv2d_dgrad_c16_body<true>(As, Bs, gradOutput, kernel, out, batch, inC, H, W, outC, outH, outW,
            kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit);
    else
        conv2d_dgrad_c16_body<false>(As, Bs, gradOutput, kernel, out, batch, inC, H, W, outC, outH, outW,
            kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit);
}
// #1313: strided input gradient by sub-pixel (phase) decomposition, for Cout % 16 == 0, unit dilation and kH >= sH,
// kW >= sW. Input pixels split into sH * sW phases by (ih + pH) mod sH and (iw + pW) mod sW; in phase (rh, rw) only the
// taps kh = rh, rh + sH, ... and kw = rw, rw + sW, ... reach an output position, and every one of them does. So each
// phase is a dense GEMM C[Cin, pixels of the phase] over K = Cout * (its taps), with no zero-filled gathers: the generic
// kernel at stride 2 spends three quarters of its B tile on taps that miss. blockIdx.z is the phase; no split-K, the
// phases write disjoint pixels of dX directly. K is ordered (tap row, tap column, co) as in the _c16 kernel.
extern ""C"" __global__ __launch_bounds__(256) void conv2d_dgrad_implicit_gemm_c16_phase(
    const float* __restrict__ gradOutput, const float* __restrict__ kernel, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    __shared__ __align__(16) float As[2][WG_BK][WG_BM + 4];
    __shared__ __align__(16) float Bs[2][WG_BK][WG_BN + 4];
    const int rh = blockIdx.z / sW, rw = blockIdx.z - rh * sW;
    // First input row / column of the phase: (ih + pH) % sH == rh.
    const int ih0 = ((rh - pH) % sH + sH) % sH, iw0 = ((rw - pW) % sW + sW) % sW;
    const int Hp = ih0 < H ? (H - ih0 + sH - 1) / sH : 0, Wp = iw0 < W ? (W - iw0 + sW - 1) / sW : 0;
    const int plane = Hp * Wp, N = batch * plane;
    const int n0 = blockIdx.x * WG_BN;
    if (n0 >= N) return;
    const int tapsH = (kH - rh + sH - 1) / sH, tapsW = (kW - rw + sW - 1) / sW;
    const int K = outC * tapsH * tapsW, HW = H * W, L = outH * outW, khw = kH * kW, P = inC * khw;
    const int m0 = blockIdx.y * WG_BM;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int lk = tid & 15, lc = tid >> 4;
    const int bn = tid & 63, bkRow = tid >> 6;
    const int n = n0 + bn;
    const bool nIn = n < N;
    int nb = 0, ihp = 0, iwp = 0;
    if (nIn) { nb = n / plane; int q = n - nb * plane; int i = q / Wp; ihp = ih0 + i * sH + pH; iwp = iw0 + (q - i * Wp) * sW + pW; }
    const float* dyBatch = gradOutput + (long long)nb * outC * L;

    int co0 = 0, kh = rh, kw = rw;
    float ra[4], rb[4];
    auto load = [&]()
    {
        const float* wRow = kernel + (long long)(co0 + lk) * P + kh * kW + kw;
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            int m = m0 + lc + 16 * r;
            ra[r] = m < inC ? wRow[(long long)m * khw] : 0.0f;
        }
        // (ihp - kh) is a multiple of sH by construction; only the edges can fall outside dY.
        int oh = (ihp - kh) / sH, ow = (iwp - kw) / sW;
        bool valid = nIn && ihp >= kh && iwp >= kw && oh < outH && ow < outW;
        const float* src = dyBatch + (long long)(co0 + bkRow) * L + (valid ? oh * outW + ow : 0);
        #pragma unroll
        for (int r = 0; r < 4; r++)
            rb[r] = valid ? src[(long long)(4 * r) * L] : 0.0f;
    };
    auto store = [&](int buf)
    {
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            As[buf][lk][lc + 16 * r] = ra[r];
            Bs[buf][bkRow + 4 * r][bn] = rb[r];
        }
    };

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    int buf = 0;
    load();
    store(0);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += WG_BK)
    {
        bool more = k0 + WG_BK < K;
        if (more)
        {
            co0 += WG_BK;
            if (co0 == outC) { co0 = 0; kw += sW; if (kw >= kW) { kw = rw; kh += sH; } }
            load();
        }
        #pragma unroll
        for (int kk = 0; kk < WG_BK; kk++)
        {
            float4 a = *reinterpret_cast<const float4*>(&As[buf][kk][ty * 4]);
            float4 bv = *reinterpret_cast<const float4*>(&Bs[buf][kk][tx * 4]);
            float av[4] = { a.x, a.y, a.z, a.w }, bw[4] = { bv.x, bv.y, bv.z, bv.w };
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += av[i] * bw[j];
        }
        if (more)
        {
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        int nn = n0 + tx * 4 + j;
        if (nn >= N) continue;
        int b = nn / plane, q = nn - b * plane, i = q / Wp;
        float* col = out + (long long)b * inC * HW + (ih0 + i * sH) * W + iw0 + (q - i * Wp) * sW;
        #pragma unroll
        for (int i2 = 0; i2 < 4; i2++)
        {
            int m = m0 + ty * 4 + i2;
            if (m < inC) col[(long long)m * HW] = acc[i2][j];
        }
    }
}

" + ConvEngineCore + @"
#define CE_FPROP_KERNEL(NAME, BM, TAPU, MAIN) \
extern ""C"" __global__ __launch_bounds__(256) void NAME( \
    const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ out, \
    int batch, int inC, int H, int W, int outC, int outH, int outW, \
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit) \
{ \
    conv2d_fprop_ig_body<BM, BM, TAPU, CeIdentityPro, CeIdentityEpi, MAIN>(input, kernel, out, batch, inC, H, W, \
        outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit); \
}
CE_FPROP_KERNEL(conv2d_fprop_ig128_tap, 128, true, 0)
CE_FPROP_KERNEL(conv2d_fprop_ig128_gen, 128, false, 0)
CE_FPROP_KERNEL(conv2d_fprop_ig64_tap, 64, true, 0)
CE_FPROP_KERNEL(conv2d_fprop_ig64_gen, 64, false, 0)
CE_FPROP_KERNEL(conv2d_fprop_ig128_tap_x3, 128, true, 1)
CE_FPROP_KERNEL(conv2d_fprop_ig128_gen_x3, 128, false, 1)
CE_FPROP_KERNEL(conv2d_fprop_ig64_tap_x3, 64, true, 1)
CE_FPROP_KERNEL(conv2d_fprop_ig64_gen_x3, 64, false, 1)
CE_FPROP_KERNEL(conv2d_fprop_ig128_tap_tf32, 128, true, 2)
CE_FPROP_KERNEL(conv2d_fprop_ig128_gen_tf32, 128, false, 2)
CE_FPROP_KERNEL(conv2d_fprop_ig64_tap_tf32, 64, true, 2)
CE_FPROP_KERNEL(conv2d_fprop_ig64_gen_tf32, 64, false, 2)

// Weight gradient on the engine core: dW[Cout, P] = dY[Cout, K] * X_gathered[K, P], P = Cin*kH*kW, K = B*oH*oW.
// Both loaders put one K element per lane (8 lanes along K, rows 32 apart), so each lane decomposes its k = (b, oh, ow)
// once per tile, dY reads coalesce along the output row, and every column's (ci, kh, kw) is fixed for the whole loop.
// Split-K slices go to out + z * Cout*P, summed in order by conv2d_wgrad_splitk_reduce.
template <int BM, int BN, int MAIN>
__device__ __forceinline__ void conv2d_wgrad_ce_body(
    const float* __restrict__ input, const float* __restrict__ gradOutput, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit)
{
    __shared__ __align__(16) float smem[2 * CE_BK * (BM + CePad<MAIN>::value) + 2 * CE_BK * (BN + CePad<MAIN>::value)];
    constexpr int TM = BM / 16, TN = BN / 16, RA = BM * CE_BK / 256, RB = BN * CE_BK / 256;
    const int L = outH * outW, khw = kH * kW, P = inC * khw, K = batch * L, HW = H * W;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int kBegin = blockIdx.z * kPerSplit, kEnd = min(K, kBegin + kPerSplit);
    const int tiles = kBegin < kEnd ? (kEnd - kBegin + CE_BK - 1) / CE_BK : 0;
    const int tid = threadIdx.x;
    const int lk = tid & 7, lr = tid >> 3;   // k lane; rows (A) / columns (B) lr + 32 r

    int cci[RB], ckh[RB], ckw[RB];
    bool cIn[RB];
    #pragma unroll
    for (int r = 0; r < RB; r++)
    {
        int p = n0 + lr + 32 * r;
        cIn[r] = p < P;
        int pp = cIn[r] ? p : 0;
        cci[r] = pp / khw;
        int t = pp - cci[r] * khw;
        ckh[r] = t / kW;
        ckw[r] = t - ckh[r] * kW;
    }

    int kc = kBegin;
    float ra[RA], rb[RB];
    auto load = [&]()
    {
        int k = kc + lk;
        bool kIn = k < kEnd;
        int b = 0, oh = 0, ow = 0;
        if (kIn) { b = k / L; int l = k - b * L; oh = l / outW; ow = l - oh * outW; }
        const float* dyRow = gradOutput + (long long)b * outC * L + oh * outW + ow;
        #pragma unroll
        for (int r = 0; r < RA; r++)
        {
            int m = m0 + lr + 32 * r;
            ra[r] = (kIn && m < outC) ? dyRow[(long long)m * L] : 0.0f;
        }
        const float* xImage = input + (long long)b * inC * HW;
        int ihBase = oh * sH - pH, iwBase = ow * sW - pW;
        #pragma unroll
        for (int r = 0; r < RB; r++)
        {
            float v = 0.0f;
            if (kIn && cIn[r])
            {
                int ih = ihBase + ckh[r] * dH, iw = iwBase + ckw[r] * dW;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) v = xImage[(long long)cci[r] * HW + ih * W + iw];
            }
            rb[r] = v;
        }
    };
    auto store = [&](auto as, auto bs)
    {
        #pragma unroll
        for (int r = 0; r < RA; r++) as[lk][lr + 32 * r] = ra[r];
        #pragma unroll
        for (int r = 0; r < RB; r++) bs[lk][lr + 32 * r] = rb[r];
    };
    auto advance = [&]() { kc += CE_BK; };

    float acc[TM][TN];
    ce_run_mainloop<BM, BN, MAIN>(smem, tiles, load, store, advance, acc);

    using Map = CeMap<BM, BN, MAIN>;
    constexpr int VEC = Map::VEC;
    float* slice = out + (long long)blockIdx.z * outC * P;
    const bool vec = (P % VEC) == 0;
    #pragma unroll
    for (int jg = 0; jg < TN; jg += VEC)
    {
        int nn = n0 + Map::col(jg);
        if (nn >= P) continue;
        #pragma unroll
        for (int i = 0; i < TM; i++)
        {
            int m = m0 + Map::row(i);
            if (m >= outC) continue;
            float* rowp = slice + (long long)m * P + nn;
            float v[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; j++) v[j] = acc[i][jg + j];
            if (vec)
                ce_store_vec<VEC>(rowp, v);
            else
            {
                #pragma unroll
                for (int j = 0; j < VEC; j++)
                    if (nn + j < P) rowp[j] = v[j];
            }
        }
    }
}

// Input gradient on the engine core, Cout % 8 == 0: dX[Cin, B*H*W] = W'[Cin, K] * dY_gathered[K, B*H*W] with K ordered
// (kh, kw, co), so a K tile shares one tap. The A loader's lanes run along ci. tapMajor: the weights arrive as
// wt[kh*kW + kw][co][ci] (conv2d_dgrad_weight_tapmajor), so those lanes read contiguous floats; else the forward layout
// w[co][ci][kh][kw], where consecutive ci are kH*kW floats apart (a 1x1 kernel's layout is the same either way). The
// host relays out only when enough column tiles reread the weights to pay for the copy. PHASE = false: unit stride or any stride with per-tile divisibility checks,
// split-K slices to out + z * B*Cin*H*W. PHASE = true (strided, unit dilation, kernel >= stride): blockIdx.z is the
// sub-pixel phase (rh, rw); the block's columns are that phase's pixels and only the taps kh = rh (mod sH),
// kw = rw (mod sW) are visited, every one of which hits; phases write disjoint pixels. With split-K, each phase's K
// is cut into ceil(K_phase / kPerSplit) slices (a 4-tap phase gets four times the slices of a 1-tap one, so every grid
// layer carries the same K) and conv2d_dgrad_phase_reduce sums each pixel's slices in order.
template <int BM, int BN, bool PHASE, int MAIN>
__device__ __forceinline__ void conv2d_dgrad_ce_body(
    const float* __restrict__ gradOutput, const float* __restrict__ kernel, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit, int tapMajor)
{
    __shared__ __align__(16) float smem[2 * CE_BK * (BM + CePad<MAIN>::value) + 2 * CE_BK * (BN + CePad<MAIN>::value)];
    constexpr int TM = BM / 16, TN = BN / 16, RA = BM * CE_BK / 256, RB = BN * CE_BK / 256, BSTEP = 256 / BN;
    const int HW = H * W, khw = kH * kW, P = inC * khw, L = outH * outW;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;

    // Column space: every pixel (PHASE = false) or the pixels of one phase.
    // PHASE: blockIdx.z enumerates (phase, split) with phase-major order and ceil(K_phase / kPerSplit) splits per
    // phase; each split takes a CE_BK-aligned K range of its phase's taps and writes a disjoint-pixel partial slice,
    // summed in order afterwards when any phase has more than one split.
    int rh = 0, rw = 0, ih0 = 0, iw0 = 0, Hp = H, Wp = W, tapsH = kH, tapsW = kW, split = 0;
    if (PHASE)
    {
        const int phases = sH * sW;
        int phase = 0, rem = blockIdx.z;
        for (; phase < phases - 1; phase++)
        {
            const int ph = phase / sW, pw = phase - ph * sW;
            const int units = (outC * ((kH - ph + sH - 1) / sH) * ((kW - pw + sW - 1) / sW) + kPerSplit - 1) / kPerSplit;
            if (rem < units) break;
            rem -= units;
        }
        split = rem;
        rh = phase / sW; rw = phase - rh * sW;
        ih0 = ((rh - pH) % sH + sH) % sH; iw0 = ((rw - pW) % sW + sW) % sW;
        Hp = ih0 < H ? (H - ih0 + sH - 1) / sH : 0; Wp = iw0 < W ? (W - iw0 + sW - 1) / sW : 0;
        tapsH = (kH - rh + sH - 1) / sH; tapsW = (kW - rw + sW - 1) / sW;
    }
    const int plane = Hp * Wp, N = batch * plane;
    if (n0 >= N) return;
    const int K = outC * tapsH * tapsW;
    const int kBegin = (PHASE ? split : (int)blockIdx.z) * kPerSplit, kEnd = min(K, kBegin + kPerSplit);
    const int tiles = kBegin < kEnd ? (kEnd - kBegin + CE_BK - 1) / CE_BK : 0;

    constexpr int ASTEP = 256 / BM;   // A tile rows (K) per pass: lane runs along m (ci), RA passes of ASTEP rows
    const int am = tid % BM, ak = tid / BM;
    const int bn = tid % BN, bk = tid / BN;
    const int n = n0 + bn;
    const bool nIn = n < N;
    int nb = 0, ihp = 0, iwp = 0;
    if (nIn)
    {
        nb = n / plane; int q = n - nb * plane; int i = q / Wp;
        ihp = ih0 + (PHASE ? i * sH : i) + pH;
        iwp = iw0 + (PHASE ? (q - i * Wp) * sW : q - i * Wp) + pW;
    }
    const float* dyBatch = gradOutput + (long long)nb * outC * L;

    // Tile state: tap (kh, kw) and first output channel co0 (K tiles never straddle a tap: Cout % 8 == 0).
    int tapIndex = kBegin / outC, co0 = kBegin - tapIndex * outC;
    int th = tapIndex / tapsW, tw = tapIndex - th * tapsW;
    int kh = PHASE ? rh + th * sH : th, kw = PHASE ? rw + tw * sW : tw;

    float ra[RA], rb[RB];
    auto load = [&]()
    {
        const int m = m0 + am;
        const int tap = kh * kW + kw;
        const float* wTap = tapMajor ? kernel + ((long long)tap * outC + co0 + ak) * inC + m
                                     : kernel + ((long long)(co0 + ak) * inC + m) * khw + tap;
        const long long rowStep = tapMajor ? (long long)ASTEP * inC : (long long)ASTEP * P;
        #pragma unroll
        for (int r = 0; r < RA; r++)
            ra[r] = m < inC ? wTap[r * rowStep] : 0.0f;
        int ty_ = ihp - kh * dH, tx_ = iwp - kw * dW;
        int oh = ty_, ow = tx_;
        bool valid = nIn && ty_ >= 0 && tx_ >= 0;
        if (valid && (sH > 1 || sW > 1))
        {
            oh = ty_ / sH; ow = tx_ / sW;
            valid = PHASE || (oh * sH == ty_ && ow * sW == tx_);
        }
        valid = valid && oh < outH && ow < outW;
        const float* src = dyBatch + (long long)(co0 + bk) * L + (valid ? oh * outW + ow : 0);
        #pragma unroll
        for (int r = 0; r < RB; r++)
            rb[r] = valid ? src[(long long)(BSTEP * r) * L] : 0.0f;
    };
    auto store = [&](auto as, auto bs)
    {
        #pragma unroll
        for (int r = 0; r < RA; r++) as[ak + ASTEP * r][am] = ra[r];
        #pragma unroll
        for (int r = 0; r < RB; r++) bs[bk + BSTEP * r][bn] = rb[r];
    };
    auto advance = [&]()
    {
        co0 += CE_BK;
        if (co0 == outC)
        {
            co0 = 0;
            if (++tw == tapsW) { tw = 0; th++; }
            kh = PHASE ? rh + th * sH : th; kw = PHASE ? rw + tw * sW : tw;
        }
    };

    float acc[TM][TN];
    ce_run_mainloop<BM, BN, MAIN>(smem, tiles, load, store, advance, acc);

    using Map = CeMap<BM, BN, MAIN>;
    constexpr int VEC = Map::VEC;
    float* slice = out + (long long)(PHASE ? split : (int)blockIdx.z) * batch * inC * HW;
    if (PHASE)
    {
        // A phase's pixels are sH apart in dX, so the register layout (a lane's 4 columns, lanes 4 columns apart) put
        // every lane of a store in its own 32-byte sector. The tile goes through shared memory 32 rows at a time instead
        // and consecutive lanes store consecutive phase pixels (32 rows * (BN + 1) floats fit the main loop's buffers).
        constexpr int CR = 32, LD = BN + 1;
        float* cs = smem;
        for (int r0 = 0; r0 < BM; r0 += CR)
        {
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < TM; i++)
            {
                const int rr = Map::row(i) - r0;
                if (rr < 0 || rr >= CR) continue;
                #pragma unroll
                for (int j = 0; j < TN; j++) cs[rr * LD + Map::col(j)] = acc[i][j];
            }
            __syncthreads();
            for (int e = tid; e < CR * BN; e += 256)
            {
                const int rr = e / BN, nl = e - rr * BN;
                const int m = m0 + r0 + rr, nj = n0 + nl;
                if (m >= inC || nj >= N) continue;
                const int b = nj / plane, q = nj - b * plane, i2 = q / Wp;
                const int pix = (ih0 + i2 * sH) * W + iw0 + (q - i2 * Wp) * sW;
                slice[((long long)b * inC + m) * HW + pix] = cs[rr * LD + nl];
            }
        }
        return;
    }
    const bool vec = (HW % VEC) == 0;
    #pragma unroll
    for (int jg = 0; jg < TN; jg += VEC)
    {
        int nn = n0 + Map::col(jg);
        if (nn >= N) continue;
        if (vec)
        {
            int b = nn / plane, q = nn - b * plane;
            float* colp = slice + (long long)b * inC * HW + q;
            #pragma unroll
            for (int i = 0; i < TM; i++)
            {
                int m = m0 + Map::row(i);
                if (m >= inC) continue;
                float v[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; j++) v[j] = acc[i][jg + j];
                ce_store_vec<VEC>(colp + (long long)m * HW, v);
            }
            continue;
        }
        #pragma unroll
        for (int j = 0; j < VEC; j++)
        {
            int nj = nn + j;
            if (nj >= N) break;
            int b = nj / plane, q = nj - b * plane, i2 = q / Wp;
            int pix = PHASE ? (ih0 + i2 * sH) * W + iw0 + (q - i2 * Wp) * sW : q;
            float* colp = slice + (long long)b * inC * HW + pix;
            #pragma unroll
            for (int i = 0; i < TM; i++)
            {
                int m = m0 + Map::row(i);
                if (m < inC) colp[(long long)m * HW] = acc[i][jg + j];
            }
        }
    }
}

#define CE_WGRAD_KERNEL(NAME, BM, MAIN) \
extern ""C"" __global__ __launch_bounds__(256) void NAME( \
    const float* __restrict__ input, const float* __restrict__ gradOutput, float* __restrict__ out, \
    int batch, int inC, int H, int W, int outC, int outH, int outW, \
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit) \
{ \
    conv2d_wgrad_ce_body<BM, BM, MAIN>(input, gradOutput, out, batch, inC, H, W, outC, outH, outW, \
        kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit); \
}
CE_WGRAD_KERNEL(conv2d_wgrad_ce128, 128, 0)
CE_WGRAD_KERNEL(conv2d_wgrad_ce64, 64, 0)
CE_WGRAD_KERNEL(conv2d_wgrad_ce128_x3, 128, 1)
CE_WGRAD_KERNEL(conv2d_wgrad_ce64_x3, 64, 1)
CE_WGRAD_KERNEL(conv2d_wgrad_ce128_tf32, 128, 2)
CE_WGRAD_KERNEL(conv2d_wgrad_ce64_tf32, 64, 2)

#define CE_DGRAD_KERNEL(NAME, BM, PHASE, MAIN) \
extern ""C"" __global__ __launch_bounds__(256) void NAME( \
    const float* __restrict__ gradOutput, const float* __restrict__ kernel, float* __restrict__ out, \
    int batch, int inC, int H, int W, int outC, int outH, int outW, \
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit, int tapMajor) \
{ \
    conv2d_dgrad_ce_body<BM, BM, PHASE, MAIN>(gradOutput, kernel, out, batch, inC, H, W, outC, outH, outW, \
        kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit, tapMajor); \
}
CE_DGRAD_KERNEL(conv2d_dgrad_ce128, 128, false, 0)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64, 64, false, 0)
CE_DGRAD_KERNEL(conv2d_dgrad_ce128_phase, 128, true, 0)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64_phase, 64, true, 0)
CE_DGRAD_KERNEL(conv2d_dgrad_ce128_x3, 128, false, 1)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64_x3, 64, false, 1)
CE_DGRAD_KERNEL(conv2d_dgrad_ce128_phase_x3, 128, true, 1)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64_phase_x3, 64, true, 1)
CE_DGRAD_KERNEL(conv2d_dgrad_ce128_tf32, 128, false, 2)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64_tf32, 64, false, 2)
CE_DGRAD_KERNEL(conv2d_dgrad_ce128_phase_tf32, 128, true, 2)
CE_DGRAD_KERNEL(conv2d_dgrad_ce64_phase_tf32, 64, true, 2)
";
        }

        /// <summary>
        /// #1325: the implicit-GEMM conv engine's shared device code (CE_BK, ce_mainloop and the forward body
        /// conv2d_fprop_ig_body with its prologue / epilogue fusion hooks), CUDA C with no extern "C" entry points.
        /// Prepend it to another NVRTC module to instantiate the engine with fused operand transforms.
        /// </summary>
        public const string ConvEngineCore = @"
// ===== #1325: implicit-GEMM conv engine (cuBLAS-class SIMT tiling, strict fp32 FMA) =====
// One main loop for every conv GEMM: a BM x BN tile of C per 256-thread block, (BM/16) x (BN/16) accumulators per
// thread (8 x 8 at 128 x 128, 4 x 4 at 64 x 64), CE_BK = 8 deep K tiles. The operand loaders of each conv kind fill
// registers for the next K tile while the current one is multiplied out of the other of two shared buffers (one
// barrier per tile). A thread's accumulator rows are ty*4 + {0..3} + 64*g and its columns tx*4 + {0..3} + 64*g, so
// every shared read is a float4.
#define CE_BK 8

template <int BM, int BN, typename Load, typename Store, typename Advance>
__device__ __forceinline__ void ce_mainloop(float* smem, int tiles, Load load, Store store, Advance advance,
    float (&acc)[BM / 16][BN / 16])
{
    constexpr int TM = BM / 16, TN = BN / 16;
    float (*As)[CE_BK][BM + 4] = reinterpret_cast<float (*)[CE_BK][BM + 4]>(smem);
    float (*Bs)[CE_BK][BN + 4] = reinterpret_cast<float (*)[CE_BK][BN + 4]>(smem + 2 * CE_BK * (BM + 4));
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
    if (tiles <= 0) return;
    int buf = 0;
    load();
    store(As[0], Bs[0]);
    __syncthreads();
    for (int t = 0; t < tiles; t++)
    {
        bool more = t + 1 < tiles;
        if (more) { advance(); load(); }
        #pragma unroll
        for (int kk = 0; kk < CE_BK; kk++)
        {
            float a[TM], b[TN];
            #pragma unroll
            for (int g = 0; g < TM / 4; g++)
            {
                float4 v = *reinterpret_cast<const float4*>(&As[buf][kk][g * 64 + ty * 4]);
                a[g * 4] = v.x; a[g * 4 + 1] = v.y; a[g * 4 + 2] = v.z; a[g * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int g = 0; g < TN / 4; g++)
            {
                float4 v = *reinterpret_cast<const float4*>(&Bs[buf][kk][g * 64 + tx * 4]);
                b[g * 4] = v.x; b[g * 4 + 1] = v.y; b[g * 4 + 2] = v.z; b[g * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
        }
        if (more)
        {
            // The other buffer was last read in the previous iteration, before its closing barrier.
            store(As[buf ^ 1], Bs[buf ^ 1]);
            __syncthreads();
            buf ^= 1;
        }
    }
}

// #1335: the tensor-core mainloop. MAIN selects the inner product of every engine body: 0 = FP32 FMA (ce_mainloop),
// 1 = 3xTF32 (each fp32 operand split into TF32 hi + lo, three mma.sync m16n8k8 per product: lo*hi + hi*lo + hi*hi,
// fp32-level accuracy), 2 = plain TF32 (hi*hi only; explicit reduced-precision opt-in). The tensor-core variants need
// sm_80+; the host dispatches them only there, and below that the helpers compile to a no-op.
template <int MAIN> struct CePad { static constexpr int value = MAIN == 0 ? 4 : 8; };

// Output coordinates of accumulator acc[i][j] within the block tile, and how many consecutive columns a thread owns
// (4 for the FMA layout, 2 for the mma C fragment). Both layouts are separable: the row depends on i only, the column
// on j only, and columns jg .. jg + VEC - 1 (jg a multiple of VEC) are consecutive.
template <int BM, int BN, int MAIN>
struct CeMap
{
    static constexpr int VEC = MAIN == 0 ? 4 : 2;
    __device__ __forceinline__ static int row(int i)
    {
        if (MAIN == 0) return (i >> 2) * 64 + (threadIdx.x >> 4) * 4 + (i & 3);
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        return (warp >> 2) * (BM / 2) + (i >> 1) * 16 + (lane >> 2) + 8 * (i & 1);
    }
    __device__ __forceinline__ static int col(int j)
    {
        if (MAIN == 0) return (j >> 2) * 64 + (threadIdx.x & 15) * 4 + (j & 3);
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        return (warp & 3) * (BN / 4) + (j >> 1) * 8 + 2 * (lane & 3) + (j & 1);
    }
};

template <int VEC> __device__ __forceinline__ void ce_store_vec(float* p, const float* v);
template <> __device__ __forceinline__ void ce_store_vec<4>(float* p, const float* v)
{
    *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
}
template <> __device__ __forceinline__ void ce_store_vec<2>(float* p, const float* v)
{
    *reinterpret_cast<float2*>(p) = make_float2(v[0], v[1]);
}

__device__ __forceinline__ unsigned int ce_tf32(float x)
{
    unsigned int r;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm(""cvt.rna.tf32.f32 %0, %1;"" : ""=r""(r) : ""f""(x));
#else
    r = __float_as_uint(x) & 0xffffe000u;
#endif
    return r;
}

// d (4 floats, the C fragment) += A (m16 x k8, 4 TF32 regs) * B (k8 x n8, 2 TF32 regs).
__device__ __forceinline__ void ce_mma_tf32(float* d, const unsigned int* a, const unsigned int* b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(""mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
        : ""+f""(d[0]), ""+f""(d[1]), ""+f""(d[2]), ""+f""(d[3])
        : ""r""(a[0]), ""r""(a[1]), ""r""(a[2]), ""r""(a[3]), ""r""(b[0]), ""r""(b[1]));
#endif
}

// Same pipeline as ce_mainloop (register prefetch, two shared buffers, one barrier per K tile; CE_BK = 8 is one mma k
// step). The 8 warps tile the block 2 x 4; a warp owns (BM/2) x (BN/4) of C as MI x NI m16n8 tiles, its accumulators
// laid out as CeMap<BM, BN, 1> describes. Rows are padded by 8 so the fragment reads (lane = 4 g + t reading
// [t][base + g]) hit 32 distinct banks.
template <int BM, int BN, bool X3, typename Load, typename Store, typename Advance>
__device__ __forceinline__ void ce_mainloop_tc(float* smem, int tiles, Load load, Store store, Advance advance,
    float (&acc)[BM / 16][BN / 16])
{
    constexpr int TM = BM / 16, TN = BN / 16, MI = BM / 32, NI = BN / 32;
    float (*As)[CE_BK][BM + 8] = reinterpret_cast<float (*)[CE_BK][BM + 8]>(smem);
    float (*Bs)[CE_BK][BN + 8] = reinterpret_cast<float (*)[CE_BK][BN + 8]>(smem + 2 * CE_BK * (BM + 8));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = (warp >> 2) * (BM / 2), wn = (warp & 3) * (BN / 4);
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
    if (tiles <= 0) return;
    int buf = 0;
    load();
    store(As[0], Bs[0]);
    __syncthreads();
    for (int tile = 0; tile < tiles; tile++)
    {
        bool more = tile + 1 < tiles;
        if (more) { advance(); load(); }
        unsigned int ahi[MI][4], alo[MI][4];
        #pragma unroll
        for (int mi = 0; mi < MI; mi++)
        {
            int r = wm + mi * 16 + g;
            float a[4] = { As[buf][t][r], As[buf][t][r + 8], As[buf][t + 4][r], As[buf][t + 4][r + 8] };
            #pragma unroll
            for (int q = 0; q < 4; q++)
            {
                ahi[mi][q] = ce_tf32(a[q]);
                alo[mi][q] = X3 ? ce_tf32(a[q] - __uint_as_float(ahi[mi][q])) : 0u;
            }
        }
        #pragma unroll
        for (int ni = 0; ni < NI; ni++)
        {
            int c = wn + ni * 8 + g;
            float bv[2] = { Bs[buf][t][c], Bs[buf][t + 4][c] };
            unsigned int bhi[2], blo[2];
            #pragma unroll
            for (int q = 0; q < 2; q++)
            {
                bhi[q] = ce_tf32(bv[q]);
                blo[q] = X3 ? ce_tf32(bv[q] - __uint_as_float(bhi[q])) : 0u;
            }
            #pragma unroll
            for (int mi = 0; mi < MI; mi++)
            {
                float d[4] = { acc[mi * 2][ni * 2], acc[mi * 2][ni * 2 + 1], acc[mi * 2 + 1][ni * 2], acc[mi * 2 + 1][ni * 2 + 1] };
                if (X3)
                {
                    // Small terms first, then the dominant hi * hi.
                    ce_mma_tf32(d, alo[mi], bhi);
                    ce_mma_tf32(d, ahi[mi], blo);
                }
                ce_mma_tf32(d, ahi[mi], bhi);
                acc[mi * 2][ni * 2] = d[0]; acc[mi * 2][ni * 2 + 1] = d[1];
                acc[mi * 2 + 1][ni * 2] = d[2]; acc[mi * 2 + 1][ni * 2 + 1] = d[3];
            }
        }
        if (more)
        {
            store(As[buf ^ 1], Bs[buf ^ 1]);
            __syncthreads();
            buf ^= 1;
        }
    }
}

template <int BM, int BN, int MAIN, typename Load, typename Store, typename Advance>
__device__ __forceinline__ void ce_run_mainloop(float* smem, int tiles, Load load, Store store, Advance advance,
    float (&acc)[BM / 16][BN / 16])
{
    if constexpr (MAIN == 0) ce_mainloop<BM, BN>(smem, tiles, load, store, advance, acc);
    else ce_mainloop_tc<BM, BN, MAIN == 1>(smem, tiles, load, store, advance, acc);
}

// Forward: Y[Cout, B*oH*oW] = W[Cout, K] * X_gathered[K, B*oH*oW], K = Cin*kH*kW. TAPU (Cin % 8 == 0) orders K as
// (ci / 8, kh, kw, ci % 8), so a K tile shares one tap: each lane's input position and bounds check are computed once per tile and
// the tap advances by addition. Otherwise K is W's natural (ci, kh, kw) order with per-element index math. Split-K
// slices go to out + z * B*Cout*oH*oW and are summed in order by conv2d_wgrad_splitk_reduce.
// Fusion hooks (no-ops here): Pro(b, ci, ih * W + iw, x) transforms each in-bounds input element as it is loaded (padding
// stays 0, so it composes with an activation applied before the convolution); Epi(b, co, l, y) transforms each output
// value before it is stored (with one split; a split-K launch stores partials, so Epi must be the identity there).
struct CeIdentityPro { __device__ __forceinline__ float operator()(int, int, int, float x) const { return x; } };
struct CeIdentityEpi { __device__ __forceinline__ float operator()(int, int, int, float y) const { return y; } };

template <int BM, int BN, bool TAPU, typename Pro = CeIdentityPro, typename Epi = CeIdentityEpi, int MAIN = 0>
__device__ __forceinline__ void conv2d_fprop_ig_body(
    const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit,
    Pro pro = Pro(), Epi epi = Epi())
{
    __shared__ __align__(16) float smem[2 * CE_BK * (BM + CePad<MAIN>::value) + 2 * CE_BK * (BN + CePad<MAIN>::value)];
    constexpr int TM = BM / 16, TN = BN / 16, RA = BM * CE_BK / 256, RB = BN * CE_BK / 256, BSTEP = 256 / BN;
    const int L = outH * outW, N = batch * L, khw = kH * kW, K = inC * khw, HW = H * W;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int kBegin = blockIdx.z * kPerSplit, kEnd = min(K, kBegin + kPerSplit);
    const int tiles = kBegin < kEnd ? (kEnd - kBegin + CE_BK - 1) / CE_BK : 0;
    const int tid = threadIdx.x;
    const int ak = tid & 7, am = tid >> 3;       // A: k column ak, rows am + 32 r
    const int bn = tid % BN, bk = tid / BN;      // B: pixel column bn, k rows bk + BSTEP r
    const int n = n0 + bn;
    const bool nIn = n < N;
    int ih0 = 0, iw0 = 0, nb = 0;
    if (nIn) { nb = n / L; int l = n - nb * L; int oh = l / outW; ih0 = oh * sH - pH; iw0 = (l - oh * outW) * sW - pW; }
    const float* xBatch = input + (long long)nb * inC * HW;

    int kc = kBegin;
    int tap = 0, ci0 = 0, kh = 0, kw = 0;
    // TAPU K order: (ci group of CE_BK, tap, ci within the group). A K tile is one (group, tap), and the kH*kW tiles of
    // a group read the weights w[co][ci0 .. ci0+7][0 .. kH*kW-1], one contiguous run per output channel, back to back,
    // so that run stays in L1 across its taps. In (tap, ci) order every tap re-fetched it from L2 or DRAM 32 bytes per
    // lane (measured: the weight loads were 18-19 us of the 256@8 and 512@4 layers).
    if (TAPU)
    {
        int g = kBegin / (CE_BK * khw);
        tap = (kBegin - g * CE_BK * khw) / CE_BK; ci0 = g * CE_BK; kh = tap / kW; kw = tap - kh * kW;
    }

    float ra[RA], rb[RB];
    auto load = [&]()
    {
        #pragma unroll
        for (int r = 0; r < RA; r++)
        {
            int m = m0 + am + 32 * r;
            if (TAPU)
                ra[r] = m < outC ? kernel[(long long)m * K + (ci0 + ak) * khw + tap] : 0.0f;
            else
            {
                int k = kc + ak;
                ra[r] = (m < outC && k < kEnd) ? kernel[(long long)m * K + k] : 0.0f;
            }
        }
        if (TAPU)
        {
            int ih = ih0 + kh * dH, iw = iw0 + kw * dW;
            bool valid = nIn && ih >= 0 && ih < H && iw >= 0 && iw < W;
            int pix = valid ? ih * W + iw : 0;
            const float* src = xBatch + (long long)(ci0 + bk) * HW + pix;
            #pragma unroll
            for (int r = 0; r < RB; r++)
                rb[r] = valid ? pro(nb, ci0 + bk + BSTEP * r, pix, src[(long long)(BSTEP * r) * HW]) : 0.0f;
        }
        else
        {
            #pragma unroll
            for (int r = 0; r < RB; r++)
            {
                int k = kc + bk + BSTEP * r;
                float v = 0.0f;
                if (nIn && k < kEnd)
                {
                    int ci = k / khw, t = k - ci * khw;
                    int th = t / kW, tw = t - th * kW;
                    int ih = ih0 + th * dH, iw = iw0 + tw * dW;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) v = pro(nb, ci, ih * W + iw, xBatch[(long long)ci * HW + ih * W + iw]);
                }
                rb[r] = v;
            }
        }
    };
    auto store = [&](auto as, auto bs)
    {
        #pragma unroll
        for (int r = 0; r < RA; r++) as[ak][am + 32 * r] = ra[r];
        #pragma unroll
        for (int r = 0; r < RB; r++) bs[bk + BSTEP * r][bn] = rb[r];
    };
    auto advance = [&]()
    {
        kc += CE_BK;
        if (TAPU) { tap++; if (++kw == kW) { kw = 0; kh++; } if (tap == khw) { tap = 0; kh = 0; ci0 += CE_BK; } }
    };

    float acc[TM][TN];
    ce_run_mainloop<BM, BN, MAIN>(smem, tiles, load, store, advance, acc);

    using Map = CeMap<BM, BN, MAIN>;
    constexpr int VEC = Map::VEC;
    float* slice = out + (long long)blockIdx.z * batch * outC * L;
    const bool vec = (L % VEC) == 0;   // then a column group is VEC consecutive pixels of one image, aligned
    #pragma unroll
    for (int jg = 0; jg < TN; jg += VEC)
    {
        int nn = n0 + Map::col(jg);
        if (nn >= N) continue;
        int b = nn / L, l = nn - b * L;
        #pragma unroll
        for (int i = 0; i < TM; i++)
        {
            int m = m0 + Map::row(i);
            if (m >= outC) continue;
            float v[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; j++) v[j] = acc[i][jg + j];
            if (vec)
            {
                #pragma unroll
                for (int j = 0; j < VEC; j++) v[j] = epi(b, m, l + j, v[j]);
                ce_store_vec<VEC>(slice + ((long long)b * outC + m) * L + l, v);
            }
            else
            {
                #pragma unroll
                for (int j = 0; j < VEC; j++)
                {
                    int nj = nn + j;
                    if (nj >= N) break;
                    int bj = nj / L, lj = nj - bj * L;
                    slice[((long long)bj * outC + m) * L + lj] = epi(bj, m, lj, v[j]);
                }
            }
        }
    }
}

";

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
                "spiral_conv_backward_weights",
                "depthwise_conv2d_backward_input",
                "depthwise_conv2d_backward_weights",
                "conv2d_wgrad_implicit_gemm",
                "conv2d_wgrad_splitk_reduce", "conv2d_dgrad_phase_reduce", "conv2d_dgrad_weight_tapmajor",
                "conv2d_dgrad_implicit_gemm",
                "conv2d_dgrad_implicit_gemm_c16",
                "conv2d_dgrad_implicit_gemm_c16_phase",
                "conv2d_fprop_ig128_tap",
                "conv2d_fprop_ig128_gen",
                "conv2d_fprop_ig64_tap",
                "conv2d_fprop_ig64_gen",
                "conv2d_wgrad_ce128",
                "conv2d_wgrad_ce64",
                "conv2d_dgrad_ce128",
                "conv2d_dgrad_ce64",
                "conv2d_dgrad_ce128_phase",
                "conv2d_dgrad_ce64_phase",
                "conv2d_fprop_ig128_tap_x3",
                "conv2d_fprop_ig128_gen_x3",
                "conv2d_fprop_ig64_tap_x3",
                "conv2d_fprop_ig64_gen_x3",
                "conv2d_wgrad_ce128_x3",
                "conv2d_wgrad_ce64_x3",
                "conv2d_dgrad_ce128_x3",
                "conv2d_dgrad_ce128_phase_x3",
                "conv2d_dgrad_ce64_x3",
                "conv2d_dgrad_ce64_phase_x3",
                "conv2d_fprop_ig128_tap_tf32",
                "conv2d_fprop_ig128_gen_tf32",
                "conv2d_fprop_ig64_tap_tf32",
                "conv2d_fprop_ig64_gen_tf32",
                "conv2d_wgrad_ce128_tf32",
                "conv2d_wgrad_ce64_tf32",
                "conv2d_dgrad_ce128_tf32",
                "conv2d_dgrad_ce128_phase_tf32",
                "conv2d_dgrad_ce64_tf32",
                "conv2d_dgrad_ce64_phase_tf32"
            };
        }
    }
}
