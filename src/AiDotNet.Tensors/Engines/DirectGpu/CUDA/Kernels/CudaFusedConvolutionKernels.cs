// Copyright (c) AiDotNet. All rights reserved.
// CUDA fused convolution kernels - Conv2D + BatchNorm/Bias + Activation in single pass.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA fused convolution kernels for CNN performance optimization.
    /// Uses device helper functions to deduplicate conv2d loop across activations.
    /// </summary>
    internal static class CudaFusedConvolutionKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

// ===========================================================================
// DEVICE HELPERS: Conv2D accumulation + activation functions
// ===========================================================================

__device__ __forceinline__ float conv2d_accumulate(
    const float* __restrict__ input, const float* __restrict__ weights,
    int b, int oc, int oh, int ow,
    int inChannels, int inHeight, int inWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    float sum = 0.0f;
    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    sum += __ldg(&input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw])
                         * __ldg(&weights[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw]);
                }
            }
        }
    }
    return sum;
}

__device__ __forceinline__ float depthwise_accumulate(
    const float* __restrict__ input, const float* __restrict__ weights,
    int b, int c, int oh, int ow,
    int channels, int inHeight, int inWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW)
{
    float sum = 0.0f;
    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;
            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                sum += __ldg(&input[((b * channels + c) * inHeight + ih) * inWidth + iw])
                     * __ldg(&weights[(c * kernelH + kh) * kernelW + kw]);
            }
        }
    }
    return sum;
}

__device__ __forceinline__ float activate_relu(float x) { return fmaxf(0.0f, x); }
__device__ __forceinline__ float activate_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float activate_gelu(float x) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}
__device__ __forceinline__ float activate_identity(float x) { return x; }

// ===========================================================================
// FUSED CONV2D + BIAS + ACTIVATION KERNELS
// ===========================================================================

#define DEFINE_CONV2D_BIAS_KERNEL(name, activation_fn) \
extern ""C"" __global__ __launch_bounds__(256) void name( \
    const float* __restrict__ input, const float* __restrict__ weights, \
    const float* __restrict__ bias, float* __restrict__ output, \
    int batch, int inChannels, int inHeight, int inWidth, \
    int outChannels, int outHeight, int outWidth, \
    int kernelH, int kernelW, int strideH, int strideW, \
    int padH, int padW, int dilationH, int dilationW) \
{ \
    int ow = blockIdx.x * blockDim.x + threadIdx.x; \
    int oh = blockIdx.y * blockDim.y + threadIdx.y; \
    int idx2 = blockIdx.z; \
    int oc = idx2 % outChannels; \
    int b = idx2 / outChannels; \
    if (ow >= outWidth || oh >= outHeight || b >= batch) return; \
    float sum = conv2d_accumulate(input, weights, b, oc, oh, ow, \
        inChannels, inHeight, inWidth, kernelH, kernelW, \
        strideH, strideW, padH, padW, dilationH, dilationW); \
    float x = sum + __ldg(&bias[oc]); \
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = activation_fn(x); \
}

DEFINE_CONV2D_BIAS_KERNEL(conv2d_bias_relu, activate_relu)
DEFINE_CONV2D_BIAS_KERNEL(conv2d_bias_gelu, activate_gelu)
DEFINE_CONV2D_BIAS_KERNEL(conv2d_bias_sigmoid, activate_sigmoid)
DEFINE_CONV2D_BIAS_KERNEL(conv2d_bias, activate_identity)

// ===========================================================================
// FUSED CONV2D + BATCHNORM + ACTIVATION (INFERENCE MODE)
// ===========================================================================

#define DEFINE_CONV2D_BN_KERNEL(name, activation_fn) \
extern ""C"" __global__ __launch_bounds__(256) void name( \
    const float* __restrict__ input, const float* __restrict__ foldedWeights, \
    const float* __restrict__ foldedBias, float* __restrict__ output, \
    int batch, int inChannels, int inHeight, int inWidth, \
    int outChannels, int outHeight, int outWidth, \
    int kernelH, int kernelW, int strideH, int strideW, \
    int padH, int padW, int dilationH, int dilationW) \
{ \
    int ow = blockIdx.x * blockDim.x + threadIdx.x; \
    int oh = blockIdx.y * blockDim.y + threadIdx.y; \
    int idx2 = blockIdx.z; \
    int oc = idx2 % outChannels; \
    int b = idx2 / outChannels; \
    if (ow >= outWidth || oh >= outHeight || b >= batch) return; \
    float sum = conv2d_accumulate(input, foldedWeights, b, oc, oh, ow, \
        inChannels, inHeight, inWidth, kernelH, kernelW, \
        strideH, strideW, padH, padW, dilationH, dilationW); \
    float x = sum + __ldg(&foldedBias[oc]); \
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = activation_fn(x); \
}

DEFINE_CONV2D_BN_KERNEL(conv2d_batchnorm_relu, activate_relu)
DEFINE_CONV2D_BN_KERNEL(conv2d_batchnorm_gelu, activate_gelu)
DEFINE_CONV2D_BN_KERNEL(conv2d_batchnorm, activate_identity)

// ===========================================================================
// DEPTHWISE FUSED KERNELS
// ===========================================================================

#define DEFINE_DEPTHWISE_KERNEL(name, activation_fn, bias_name) \
extern ""C"" __global__ __launch_bounds__(256) void name( \
    const float* __restrict__ input, const float* __restrict__ weights, \
    const float* __restrict__ bias_name, float* __restrict__ output, \
    int batch, int channels, int inHeight, int inWidth, \
    int outHeight, int outWidth, int kernelH, int kernelW, \
    int strideH, int strideW, int padH, int padW) \
{ \
    int ow = blockIdx.x * blockDim.x + threadIdx.x; \
    int oh = blockIdx.y * blockDim.y + threadIdx.y; \
    int idx2 = blockIdx.z; \
    int c = idx2 % channels; \
    int b = idx2 / channels; \
    if (ow >= outWidth || oh >= outHeight || b >= batch) return; \
    float sum = depthwise_accumulate(input, weights, b, c, oh, ow, \
        channels, inHeight, inWidth, kernelH, kernelW, \
        strideH, strideW, padH, padW); \
    float x = sum + __ldg(&bias_name[c]); \
    output[((b * channels + c) * outHeight + oh) * outWidth + ow] = activation_fn(x); \
}

DEFINE_DEPTHWISE_KERNEL(depthwise_conv2d_bias_relu, activate_relu, bias)
DEFINE_DEPTHWISE_KERNEL(depthwise_conv2d_batchnorm_relu, activate_relu, foldedBias)
";
        }

        public static string[] GetKernelNames()
        {
            return new string[]
            {
                "conv2d_bias_relu",
                "conv2d_bias_gelu",
                "conv2d_bias_sigmoid",
                "conv2d_bias",
                "conv2d_batchnorm_relu",
                "conv2d_batchnorm_gelu",
                "conv2d_batchnorm",
                "depthwise_conv2d_bias_relu",
                "depthwise_conv2d_batchnorm_relu"
            };
        }
    }
}
