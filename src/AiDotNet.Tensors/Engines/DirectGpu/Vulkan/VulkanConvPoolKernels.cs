// Copyright (c) AiDotNet. All rights reserved.
// GLSL compute kernels for the Vulkan conv/pool family (issue #646). Each is a 1D-flattened port of the
// corresponding VERIFIED OpenCL kernel (one thread per output element; index decoded from gl_GlobalInvocationID.x),
// so the math matches the OpenCL/CUDA references. Dispatched via VulkanBackend.TryDispatchConvPoolGlsl with the
// param ints supplied as a push-constant block in declaration order. These require libshaderc at runtime and are
// validated on a Vulkan runner (not on the dev box).

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static class VulkanConvPoolKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    // ---- Average pooling 2D (forward, gather) ----
    public const string AvgPool2D = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float outp[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int inHeight; int inWidth;
    int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int countIncludePad;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth; int t = idx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int c = t % channels; int b = t / channels;
    float sum = 0.0; int count = 0;
    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;
            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                sum += inp[((b * channels + c) * inHeight + ih) * inWidth + iw]; count++;
            } else if (countIncludePad != 0) { count++; }
        }
    }
    int divisor = (countIncludePad != 0) ? (kernelH * kernelW) : max(count, 1);
    outp[((b * channels + c) * outHeight + oh) * outWidth + ow] = sum / float(divisor);
}";

    // ---- Average pooling 2D (backward, GATHER per input element — race-free) ----
    public const string AvgPool2DBackward = Header + @"
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { float gradInput[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int inHeight; int inWidth;
    int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int countIncludePad;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels * inHeight * inWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int iw = idx % inWidth; int t = idx / inWidth;
    int ih = t % inHeight; t = t / inHeight;
    int c = t % channels; int b = t / channels;
    float sum = 0.0;
    for (int oh = 0; oh < outHeight; oh++) {
        for (int ow = 0; ow < outWidth; ow++) {
            int hStart = oh * strideH - padH;
            int wStart = ow * strideW - padW;
            int hEnd = hStart + kernelH;
            int wEnd = wStart + kernelW;
            if (ih >= hStart && ih < hEnd && iw >= wStart && iw < wEnd) {
                int poolSize;
                if (countIncludePad != 0) {
                    poolSize = kernelH * kernelW;
                } else {
                    int hs = max(hStart, 0); int he = min(hEnd, inHeight);
                    int ws = max(wStart, 0); int we = min(wEnd, inWidth);
                    poolSize = (he - hs) * (we - ws);
                }
                sum += gradOutput[((b * channels + c) * outHeight + oh) * outWidth + ow] / float(max(poolSize, 1));
            }
        }
    }
    gradInput[((b * channels + c) * inHeight + ih) * inWidth + iw] = sum;
}";

    // ---- Max pooling 2D (forward, gather; writes argmax indices as raw int bits) ----
    public const string MaxPool2D = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float outp[]; };
layout(set=0, binding=2) buffer B2 { int indices[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int inHeight; int inWidth;
    int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth; int t = idx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int c = t % channels; int b = t / channels;
    float maxVal = -3.402823466e+38;
    int maxIdx = 0;
    for (int kh = 0; kh < kernelH; kh++) {
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;
            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                float v = inp[((b * channels + c) * inHeight + ih) * inWidth + iw];
                if (v > maxVal) { maxVal = v; maxIdx = ih * inWidth + iw; }
            }
        }
    }
    int outIdx = ((b * channels + c) * outHeight + oh) * outWidth + ow;
    outp[outIdx] = maxVal;
    indices[outIdx] = maxIdx;
}";

    // ---- Max pooling 2D (backward). One thread per OUTPUT scatters its gradient to its argmax input. Multiple
    // windows can share an argmax, so the scatter-add uses atomicAdd (correct, unlike the OpenCL reference's racy
    // `+=`). Requires GL_EXT_shader_atomic_float; if the device lacks it the pipeline fails to build and the caller
    // falls back to CPU. gradInput MUST be zero-initialized by the caller (VulkanBackend.Fill) before dispatch. ----
    public const string MaxPool2DBackward = @"#version 450
#extension GL_EXT_shader_atomic_float : require
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { int indices[]; };
layout(set=0, binding=2) buffer B2 { float gradInput[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int inHeight; int inWidth; int outHeight; int outWidth;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int outIdx = int(gid);
    int ow = outIdx % outWidth; int t = outIdx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int c = t % channels; int b = t / channels;
    float grad = gradOutput[outIdx];
    int maxIdx = indices[outIdx];
    int ih = maxIdx / inWidth;
    int iw = maxIdx % inWidth;
    atomicAdd(gradInput[((b * channels + c) * inHeight + ih) * inWidth + iw], grad);
}";

    // ---- Global average pooling 2D (forward): one thread per (b,c) ----
    public const string GlobalAvgPool2D = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float outp[]; };
layout(push_constant) uniform PC { int batch; int channels; int height; int width; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int c = idx % channels; int b = idx / channels;
    float sum = 0.0; int spatial = height * width;
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            sum += inp[((b * channels + c) * height + h) * width + w];
    outp[b * channels + c] = sum / float(spatial);
}";

    // ---- Global average pooling 2D (backward, gather): one thread per input element ----
    public const string GlobalAvgPool2DBackward = Header + @"
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { float gradInput[]; };
layout(push_constant) uniform PC { int batch; int channels; int height; int width; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int spatial = height * width;
    int total = batch * channels * spatial;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int c = (idx / spatial) % channels;
    int b = idx / (channels * spatial);
    gradInput[idx] = gradOutput[b * channels + c] / float(spatial);
}";

    // ---- Global max pooling 2D (forward, no indices): one thread per (b,c) ----
    public const string GlobalMaxPool2DNoIndices = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float outp[]; };
layout(push_constant) uniform PC { int batch; int channels; int height; int width; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int c = idx % channels; int b = idx / channels;
    float maxVal = -3.402823466e+38;
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            maxVal = max(maxVal, inp[((b * channels + c) * height + h) * width + w]);
    outp[b * channels + c] = maxVal;
}";

    // ---- Global max pooling 2D (forward, with argmax indices): one thread per (b,c) ----
    public const string GlobalMaxPool2D = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float outp[]; };
layout(set=0, binding=2) buffer B2 { int indices[]; };
layout(push_constant) uniform PC { int batch; int channels; int height; int width; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int c = idx % channels; int b = idx / channels;
    float maxVal = -3.402823466e+38; int maxIdx = 0;
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++) {
            float v = inp[((b * channels + c) * height + h) * width + w];
            if (v > maxVal) { maxVal = v; maxIdx = h * width + w; }
        }
    outp[b * channels + c] = maxVal;
    indices[b * channels + c] = maxIdx;
}";

    // ---- Global max pooling 2D (backward): one thread per (b,c) writes exactly one input position. Distinct (b,c)
    // map to distinct offsets, so there is no collision — a plain write suffices (caller zero-inits gradInput). ----
    public const string GlobalMaxPool2DBackward = Header + @"
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { int indices[]; };
layout(set=0, binding=2) buffer B2 { float gradInput[]; };
layout(push_constant) uniform PC { int batch; int channels; int height; int width; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int c = idx % channels; int b = idx / channels;
    int spatial = height * width;
    gradInput[(b * channels + c) * spatial + indices[idx]] = gradOutput[idx];
}";

    // ---- Adaptive average pooling 2D (forward, gather): one thread per output element ----
    public const string AdaptiveAvgPool2D = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float outp[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int inHeight; int inWidth; int outHeight; int outWidth;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth; int t = idx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int c = t % channels; int b = t / channels;
    int hStart = (oh * inHeight) / outHeight;
    int hEnd = ((oh + 1) * inHeight) / outHeight;
    int wStart = (ow * inWidth) / outWidth;
    int wEnd = ((ow + 1) * inWidth) / outWidth;
    float sum = 0.0; int count = 0;
    for (int ih = hStart; ih < hEnd; ih++)
        for (int iw = wStart; iw < wEnd; iw++) {
            sum += inp[((b * channels + c) * inHeight + ih) * inWidth + iw]; count++;
        }
    outp[((b * channels + c) * outHeight + oh) * outWidth + ow] = sum / float(max(count, 1));
}";

    // ---- Max pooling 3D (forward, NCDHW, with argmax indices): one thread per output element ----
    public const string MaxPool3D = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float outp[]; };
layout(set=0, binding=2) buffer B2 { int indices[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int inDepth; int inHeight; int inWidth;
    int outDepth; int outHeight; int outWidth;
    int kernelD; int kernelH; int kernelW; int strideD; int strideH; int strideW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels * outDepth * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth; int t = idx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int od = t % outDepth; t = t / outDepth;
    int c = t % channels; int b = t / channels;
    float maxVal = -3.402823466e+38; int maxIdx = 0;
    for (int kd = 0; kd < kernelD; kd++) {
        int id = od * strideD + kd; if (id >= inDepth) continue;
        for (int kh = 0; kh < kernelH; kh++) {
            int ih = oh * strideH + kh; if (ih >= inHeight) continue;
            for (int kw = 0; kw < kernelW; kw++) {
                int iw = ow * strideW + kw; if (iw >= inWidth) continue;
                float v = inp[((b * channels + c) * inDepth + id) * inHeight * inWidth + ih * inWidth + iw];
                if (v > maxVal) { maxVal = v; maxIdx = id * inHeight * inWidth + ih * inWidth + iw; }
            }
        }
    }
    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth + oh * outWidth + ow;
    outp[outIdx] = maxVal;
    indices[outIdx] = maxIdx;
}";

    // ---- Max pooling 3D (backward): one thread per output scatter-adds to its argmax input (atomic; caller
    // zero-inits gradInput). Requires GL_EXT_shader_atomic_float. ----
    public const string MaxPool3DBackward = @"#version 450
#extension GL_EXT_shader_atomic_float : require
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { int indices[]; };
layout(set=0, binding=2) buffer B2 { float gradInput[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int inDepth; int inHeight; int inWidth;
    int outDepth; int outHeight; int outWidth;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels * outDepth * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int outIdx = int(gid);
    int ow = outIdx % outWidth; int t = outIdx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int od = t % outDepth; t = t / outDepth;
    int c = t % channels; int b = t / channels;
    float grad = gradOutput[outIdx];
    int maxIdx = indices[outIdx];
    int spatialHW = inHeight * inWidth;
    int id = maxIdx / spatialHW; int rem = maxIdx % spatialHW;
    int ih = rem / inWidth; int iw = rem % inWidth;
    atomicAdd(gradInput[((b * channels + c) * inDepth + id) * inHeight * inWidth + ih * inWidth + iw], grad);
}";

    // ---- Conv2D (forward, gather): one thread per output element ----
    public const string Conv2D = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float wgt[]; };
layout(set=0, binding=2) buffer B2 { float outp[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int dilationH; int dilationW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * outChannels * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth; int t = idx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int oc = t % outChannels; int b = t / outChannels;
    float sum = 0.0;
    for (int ic = 0; ic < inChannels; ic++)
        for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                    sum += inp[((b * inChannels + ic) * inHeight + ih) * inWidth + iw]
                         * wgt[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
            }
    outp[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}";

    // ---- Conv2D backward input (gather per input element) ----
    public const string Conv2DBackwardInput = Header + @"
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { float wgt[]; };
layout(set=0, binding=2) buffer B2 { float gradInput[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int dilationH; int dilationW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * inChannels * inHeight * inWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int iw = idx % inWidth; int t = idx / inWidth;
    int ih = t % inHeight; t = t / inHeight;
    int ic = t % inChannels; int b = t / inChannels;
    float sum = 0.0;
    for (int oc = 0; oc < outChannels; oc++)
        for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++) {
                int ohb = ih + padH - kh * dilationH;
                int owb = iw + padW - kw * dilationW;
                if ((ohb % strideH) == 0 && (owb % strideW) == 0) {
                    int oh = ohb / strideH; int ow = owb / strideW;
                    if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth)
                        sum += gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow]
                             * wgt[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw];
                }
            }
    gradInput[((b * inChannels + ic) * inHeight + ih) * inWidth + iw] = sum;
}";

    // ---- Conv2D backward weights (one thread per kernel weight) ----
    public const string Conv2DBackwardKernel = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float gradOutput[]; };
layout(set=0, binding=2) buffer B2 { float gradKernel[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int dilationH; int dilationW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = outChannels * inChannels * kernelH * kernelW;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int kw = idx % kernelW; int t = idx / kernelW;
    int kh = t % kernelH; t = t / kernelH;
    int ic = t % inChannels; int oc = t / inChannels;
    float sum = 0.0;
    for (int b = 0; b < batch; b++)
        for (int oh = 0; oh < outHeight; oh++)
            for (int ow = 0; ow < outWidth; ow++) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                    sum += inp[((b * inChannels + ic) * inHeight + ih) * inWidth + iw]
                         * gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
            }
    gradKernel[((oc * inChannels + ic) * kernelH + kh) * kernelW + kw] = sum;
}";

    // ---- Depthwise Conv2D (forward, gather; weights [channels, kH, kW]) ----
    public const string DepthwiseConv2D = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float wgt[]; };
layout(set=0, binding=2) buffer B2 { float outp[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int inHeight; int inWidth;
    int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth; int t = idx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int c = t % channels; int b = t / channels;
    float sum = 0.0;
    for (int kh = 0; kh < kernelH; kh++)
        for (int kw = 0; kw < kernelW; kw++) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;
            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                sum += inp[((b * channels + c) * inHeight + ih) * inWidth + iw]
                     * wgt[(c * kernelH + kh) * kernelW + kw];
        }
    outp[((b * channels + c) * outHeight + oh) * outWidth + ow] = sum;
}";

    // ---- Transposed Conv2D backward weights (one thread per kernel weight; layout [inChannels,outChannels,kH,kW]) ----
    public const string ConvTranspose2DBackwardWeights = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float gradOutput[]; };
layout(set=0, binding=2) buffer B2 { float gradWeights[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int outputPadH; int outputPadW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = inChannels * outChannels * kernelH * kernelW;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int kw = idx % kernelW; int t = idx / kernelW;
    int kh = t % kernelH; t = t / kernelH;
    int oc = t % outChannels; int ic = t / outChannels;
    float sum = 0.0;
    for (int b = 0; b < batch; b++)
        for (int ih = 0; ih < inHeight; ih++)
            for (int iw = 0; iw < inWidth; iw++) {
                int oh = ih * strideH - padH + kh;
                int ow = iw * strideW - padW + kw;
                if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth)
                    sum += inp[((b * inChannels + ic) * inHeight + ih) * inWidth + iw]
                         * gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
            }
    gradWeights[idx] = sum;
}";

    // ---- Conv3D (forward, gather; weights [outC, inC, kD, kH, kW]) ----
    public const string Conv3D = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float wgt[]; };
layout(set=0, binding=2) buffer B2 { float outp[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inDepth; int inHeight; int inWidth;
    int outChannels; int outDepth; int outHeight; int outWidth;
    int kernelD; int kernelH; int kernelW; int strideD; int strideH; int strideW;
    int padD; int padH; int padW; int dilationD; int dilationH; int dilationW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * outChannels * outDepth * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth; int t = idx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int od = t % outDepth; t = t / outDepth;
    int oc = t % outChannels; int b = t / outChannels;
    float sum = 0.0;
    for (int ic = 0; ic < inChannels; ic++)
        for (int kd = 0; kd < kernelD; kd++)
            for (int kh = 0; kh < kernelH; kh++)
                for (int kw = 0; kw < kernelW; kw++) {
                    int id = od * strideD - padD + kd * dilationD;
                    int ih = oh * strideH - padH + kh * dilationH;
                    int iw = ow * strideW - padW + kw * dilationW;
                    if (id >= 0 && id < inDepth && ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                        sum += inp[(((b * inChannels + ic) * inDepth + id) * inHeight + ih) * inWidth + iw]
                             * wgt[(((oc * inChannels + ic) * kernelD + kd) * kernelH + kh) * kernelW + kw];
                }
    outp[(((b * outChannels + oc) * outDepth + od) * outHeight + oh) * outWidth + ow] = sum;
}";

    // ---- Unfold (Vulkan column layout [b, colRow=(c*kH+ki)*kW+kj, colLen=outH*outW], no dilation): one thread
    // per column element. Matches VulkanBackend.Unfold's CPU layout exactly (NOT the OpenCL patch-major im2col). ----
    public const string Unfold = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float outp[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int height; int width;
    int kernelH; int kernelW; int strideH; int strideW; int padH; int padW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int outH = (height + 2 * padH - kernelH) / strideH + 1;
    int outW = (width + 2 * padW - kernelW) / strideW + 1;
    int colLen = outH * outW;
    int colCh = channels * kernelH * kernelW;
    int total = batch * colCh * colLen;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int b = idx / (colCh * colLen);
    int r = idx % (colCh * colLen);
    int colRow = r / colLen;
    int pos = r % colLen;
    int oh = pos / outW; int ow = pos % outW;
    int c = colRow / (kernelH * kernelW);
    int k = colRow % (kernelH * kernelW);
    int ki = k / kernelW; int kj = k % kernelW;
    int ih = oh * strideH + ki - padH;
    int iw = ow * strideW + kj - padW;
    float val = 0.0;
    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
        val = inp[(b * channels + c) * height * width + ih * width + iw];
    outp[idx] = val;
}";

    // ---- Fold (Vulkan layout, GATHER per output pixel — race-free inverse of Unfold's scatter) ----
    public const string Fold = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float outp[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int outputH; int outputW;
    int kernelH; int kernelW; int strideH; int strideW; int padH; int padW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * channels * outputH * outputW;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int iw = idx % outputW; int t = idx / outputW;
    int ih = t % outputH; t = t / outputH;
    int c = t % channels; int b = t / channels;
    int unfoldH = (outputH + 2 * padH - kernelH) / strideH + 1;
    int unfoldW = (outputW + 2 * padW - kernelW) / strideW + 1;
    int colLen = unfoldH * unfoldW;
    int colCh = channels * kernelH * kernelW;
    float sum = 0.0;
    for (int ki = 0; ki < kernelH; ki++)
        for (int kj = 0; kj < kernelW; kj++) {
            int ohn = ih - ki + padH;
            int own = iw - kj + padW;
            if ((ohn % strideH) == 0 && (own % strideW) == 0) {
                int oh = ohn / strideH; int ow = own / strideW;
                if (oh >= 0 && oh < unfoldH && ow >= 0 && ow < unfoldW) {
                    int colRow = (c * kernelH + ki) * kernelW + kj;
                    sum += inp[b * colCh * colLen + colRow * colLen + oh * unfoldW + ow];
                }
            }
        }
    outp[idx] = sum;
}";
}
