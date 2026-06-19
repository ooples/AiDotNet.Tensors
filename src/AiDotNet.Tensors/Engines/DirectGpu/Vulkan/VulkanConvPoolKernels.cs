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
}
