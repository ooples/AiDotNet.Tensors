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

    // ---- Max pooling 2D (backward). Mirrors the OpenCL reference's non-atomic scatter-add: each output routes its
    // gradient to its argmax input. Collisions (two windows sharing an argmax) race exactly as the OpenCL kernel
    // does; parity with the validated reference is the contract. ----
    public const string MaxPool2DBackward = Header + @"
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
    gradInput[((b * channels + c) * inHeight + ih) * inWidth + iw] += grad;
}";
}
