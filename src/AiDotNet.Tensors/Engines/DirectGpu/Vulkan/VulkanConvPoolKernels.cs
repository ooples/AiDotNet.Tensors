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

    // ---- FUSED im2col + FP32->FP16, TRANSPOSED [K,N] layout (#1650/#638). 1-D-flattened port of the VALIDATED
    // CUDA im2col_kn_fp16hw (primitive relL2 0.028%). K = channels*kernelH*kernelW, N = batch*outH*outW; the col
    // matrix is written so the conv becomes a plain NN GEMM out[outC,N] = weights[outC,K] · col[K,N] fed to
    // GemmFp16In32fOut. The Vulkan backend stores FP16 packed TWO halves per 32-bit word (the layout
    // VulkanBackend.ConvertToFp16 produces and GemmFp16In32fOut consumes via unpackHalf2x16), so — unlike the CUDA
    // kernel which writes one ushort per thread — this kernel dispatches ONE THREAD PER OUTPUT WORD (ceil(N*K/2)
    // threads, exactly the per-word pattern of the FP16-native kernels) and writes one packHalf2x16 per word. That
    // makes the two-halves-share-a-word writes race-free (each word is written by exactly one thread). The two
    // col elements a word holds are the consecutive linear indices t0=2*w (low/even half) and t1=2*w+1 (high/odd
    // half); each is decoded + gathered with the SAME math as the CUDA reference (t = k*N + n). The high half of
    // the final word when N*K is odd is padding (val=0) the GEMM never reads (it indexes strictly < K*N).
    // packHalf2x16 is a core GLSL built-in (no float16/extension needed), matching GemmFp16In32fOut. outH/outW are
    // host-computed and passed in (declaration-order push constants, as int — same convention as the conv family).
    public const string Im2colKnFp16Hw = Header + @"
layout(set=0, binding=0) readonly buffer B0 { float inp[]; };
layout(set=0, binding=1) writeonly buffer B1 { uint outp[]; };
layout(push_constant) uniform PC {
    int batch; int channels; int height; int width;
    int kernelH; int kernelW; int strideH; int strideW;
    int padH; int padW; int dilationH; int dilationW; int outH; int outW;
};
float gather(int t, int N, int K) {
    // EXACT math of the CUDA im2col_kn_fp16hw, per col element t in [0, N*K).
    int n = t % N;     // output position (fastest — coalesced)
    int k = t / N;     // unrolled (c,kh,kw) row
    int b = n / (outH * outW);
    int rem = n % (outH * outW);
    int oh = rem / outW;
    int ow = rem % outW;
    int kw = k % kernelW;
    int kh = (k / kernelW) % kernelH;
    int c  = k / (kernelW * kernelH);
    int ih = oh * strideH - padH + kh * dilationH;
    int iw = ow * strideW - padW + kw * dilationW;
    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
        return inp[((b * channels + c) * height + ih) * width + iw];
    return 0.0;
}
void main() {
    int N = batch * outH * outW;
    int K = channels * kernelH * kernelW;
    int total = N * K;
    int w = int(gl_GlobalInvocationID.x);
    int numWords = (total + 1) / 2;     // ceil — packed two halves per 32-bit word
    if (w >= numWords) return;
    int t0 = 2 * w;
    int t1 = t0 + 1;
    float v0 = gather(t0, N, K);
    float v1 = (t1 < total) ? gather(t1, N, K) : 0.0;   // odd tail: high half is unread padding
    outp[w] = packHalf2x16(vec2(v0, v1));
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

    // ---- Locally connected Conv2D (forward, gather; weights [outH,outW,outC,inC,kH,kW]; bias [outC]; no padding) ----
    public const string LocallyConnectedConv2D = Header + @"
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float wgt[]; };
layout(set=0, binding=2) buffer B2 { float bias[]; };
layout(set=0, binding=3) buffer B3 { float outp[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW; int strideH; int strideW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * outChannels * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth; int t = idx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int oc = t % outChannels; int b = t / outChannels;
    int wBase = ((oh * outWidth + ow) * outChannels + oc) * inChannels * kernelH * kernelW;
    float sum = bias[oc];
    for (int ic = 0; ic < inChannels; ic++)
        for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH + kh; int iw = ow * strideW + kw;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                    sum += inp[((b * inChannels + ic) * inHeight + ih) * inWidth + iw]
                         * wgt[wBase + (ic * kernelH + kh) * kernelW + kw];
            }
    outp[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}";

    // ---- Locally connected backward input (gather per input element) ----
    public const string LocallyConnectedConv2DBackwardInput = Header + @"
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { float wgt[]; };
layout(set=0, binding=2) buffer B2 { float gradInput[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW; int strideH; int strideW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * inChannels * inHeight * inWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int b = idx / (inChannels * inHeight * inWidth);
    int r = idx % (inChannels * inHeight * inWidth);
    int ic = r / (inHeight * inWidth);
    int r2 = r % (inHeight * inWidth);
    int ih = r2 / inWidth; int iw = r2 % inWidth;
    float sum = 0.0;
    for (int oh = 0; oh < outHeight; oh++)
        for (int ow = 0; ow < outWidth; ow++) {
            int khr = ih - oh * strideH; int kwr = iw - ow * strideW;
            if (khr >= 0 && khr < kernelH && kwr >= 0 && kwr < kernelW) {
                for (int oc = 0; oc < outChannels; oc++) {
                    float g = gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
                    int wIdx = (((oh * outWidth + ow) * outChannels + oc) * inChannels + ic) * kernelH * kernelW + khr * kernelW + kwr;
                    sum += g * wgt[wIdx];
                }
            }
        }
    gradInput[idx] = sum;
}";

    // ---- Locally connected backward weights (one thread per weight; layout [outH,outW,outC,inC,kH,kW]) ----
    public const string LocallyConnectedConv2DBackwardWeights = Header + @"
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { float inp[]; };
layout(set=0, binding=2) buffer B2 { float gradWeights[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW; int strideH; int strideW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int tmp = idx;
    int kw = tmp % kernelW; tmp /= kernelW;
    int kh = tmp % kernelH; tmp /= kernelH;
    int ic = tmp % inChannels; tmp /= inChannels;
    int oc = tmp % outChannels; tmp /= outChannels;
    int ow = tmp % outWidth; tmp /= outWidth;
    int oh = tmp;
    int ih = oh * strideH + kh; int iw = ow * strideW + kw;
    float sum = 0.0;
    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
        for (int b = 0; b < batch; b++)
            sum += gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow]
                 * inp[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
    gradWeights[idx] = sum;
}";

    // ---- Locally connected backward bias (one thread per output channel) ----
    public const string LocallyConnectedConv2DBackwardBias = Header + @"
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { float gradBias[]; };
layout(push_constant) uniform PC { int batch; int outChannels; int outHeight; int outWidth; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(outChannels)) return;
    int oc = int(gid);
    float sum = 0.0;
    for (int b = 0; b < batch; b++)
        for (int oh = 0; oh < outHeight; oh++)
            for (int ow = 0; ow < outWidth; ow++)
                sum += gradOutput[((b * outChannels + oc) * outHeight + oh) * outWidth + ow];
    gradBias[oc] = sum;
}";

    // ---- Deformable Conv2D (forward, gather; DCNv2 — mask required on the GPU path) ----
    public const string DeformableConv2D = @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float wgt[]; };
layout(set=0, binding=2) buffer B2 { float offsets[]; };
layout(set=0, binding=3) buffer B3 { float mask[]; };
layout(set=0, binding=4) buffer B4 { float outp[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int dilationH; int dilationW; int groups; int deformGroups;
};
float bilinearSample(int b, int c, float h, float w) {
    int hl = int(floor(h)); int wl = int(floor(w)); int hh = hl + 1; int wh = wl + 1;
    float lh = h - float(hl); float lw = w - float(wl); float hhf = 1.0 - lh; float hwf = 1.0 - lw;
    float v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0;
    if (hl >= 0 && hl < inHeight && wl >= 0 && wl < inWidth) v1 = inp[((b*inChannels+c)*inHeight+hl)*inWidth+wl];
    if (hl >= 0 && hl < inHeight && wh >= 0 && wh < inWidth) v2 = inp[((b*inChannels+c)*inHeight+hl)*inWidth+wh];
    if (hh >= 0 && hh < inHeight && wl >= 0 && wl < inWidth) v3 = inp[((b*inChannels+c)*inHeight+hh)*inWidth+wl];
    if (hh >= 0 && hh < inHeight && wh >= 0 && wh < inWidth) v4 = inp[((b*inChannels+c)*inHeight+hh)*inWidth+wh];
    return hhf*hwf*v1 + hhf*lw*v2 + lh*hwf*v3 + lh*lw*v4;
}
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * outChannels * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth; int t = idx / outWidth;
    int oh = t % outHeight; t = t / outHeight;
    int oc = t % outChannels; int b = t / outChannels;
    int g = oc / (outChannels / groups);
    int dg = oc / (outChannels / deformGroups);
    int icpg = inChannels / groups;
    int ks = kernelH * kernelW;
    int baseH = oh * strideH - padH; int baseW = ow * strideW - padW;
    float sum = 0.0;
    for (int ic = 0; ic < icpg; ic++) {
        int aic = g * icpg + ic;
        for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++) {
                int ki = kh * kernelW + kw;
                int oIx = ((b*deformGroups+dg)*2*ks + ki)*outHeight*outWidth + oh*outWidth + ow;
                int oIy = ((b*deformGroups+dg)*2*ks + ks + ki)*outHeight*outWidth + oh*outWidth + ow;
                float h = float(baseH) + float(kh*dilationH) + offsets[oIx];
                float w = float(baseW) + float(kw*dilationW) + offsets[oIy];
                float val = bilinearSample(b, aic, h, w);
                int mIx = ((b*deformGroups+dg)*ks + ki)*outHeight*outWidth + oh*outWidth + ow;
                val *= mask[mIx];
                sum += val * wgt[((oc*icpg+ic)*kernelH+kh)*kernelW+kw];
            }
    }
    outp[((b*outChannels+oc)*outHeight+oh)*outWidth+ow] = sum;
}";

    // ---- Deformable backward input (gather per input element) ----
    public const string DeformableConv2DBackwardInput = @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { float wgt[]; };
layout(set=0, binding=2) buffer B2 { float offsets[]; };
layout(set=0, binding=3) buffer B3 { float mask[]; };
layout(set=0, binding=4) buffer B4 { float gradInput[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int dilationH; int dilationW; int groups; int deformGroups;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * inChannels * inHeight * inWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int b = idx / (inChannels*inHeight*inWidth);
    int r1 = idx % (inChannels*inHeight*inWidth);
    int ic = r1 / (inHeight*inWidth);
    int r2 = r1 % (inHeight*inWidth);
    int ih = r2 / inWidth; int iw = r2 % inWidth;
    int g = ic / (inChannels / groups);
    int icpg = inChannels / groups; int ocpg = outChannels / groups; int ks = kernelH * kernelW;
    float sum = 0.0;
    for (int oc = g*ocpg; oc < (g+1)*ocpg; oc++) {
        int dg = oc / (outChannels / deformGroups);
        int icLocal = ic - g*icpg;
        for (int oh = 0; oh < outHeight; oh++)
            for (int ow = 0; ow < outWidth; ow++) {
                int baseH = oh*strideH - padH; int baseW = ow*strideW - padW;
                for (int kh = 0; kh < kernelH; kh++)
                    for (int kw = 0; kw < kernelW; kw++) {
                        int ki = kh*kernelW + kw;
                        int oIx = ((b*deformGroups+dg)*2*ks + ki)*outHeight*outWidth + oh*outWidth + ow;
                        int oIy = ((b*deformGroups+dg)*2*ks + ks + ki)*outHeight*outWidth + oh*outWidth + ow;
                        float h = float(baseH) + float(kh*dilationH) + offsets[oIx];
                        float w = float(baseW) + float(kw*dilationW) + offsets[oIy];
                        int hl = int(floor(h)); int wl = int(floor(w)); int hh = hl+1; int wh = wl+1;
                        float lh = h - float(hl); float lw = w - float(wl); float hhf = 1.0-lh; float hwf = 1.0-lw;
                        float wc = 0.0;
                        if (ih == hl && iw == wl) wc = hhf*hwf;
                        else if (ih == hl && iw == wh) wc = hhf*lw;
                        else if (ih == hh && iw == wl) wc = lh*hwf;
                        else if (ih == hh && iw == wh) wc = lh*lw;
                        else continue;
                        float go = gradOutput[((b*outChannels+oc)*outHeight+oh)*outWidth+ow];
                        float contrib = go * wgt[((oc*icpg+icLocal)*kernelH+kh)*kernelW+kw] * wc;
                        int mIx = ((b*deformGroups+dg)*ks + ki)*outHeight*outWidth + oh*outWidth + ow;
                        contrib *= mask[mIx];
                        sum += contrib;
                    }
            }
    }
    gradInput[idx] = sum;
}";

    // ---- Deformable backward weights (one thread per weight) ----
    public const string DeformableConv2DBackwardWeights = @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { float inp[]; };
layout(set=0, binding=2) buffer B2 { float offsets[]; };
layout(set=0, binding=3) buffer B3 { float mask[]; };
layout(set=0, binding=4) buffer B4 { float gradWeights[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int dilationH; int dilationW; int groups; int deformGroups;
};
float bilinearSample(int b, int c, float h, float w) {
    int hl = int(floor(h)); int wl = int(floor(w)); int hh = hl + 1; int wh = wl + 1;
    float lh = h - float(hl); float lw = w - float(wl); float hhf = 1.0 - lh; float hwf = 1.0 - lw;
    float v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0;
    if (hl >= 0 && hl < inHeight && wl >= 0 && wl < inWidth) v1 = inp[((b*inChannels+c)*inHeight+hl)*inWidth+wl];
    if (hl >= 0 && hl < inHeight && wh >= 0 && wh < inWidth) v2 = inp[((b*inChannels+c)*inHeight+hl)*inWidth+wh];
    if (hh >= 0 && hh < inHeight && wl >= 0 && wl < inWidth) v3 = inp[((b*inChannels+c)*inHeight+hh)*inWidth+wl];
    if (hh >= 0 && hh < inHeight && wh >= 0 && wh < inWidth) v4 = inp[((b*inChannels+c)*inHeight+hh)*inWidth+wh];
    return hhf*hwf*v1 + hhf*lw*v2 + lh*hwf*v3 + lh*lw*v4;
}
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int icpg = inChannels / groups;
    int total = outChannels * icpg * kernelH * kernelW;
    if (gid >= uint(total)) return;
    int idx = int(gid); int tmp = idx;
    int kw = tmp % kernelW; tmp /= kernelW;
    int kh = tmp % kernelH; tmp /= kernelH;
    int icLocal = tmp % icpg; tmp /= icpg;
    int oc = tmp;
    int g = oc / (outChannels / groups);
    int dg = oc / (outChannels / deformGroups);
    int ic = g*icpg + icLocal; int ks = kernelH*kernelW; int ki = kh*kernelW + kw;
    float sum = 0.0;
    for (int b = 0; b < batch; b++)
        for (int oh = 0; oh < outHeight; oh++)
            for (int ow = 0; ow < outWidth; ow++) {
                int baseH = oh*strideH - padH; int baseW = ow*strideW - padW;
                int oIx = ((b*deformGroups+dg)*2*ks + ki)*outHeight*outWidth + oh*outWidth + ow;
                int oIy = ((b*deformGroups+dg)*2*ks + ks + ki)*outHeight*outWidth + oh*outWidth + ow;
                float h = float(baseH) + float(kh*dilationH) + offsets[oIx];
                float w = float(baseW) + float(kw*dilationW) + offsets[oIy];
                float iv = bilinearSample(b, ic, h, w);
                int mIx = ((b*deformGroups+dg)*ks + ki)*outHeight*outWidth + oh*outWidth + ow;
                iv *= mask[mIx];
                sum += gradOutput[((b*outChannels+oc)*outHeight+oh)*outWidth+ow] * iv;
            }
    gradWeights[idx] = sum;
}";

    // ---- Deformable backward offset (one thread per offset element) ----
    public const string DeformableConv2DBackwardOffset = @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { float inp[]; };
layout(set=0, binding=2) buffer B2 { float wgt[]; };
layout(set=0, binding=3) buffer B3 { float offsets[]; };
layout(set=0, binding=4) buffer B4 { float mask[]; };
layout(set=0, binding=5) buffer B5 { float gradOffsets[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int dilationH; int dilationW; int groups; int deformGroups;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int ks = kernelH * kernelW;
    int total = batch * deformGroups * 2 * ks * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid); int tmp = idx;
    int ow = tmp % outWidth; tmp /= outWidth;
    int oh = tmp % outHeight; tmp /= outHeight;
    int comp = tmp % (2*ks); tmp /= (2*ks);
    int dg = tmp % deformGroups; tmp /= deformGroups;
    int b = tmp;
    int isY = comp >= ks ? 1 : 0;
    int ki = comp % ks; int kh = ki / kernelW; int kw = ki % kernelW;
    int oIx = ((b*deformGroups+dg)*2*ks + ki)*outHeight*outWidth + oh*outWidth + ow;
    int oIy = ((b*deformGroups+dg)*2*ks + ks + ki)*outHeight*outWidth + oh*outWidth + ow;
    float oh_off = offsets[oIx]; float ow_off = offsets[oIy];
    int baseH = oh*strideH - padH; int baseW = ow*strideW - padW;
    float h = float(baseH) + float(kh*dilationH) + oh_off;
    float w = float(baseW) + float(kw*dilationW) + ow_off;
    int hl = int(floor(h)); int wl = int(floor(w)); int hh = hl+1; int wh = wl+1;
    float lh = h - float(hl); float lw = w - float(wl);
    float sum = 0.0;
    int icpg = inChannels / groups; int ocpg = outChannels / groups;
    for (int oco = 0; oco < outChannels / deformGroups; oco++) {
        int oc = dg * (outChannels / deformGroups) + oco;
        int g = oc / ocpg;
        float go = gradOutput[((b*outChannels+oc)*outHeight+oh)*outWidth+ow];
        int mIx = ((b*deformGroups+dg)*ks + ki)*outHeight*outWidth + oh*outWidth + ow;
        go *= mask[mIx];
        for (int ic = g*icpg; ic < (g+1)*icpg; ic++) {
            int icLocal = ic - g*icpg;
            float wv = wgt[((oc*icpg+icLocal)*kernelH+kh)*kernelW+kw];
            float v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0;
            if (hl >= 0 && hl < inHeight && wl >= 0 && wl < inWidth) v1 = inp[((b*inChannels+ic)*inHeight+hl)*inWidth+wl];
            if (hl >= 0 && hl < inHeight && wh >= 0 && wh < inWidth) v2 = inp[((b*inChannels+ic)*inHeight+hl)*inWidth+wh];
            if (hh >= 0 && hh < inHeight && wl >= 0 && wl < inWidth) v3 = inp[((b*inChannels+ic)*inHeight+hh)*inWidth+wl];
            if (hh >= 0 && hh < inHeight && wh >= 0 && wh < inWidth) v4 = inp[((b*inChannels+ic)*inHeight+hh)*inWidth+wh];
            if (isY == 0) sum += go * wv * ((1.0-lw)*(v3-v1) + lw*(v4-v2));
            else sum += go * wv * ((1.0-lh)*(v2-v1) + lh*(v4-v3));
        }
    }
    gradOffsets[idx] = sum;
}";

    // ---- Deformable backward mask (one thread per mask element) ----
    public const string DeformableConv2DBackwardMask = @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float gradOutput[]; };
layout(set=0, binding=1) buffer B1 { float inp[]; };
layout(set=0, binding=2) buffer B2 { float wgt[]; };
layout(set=0, binding=3) buffer B3 { float offsets[]; };
layout(set=0, binding=4) buffer B4 { float gradMask[]; };
layout(push_constant) uniform PC {
    int batch; int inChannels; int inHeight; int inWidth;
    int outChannels; int outHeight; int outWidth; int kernelH; int kernelW;
    int strideH; int strideW; int padH; int padW; int dilationH; int dilationW; int groups; int deformGroups;
};
float bilinearSample(int b, int c, float h, float w) {
    int hl = int(floor(h)); int wl = int(floor(w)); int hh = hl + 1; int wh = wl + 1;
    float lh = h - float(hl); float lw = w - float(wl); float hhf = 1.0 - lh; float hwf = 1.0 - lw;
    float v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0;
    if (hl >= 0 && hl < inHeight && wl >= 0 && wl < inWidth) v1 = inp[((b*inChannels+c)*inHeight+hl)*inWidth+wl];
    if (hl >= 0 && hl < inHeight && wh >= 0 && wh < inWidth) v2 = inp[((b*inChannels+c)*inHeight+hl)*inWidth+wh];
    if (hh >= 0 && hh < inHeight && wl >= 0 && wl < inWidth) v3 = inp[((b*inChannels+c)*inHeight+hh)*inWidth+wl];
    if (hh >= 0 && hh < inHeight && wh >= 0 && wh < inWidth) v4 = inp[((b*inChannels+c)*inHeight+hh)*inWidth+wh];
    return hhf*hwf*v1 + hhf*lw*v2 + lh*hwf*v3 + lh*lw*v4;
}
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int ks = kernelH * kernelW;
    int total = batch * deformGroups * ks * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid); int tmp = idx;
    int ow = tmp % outWidth; tmp /= outWidth;
    int oh = tmp % outHeight; tmp /= outHeight;
    int ki = tmp % ks; tmp /= ks;
    int dg = tmp % deformGroups; tmp /= deformGroups;
    int b = tmp;
    int kh = ki / kernelW; int kw = ki % kernelW;
    int oIx = ((b*deformGroups+dg)*2*ks + ki)*outHeight*outWidth + oh*outWidth + ow;
    int oIy = ((b*deformGroups+dg)*2*ks + ks + ki)*outHeight*outWidth + oh*outWidth + ow;
    float h = float(oh*strideH - padH) + float(kh*dilationH) + offsets[oIx];
    float w = float(ow*strideW - padW) + float(kw*dilationW) + offsets[oIy];
    float sum = 0.0;
    int icpg = inChannels / groups; int ocpg = outChannels / groups;
    for (int oco = 0; oco < outChannels / deformGroups; oco++) {
        int oc = dg * (outChannels / deformGroups) + oco;
        int g = oc / ocpg;
        float go = gradOutput[((b*outChannels+oc)*outHeight+oh)*outWidth+ow];
        for (int ic = g*icpg; ic < (g+1)*icpg; ic++) {
            int icLocal = ic - g*icpg;
            float wv = wgt[((oc*icpg+icLocal)*kernelH+kh)*kernelW+kw];
            sum += go * wv * bilinearSample(b, ic, h, w);
        }
    }
    gradMask[idx] = sum;
}";
}
