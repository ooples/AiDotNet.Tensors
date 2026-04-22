// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for the RoI family (torchvision roi_align / roi_pool) —
// tail of Issue #217. One thread per output element.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels
{
    public static class HipRoiKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "roi_align",
            "roi_pool",
            "ps_roi_align",
            "ps_roi_pool",
        };

        public static string GetSource() => @"
#include <math.h>
#include <float.h>

__device__ __forceinline__ float bilinear_sample(
    const float* __restrict__ src, int planeBase,
    float y, float x, int H, int W)
{
    if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W) return 0.0f;
    if (y <= 0) y = 0;
    if (x <= 0) x = 0;
    int y0 = (int)y;
    int x0 = (int)x;
    int y1 = y0 + 1 >= H ? H - 1 : y0 + 1;
    int x1 = x0 + 1 >= W ? W - 1 : x0 + 1;
    if (y0 >= H - 1) { y0 = y1 = H - 1; y = (float)y0; }
    if (x0 >= W - 1) { x0 = x1 = W - 1; x = (float)x0; }
    float ly = y - y0, lx = x - x0;
    float hy = 1.0f - ly, hx = 1.0f - lx;
    float v00 = src[planeBase + y0 * W + x0];
    float v01 = src[planeBase + y0 * W + x1];
    float v10 = src[planeBase + y1 * W + x0];
    float v11 = src[planeBase + y1 * W + x1];
    return hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11;
}

extern ""C"" __global__ __launch_bounds__(256) void roi_align(
    const float* __restrict__ input, const float* __restrict__ boxes,
    float* __restrict__ output,
    int N, int C, int H, int W, int K,
    int outH, int outW,
    float spatialScale, int samplingRatio, int aligned)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * C * outH * outW;
    if (gid >= total) return;
    int pw = gid % outW; int t1 = gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int c = t2 % C; int k = t2 / C;

    int n = (int)boxes[k * 5];
    if (n < 0 || n >= N) { output[gid] = 0.0f; return; }
    float offset = aligned ? 0.5f : 0.0f;
    float x1 = boxes[k * 5 + 1] * spatialScale - offset;
    float y1 = boxes[k * 5 + 2] * spatialScale - offset;
    float x2 = boxes[k * 5 + 3] * spatialScale - offset;
    float y2 = boxes[k * 5 + 4] * spatialScale - offset;
    float roiW = aligned ? (x2 - x1) : fmaxf(x2 - x1, 1.0f);
    float roiH = aligned ? (y2 - y1) : fmaxf(y2 - y1, 1.0f);
    float binH = roiH / outH;
    float binW = roiW / outW;

    int ry = samplingRatio > 0 ? samplingRatio : (int)ceilf(roiH / outH);
    int rx = samplingRatio > 0 ? samplingRatio : (int)ceilf(roiW / outW);
    if (ry < 1) ry = 1;
    if (rx < 1) rx = 1;

    int planeBase = (n * C + c) * H * W;
    float acc = 0;
    for (int iy = 0; iy < ry; iy++) {
        float sy = y1 + ph * binH + (iy + 0.5f) * binH / ry;
        for (int ix = 0; ix < rx; ix++) {
            float sx = x1 + pw * binW + (ix + 0.5f) * binW / rx;
            acc += bilinear_sample(input, planeBase, sy, sx, H, W);
        }
    }
    output[gid] = acc / (ry * rx);
}

extern ""C"" __global__ __launch_bounds__(256) void roi_pool(
    const float* __restrict__ input, const float* __restrict__ boxes,
    float* __restrict__ output,
    int N, int C, int H, int W, int K,
    int outH, int outW, float spatialScale)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * C * outH * outW;
    if (gid >= total) return;
    int pw = gid % outW; int t1 = gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int c = t2 % C; int k = t2 / C;

    int n = (int)boxes[k * 5];
    if (n < 0 || n >= N) { output[gid] = 0.0f; return; }
    int x1 = (int)roundf(boxes[k * 5 + 1] * spatialScale);
    int y1 = (int)roundf(boxes[k * 5 + 2] * spatialScale);
    int x2 = (int)roundf(boxes[k * 5 + 3] * spatialScale);
    int y2 = (int)roundf(boxes[k * 5 + 4] * spatialScale);
    int roiW = x2 - x1 + 1; if (roiW < 1) roiW = 1;
    int roiH = y2 - y1 + 1; if (roiH < 1) roiH = 1;
    float binH = (float)roiH / outH;
    float binW = (float)roiW / outW;

    int hstart = (int)floorf(ph * binH) + y1;
    int hend = (int)ceilf((ph + 1) * binH) + y1;
    int wstart = (int)floorf(pw * binW) + x1;
    int wend = (int)ceilf((pw + 1) * binW) + x1;
    if (hstart < 0) hstart = 0; if (hstart > H) hstart = H;
    if (hend < 0) hend = 0; if (hend > H) hend = H;
    if (wstart < 0) wstart = 0; if (wstart > W) wstart = W;
    if (wend < 0) wend = 0; if (wend > W) wend = W;

    int planeBase = (n * C + c) * H * W;
    bool empty = hend <= hstart || wend <= wstart;
    if (empty) { output[gid] = 0.0f; return; }
    float best = -FLT_MAX;
    for (int yy = hstart; yy < hend; yy++)
        for (int xx = wstart; xx < wend; xx++) {
            float v = input[planeBase + yy * W + xx];
            if (v > best) best = v;
        }
    output[gid] = best;
}

// ----------------------------------------------------------------------------
// Position-sensitive RoIAlign / RoIPool (R-FCN). Input channel layout:
//   C = outputChannels * outH * outW. Per output (k, co, ph, pw), pull
//   from channel c = (co * outH + ph) * outW + pw.
// ----------------------------------------------------------------------------

extern ""C"" __global__ __launch_bounds__(256) void ps_roi_align(
    const float* __restrict__ input, const float* __restrict__ boxes,
    float* __restrict__ output,
    int N, int C, int H, int W, int K,
    int outH, int outW, int outputChannels,
    float spatialScale, int samplingRatio)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * outputChannels * outH * outW;
    if (gid >= total) return;
    int pw = gid % outW; int t1 = gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int co = t2 % outputChannels; int k = t2 / outputChannels;

    int n = (int)boxes[k * 5];
    if (n < 0 || n >= N) { output[gid] = 0.0f; return; }
    float x1 = boxes[k * 5 + 1] * spatialScale;
    float y1 = boxes[k * 5 + 2] * spatialScale;
    float x2 = boxes[k * 5 + 3] * spatialScale;
    float y2 = boxes[k * 5 + 4] * spatialScale;
    float roiW = fmaxf(x2 - x1, 0.1f);
    float roiH = fmaxf(y2 - y1, 0.1f);
    float binH = roiH / outH;
    float binW = roiW / outW;
    int ry = samplingRatio > 0 ? samplingRatio : (int)ceilf(roiH / outH);
    int rx = samplingRatio > 0 ? samplingRatio : (int)ceilf(roiW / outW);
    if (ry < 1) ry = 1;
    if (rx < 1) rx = 1;

    int c = (co * outH + ph) * outW + pw;
    int planeBase = (n * C + c) * H * W;
    float acc = 0;
    for (int iy = 0; iy < ry; iy++) {
        float sy = y1 + ph * binH + (iy + 0.5f) * binH / ry;
        for (int ix = 0; ix < rx; ix++) {
            float sx = x1 + pw * binW + (ix + 0.5f) * binW / rx;
            acc += bilinear_sample(input, planeBase, sy, sx, H, W);
        }
    }
    output[gid] = acc / (ry * rx);
}

extern ""C"" __global__ __launch_bounds__(256) void ps_roi_pool(
    const float* __restrict__ input, const float* __restrict__ boxes,
    float* __restrict__ output,
    int N, int C, int H, int W, int K,
    int outH, int outW, int outputChannels, float spatialScale)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * outputChannels * outH * outW;
    if (gid >= total) return;
    int pw = gid % outW; int t1 = gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int co = t2 % outputChannels; int k = t2 / outputChannels;

    int n = (int)boxes[k * 5];
    if (n < 0 || n >= N) { output[gid] = 0.0f; return; }
    float x1 = boxes[k * 5 + 1] * spatialScale;
    float y1 = boxes[k * 5 + 2] * spatialScale;
    float x2 = boxes[k * 5 + 3] * spatialScale;
    float y2 = boxes[k * 5 + 4] * spatialScale;
    float binH = fmaxf(y2 - y1, 0.1f) / outH;
    float binW = fmaxf(x2 - x1, 0.1f) / outW;

    int c = (co * outH + ph) * outW + pw;
    int planeBase = (n * C + c) * H * W;
    int hs = (int)fmaxf(0.0f, floorf(y1 + ph * binH));
    int he = (int)fminf((float)H, ceilf(y1 + (ph + 1) * binH));
    int ws = (int)fmaxf(0.0f, floorf(x1 + pw * binW));
    int we = (int)fminf((float)W, ceilf(x1 + (pw + 1) * binW));
    float acc = 0; int cnt = 0;
    for (int yy = hs; yy < he; yy++)
        for (int xx = ws; xx < we; xx++) {
            acc += input[planeBase + yy * W + xx];
            cnt++;
        }
    output[gid] = cnt > 0 ? acc / cnt : 0.0f;
}
";
    }
}
