// Copyright (c) AiDotNet. All rights reserved.
// OpenCL RoI kernels (Issue #217 tail).
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public static class OpenClRoiKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "roi_align", "roi_pool", "ps_roi_align", "ps_roi_pool",
    };

    public static string GetSource() => @"
inline float bilinear_sample(__global const float* src, int planeBase,
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
    return hy * hx * src[planeBase + y0 * W + x0]
         + hy * lx * src[planeBase + y0 * W + x1]
         + ly * hx * src[planeBase + y1 * W + x0]
         + ly * lx * src[planeBase + y1 * W + x1];
}

__kernel void roi_align(
    __global const float* input, __global const float* boxes,
    __global float* output,
    const int N, const int C, const int H, const int W, const int K,
    const int outH, const int outW,
    const float spatialScale, const int samplingRatio, const int aligned)
{
    int gid = get_global_id(0);
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
    float roiW = aligned ? (x2 - x1) : fmax(x2 - x1, 1.0f);
    float roiH = aligned ? (y2 - y1) : fmax(y2 - y1, 1.0f);
    float binH = roiH / outH;
    float binW = roiW / outW;
    int ry = samplingRatio > 0 ? samplingRatio : (int)ceil(roiH / outH);
    int rx = samplingRatio > 0 ? samplingRatio : (int)ceil(roiW / outW);
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

__kernel void roi_pool(
    __global const float* input, __global const float* boxes,
    __global float* output,
    const int N, const int C, const int H, const int W, const int K,
    const int outH, const int outW, const float spatialScale)
{
    int gid = get_global_id(0);
    int total = K * C * outH * outW;
    if (gid >= total) return;
    int pw = gid % outW; int t1 = gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int c = t2 % C; int k = t2 / C;

    int n = (int)boxes[k * 5];
    if (n < 0 || n >= N) { output[gid] = 0.0f; return; }
    int x1 = (int)round(boxes[k * 5 + 1] * spatialScale);
    int y1 = (int)round(boxes[k * 5 + 2] * spatialScale);
    int x2 = (int)round(boxes[k * 5 + 3] * spatialScale);
    int y2 = (int)round(boxes[k * 5 + 4] * spatialScale);
    int roiW = x2 - x1 + 1; if (roiW < 1) roiW = 1;
    int roiH = y2 - y1 + 1; if (roiH < 1) roiH = 1;
    float binH = (float)roiH / outH;
    float binW = (float)roiW / outW;

    int hstart = (int)floor(ph * binH) + y1;
    int hend = (int)ceil((ph + 1) * binH) + y1;
    int wstart = (int)floor(pw * binW) + x1;
    int wend = (int)ceil((pw + 1) * binW) + x1;
    if (hstart < 0) hstart = 0; if (hstart > H) hstart = H;
    if (hend < 0) hend = 0; if (hend > H) hend = H;
    if (wstart < 0) wstart = 0; if (wstart > W) wstart = W;
    if (wend < 0) wend = 0; if (wend > W) wend = W;

    int planeBase = (n * C + c) * H * W;
    if (hend <= hstart || wend <= wstart) { output[gid] = 0.0f; return; }
    float best = -FLT_MAX;
    for (int yy = hstart; yy < hend; yy++)
        for (int xx = wstart; xx < wend; xx++) {
            float v = input[planeBase + yy * W + xx];
            if (v > best) best = v;
        }
    output[gid] = best;
}

// Position-sensitive RoIAlign (R-FCN).
__kernel void ps_roi_align(
    __global const float* input, __global const float* boxes,
    __global float* output,
    const int N, const int C, const int H, const int W, const int K,
    const int outH, const int outW, const int outputChannels,
    const float spatialScale, const int samplingRatio)
{
    int gid = get_global_id(0);
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
    float roiW = fmax(x2 - x1, 0.1f);
    float roiH = fmax(y2 - y1, 0.1f);
    float binH = roiH / outH;
    float binW = roiW / outW;
    int ry = samplingRatio > 0 ? samplingRatio : (int)ceil(roiH / outH);
    int rx = samplingRatio > 0 ? samplingRatio : (int)ceil(roiW / outW);
    if (ry < 1) ry = 1;
    if (rx < 1) rx = 1;

    int c = (co * outH + ph) * outW + pw;
    if (c >= C) { output[gid] = 0.0f; return; }
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

__kernel void ps_roi_pool(
    __global const float* input, __global const float* boxes,
    __global float* output,
    const int N, const int C, const int H, const int W, const int K,
    const int outH, const int outW, const int outputChannels,
    const float spatialScale)
{
    int gid = get_global_id(0);
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
    float binH = fmax(y2 - y1, 0.1f) / outH;
    float binW = fmax(x2 - x1, 0.1f) / outW;

    int c = (co * outH + ph) * outW + pw;
    if (c >= C) { output[gid] = 0.0f; return; }
    int planeBase = (n * C + c) * H * W;
    int hs = (int)fmax(0.0f, floor(y1 + ph * binH));
    int he = (int)fmin((float)H, ceil(y1 + (ph + 1) * binH));
    int ws = (int)fmax(0.0f, floor(x1 + pw * binW));
    int we = (int)fmin((float)W, ceil(x1 + (pw + 1) * binW));
    float acc = 0; int cnt = 0;
    for (int yy = hs; yy < he; yy++)
        for (int xx = ws; xx < we; xx++) { acc += input[planeBase + yy * W + xx]; cnt++; }
    output[gid] = cnt > 0 ? acc / cnt : 0.0f;
}
";
}
#endif
