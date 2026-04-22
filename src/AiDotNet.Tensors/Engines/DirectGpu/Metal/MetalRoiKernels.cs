// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    public static class MetalRoiKernels
    {
        public static string[] GetKernelNames() => new[] { "roi_align", "roi_pool" };

        public const string Source = @"
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

inline float bilinear_sample_msl(device const float* src, int planeBase,
    float y, float x, int H, int W)
{
    if (y < -1.0 || y > float(H) || x < -1.0 || x > float(W)) return 0.0;
    if (y <= 0) y = 0;
    if (x <= 0) x = 0;
    int y0 = int(y);
    int x0 = int(x);
    int y1 = y0 + 1 >= H ? H - 1 : y0 + 1;
    int x1 = x0 + 1 >= W ? W - 1 : x0 + 1;
    if (y0 >= H - 1) { y0 = y1 = H - 1; y = float(y0); }
    if (x0 >= W - 1) { x0 = x1 = W - 1; x = float(x0); }
    float ly = y - y0, lx = x - x0;
    float hy = 1.0 - ly, hx = 1.0 - lx;
    return hy * hx * src[planeBase + y0 * W + x0]
         + hy * lx * src[planeBase + y0 * W + x1]
         + ly * hx * src[planeBase + y1 * W + x0]
         + ly * lx * src[planeBase + y1 * W + x1];
}

kernel void roi_align(
    device const float* input [[buffer(0)]],
    device const float* boxes [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& C [[buffer(4)]],
    constant int& H [[buffer(5)]], constant int& W [[buffer(6)]],
    constant int& K [[buffer(7)]], constant int& outH [[buffer(8)]],
    constant int& outW [[buffer(9)]],
    constant float& spatialScale [[buffer(10)]],
    constant int& samplingRatio [[buffer(11)]], constant int& aligned [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    int total = K * C * outH * outW;
    if ((int)gid >= total) return;
    int pw = (int)gid % outW; int t1 = (int)gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int c = t2 % C; int k = t2 / C;
    int n = (int)boxes[k * 5];
    if (n < 0 || n >= N) { output[gid] = 0.0; return; }
    float offset = aligned ? 0.5 : 0.0;
    float x1 = boxes[k * 5 + 1] * spatialScale - offset;
    float y1 = boxes[k * 5 + 2] * spatialScale - offset;
    float x2 = boxes[k * 5 + 3] * spatialScale - offset;
    float y2 = boxes[k * 5 + 4] * spatialScale - offset;
    float roiW = aligned ? (x2 - x1) : max(x2 - x1, 1.0f);
    float roiH = aligned ? (y2 - y1) : max(y2 - y1, 1.0f);
    float binH = roiH / outH;
    float binW = roiW / outW;
    int ry = samplingRatio > 0 ? samplingRatio : int(ceil(roiH / outH));
    int rx = samplingRatio > 0 ? samplingRatio : int(ceil(roiW / outW));
    if (ry < 1) ry = 1;
    if (rx < 1) rx = 1;
    int planeBase = (n * C + c) * H * W;
    float acc = 0;
    for (int iy = 0; iy < ry; iy++) {
        float sy = y1 + ph * binH + (iy + 0.5f) * binH / ry;
        for (int ix = 0; ix < rx; ix++) {
            float sx = x1 + pw * binW + (ix + 0.5f) * binW / rx;
            acc += bilinear_sample_msl(input, planeBase, sy, sx, H, W);
        }
    }
    output[gid] = acc / (ry * rx);
}

kernel void roi_pool(
    device const float* input [[buffer(0)]],
    device const float* boxes [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& C [[buffer(4)]],
    constant int& H [[buffer(5)]], constant int& W [[buffer(6)]],
    constant int& K [[buffer(7)]], constant int& outH [[buffer(8)]],
    constant int& outW [[buffer(9)]],
    constant float& spatialScale [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    int total = K * C * outH * outW;
    if ((int)gid >= total) return;
    int pw = (int)gid % outW; int t1 = (int)gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int c = t2 % C; int k = t2 / C;
    int n = (int)boxes[k * 5];
    if (n < 0 || n >= N) { output[gid] = 0.0; return; }
    int x1 = int(round(boxes[k * 5 + 1] * spatialScale));
    int y1 = int(round(boxes[k * 5 + 2] * spatialScale));
    int x2 = int(round(boxes[k * 5 + 3] * spatialScale));
    int y2 = int(round(boxes[k * 5 + 4] * spatialScale));
    int roiW = x2 - x1 + 1; if (roiW < 1) roiW = 1;
    int roiH = y2 - y1 + 1; if (roiH < 1) roiH = 1;
    float binH = float(roiH) / outH;
    float binW = float(roiW) / outW;
    int hstart = int(floor(ph * binH)) + y1;
    int hend = int(ceil((ph + 1) * binH)) + y1;
    int wstart = int(floor(pw * binW)) + x1;
    int wend = int(ceil((pw + 1) * binW)) + x1;
    if (hstart < 0) hstart = 0; if (hstart > H) hstart = H;
    if (hend < 0) hend = 0; if (hend > H) hend = H;
    if (wstart < 0) wstart = 0; if (wstart > W) wstart = W;
    if (wend < 0) wend = 0; if (wend > W) wend = W;
    int planeBase = (n * C + c) * H * W;
    if (hend <= hstart || wend <= wstart) { output[gid] = 0.0; return; }
    float best = -3.4e38;
    for (int yy = hstart; yy < hend; yy++)
        for (int xx = wstart; xx < wend; xx++) {
            float v = input[planeBase + yy * W + xx];
            if (v > best) best = v;
        }
    output[gid] = best;
}
";
    }
}
