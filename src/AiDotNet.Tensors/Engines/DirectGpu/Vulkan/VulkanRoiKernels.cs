namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public static class VulkanRoiKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    private const string Helper = @"
float bilinear_sample(uint planeBase, float y, float x, int H, int W) {
    if (H <= 0 || W <= 0) return 0.0;
    if (y < -1.0 || y > float(H) || x < -1.0 || x > float(W)) return 0.0;
    if (y <= 0.0) y = 0.0;
    if (x <= 0.0) x = 0.0;
    int y0 = int(y);
    int x0 = int(x);
    int y1_ = y0 + 1 >= H ? H - 1 : y0 + 1;
    int x1_ = x0 + 1 >= W ? W - 1 : x0 + 1;
    if (y0 >= H - 1) { y0 = y1_ = H - 1; y = float(y0); }
    if (x0 >= W - 1) { x0 = x1_ = W - 1; x = float(x0); }
    float ly = y - y0, lx = x - x0;
    float hy = 1.0 - ly, hx = 1.0 - lx;
    return hy * hx * input_[planeBase + uint(y0 * W + x0)]
         + hy * lx * input_[planeBase + uint(y0 * W + x1_)]
         + ly * hx * input_[planeBase + uint(y1_ * W + x0)]
         + ly * lx * input_[planeBase + uint(y1_ * W + x1_)];
}
";

    public static string RoIAlign => Header + @"
layout(set = 0, binding = 0) readonly buffer I { float input_[]; };
layout(set = 0, binding = 1) readonly buffer B { float boxes[]; };
layout(set = 0, binding = 2) writeonly buffer O { float output_[]; };
layout(push_constant) uniform P {
    int N; int C; int H; int W; int K; int outH; int outW;
    float spatialScale; int samplingRatio; int aligned;
};
" + Helper + @"
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = K * C * outH * outW;
    if (gid >= total) return;
    int pw = gid % outW; int t1 = gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int c = t2 % C; int k = t2 / C;
    int n = int(boxes[k * 5]);
    if (n < 0 || n >= N) { output_[gid] = 0.0; return; }
    float offset = aligned != 0 ? 0.5 : 0.0;
    float x1 = boxes[k * 5 + 1] * spatialScale - offset;
    float y1 = boxes[k * 5 + 2] * spatialScale - offset;
    float x2 = boxes[k * 5 + 3] * spatialScale - offset;
    float y2 = boxes[k * 5 + 4] * spatialScale - offset;
    float roiW = aligned != 0 ? (x2 - x1) : max(x2 - x1, 1.0);
    float roiH = aligned != 0 ? (y2 - y1) : max(y2 - y1, 1.0);
    float binH = roiH / outH;
    float binW = roiW / outW;
    int ry = samplingRatio > 0 ? samplingRatio : int(ceil(roiH / outH));
    int rx = samplingRatio > 0 ? samplingRatio : int(ceil(roiW / outW));
    if (ry < 1) ry = 1;
    if (rx < 1) rx = 1;
    uint planeBase = uint((n * C + c) * H * W);
    float acc = 0.0;
    for (int iy = 0; iy < ry; iy++) {
        float sy = y1 + ph * binH + (float(iy) + 0.5) * binH / ry;
        for (int ix = 0; ix < rx; ix++) {
            float sx = x1 + pw * binW + (float(ix) + 0.5) * binW / rx;
            acc += bilinear_sample(planeBase, sy, sx, H, W);
        }
    }
    output_[gid] = acc / float(ry * rx);
}
";

    public static string PsRoIAlign => Header + @"
layout(set = 0, binding = 0) readonly buffer I { float input_[]; };
layout(set = 0, binding = 1) readonly buffer B { float boxes[]; };
layout(set = 0, binding = 2) writeonly buffer O { float output_[]; };
layout(push_constant) uniform P {
    int N; int C; int H; int W; int K; int outH; int outW; int outputChannels;
    float spatialScale; int samplingRatio;
};
" + Helper + @"
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = K * outputChannels * outH * outW;
    if (gid >= total) return;
    int pw = gid % outW; int t1 = gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int co = t2 % outputChannels; int k = t2 / outputChannels;
    int n = int(boxes[k * 5]);
    if (n < 0 || n >= N) { output_[gid] = 0.0; return; }
    float x1 = boxes[k * 5 + 1] * spatialScale;
    float y1 = boxes[k * 5 + 2] * spatialScale;
    float x2 = boxes[k * 5 + 3] * spatialScale;
    float y2 = boxes[k * 5 + 4] * spatialScale;
    float roiW = max(x2 - x1, 0.1);
    float roiH = max(y2 - y1, 0.1);
    float binH = roiH / outH;
    float binW = roiW / outW;
    int ry = samplingRatio > 0 ? samplingRatio : int(ceil(roiH / outH));
    int rx = samplingRatio > 0 ? samplingRatio : int(ceil(roiW / outW));
    if (ry < 1) ry = 1;
    if (rx < 1) rx = 1;
    int c = (co * outH + ph) * outW + pw;
    if (c >= C) { output_[gid] = 0.0; return; }
    uint planeBase = uint((n * C + c) * H * W);
    float acc = 0.0;
    for (int iy = 0; iy < ry; iy++) {
        float sy = y1 + ph * binH + (float(iy) + 0.5) * binH / ry;
        for (int ix = 0; ix < rx; ix++) {
            float sx = x1 + pw * binW + (float(ix) + 0.5) * binW / rx;
            acc += bilinear_sample(planeBase, sy, sx, H, W);
        }
    }
    output_[gid] = acc / float(ry * rx);
}
";

    public static string PsRoIPool => Header + @"
layout(set = 0, binding = 0) readonly buffer I { float input_[]; };
layout(set = 0, binding = 1) readonly buffer B { float boxes[]; };
layout(set = 0, binding = 2) writeonly buffer O { float output_[]; };
layout(push_constant) uniform P {
    int N; int C; int H; int W; int K; int outH; int outW; int outputChannels; float spatialScale;
};
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = K * outputChannels * outH * outW;
    if (gid >= total) return;
    int pw = gid % outW; int t1 = gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int co = t2 % outputChannels; int k = t2 / outputChannels;
    int n = int(boxes[k * 5]);
    if (n < 0 || n >= N) { output_[gid] = 0.0; return; }
    float x1 = boxes[k * 5 + 1] * spatialScale;
    float y1 = boxes[k * 5 + 2] * spatialScale;
    float x2 = boxes[k * 5 + 3] * spatialScale;
    float y2 = boxes[k * 5 + 4] * spatialScale;
    float binH = max(y2 - y1, 0.1) / outH;
    float binW = max(x2 - x1, 0.1) / outW;
    int c = (co * outH + ph) * outW + pw;
    if (c >= C) { output_[gid] = 0.0; return; }
    int hs = int(max(0.0, floor(y1 + ph * binH)));
    int he = int(min(float(H), ceil(y1 + (ph + 1) * binH)));
    int ws = int(max(0.0, floor(x1 + pw * binW)));
    int we = int(min(float(W), ceil(x1 + (pw + 1) * binW)));
    uint planeBase = uint((n * C + c) * H * W);
    float acc = 0.0; int cnt = 0;
    for (int yy = hs; yy < he; yy++)
        for (int xx = ws; xx < we; xx++) { acc += input_[planeBase + uint(yy * W + xx)]; cnt++; }
    output_[gid] = cnt > 0 ? acc / float(cnt) : 0.0;
}
";

    public static string RoIPool => Header + @"
layout(set = 0, binding = 0) readonly buffer I { float input_[]; };
layout(set = 0, binding = 1) readonly buffer B { float boxes[]; };
layout(set = 0, binding = 2) writeonly buffer O { float output_[]; };
layout(push_constant) uniform P {
    int N; int C; int H; int W; int K; int outH; int outW; float spatialScale;
};
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = K * C * outH * outW;
    if (gid >= total) return;
    int pw = gid % outW; int t1 = gid / outW;
    int ph = t1 % outH; int t2 = t1 / outH;
    int c = t2 % C; int k = t2 / C;
    int n = int(boxes[k * 5]);
    if (n < 0 || n >= N) { output_[gid] = 0.0; return; }
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
    uint planeBase = uint((n * C + c) * H * W);
    if (hend <= hstart || wend <= wstart) { output_[gid] = 0.0; return; }
    float best = -3.402823466e+38;  // full -FLT_MAX
    for (int yy = hstart; yy < hend; yy++)
        for (int xx = wstart; xx < wend; xx++) {
            float v = input_[planeBase + uint(yy * W + xx)];
            if (v > best) best = v;
        }
    output_[gid] = best;
}
";
}
