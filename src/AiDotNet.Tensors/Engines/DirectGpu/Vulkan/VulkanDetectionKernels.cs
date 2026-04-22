namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GLSL compute shaders (Vulkan) for the vision detection ops added by Issue
/// #217. Mirrors CudaDetectionKernels / HipDetectionKernels / MetalDetectionKernels
/// function-for-function. Each shader is compiled to SPIR-V at runtime via
/// <see cref="VulkanGlslCompiler"/> and dispatched with 1-D workgroups of
/// local_size_x = 256.
/// </summary>
public static class VulkanDetectionKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    private const string TwoBufBoxes = @"
layout(set = 0, binding = 0) readonly buffer Boxes { float boxes[]; };
layout(set = 0, binding = 1) writeonly buffer Out { float o[]; };
";

    private const string ThreeBufIoU = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float b[]; };
layout(set = 0, binding = 2) writeonly buffer Out { float o[]; };
";

    // -----------------------------------------------------------------------
    // Pairwise IoU family — 3 SSBO bindings (a, b, output) + push constants
    // (n, m). Dispatched over n*m total cells.
    // -----------------------------------------------------------------------

    public static string BoxIou => Header + ThreeBufIoU + @"
layout(push_constant) uniform P { int n; int m; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = n * m;
    if (gid >= total) return;
    int i = gid / m;
    int j = gid % m;
    float ax1 = a[i * 4]; float ay1 = a[i * 4 + 1]; float ax2 = a[i * 4 + 2]; float ay2 = a[i * 4 + 3];
    float bx1 = b[j * 4]; float by1 = b[j * 4 + 1]; float bx2 = b[j * 4 + 2]; float by2 = b[j * 4 + 3];

    float aw = max(ax2 - ax1, 0.0); float ah = max(ay2 - ay1, 0.0);
    float bw = max(bx2 - bx1, 0.0); float bh = max(by2 - by1, 0.0);
    float areaA = aw * ah; float areaB = bw * bh;

    float ix1 = max(ax1, bx1); float iy1 = max(ay1, by1);
    float ix2 = min(ax2, bx2); float iy2 = min(ay2, by2);
    float iw = max(ix2 - ix1, 0.0); float ih = max(iy2 - iy1, 0.0);
    float inter = iw * ih;
    float u = areaA + areaB - inter;
    o[gid] = u > 0.0 ? inter / u : 0.0;
}";

    public static string GeneralizedBoxIou => Header + ThreeBufIoU + @"
layout(push_constant) uniform P { int n; int m; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = n * m;
    if (gid >= total) return;
    int i = gid / m;
    int j = gid % m;
    float ax1 = a[i * 4]; float ay1 = a[i * 4 + 1]; float ax2 = a[i * 4 + 2]; float ay2 = a[i * 4 + 3];
    float bx1 = b[j * 4]; float by1 = b[j * 4 + 1]; float bx2 = b[j * 4 + 2]; float by2 = b[j * 4 + 3];

    float areaA = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0);
    float areaB = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0);
    float ix1 = max(ax1, bx1); float iy1 = max(ay1, by1);
    float ix2 = min(ax2, bx2); float iy2 = min(ay2, by2);
    float inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0);
    float u = areaA + areaB - inter;
    float iou = u > 0.0 ? inter / u : 0.0;

    float ex1 = min(ax1, bx1); float ey1 = min(ay1, by1);
    float ex2 = max(ax2, bx2); float ey2 = max(ay2, by2);
    float enclose = max(ex2 - ex1, 0.0) * max(ey2 - ey1, 0.0);
    o[gid] = enclose > 0.0 ? iou - (enclose - u) / enclose : iou;
}";

    public static string DistanceBoxIou => Header + ThreeBufIoU + @"
layout(push_constant) uniform P { int n; int m; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = n * m;
    if (gid >= total) return;
    int i = gid / m;
    int j = gid % m;
    float ax1 = a[i * 4]; float ay1 = a[i * 4 + 1]; float ax2 = a[i * 4 + 2]; float ay2 = a[i * 4 + 3];
    float bx1 = b[j * 4]; float by1 = b[j * 4 + 1]; float bx2 = b[j * 4 + 2]; float by2 = b[j * 4 + 3];

    float areaA = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0);
    float areaB = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0);
    float ix1 = max(ax1, bx1); float iy1 = max(ay1, by1);
    float ix2 = min(ax2, bx2); float iy2 = min(ay2, by2);
    float inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0);
    float u = areaA + areaB - inter;
    float iou = u > 0.0 ? inter / u : 0.0;

    float acx = (ax1 + ax2) * 0.5; float acy = (ay1 + ay2) * 0.5;
    float bcx = (bx1 + bx2) * 0.5; float bcy = (by1 + by2) * 0.5;
    float dx = acx - bcx; float dy = acy - bcy;
    float centreSq = dx * dx + dy * dy;
    float ex1 = min(ax1, bx1); float ey1 = min(ay1, by1);
    float ex2 = max(ax2, bx2); float ey2 = max(ay2, by2);
    float ew = ex2 - ex1; float eh = ey2 - ey1;
    float diagSq = ew * ew + eh * eh;
    o[gid] = diagSq > 0.0 ? iou - centreSq / diagSq : iou;
}";

    public static string CompleteBoxIou => Header + ThreeBufIoU + @"
layout(push_constant) uniform P { int n; int m; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = n * m;
    if (gid >= total) return;
    int i = gid / m;
    int j = gid % m;
    float ax1 = a[i * 4]; float ay1 = a[i * 4 + 1]; float ax2 = a[i * 4 + 2]; float ay2 = a[i * 4 + 3];
    float bx1 = b[j * 4]; float by1 = b[j * 4 + 1]; float bx2 = b[j * 4 + 2]; float by2 = b[j * 4 + 3];

    float aw = ax2 - ax1; float ah = ay2 - ay1;
    float bw = bx2 - bx1; float bh = by2 - by1;
    float areaA = max(aw, 0.0) * max(ah, 0.0);
    float areaB = max(bw, 0.0) * max(bh, 0.0);
    float ix1 = max(ax1, bx1); float iy1 = max(ay1, by1);
    float ix2 = min(ax2, bx2); float iy2 = min(ay2, by2);
    float inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0);
    float u = areaA + areaB - inter;
    float iou = u > 0.0 ? inter / u : 0.0;

    float acx = (ax1 + ax2) * 0.5; float acy = (ay1 + ay2) * 0.5;
    float bcx = (bx1 + bx2) * 0.5; float bcy = (by1 + by2) * 0.5;
    float dx = acx - bcx; float dy = acy - bcy;
    float centreSq = dx * dx + dy * dy;
    float ex1 = min(ax1, bx1); float ey1 = min(ay1, by1);
    float ex2 = max(ax2, bx2); float ey2 = max(ay2, by2);
    float ew = ex2 - ex1; float eh = ey2 - ey1;
    float diagSq = ew * ew + eh * eh;
    float diou = diagSq > 0.0 ? iou - centreSq / diagSq : iou;

    float v = 0.0; float alpha = 0.0;
    if (ah > 0.0 && bh > 0.0) {
        float aspectA = atan(aw / ah);
        float aspectB = atan(bw / bh);
        float diff = aspectA - aspectB;
        const float PI = 3.14159265358979323846;
        const float invPiSq = 4.0 / (PI * PI);
        v = invPiSq * diff * diff;
        float denom = (1.0 - iou) + v;
        alpha = denom > 0.0 ? v / denom : 0.0;
    }
    o[gid] = diou - alpha * v;
}";

    // -----------------------------------------------------------------------
    // Per-box ops — 2 SSBO bindings (boxes, output).
    // -----------------------------------------------------------------------

    public static string BoxArea => Header + TwoBufBoxes + @"
layout(push_constant) uniform P { int n; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= n) return;
    float w = max(boxes[gid * 4 + 2] - boxes[gid * 4], 0.0);
    float h = max(boxes[gid * 4 + 3] - boxes[gid * 4 + 1], 0.0);
    o[gid] = w * h;
}";

    public static string BoxConvert => Header + TwoBufBoxes + @"
layout(push_constant) uniform P { int n; int fromFormat; int toFormat; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= n) return;
    int oi = gid * 4;
    float v0 = boxes[oi]; float v1 = boxes[oi + 1]; float v2 = boxes[oi + 2]; float v3 = boxes[oi + 3];

    float x1; float y1; float x2; float y2;
    if (fromFormat == 0) { x1 = v0; y1 = v1; x2 = v2; y2 = v3; }
    else if (fromFormat == 1) { x1 = v0; y1 = v1; x2 = v0 + v2; y2 = v1 + v3; }
    else { float hw = v2 * 0.5; float hh = v3 * 0.5;
           x1 = v0 - hw; y1 = v1 - hh; x2 = v0 + hw; y2 = v1 + hh; }

    if (toFormat == 0) { o[oi] = x1; o[oi + 1] = y1; o[oi + 2] = x2; o[oi + 3] = y2; }
    else if (toFormat == 1) {
        o[oi] = x1; o[oi + 1] = y1;
        o[oi + 2] = x2 - x1; o[oi + 3] = y2 - y1;
    } else {
        float w = x2 - x1; float h = y2 - y1;
        o[oi] = x1 + w * 0.5; o[oi + 1] = y1 + h * 0.5;
        o[oi + 2] = w; o[oi + 3] = h;
    }
}";

    // -----------------------------------------------------------------------
    // IoU family backward — Issue #217. Atomics-free two-kernel split:
    // gradA_kernel owns rows of N, gradB_kernel owns rows of M. Each kernel
    // has 4 SSBO bindings (gradOutput, a, b, gradA|gradB) and 3 push
    // constants (n, m, variant). GLSL here is the ASCII-safe equivalent of
    // the CudaDetectionKernels backward — no atomics needed.
    // -----------------------------------------------------------------------

    private const string QuadBufBackwardA = @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOutput[]; };
layout(set = 0, binding = 1) readonly buffer A { float a[]; };
layout(set = 0, binding = 2) readonly buffer B { float b[]; };
layout(set = 0, binding = 3) writeonly buffer GA { float gradA[]; };
";

    private const string QuadBufBackwardB = @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOutput[]; };
layout(set = 0, binding = 1) readonly buffer A { float a[]; };
layout(set = 0, binding = 2) readonly buffer B { float b[]; };
layout(set = 0, binding = 3) writeonly buffer GB { float gradB[]; };
";

    // The per-cell computation is embedded inline in both kernels. We use a
    // helper with inout out-params in GLSL. (GLSL functions can take inout
    // parameters for pointer-like semantics.)
    private const string IouCellFn = @"
const float DETECTION_INV_PI_SQ = 4.0 / (3.14159265358979323846 * 3.14159265358979323846);

void compute_cell_grads_iou(
    float ax1, float ay1, float ax2, float ay2,
    float bx1, float by1, float bx2, float by2,
    float g, int variant,
    out float gAx1, out float gAy1, out float gAx2, out float gAy2,
    out float gBx1, out float gBy1, out float gBx2, out float gBy2)
{
    gAx1 = 0.0; gAy1 = 0.0; gAx2 = 0.0; gAy2 = 0.0;
    gBx1 = 0.0; gBy1 = 0.0; gBx2 = 0.0; gBy2 = 0.0;
    if (g == 0.0) return;

    float awRaw = ax2 - ax1; float ahRaw = ay2 - ay1;
    float bwRaw = bx2 - bx1; float bhRaw = by2 - by1;
    float aw = max(awRaw, 0.0); float ah = max(ahRaw, 0.0);
    float bw = max(bwRaw, 0.0); float bh = max(bhRaw, 0.0);
    float areaA = aw * ah; float areaB = bw * bh;

    float ix1 = max(ax1, bx1); float ix2 = min(ax2, bx2);
    float iy1 = max(ay1, by1); float iy2 = min(ay2, by2);
    float iwRaw = ix2 - ix1; float ihRaw = iy2 - iy1;
    float iw = max(iwRaw, 0.0); float ih = max(ihRaw, 0.0);
    float inter = iw * ih;
    float u = areaA + areaB - inter;
    float iou = u > 0.0 ? inter / u : 0.0;

    float gIou = g;
    float gInter = 0.0; float gAreaA = 0.0; float gAreaB = 0.0;

    if (variant == 1) {
        float ex1 = min(ax1, bx1); float ex2 = max(ax2, bx2);
        float ey1 = min(ay1, by1); float ey2 = max(ay2, by2);
        float ewRaw = ex2 - ex1; float ehRaw = ey2 - ey1;
        float ew = max(ewRaw, 0.0); float eh = max(ehRaw, 0.0);
        float enclose = ew * eh;
        if (enclose > 0.0) {
            float gUnion = g / enclose;
            float gEnclose = g * (-u / (enclose * enclose));
            gAreaA += gUnion; gAreaB += gUnion; gInter += -gUnion;
            float gEw = gEnclose * eh; float gEh = gEnclose * ew;
            float gEwRaw = ewRaw > 0.0 ? gEw : 0.0;
            float gEhRaw = ehRaw > 0.0 ? gEh : 0.0;
            float gEx1 = -gEwRaw; float gEx2 = gEwRaw;
            float gEy1 = -gEhRaw; float gEy2 = gEhRaw;
            if (ax1 <= bx1) gAx1 += gEx1; else gBx1 += gEx1;
            if (ay1 <= by1) gAy1 += gEy1; else gBy1 += gEy1;
            if (ax2 >= bx2) gAx2 += gEx2; else gBx2 += gEx2;
            if (ay2 >= by2) gAy2 += gEy2; else gBy2 += gEy2;
        }
    } else if (variant == 2 || variant == 3) {
        float acx = (ax1 + ax2) * 0.5; float acy = (ay1 + ay2) * 0.5;
        float bcx = (bx1 + bx2) * 0.5; float bcy = (by1 + by2) * 0.5;
        float dcx = acx - bcx; float dcy = acy - bcy;
        float centreSq = dcx * dcx + dcy * dcy;
        float ex1 = min(ax1, bx1); float ex2 = max(ax2, bx2);
        float ey1 = min(ay1, by1); float ey2 = max(ay2, by2);
        float ew = ex2 - ex1; float eh = ey2 - ey1;
        float diagSq = ew * ew + eh * eh;

        float gCentreSq = 0.0; float gDiagSq = 0.0;
        if (diagSq > 0.0) {
            gCentreSq = g * (-1.0 / diagSq);
            gDiagSq = g * centreSq / (diagSq * diagSq);
        }

        if (variant == 3 && ah > 0.0 && bh > 0.0) {
            float aspectA = atan(aw / ah);
            float aspectB = atan(bw / bh);
            float diff = aspectA - aspectB;
            float v = DETECTION_INV_PI_SQ * diff * diff;
            float denom = (1.0 - iou) + v;
            float alpha = denom > 0.0 ? v / denom : 0.0;
            float gV = g * (-alpha);
            float gDiff = gV * 2.0 * DETECTION_INV_PI_SQ * diff;
            float gAspectA = gDiff; float gAspectB = -gDiff;
            float aDen = aw * aw + ah * ah;
            float bDen = bw * bw + bh * bh;
            if (aDen > 0.0) {
                float gAw = gAspectA * (ah / aDen);
                float gAh = gAspectA * (-aw / aDen);
                if (awRaw > 0.0) { gAx2 += gAw; gAx1 += -gAw; }
                if (ahRaw > 0.0) { gAy2 += gAh; gAy1 += -gAh; }
            }
            if (bDen > 0.0) {
                float gBw = gAspectB * (bh / bDen);
                float gBh = gAspectB * (-bw / bDen);
                if (bwRaw > 0.0) { gBx2 += gBw; gBx1 += -gBw; }
                if (bhRaw > 0.0) { gBy2 += gBh; gBy1 += -gBh; }
            }
        }

        float gAcx = gCentreSq * 2.0 * dcx;
        float gAcy = gCentreSq * 2.0 * dcy;
        gAx1 += gAcx * 0.5; gAx2 += gAcx * 0.5;
        gAy1 += gAcy * 0.5; gAy2 += gAcy * 0.5;
        gBx1 += -gAcx * 0.5; gBx2 += -gAcx * 0.5;
        gBy1 += -gAcy * 0.5; gBy2 += -gAcy * 0.5;

        float gEw = gDiagSq * 2.0 * ew;
        float gEh = gDiagSq * 2.0 * eh;
        float gEx1 = -gEw; float gEx2 = gEw;
        float gEy1 = -gEh; float gEy2 = gEh;
        if (ax1 <= bx1) gAx1 += gEx1; else gBx1 += gEx1;
        if (ay1 <= by1) gAy1 += gEy1; else gBy1 += gEy1;
        if (ax2 >= bx2) gAx2 += gEx2; else gBx2 += gEx2;
        if (ay2 >= by2) gAy2 += gEy2; else gBy2 += gEy2;
    }

    if (u > 0.0) {
        float uSq = u * u;
        gInter += gIou * (u + inter) / uSq;
        gAreaA += gIou * (-inter) / uSq;
        gAreaB += gIou * (-inter) / uSq;
    }

    float gIw = gInter * ih; float gIh = gInter * iw;
    float gIwRaw = iwRaw > 0.0 ? gIw : 0.0;
    float gIhRaw = ihRaw > 0.0 ? gIh : 0.0;
    float gIx2 = gIwRaw; float gIx1 = -gIwRaw;
    float gIy2 = gIhRaw; float gIy1 = -gIhRaw;
    if (ax1 >= bx1) gAx1 += gIx1; else gBx1 += gIx1;
    if (ay1 >= by1) gAy1 += gIy1; else gBy1 += gIy1;
    if (ax2 <= bx2) gAx2 += gIx2; else gBx2 += gIx2;
    if (ay2 <= by2) gAy2 += gIy2; else gBy2 += gIy2;

    float gAw2 = gAreaA * ah; float gAh2 = gAreaA * aw;
    float gBw2 = gAreaB * bh; float gBh2 = gAreaB * bw;
    if (awRaw > 0.0) { gAx2 += gAw2; gAx1 += -gAw2; }
    if (ahRaw > 0.0) { gAy2 += gAh2; gAy1 += -gAh2; }
    if (bwRaw > 0.0) { gBx2 += gBw2; gBx1 += -gBw2; }
    if (bhRaw > 0.0) { gBy2 += gBh2; gBy1 += -gBh2; }
}
";

    public static string IouBackwardA => Header + QuadBufBackwardA + IouCellFn + @"
layout(push_constant) uniform P { int n; int m; int variant; };
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= n) return;
    float ax1 = a[i * 4]; float ay1 = a[i * 4 + 1]; float ax2 = a[i * 4 + 2]; float ay2 = a[i * 4 + 3];
    float sumX1 = 0.0; float sumY1 = 0.0; float sumX2 = 0.0; float sumY2 = 0.0;
    for (int j = 0; j < m; j++) {
        float bx1 = b[j * 4]; float by1 = b[j * 4 + 1]; float bx2 = b[j * 4 + 2]; float by2 = b[j * 4 + 3];
        float g = gradOutput[i * m + j];
        float gAx1; float gAy1; float gAx2; float gAy2;
        float gBx1; float gBy1; float gBx2; float gBy2;
        compute_cell_grads_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, g, variant,
            gAx1, gAy1, gAx2, gAy2, gBx1, gBy1, gBx2, gBy2);
        sumX1 += gAx1; sumY1 += gAy1; sumX2 += gAx2; sumY2 += gAy2;
    }
    gradA[i * 4]     = sumX1;
    gradA[i * 4 + 1] = sumY1;
    gradA[i * 4 + 2] = sumX2;
    gradA[i * 4 + 3] = sumY2;
}";

    public static string IouBackwardB => Header + QuadBufBackwardB + IouCellFn + @"
layout(push_constant) uniform P { int n; int m; int variant; };
void main() {
    int j = int(gl_GlobalInvocationID.x);
    if (j >= m) return;
    float bx1 = b[j * 4]; float by1 = b[j * 4 + 1]; float bx2 = b[j * 4 + 2]; float by2 = b[j * 4 + 3];
    float sumX1 = 0.0; float sumY1 = 0.0; float sumX2 = 0.0; float sumY2 = 0.0;
    for (int i = 0; i < n; i++) {
        float ax1 = a[i * 4]; float ay1 = a[i * 4 + 1]; float ax2 = a[i * 4 + 2]; float ay2 = a[i * 4 + 3];
        float g = gradOutput[i * m + j];
        float gAx1; float gAy1; float gAx2; float gAy2;
        float gBx1; float gBy1; float gBx2; float gBy2;
        compute_cell_grads_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, g, variant,
            gAx1, gAy1, gAx2, gAy2, gBx1, gBy1, gBx2, gBy2);
        sumX1 += gBx1; sumY1 += gBy1; sumX2 += gBx2; sumY2 += gBy2;
    }
    gradB[j * 4]     = sumX1;
    gradB[j * 4 + 1] = sumY1;
    gradB[j * 4 + 2] = sumX2;
    gradB[j * 4 + 3] = sumY2;
}";
}
