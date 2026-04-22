// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for the vision detection ops added by Issue #217.
// Mirrors CudaDetectionKernels function-for-function — HIP's hiprtc accepts
// the same CUDA-style source, so we keep the two files structurally identical
// to share test harnesses and reference math.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels
{
    /// <summary>
    /// Mirror of <c>CudaDetectionKernels</c> for HIP/ROCm. Pairwise IoU family
    /// + per-box ops. fp32 only — DirectGpuTensorEngine gates GPU dispatch on
    /// T = float, so non-float tensors fall through to the CpuEngine path.
    /// </summary>
    internal static class HipDetectionKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "detection_box_iou",
            "detection_generalized_box_iou",
            "detection_distance_box_iou",
            "detection_complete_box_iou",
            "detection_box_area",
            "detection_box_convert",
            "detection_iou_backward_a",
            "detection_iou_backward_b",
        };

        public static string GetSource() => @"
// HIP-RTC device defines — no #include needed.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------------------------------------------------------------------
// Pairwise IoU family — one thread per (i, j) cell.
// ----------------------------------------------------------------------------

extern ""C"" __global__ __launch_bounds__(256) void detection_box_iou(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int n, int m)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n * m) return;
    int i = gid / m;
    int j = gid % m;
    float ax1 = a[i * 4], ay1 = a[i * 4 + 1], ax2 = a[i * 4 + 2], ay2 = a[i * 4 + 3];
    float bx1 = b[j * 4], by1 = b[j * 4 + 1], bx2 = b[j * 4 + 2], by2 = b[j * 4 + 3];

    float aw = fmaxf(ax2 - ax1, 0.0f);
    float ah = fmaxf(ay2 - ay1, 0.0f);
    float bw = fmaxf(bx2 - bx1, 0.0f);
    float bh = fmaxf(by2 - by1, 0.0f);
    float areaA = aw * ah;
    float areaB = bw * bh;

    float ix1 = fmaxf(ax1, bx1);
    float iy1 = fmaxf(ay1, by1);
    float ix2 = fminf(ax2, bx2);
    float iy2 = fminf(ay2, by2);
    float iw = fmaxf(ix2 - ix1, 0.0f);
    float ih = fmaxf(iy2 - iy1, 0.0f);
    float inter = iw * ih;
    float u = areaA + areaB - inter;
    output[gid] = u > 0.0f ? inter / u : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void detection_generalized_box_iou(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int n, int m)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n * m) return;
    int i = gid / m;
    int j = gid % m;
    float ax1 = a[i * 4], ay1 = a[i * 4 + 1], ax2 = a[i * 4 + 2], ay2 = a[i * 4 + 3];
    float bx1 = b[j * 4], by1 = b[j * 4 + 1], bx2 = b[j * 4 + 2], by2 = b[j * 4 + 3];

    float areaA = fmaxf(ax2 - ax1, 0.0f) * fmaxf(ay2 - ay1, 0.0f);
    float areaB = fmaxf(bx2 - bx1, 0.0f) * fmaxf(by2 - by1, 0.0f);
    float ix1 = fmaxf(ax1, bx1), iy1 = fmaxf(ay1, by1);
    float ix2 = fminf(ax2, bx2), iy2 = fminf(ay2, by2);
    float inter = fmaxf(ix2 - ix1, 0.0f) * fmaxf(iy2 - iy1, 0.0f);
    float u = areaA + areaB - inter;
    float iou = u > 0.0f ? inter / u : 0.0f;

    float ex1 = fminf(ax1, bx1), ey1 = fminf(ay1, by1);
    float ex2 = fmaxf(ax2, bx2), ey2 = fmaxf(ay2, by2);
    float enclose = fmaxf(ex2 - ex1, 0.0f) * fmaxf(ey2 - ey1, 0.0f);
    output[gid] = enclose > 0.0f ? iou - (enclose - u) / enclose : iou;
}

extern ""C"" __global__ __launch_bounds__(256) void detection_distance_box_iou(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int n, int m)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n * m) return;
    int i = gid / m;
    int j = gid % m;
    float ax1 = a[i * 4], ay1 = a[i * 4 + 1], ax2 = a[i * 4 + 2], ay2 = a[i * 4 + 3];
    float bx1 = b[j * 4], by1 = b[j * 4 + 1], bx2 = b[j * 4 + 2], by2 = b[j * 4 + 3];

    float areaA = fmaxf(ax2 - ax1, 0.0f) * fmaxf(ay2 - ay1, 0.0f);
    float areaB = fmaxf(bx2 - bx1, 0.0f) * fmaxf(by2 - by1, 0.0f);
    float ix1 = fmaxf(ax1, bx1), iy1 = fmaxf(ay1, by1);
    float ix2 = fminf(ax2, bx2), iy2 = fminf(ay2, by2);
    float inter = fmaxf(ix2 - ix1, 0.0f) * fmaxf(iy2 - iy1, 0.0f);
    float u = areaA + areaB - inter;
    float iou = u > 0.0f ? inter / u : 0.0f;

    float acx = (ax1 + ax2) * 0.5f, acy = (ay1 + ay2) * 0.5f;
    float bcx = (bx1 + bx2) * 0.5f, bcy = (by1 + by2) * 0.5f;
    float dx = acx - bcx, dy = acy - bcy;
    float centreSq = dx * dx + dy * dy;
    float ex1 = fminf(ax1, bx1), ey1 = fminf(ay1, by1);
    float ex2 = fmaxf(ax2, bx2), ey2 = fmaxf(ay2, by2);
    float ew = ex2 - ex1, eh = ey2 - ey1;
    float diagSq = ew * ew + eh * eh;
    output[gid] = diagSq > 0.0f ? iou - centreSq / diagSq : iou;
}

extern ""C"" __global__ __launch_bounds__(256) void detection_complete_box_iou(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int n, int m)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n * m) return;
    int i = gid / m;
    int j = gid % m;
    float ax1 = a[i * 4], ay1 = a[i * 4 + 1], ax2 = a[i * 4 + 2], ay2 = a[i * 4 + 3];
    float bx1 = b[j * 4], by1 = b[j * 4 + 1], bx2 = b[j * 4 + 2], by2 = b[j * 4 + 3];

    float aw = ax2 - ax1, ah = ay2 - ay1;
    float bw = bx2 - bx1, bh = by2 - by1;
    float areaA = fmaxf(aw, 0.0f) * fmaxf(ah, 0.0f);
    float areaB = fmaxf(bw, 0.0f) * fmaxf(bh, 0.0f);
    float ix1 = fmaxf(ax1, bx1), iy1 = fmaxf(ay1, by1);
    float ix2 = fminf(ax2, bx2), iy2 = fminf(ay2, by2);
    float inter = fmaxf(ix2 - ix1, 0.0f) * fmaxf(iy2 - iy1, 0.0f);
    float u = areaA + areaB - inter;
    float iou = u > 0.0f ? inter / u : 0.0f;

    float acx = (ax1 + ax2) * 0.5f, acy = (ay1 + ay2) * 0.5f;
    float bcx = (bx1 + bx2) * 0.5f, bcy = (by1 + by2) * 0.5f;
    float dx = acx - bcx, dy = acy - bcy;
    float centreSq = dx * dx + dy * dy;
    float ex1 = fminf(ax1, bx1), ey1 = fminf(ay1, by1);
    float ex2 = fmaxf(ax2, bx2), ey2 = fmaxf(ay2, by2);
    float ew = ex2 - ex1, eh = ey2 - ey1;
    float diagSq = ew * ew + eh * eh;
    float diou = diagSq > 0.0f ? iou - centreSq / diagSq : iou;

    float v = 0.0f, alpha = 0.0f;
    if (ah > 0.0f && bh > 0.0f) {
        float aspectA = atanf(aw / ah);
        float aspectB = atanf(bw / bh);
        float diff = aspectA - aspectB;
        const float invPiSq = 4.0f / ((float)M_PI * (float)M_PI);
        v = invPiSq * diff * diff;
        float denom = (1.0f - iou) + v;
        alpha = denom > 0.0f ? v / denom : 0.0f;
    }
    output[gid] = diou - alpha * v;
}

// ----------------------------------------------------------------------------
// BoxArea — one thread per box.
// ----------------------------------------------------------------------------

extern ""C"" __global__ __launch_bounds__(256) void detection_box_area(
    const float* __restrict__ boxes, float* __restrict__ output, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float w = fmaxf(boxes[gid * 4 + 2] - boxes[gid * 4], 0.0f);
    float h = fmaxf(boxes[gid * 4 + 3] - boxes[gid * 4 + 1], 0.0f);
    output[gid] = w * h;
}

// ----------------------------------------------------------------------------
// BoxConvert — fromFormat/toFormat are int codes:
//   0 = XYXY, 1 = XYWH, 2 = CXCYWH.
// One thread per box.
// ----------------------------------------------------------------------------

extern ""C"" __global__ __launch_bounds__(256) void detection_box_convert(
    const float* __restrict__ boxes, float* __restrict__ output,
    int n, int fromFormat, int toFormat)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int o = gid * 4;
    float v0 = boxes[o], v1 = boxes[o + 1], v2 = boxes[o + 2], v3 = boxes[o + 3];

    float x1, y1, x2, y2;
    if (fromFormat == 0) { x1 = v0; y1 = v1; x2 = v2; y2 = v3; }
    else if (fromFormat == 1) { x1 = v0; y1 = v1; x2 = v0 + v2; y2 = v1 + v3; }
    else { float hw = v2 * 0.5f, hh = v3 * 0.5f;
           x1 = v0 - hw; y1 = v1 - hh; x2 = v0 + hw; y2 = v1 + hh; }

    if (toFormat == 0) { output[o] = x1; output[o + 1] = y1; output[o + 2] = x2; output[o + 3] = y2; }
    else if (toFormat == 1) {
        output[o] = x1; output[o + 1] = y1;
        output[o + 2] = x2 - x1; output[o + 3] = y2 - y1;
    } else {
        float w = x2 - x1, h = y2 - y1;
        output[o] = x1 + w * 0.5f; output[o + 1] = y1 + h * 0.5f;
        output[o + 2] = w; output[o + 3] = h;
    }
}

// ----------------------------------------------------------------------------
// IoU family backward — Issue #217. Mirrors CudaDetectionKernels (atomics-
// free two-kernel split). See CudaDetectionKernels.cs for derivation notes.
// ----------------------------------------------------------------------------

__device__ __forceinline__ void compute_cell_grads_iou(
    float ax1, float ay1, float ax2, float ay2,
    float bx1, float by1, float bx2, float by2,
    float g, int variant,
    float* gAx1, float* gAy1, float* gAx2, float* gAy2,
    float* gBx1, float* gBy1, float* gBx2, float* gBy2)
{
    *gAx1 = 0.0f; *gAy1 = 0.0f; *gAx2 = 0.0f; *gAy2 = 0.0f;
    *gBx1 = 0.0f; *gBy1 = 0.0f; *gBx2 = 0.0f; *gBy2 = 0.0f;
    if (g == 0.0f) return;

    const float INV_PI_SQ = 4.0f / (3.14159265358979323846f * 3.14159265358979323846f);

    float awRaw = ax2 - ax1, ahRaw = ay2 - ay1;
    float bwRaw = bx2 - bx1, bhRaw = by2 - by1;
    float aw = fmaxf(awRaw, 0.0f), ah = fmaxf(ahRaw, 0.0f);
    float bw = fmaxf(bwRaw, 0.0f), bh = fmaxf(bhRaw, 0.0f);
    float areaA = aw * ah, areaB = bw * bh;

    float ix1 = fmaxf(ax1, bx1), ix2 = fminf(ax2, bx2);
    float iy1 = fmaxf(ay1, by1), iy2 = fminf(ay2, by2);
    float iwRaw = ix2 - ix1, ihRaw = iy2 - iy1;
    float iw = fmaxf(iwRaw, 0.0f), ih = fmaxf(ihRaw, 0.0f);
    float inter = iw * ih;
    float u = areaA + areaB - inter;
    float iou = u > 0.0f ? inter / u : 0.0f;

    float gIou = g;
    float gInter = 0.0f, gAreaA = 0.0f, gAreaB = 0.0f;

    if (variant == 1) {
        float ex1 = fminf(ax1, bx1), ex2 = fmaxf(ax2, bx2);
        float ey1 = fminf(ay1, by1), ey2 = fmaxf(ay2, by2);
        float ewRaw = ex2 - ex1, ehRaw = ey2 - ey1;
        float ew = fmaxf(ewRaw, 0.0f), eh = fmaxf(ehRaw, 0.0f);
        float enclose = ew * eh;
        if (enclose > 0.0f) {
            float gUnion = g / enclose;
            float gEnclose = g * (-u / (enclose * enclose));
            gAreaA += gUnion; gAreaB += gUnion; gInter += -gUnion;
            float gEw = gEnclose * eh, gEh = gEnclose * ew;
            float gEwRaw = ewRaw > 0.0f ? gEw : 0.0f;
            float gEhRaw = ehRaw > 0.0f ? gEh : 0.0f;
            float gEx1 = -gEwRaw, gEx2 = gEwRaw;
            float gEy1 = -gEhRaw, gEy2 = gEhRaw;
            if (ax1 <= bx1) *gAx1 += gEx1; else *gBx1 += gEx1;
            if (ay1 <= by1) *gAy1 += gEy1; else *gBy1 += gEy1;
            if (ax2 >= bx2) *gAx2 += gEx2; else *gBx2 += gEx2;
            if (ay2 >= by2) *gAy2 += gEy2; else *gBy2 += gEy2;
        }
    } else if (variant == 2 || variant == 3) {
        float acx = (ax1 + ax2) * 0.5f, acy = (ay1 + ay2) * 0.5f;
        float bcx = (bx1 + bx2) * 0.5f, bcy = (by1 + by2) * 0.5f;
        float dcx = acx - bcx, dcy = acy - bcy;
        float centreSq = dcx * dcx + dcy * dcy;
        float ex1 = fminf(ax1, bx1), ex2 = fmaxf(ax2, bx2);
        float ey1 = fminf(ay1, by1), ey2 = fmaxf(ay2, by2);
        float ew = ex2 - ex1, eh = ey2 - ey1;
        float diagSq = ew * ew + eh * eh;

        float gCentreSq = 0.0f, gDiagSq = 0.0f;
        if (diagSq > 0.0f) {
            gCentreSq = g * (-1.0f / diagSq);
            gDiagSq = g * centreSq / (diagSq * diagSq);
        }

        if (variant == 3 && ah > 0.0f && bh > 0.0f) {
            float aspectA = atanf(aw / ah);
            float aspectB = atanf(bw / bh);
            float diff = aspectA - aspectB;
            float v = INV_PI_SQ * diff * diff;
            float denom = (1.0f - iou) + v;
            float alpha = denom > 0.0f ? v / denom : 0.0f;
            float gV = g * (-alpha);
            float gDiff = gV * 2.0f * INV_PI_SQ * diff;
            float gAspectA = gDiff, gAspectB = -gDiff;
            float aDen = aw * aw + ah * ah;
            float bDen = bw * bw + bh * bh;
            if (aDen > 0.0f) {
                float gAw = gAspectA * (ah / aDen);
                float gAh = gAspectA * (-aw / aDen);
                if (awRaw > 0.0f) { *gAx2 += gAw; *gAx1 += -gAw; }
                if (ahRaw > 0.0f) { *gAy2 += gAh; *gAy1 += -gAh; }
            }
            if (bDen > 0.0f) {
                float gBw = gAspectB * (bh / bDen);
                float gBh = gAspectB * (-bw / bDen);
                if (bwRaw > 0.0f) { *gBx2 += gBw; *gBx1 += -gBw; }
                if (bhRaw > 0.0f) { *gBy2 += gBh; *gBy1 += -gBh; }
            }
        }

        float gAcx = gCentreSq * 2.0f * dcx;
        float gAcy = gCentreSq * 2.0f * dcy;
        *gAx1 += gAcx * 0.5f; *gAx2 += gAcx * 0.5f;
        *gAy1 += gAcy * 0.5f; *gAy2 += gAcy * 0.5f;
        *gBx1 += -gAcx * 0.5f; *gBx2 += -gAcx * 0.5f;
        *gBy1 += -gAcy * 0.5f; *gBy2 += -gAcy * 0.5f;

        float gEw = gDiagSq * 2.0f * ew;
        float gEh = gDiagSq * 2.0f * eh;
        float gEx1 = -gEw, gEx2 = gEw;
        float gEy1 = -gEh, gEy2 = gEh;
        if (ax1 <= bx1) *gAx1 += gEx1; else *gBx1 += gEx1;
        if (ay1 <= by1) *gAy1 += gEy1; else *gBy1 += gEy1;
        if (ax2 >= bx2) *gAx2 += gEx2; else *gBx2 += gEx2;
        if (ay2 >= by2) *gAy2 += gEy2; else *gBy2 += gEy2;
    }

    if (u > 0.0f) {
        float uSq = u * u;
        gInter += gIou * (u + inter) / uSq;
        gAreaA += gIou * (-inter) / uSq;
        gAreaB += gIou * (-inter) / uSq;
    }

    float gIw = gInter * ih, gIh = gInter * iw;
    float gIwRaw = iwRaw > 0.0f ? gIw : 0.0f;
    float gIhRaw = ihRaw > 0.0f ? gIh : 0.0f;
    float gIx2 = gIwRaw, gIx1 = -gIwRaw;
    float gIy2 = gIhRaw, gIy1 = -gIhRaw;
    if (ax1 >= bx1) *gAx1 += gIx1; else *gBx1 += gIx1;
    if (ay1 >= by1) *gAy1 += gIy1; else *gBy1 += gIy1;
    if (ax2 <= bx2) *gAx2 += gIx2; else *gBx2 += gIx2;
    if (ay2 <= by2) *gAy2 += gIy2; else *gBy2 += gIy2;

    float gAw2 = gAreaA * ah, gAh2 = gAreaA * aw;
    float gBw2 = gAreaB * bh, gBh2 = gAreaB * bw;
    if (awRaw > 0.0f) { *gAx2 += gAw2; *gAx1 += -gAw2; }
    if (ahRaw > 0.0f) { *gAy2 += gAh2; *gAy1 += -gAh2; }
    if (bwRaw > 0.0f) { *gBx2 += gBw2; *gBx1 += -gBw2; }
    if (bhRaw > 0.0f) { *gBy2 += gBh2; *gBy1 += -gBh2; }
}

extern ""C"" __global__ __launch_bounds__(256) void detection_iou_backward_a(
    const float* __restrict__ gradOutput,
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ gradA,
    int n, int m, int variant)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ax1 = a[i * 4], ay1 = a[i * 4 + 1], ax2 = a[i * 4 + 2], ay2 = a[i * 4 + 3];
    float sumX1 = 0.0f, sumY1 = 0.0f, sumX2 = 0.0f, sumY2 = 0.0f;
    for (int j = 0; j < m; j++) {
        float bx1 = b[j * 4], by1 = b[j * 4 + 1], bx2 = b[j * 4 + 2], by2 = b[j * 4 + 3];
        float g = gradOutput[i * m + j];
        float gAx1, gAy1, gAx2, gAy2, gBx1, gBy1, gBx2, gBy2;
        compute_cell_grads_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, g, variant,
            &gAx1, &gAy1, &gAx2, &gAy2, &gBx1, &gBy1, &gBx2, &gBy2);
        sumX1 += gAx1; sumY1 += gAy1; sumX2 += gAx2; sumY2 += gAy2;
    }
    gradA[i * 4] = sumX1;
    gradA[i * 4 + 1] = sumY1;
    gradA[i * 4 + 2] = sumX2;
    gradA[i * 4 + 3] = sumY2;
}

extern ""C"" __global__ __launch_bounds__(256) void detection_iou_backward_b(
    const float* __restrict__ gradOutput,
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ gradB,
    int n, int m, int variant)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    float bx1 = b[j * 4], by1 = b[j * 4 + 1], bx2 = b[j * 4 + 2], by2 = b[j * 4 + 3];
    float sumX1 = 0.0f, sumY1 = 0.0f, sumX2 = 0.0f, sumY2 = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax1 = a[i * 4], ay1 = a[i * 4 + 1], ax2 = a[i * 4 + 2], ay2 = a[i * 4 + 3];
        float g = gradOutput[i * m + j];
        float gAx1, gAy1, gAx2, gAy2, gBx1, gBy1, gBx2, gBy2;
        compute_cell_grads_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, g, variant,
            &gAx1, &gAy1, &gAx2, &gAy2, &gBx1, &gBy1, &gBx2, &gBy2);
        sumX1 += gBx1; sumY1 += gBy1; sumX2 += gBx2; sumY2 += gBy2;
    }
    gradB[j * 4] = sumX1;
    gradB[j * 4 + 1] = sumY1;
    gradB[j * 4 + 2] = sumX2;
    gradB[j * 4 + 3] = sumY2;
}
";
    }
}
