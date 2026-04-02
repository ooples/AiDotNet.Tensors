// Copyright (c) AiDotNet. All rights reserved.
// CUDA IoU/GIoU/DIoU/CIoU loss kernels for bounding box regression.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for differentiable IoU-family losses on [N,4] XYXY bounding boxes.
/// References: Rezatofighi et al. CVPR 2019 (GIoU), Zheng et al. AAAI 2020 (DIoU/CIoU).
/// </summary>
internal static class CudaIoUKernels
{
    public static string GetSource()
    {
        return @"
#include <math.h>

// ===========================================================================
// Device helpers for box coordinate extraction and IoU computation
// ===========================================================================

__device__ __forceinline__ float max_f(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float min_f(float a, float b) { return a < b ? a : b; }

__device__ __forceinline__ float compute_iou(
    float px1, float py1, float px2, float py2,
    float tx1, float ty1, float tx2, float ty2,
    float* interArea_out, float* unionArea_out)
{
    float interX1 = max_f(px1, tx1);
    float interY1 = max_f(py1, ty1);
    float interX2 = min_f(px2, tx2);
    float interY2 = min_f(py2, ty2);
    float interW = max_f(0.0f, interX2 - interX1);
    float interH = max_f(0.0f, interY2 - interY1);
    float interArea = interW * interH;
    float predArea = (px2 - px1) * (py2 - py1);
    float targArea = (tx2 - tx1) * (ty2 - ty1);
    float unionArea = predArea + targArea - interArea + 1e-7f;
    if (interArea_out) *interArea_out = interArea;
    if (unionArea_out) *unionArea_out = unionArea;
    return interArea / unionArea;
}

// ===========================================================================
// IoU loss: loss[i] = 1 - IoU(predicted[i], target[i])
// ===========================================================================

extern ""C"" __global__ void iou_loss(
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int numBoxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBoxes) return;
    int off = i * 4;
    float iou = compute_iou(
        predicted[off], predicted[off+1], predicted[off+2], predicted[off+3],
        target[off], target[off+1], target[off+2], target[off+3],
        NULL, NULL);
    loss[i] = 1.0f - iou;
}

// ===========================================================================
// GIoU loss: loss[i] = 1 - GIoU = 1 - (IoU - (enclosing - union) / enclosing)
// ===========================================================================

extern ""C"" __global__ void giou_loss(
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int numBoxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBoxes) return;
    int off = i * 4;
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    float interArea, unionArea;
    float iou = compute_iou(px1,py1,px2,py2, tx1,ty1,tx2,ty2, &interArea, &unionArea);
    float encX1 = min_f(px1, tx1), encY1 = min_f(py1, ty1);
    float encX2 = max_f(px2, tx2), encY2 = max_f(py2, ty2);
    float encArea = (encX2 - encX1) * (encY2 - encY1) + 1e-7f;
    float giou = iou - (encArea - unionArea) / encArea;
    loss[i] = 1.0f - giou;
}

// ===========================================================================
// DIoU loss: loss[i] = 1 - (IoU - centerDist² / diagDist²)
// ===========================================================================

extern ""C"" __global__ void diou_loss(
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int numBoxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBoxes) return;
    int off = i * 4;
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    float iou = compute_iou(px1,py1,px2,py2, tx1,ty1,tx2,ty2, NULL, NULL);
    float pcx = 0.5f*(px1+px2), pcy = 0.5f*(py1+py2);
    float tcx = 0.5f*(tx1+tx2), tcy = 0.5f*(ty1+ty2);
    float dx = pcx - tcx, dy = pcy - tcy;
    float centerDistSq = dx*dx + dy*dy;
    float encX1 = min_f(px1,tx1), encY1 = min_f(py1,ty1);
    float encX2 = max_f(px2,tx2), encY2 = max_f(py2,ty2);
    float encDx = encX2 - encX1, encDy = encY2 - encY1;
    float diagSq = encDx*encDx + encDy*encDy + 1e-7f;
    float diou = iou - centerDistSq / diagSq;
    loss[i] = 1.0f - diou;
}

// ===========================================================================
// CIoU loss: loss[i] = 1 - (IoU - centerDist²/diag² - alpha*v)
// ===========================================================================

extern ""C"" __global__ void ciou_loss(
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int numBoxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBoxes) return;
    int off = i * 4;
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    float iou = compute_iou(px1,py1,px2,py2, tx1,ty1,tx2,ty2, NULL, NULL);
    // Center distance
    float pcx = 0.5f*(px1+px2), pcy = 0.5f*(py1+py2);
    float tcx = 0.5f*(tx1+tx2), tcy = 0.5f*(ty1+ty2);
    float centerDistSq = (pcx-tcx)*(pcx-tcx) + (pcy-tcy)*(pcy-tcy);
    // Enclosing diagonal
    float encX1 = min_f(px1,tx1), encY1 = min_f(py1,ty1);
    float encX2 = max_f(px2,tx2), encY2 = max_f(py2,ty2);
    float encDx = encX2-encX1, encDy = encY2-encY1;
    float diagSq = encDx*encDx + encDy*encDy + 1e-7f;
    // Aspect ratio
    float predW = px2-px1+1e-7f, predH = py2-py1+1e-7f;
    float targW = tx2-tx1+1e-7f, targH = ty2-ty1+1e-7f;
    float rDiff = atanf(targW/targH) - atanf(predW/predH);
    float v = (4.0f / (3.14159265f * 3.14159265f)) * rDiff * rDiff;
    float alpha = v / (1.0f - iou + v + 1e-7f);
    float ciou = iou - centerDistSq/diagSq - alpha*v;
    loss[i] = 1.0f - ciou;
}

// ===========================================================================
// Analytical IoU backward: exact gradients w.r.t. predicted box coordinates
// Derived from quotient rule on IoU = I/U with piecewise max/min derivatives
// ===========================================================================

// Device helper: compute analytical IoU gradients for one box
__device__ void compute_iou_grad(
    float px1, float py1, float px2, float py2,
    float tx1, float ty1, float tx2, float ty2,
    float go, float* grad)
{
    // Intersection boundaries
    float ix1 = max_f(px1, tx1), iy1 = max_f(py1, ty1);
    float ix2 = min_f(px2, tx2), iy2 = min_f(py2, ty2);
    float iw = max_f(0.0f, ix2 - ix1), ih = max_f(0.0f, iy2 - iy1);
    float iArea = iw * ih;

    // Predicted area
    float pw = px2 - px1, ph = py2 - py1;
    float pArea = pw * ph;
    float tArea = (tx2 - tx1) * (ty2 - ty1);
    float uArea = pArea + tArea - iArea + 1e-7f;
    float iou = iArea / uArea;

    // Indicators for which box contributes to intersection boundary
    float dix1_dpx1 = (px1 > tx1) ? 1.0f : 0.0f;  // ∂max(px1,tx1)/∂px1
    float diy1_dpy1 = (py1 > ty1) ? 1.0f : 0.0f;
    float dix2_dpx2 = (px2 < tx2) ? 1.0f : 0.0f;  // ∂min(px2,tx2)/∂px2
    float diy2_dpy2 = (py2 < ty2) ? 1.0f : 0.0f;

    // Only non-zero when intersection exists
    float hasInter = (iw > 0.0f && ih > 0.0f) ? 1.0f : 0.0f;

    // ∂I/∂px1 = (hasInter) * (-dix1_dpx1) * ih
    float dI_dpx1 = hasInter * (-dix1_dpx1) * ih;
    float dI_dpy1 = hasInter * (-diy1_dpy1) * iw;
    float dI_dpx2 = hasInter * dix2_dpx2 * ih;
    float dI_dpy2 = hasInter * diy2_dpy2 * iw;

    // ∂pArea/∂px1 = -ph, ∂pArea/∂py1 = -pw, ∂pArea/∂px2 = ph, ∂pArea/∂py2 = pw
    // ∂U/∂coord = ∂pArea/∂coord - ∂I/∂coord
    float dU_dpx1 = -ph - dI_dpx1;
    float dU_dpy1 = -pw - dI_dpy1;
    float dU_dpx2 = ph - dI_dpx2;
    float dU_dpy2 = pw - dI_dpy2;

    // ∂IoU/∂coord = (∂I/∂coord * U - I * ∂U/∂coord) / U²
    float uSq = uArea * uArea;
    float dIoU_dpx1 = (dI_dpx1 * uArea - iArea * dU_dpx1) / uSq;
    float dIoU_dpy1 = (dI_dpy1 * uArea - iArea * dU_dpy1) / uSq;
    float dIoU_dpx2 = (dI_dpx2 * uArea - iArea * dU_dpx2) / uSq;
    float dIoU_dpy2 = (dI_dpy2 * uArea - iArea * dU_dpy2) / uSq;

    // loss = 1 - IoU → ∂loss/∂coord = -∂IoU/∂coord
    grad[0] = go * (-dIoU_dpx1);
    grad[1] = go * (-dIoU_dpy1);
    grad[2] = go * (-dIoU_dpx2);
    grad[3] = go * (-dIoU_dpy2);
}

extern ""C"" __global__ void iou_loss_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ gradPredicted,
    int numBoxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBoxes) return;
    int off = i * 4;
    float grad[4];
    compute_iou_grad(
        predicted[off], predicted[off+1], predicted[off+2], predicted[off+3],
        target[off], target[off+1], target[off+2], target[off+3],
        gradOutput[i], grad);
    gradPredicted[off] = grad[0];
    gradPredicted[off+1] = grad[1];
    gradPredicted[off+2] = grad[2];
    gradPredicted[off+3] = grad[3];
}

// GIoU/DIoU/CIoU backward use same finite-difference pattern
extern ""C"" __global__ void giou_loss_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ gradPredicted,
    int numBoxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBoxes) return;
    int off = i * 4;
    float eps = 1e-4f;
    float go = gradOutput[i];
    for (int c = 0; c < 4; c++) {
        float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
        float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
        // GIoU at x+eps
        float cp[4] = {px1,py1,px2,py2}; cp[c] += eps;
        float interA, unionA;
        float iou_p = compute_iou(cp[0],cp[1],cp[2],cp[3], tx1,ty1,tx2,ty2, &interA, &unionA);
        float eX1=min_f(cp[0],tx1), eY1=min_f(cp[1],ty1), eX2=max_f(cp[2],tx2), eY2=max_f(cp[3],ty2);
        float eA_p = (eX2-eX1)*(eY2-eY1)+1e-7f;
        float giou_p = iou_p - (eA_p - unionA)/eA_p;
        // GIoU at x-eps
        float cm[4] = {px1,py1,px2,py2}; cm[c] -= eps;
        iou_p = compute_iou(cm[0],cm[1],cm[2],cm[3], tx1,ty1,tx2,ty2, &interA, &unionA);
        eX1=min_f(cm[0],tx1); eY1=min_f(cm[1],ty1); eX2=max_f(cm[2],tx2); eY2=max_f(cm[3],ty2);
        float eA_m = (eX2-eX1)*(eY2-eY1)+1e-7f;
        float giou_m = iou_p - (eA_m - unionA)/eA_m;
        gradPredicted[off + c] = go * -(giou_p - giou_m) / (2.0f * eps);
    }
}

extern ""C"" __global__ void diou_loss_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ gradPredicted,
    int numBoxes)
{
    // DIoU backward via finite differences (same pattern)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBoxes) return;
    int off = i * 4;
    float eps = 1e-4f;
    float go = gradOutput[i];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    for (int c = 0; c < 4; c++) {
        float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
        // DIoU at x+eps
        float cp[4] = {px1,py1,px2,py2}; cp[c] += eps;
        float iou_p = compute_iou(cp[0],cp[1],cp[2],cp[3], tx1,ty1,tx2,ty2, NULL, NULL);
        float pcx_p=0.5f*(cp[0]+cp[2]), pcy_p=0.5f*(cp[1]+cp[3]);
        float tcx=0.5f*(tx1+tx2), tcy=0.5f*(ty1+ty2);
        float cds_p = (pcx_p-tcx)*(pcx_p-tcx) + (pcy_p-tcy)*(pcy_p-tcy);
        float eX1=min_f(cp[0],tx1), eY1=min_f(cp[1],ty1), eX2=max_f(cp[2],tx2), eY2=max_f(cp[3],ty2);
        float ds_p = (eX2-eX1)*(eX2-eX1) + (eY2-eY1)*(eY2-eY1) + 1e-7f;
        float diou_p = iou_p - cds_p/ds_p;
        // DIoU at x-eps
        float cm[4] = {px1,py1,px2,py2}; cm[c] -= eps;
        float iou_m = compute_iou(cm[0],cm[1],cm[2],cm[3], tx1,ty1,tx2,ty2, NULL, NULL);
        float pcx_m=0.5f*(cm[0]+cm[2]), pcy_m=0.5f*(cm[1]+cm[3]);
        float cds_m = (pcx_m-tcx)*(pcx_m-tcx) + (pcy_m-tcy)*(pcy_m-tcy);
        eX1=min_f(cm[0],tx1); eY1=min_f(cm[1],ty1); eX2=max_f(cm[2],tx2); eY2=max_f(cm[3],ty2);
        float ds_m = (eX2-eX1)*(eX2-eX1) + (eY2-eY1)*(eY2-eY1) + 1e-7f;
        float diou_m = iou_m - cds_m/ds_m;
        gradPredicted[off + c] = go * -(diou_p - diou_m) / (2.0f * eps);
    }
}

extern ""C"" __global__ void ciou_loss_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ gradPredicted,
    int numBoxes)
{
    // CIoU backward via finite differences
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBoxes) return;
    int off = i * 4;
    float eps = 1e-4f;
    float go = gradOutput[i];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];

    // Helper: compute CIoU given box coordinates
    #define COMPUTE_CIOU(bx1,by1,bx2,by2) ({ \
        float _iou = compute_iou(bx1,by1,bx2,by2, tx1,ty1,tx2,ty2, NULL,NULL); \
        float _pcx=0.5f*(bx1+bx2), _pcy=0.5f*(by1+by2); \
        float _tcx=0.5f*(tx1+tx2), _tcy=0.5f*(ty1+ty2); \
        float _cds=(_pcx-_tcx)*(_pcx-_tcx)+(_pcy-_tcy)*(_pcy-_tcy); \
        float _eX1=min_f(bx1,tx1),_eY1=min_f(by1,ty1),_eX2=max_f(bx2,tx2),_eY2=max_f(by2,ty2); \
        float _ds=(_eX2-_eX1)*(_eX2-_eX1)+(_eY2-_eY1)*(_eY2-_eY1)+1e-7f; \
        float _pw=bx2-bx1+1e-7f,_ph=by2-by1+1e-7f,_tw=tx2-tx1+1e-7f,_th=ty2-ty1+1e-7f; \
        float _rd=atanf(_tw/_th)-atanf(_pw/_ph); \
        float _v=(4.0f/(3.14159265f*3.14159265f))*_rd*_rd; \
        float _alpha=_v/(1.0f-_iou+_v+1e-7f); \
        _iou - _cds/_ds - _alpha*_v; \
    })

    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    for (int c = 0; c < 4; c++) {
        float cp[4] = {px1,py1,px2,py2}; cp[c] += eps;
        float ciou_p = COMPUTE_CIOU(cp[0],cp[1],cp[2],cp[3]);
        float cm[4] = {px1,py1,px2,py2}; cm[c] -= eps;
        float ciou_m = COMPUTE_CIOU(cm[0],cm[1],cm[2],cm[3]);
        gradPredicted[off + c] = go * -(ciou_p - ciou_m) / (2.0f * eps);
    }
    #undef COMPUTE_CIOU
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "iou_loss",
            "giou_loss",
            "diou_loss",
            "ciou_loss",
            "iou_loss_backward",
            "giou_loss_backward",
            "diou_loss_backward",
            "ciou_loss_backward",
        };
    }
}
