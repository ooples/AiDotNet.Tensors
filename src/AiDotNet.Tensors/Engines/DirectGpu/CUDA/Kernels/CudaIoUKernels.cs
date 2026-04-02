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
// IoU loss backward: gradients w.r.t. predicted box coordinates
// Uses finite differences for simplicity and correctness
// ===========================================================================

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
    float eps = 1e-4f;
    float go = gradOutput[i];
    // Compute gradient for each of the 4 box coordinates via finite differences
    for (int c = 0; c < 4; c++) {
        float orig = predicted[off + c];
        // f(x + eps)
        float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
        float coords_plus[4] = {px1, py1, px2, py2};
        coords_plus[c] += eps;
        float iou_plus = compute_iou(coords_plus[0],coords_plus[1],coords_plus[2],coords_plus[3],
            target[off],target[off+1],target[off+2],target[off+3], NULL, NULL);
        // f(x - eps)
        float coords_minus[4] = {px1, py1, px2, py2};
        coords_minus[c] -= eps;
        float iou_minus = compute_iou(coords_minus[0],coords_minus[1],coords_minus[2],coords_minus[3],
            target[off],target[off+1],target[off+2],target[off+3], NULL, NULL);
        // loss = 1 - iou, d(loss)/d(coord) = -d(iou)/d(coord)
        float grad = -(iou_plus - iou_minus) / (2.0f * eps);
        gradPredicted[off + c] = go * grad;
    }
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
