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
    float predArea = max_f(0.0f, px2 - px1) * max_f(0.0f, py2 - py1);
    float targArea = max_f(0.0f, tx2 - tx1) * max_f(0.0f, ty2 - ty1);
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

// Analytical GIoU backward
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
    float go = gradOutput[i];
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];

    // IoU gradients (reuse helper)
    float iouGrad[4];
    compute_iou_grad(px1,py1,px2,py2, tx1,ty1,tx2,ty2, 1.0f, iouGrad);

    // Enclosing box
    float encX1=min_f(px1,tx1), encY1=min_f(py1,ty1);
    float encX2=max_f(px2,tx2), encY2=max_f(py2,ty2);
    float encW=encX2-encX1, encH=encY2-encY1;
    float encA=encW*encH+1e-7f;

    // IoU components for penalty term
    float ix1=max_f(px1,tx1), iy1=max_f(py1,ty1), ix2=min_f(px2,tx2), iy2=min_f(py2,ty2);
    float iw=max_f(0.0f,ix2-ix1), ih=max_f(0.0f,iy2-iy1), iA=iw*ih;
    float pw=px2-px1, ph=py2-py1, pA=pw*ph, tA=(tx2-tx1)*(ty2-ty1);
    float uA=pA+tA-iA+1e-7f;
    float hasInter=(iw>0.0f&&ih>0.0f)?1.0f:0.0f;

    // ∂encA/∂coord
    float dEncA[4];
    dEncA[0] = -(px1<tx1?1.0f:0.0f) * encH;  // ∂encA/∂px1
    dEncA[1] = -(py1<ty1?1.0f:0.0f) * encW;  // ∂encA/∂py1
    dEncA[2] = (px2>tx2?1.0f:0.0f) * encH;   // ∂encA/∂px2
    dEncA[3] = (py2>ty2?1.0f:0.0f) * encW;   // ∂encA/∂py2

    // ∂I/∂coord and ∂U/∂coord
    float dI[4], dU[4];
    dI[0] = hasInter * (-(px1>tx1?1.0f:0.0f)) * ih;
    dI[1] = hasInter * (-(py1>ty1?1.0f:0.0f)) * iw;
    dI[2] = hasInter * (px2<tx2?1.0f:0.0f) * ih;
    dI[3] = hasInter * (py2<ty2?1.0f:0.0f) * iw;
    dU[0] = -ph - dI[0]; dU[1] = -pw - dI[1]; dU[2] = ph - dI[2]; dU[3] = pw - dI[3];

    // GIoU = IoU - P where P = (encA - U)/encA
    // loss = 1 - GIoU = 1 - IoU + P
    // ∂loss/∂coord = -∂IoU/∂coord + ∂P/∂coord
    // compute_iou_grad(..., 1.0f, iouGrad) gives iouGrad[c] = -∂IoU/∂coord (go=1 inside helper)
    float encASq = encA * encA;
    for (int c = 0; c < 4; c++) {
        float dP_dc = -(dU[c] * encA - uA * dEncA[c]) / encASq;
        // iouGrad[c] = -∂IoU/∂coord, so ∂(1-IoU)/∂coord = iouGrad[c]
        // ∂loss/∂coord = ∂(1-IoU)/∂coord + ∂P/∂coord = iouGrad[c] + dP_dc
        gradPredicted[off+c] = go * (iouGrad[c] + dP_dc);
    }
}

// Analytical DIoU backward
extern ""C"" __global__ void diou_loss_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ gradPredicted,
    int numBoxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBoxes) return;
    int off = i * 4;
    float go = gradOutput[i];
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];

    // IoU gradients
    float iouGrad[4];
    compute_iou_grad(px1,py1,px2,py2, tx1,ty1,tx2,ty2, 1.0f, iouGrad);

    // Center distance: ρ² = (pcx-tcx)² + (pcy-tcy)²
    float pcx=0.5f*(px1+px2), pcy=0.5f*(py1+py2);
    float tcx=0.5f*(tx1+tx2), tcy=0.5f*(ty1+ty2);
    float dx=pcx-tcx, dy=pcy-tcy;
    float rhoSq=dx*dx+dy*dy;

    // ∂ρ²/∂coord: pcx depends on px1,px2; pcy depends on py1,py2
    float dRho[4];
    dRho[0] = 2.0f*dx*0.5f;        // ∂ρ²/∂px1 = dx (pcx = 0.5*(px1+px2))
    dRho[1] = 2.0f*dy*0.5f;        // ∂ρ²/∂py1 = dy
    dRho[2] = 2.0f*dx*0.5f;        // ∂ρ²/∂px2 = dx
    dRho[3] = 2.0f*dy*0.5f;        // ∂ρ²/∂py2 = dy

    // Enclosing diagonal: c² = encDx² + encDy²
    float encX1=min_f(px1,tx1), encY1=min_f(py1,ty1);
    float encX2=max_f(px2,tx2), encY2=max_f(py2,ty2);
    float encDx=encX2-encX1, encDy=encY2-encY1;
    float cSq=encDx*encDx+encDy*encDy+1e-7f;

    // ∂c²/∂coord
    float dCSq[4];
    dCSq[0] = 2.0f*encDx*(-(px1<tx1?1.0f:0.0f));  // encX1 = min(px1,tx1)
    dCSq[1] = 2.0f*encDy*(-(py1<ty1?1.0f:0.0f));
    dCSq[2] = 2.0f*encDx*(px2>tx2?1.0f:0.0f);      // encX2 = max(px2,tx2)
    dCSq[3] = 2.0f*encDy*(py2>ty2?1.0f:0.0f);

    // DIoU penalty: P = ρ²/c²
    // ∂P/∂coord = (∂ρ²/∂coord * c² - ρ² * ∂c²/∂coord) / c⁴
    float cSqSq = cSq * cSq;
    for (int c = 0; c < 4; c++) {
        float dP = (dRho[c] * cSq - rhoSq * dCSq[c]) / cSqSq;
        // loss = 1 - DIoU = 1 - IoU + P → ∂loss = -∂IoU + ∂P
        // iouGrad[c] = -∂IoU/∂coord = ∂(1-IoU)/∂coord, so ∂loss = iouGrad[c] + dP
        gradPredicted[off+c] = go * (iouGrad[c] + dP);
    }
}

// Analytical CIoU backward
extern ""C"" __global__ void ciou_loss_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ gradPredicted,
    int numBoxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBoxes) return;
    int off = i * 4;
    float go = gradOutput[i];
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];

    // IoU + DIoU components (reuse from DIoU backward)
    float iouGrad[4];
    compute_iou_grad(px1,py1,px2,py2, tx1,ty1,tx2,ty2, 1.0f, iouGrad);

    // Center distance
    float pcx=0.5f*(px1+px2), pcy=0.5f*(py1+py2);
    float tcx=0.5f*(tx1+tx2), tcy=0.5f*(ty1+ty2);
    float dx=pcx-tcx, dy=pcy-tcy, rhoSq=dx*dx+dy*dy;
    float dRho[4] = {dx, dy, dx, dy};

    // Enclosing diagonal
    float encDx=max_f(px2,tx2)-min_f(px1,tx1), encDy=max_f(py2,ty2)-min_f(py1,ty1);
    float cSq=encDx*encDx+encDy*encDy+1e-7f;
    float dCSq[4];
    dCSq[0] = 2.0f*encDx*(-(px1<tx1?1.0f:0.0f));
    dCSq[1] = 2.0f*encDy*(-(py1<ty1?1.0f:0.0f));
    dCSq[2] = 2.0f*encDx*(px2>tx2?1.0f:0.0f);
    dCSq[3] = 2.0f*encDy*(py2>ty2?1.0f:0.0f);

    // Aspect ratio: v = (4/π²) * (atan(tw/th) - atan(pw/ph))²
    float pw=px2-px1+1e-7f, ph=py2-py1+1e-7f;
    float tw=tx2-tx1+1e-7f, th=ty2-ty1+1e-7f;
    float atanDiff = atanf(tw/th) - atanf(pw/ph);
    float fourOverPiSq = 4.0f / (3.14159265f * 3.14159265f);
    float v = fourOverPiSq * atanDiff * atanDiff;

    // IoU value for alpha
    float ix1=max_f(px1,tx1), iy1=max_f(py1,ty1), ix2=min_f(px2,tx2), iy2=min_f(py2,ty2);
    float iw=max_f(0.0f,ix2-ix1), ih=max_f(0.0f,iy2-iy1), iA=iw*ih;
    float pA=pw*ph, tA=tw*th, uA=pA+tA-iA+1e-7f;
    float iou=iA/uA;
    float alpha = v / (1.0f - iou + v + 1e-7f);

    // ∂atan(pw/ph)/∂pw = ph/(pw²+ph²), ∂atan(pw/ph)/∂ph = -pw/(pw²+ph²)
    float ratioSumSq = pw*pw + ph*ph;
    float dAtanPw = ph / ratioSumSq;   // ∂atan(pw/ph)/∂pw
    float dAtanPh = -pw / ratioSumSq;  // ∂atan(pw/ph)/∂ph

    // ∂v/∂coord = 2*fourOverPiSq * atanDiff * (-∂atan(pw/ph)/∂coord)
    // pw=px2-px1 → ∂pw/∂px1=-1, ∂pw/∂px2=1; ph=py2-py1 → ∂ph/∂py1=-1, ∂ph/∂py2=1
    float dV[4];
    dV[0] = 2.0f * fourOverPiSq * atanDiff * (-dAtanPw * (-1.0f));  // ∂v/∂px1 via pw
    dV[1] = 2.0f * fourOverPiSq * atanDiff * (-dAtanPh * (-1.0f));  // ∂v/∂py1 via ph
    dV[2] = 2.0f * fourOverPiSq * atanDiff * (-dAtanPw * 1.0f);     // ∂v/∂px2 via pw
    dV[3] = 2.0f * fourOverPiSq * atanDiff * (-dAtanPh * 1.0f);     // ∂v/∂py2 via ph

    float cSqSq = cSq * cSq;
    for (int c = 0; c < 4; c++) {
        float dDistPenalty = (dRho[c] * cSq - rhoSq * dCSq[c]) / cSqSq;
        float dAspectPenalty = alpha * dV[c];  // alpha detached per CIoU paper
        // loss = 1 - CIoU = 1 - IoU + distPenalty + aspectPenalty
        // iouGrad[c] = -∂IoU/∂coord = ∂(1-IoU)/∂coord
        gradPredicted[off+c] = go * (iouGrad[c] + dDistPenalty + dAspectPenalty);
    }
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
