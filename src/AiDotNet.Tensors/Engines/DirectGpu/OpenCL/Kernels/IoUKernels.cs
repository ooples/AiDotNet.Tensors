// Copyright (c) AiDotNet. All rights reserved.
// OpenCL IoU/GIoU/DIoU/CIoU loss kernels for bounding box regression.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class IoUKernels
{
    public static string GetSource()
    {
        return @"
inline float compute_iou_ocl(
    float px1, float py1, float px2, float py2,
    float tx1, float ty1, float tx2, float ty2,
    float* interArea_out, float* unionArea_out)
{
    float interX1 = max(px1, tx1), interY1 = max(py1, ty1);
    float interX2 = min(px2, tx2), interY2 = min(py2, ty2);
    float interW = max(0.0f, interX2 - interX1);
    float interH = max(0.0f, interY2 - interY1);
    float interArea = interW * interH;
    float predArea = (px2 - px1) * (py2 - py1);
    float targArea = (tx2 - tx1) * (ty2 - ty1);
    float unionArea = predArea + targArea - interArea + 1e-7f;
    if (interArea_out) *interArea_out = interArea;
    if (unionArea_out) *unionArea_out = unionArea;
    return interArea / unionArea;
}

__kernel void iou_loss(
    __global const float* predicted, __global const float* target,
    __global float* loss, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float iou = compute_iou_ocl(
        predicted[off], predicted[off+1], predicted[off+2], predicted[off+3],
        target[off], target[off+1], target[off+2], target[off+3], 0, 0);
    loss[i] = 1.0f - iou;
}

__kernel void giou_loss(
    __global const float* predicted, __global const float* target,
    __global float* loss, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    float interArea, unionArea;
    float iou = compute_iou_ocl(px1,py1,px2,py2, tx1,ty1,tx2,ty2, &interArea, &unionArea);
    float encX1=min(px1,tx1), encY1=min(py1,ty1), encX2=max(px2,tx2), encY2=max(py2,ty2);
    float encArea = (encX2-encX1)*(encY2-encY1) + 1e-7f;
    float giou = iou - (encArea - unionArea) / encArea;
    loss[i] = 1.0f - giou;
}

__kernel void diou_loss(
    __global const float* predicted, __global const float* target,
    __global float* loss, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    float iou = compute_iou_ocl(px1,py1,px2,py2, tx1,ty1,tx2,ty2, 0, 0);
    float pcx=0.5f*(px1+px2), pcy=0.5f*(py1+py2);
    float tcx=0.5f*(tx1+tx2), tcy=0.5f*(ty1+ty2);
    float dx=pcx-tcx, dy=pcy-tcy;
    float centerDistSq = dx*dx + dy*dy;
    float encX1=min(px1,tx1), encY1=min(py1,ty1), encX2=max(px2,tx2), encY2=max(py2,ty2);
    float encDx=encX2-encX1, encDy=encY2-encY1;
    float diagSq = encDx*encDx + encDy*encDy + 1e-7f;
    loss[i] = 1.0f - (iou - centerDistSq / diagSq);
}

__kernel void ciou_loss(
    __global const float* predicted, __global const float* target,
    __global float* loss, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    float iou = compute_iou_ocl(px1,py1,px2,py2, tx1,ty1,tx2,ty2, 0, 0);
    float pcx=0.5f*(px1+px2), pcy=0.5f*(py1+py2);
    float tcx=0.5f*(tx1+tx2), tcy=0.5f*(ty1+ty2);
    float centerDistSq = (pcx-tcx)*(pcx-tcx) + (pcy-tcy)*(pcy-tcy);
    float encX1=min(px1,tx1), encY1=min(py1,ty1), encX2=max(px2,tx2), encY2=max(py2,ty2);
    float encDx=encX2-encX1, encDy=encY2-encY1;
    float diagSq = encDx*encDx + encDy*encDy + 1e-7f;
    float predW=px2-px1+1e-7f, predH=py2-py1+1e-7f;
    float targW=tx2-tx1+1e-7f, targH=ty2-ty1+1e-7f;
    float rDiff = atan(targW/targH) - atan(predW/predH);
    float v = (4.0f / (3.14159265f * 3.14159265f)) * rDiff * rDiff;
    float alpha = v / (1.0f - iou + v + 1e-7f);
    loss[i] = 1.0f - (iou - centerDistSq/diagSq - alpha*v);
}

// IoU backward via finite differences
__kernel void iou_loss_backward(
    __global const float* gradOutput, __global const float* predicted,
    __global const float* target, __global float* gradPredicted, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float eps = 1e-4f;
    float go = gradOutput[i];
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    for (int c = 0; c < 4; c++) {
        float cp[4] = {px1, py1, px2, py2};
        cp[c] += eps;
        float iou_p = compute_iou_ocl(cp[0],cp[1],cp[2],cp[3], tx1,ty1,tx2,ty2, 0, 0);
        float cm[4] = {px1, py1, px2, py2};
        cm[c] -= eps;
        float iou_m = compute_iou_ocl(cm[0],cm[1],cm[2],cm[3], tx1,ty1,tx2,ty2, 0, 0);
        gradPredicted[off + c] = go * -(iou_p - iou_m) / (2.0f * eps);
    }
}

__kernel void giou_loss_backward(
    __global const float* gradOutput, __global const float* predicted,
    __global const float* target, __global float* gradPredicted, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float eps = 1e-4f;
    float go = gradOutput[i];
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    for (int c = 0; c < 4; c++) {
        float cp[4] = {px1,py1,px2,py2}; cp[c] += eps;
        float interA, unionA;
        float iou_p = compute_iou_ocl(cp[0],cp[1],cp[2],cp[3], tx1,ty1,tx2,ty2, &interA, &unionA);
        float eX1=min(cp[0],tx1), eY1=min(cp[1],ty1), eX2=max(cp[2],tx2), eY2=max(cp[3],ty2);
        float eA_p = (eX2-eX1)*(eY2-eY1)+1e-7f;
        float giou_p = iou_p - (eA_p - unionA)/eA_p;
        float cm[4] = {px1,py1,px2,py2}; cm[c] -= eps;
        float iou_m = compute_iou_ocl(cm[0],cm[1],cm[2],cm[3], tx1,ty1,tx2,ty2, &interA, &unionA);
        eX1=min(cm[0],tx1); eY1=min(cm[1],ty1); eX2=max(cm[2],tx2); eY2=max(cm[3],ty2);
        float eA_m = (eX2-eX1)*(eY2-eY1)+1e-7f;
        float giou_m = iou_m - (eA_m - unionA)/eA_m;
        gradPredicted[off + c] = go * -(giou_p - giou_m) / (2.0f * eps);
    }
}

__kernel void diou_loss_backward(
    __global const float* gradOutput, __global const float* predicted,
    __global const float* target, __global float* gradPredicted, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float eps = 1e-4f;
    float go = gradOutput[i];
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    for (int c = 0; c < 4; c++) {
        float cp[4] = {px1,py1,px2,py2}; cp[c] += eps;
        float iou_p = compute_iou_ocl(cp[0],cp[1],cp[2],cp[3], tx1,ty1,tx2,ty2, 0, 0);
        float pcx_p=0.5f*(cp[0]+cp[2]), pcy_p=0.5f*(cp[1]+cp[3]);
        float tcx=0.5f*(tx1+tx2), tcy=0.5f*(ty1+ty2);
        float cds_p=(pcx_p-tcx)*(pcx_p-tcx)+(pcy_p-tcy)*(pcy_p-tcy);
        float eX1=min(cp[0],tx1),eY1=min(cp[1],ty1),eX2=max(cp[2],tx2),eY2=max(cp[3],ty2);
        float ds_p=(eX2-eX1)*(eX2-eX1)+(eY2-eY1)*(eY2-eY1)+1e-7f;
        float diou_p = iou_p - cds_p/ds_p;
        float cm[4] = {px1,py1,px2,py2}; cm[c] -= eps;
        float iou_m = compute_iou_ocl(cm[0],cm[1],cm[2],cm[3], tx1,ty1,tx2,ty2, 0, 0);
        float pcx_m=0.5f*(cm[0]+cm[2]), pcy_m=0.5f*(cm[1]+cm[3]);
        float cds_m=(pcx_m-tcx)*(pcx_m-tcx)+(pcy_m-tcy)*(pcy_m-tcy);
        eX1=min(cm[0],tx1);eY1=min(cm[1],ty1);eX2=max(cm[2],tx2);eY2=max(cm[3],ty2);
        float ds_m=(eX2-eX1)*(eX2-eX1)+(eY2-eY1)*(eY2-eY1)+1e-7f;
        float diou_m = iou_m - cds_m/ds_m;
        gradPredicted[off + c] = go * -(diou_p - diou_m) / (2.0f * eps);
    }
}

__kernel void ciou_loss_backward(
    __global const float* gradOutput, __global const float* predicted,
    __global const float* target, __global float* gradPredicted, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float eps = 1e-4f;
    float go = gradOutput[i];
    float px1=predicted[off], py1=predicted[off+1], px2=predicted[off+2], py2=predicted[off+3];
    float tx1=target[off], ty1=target[off+1], tx2=target[off+2], ty2=target[off+3];
    for (int c = 0; c < 4; c++) {
        float cp[4] = {px1,py1,px2,py2}; cp[c] += eps;
        float iou_p = compute_iou_ocl(cp[0],cp[1],cp[2],cp[3], tx1,ty1,tx2,ty2, 0, 0);
        float pcx_p=0.5f*(cp[0]+cp[2]),pcy_p=0.5f*(cp[1]+cp[3]);
        float tcx=0.5f*(tx1+tx2),tcy=0.5f*(ty1+ty2);
        float cds_p=(pcx_p-tcx)*(pcx_p-tcx)+(pcy_p-tcy)*(pcy_p-tcy);
        float eX1=min(cp[0],tx1),eY1=min(cp[1],ty1),eX2=max(cp[2],tx2),eY2=max(cp[3],ty2);
        float ds_p=(eX2-eX1)*(eX2-eX1)+(eY2-eY1)*(eY2-eY1)+1e-7f;
        float pw=cp[2]-cp[0]+1e-7f,ph=cp[3]-cp[1]+1e-7f;
        float tw=tx2-tx1+1e-7f,th=ty2-ty1+1e-7f;
        float rd_p=atan(tw/th)-atan(pw/ph);
        float v_p=(4.0f/(3.14159265f*3.14159265f))*rd_p*rd_p;
        float alpha_p=v_p/(1.0f-iou_p+v_p+1e-7f);
        float ciou_p = iou_p - cds_p/ds_p - alpha_p*v_p;

        float cm[4] = {px1,py1,px2,py2}; cm[c] -= eps;
        float iou_m = compute_iou_ocl(cm[0],cm[1],cm[2],cm[3], tx1,ty1,tx2,ty2, 0, 0);
        float pcx_m=0.5f*(cm[0]+cm[2]),pcy_m=0.5f*(cm[1]+cm[3]);
        float cds_m=(pcx_m-tcx)*(pcx_m-tcx)+(pcy_m-tcy)*(pcy_m-tcy);
        eX1=min(cm[0],tx1);eY1=min(cm[1],ty1);eX2=max(cm[2],tx2);eY2=max(cm[3],ty2);
        float ds_m=(eX2-eX1)*(eX2-eX1)+(eY2-eY1)*(eY2-eY1)+1e-7f;
        pw=cm[2]-cm[0]+1e-7f;ph=cm[3]-cm[1]+1e-7f;
        float rd_m=atan(tw/th)-atan(pw/ph);
        float v_m=(4.0f/(3.14159265f*3.14159265f))*rd_m*rd_m;
        float alpha_m=v_m/(1.0f-iou_m+v_m+1e-7f);
        float ciou_m = iou_m - cds_m/ds_m - alpha_m*v_m;

        gradPredicted[off + c] = go * -(ciou_p - ciou_m) / (2.0f * eps);
    }
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "iou_loss", "giou_loss", "diou_loss", "ciou_loss",
            "iou_loss_backward", "giou_loss_backward", "diou_loss_backward", "ciou_loss_backward",
        };
    }
}
