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
    float predArea = max(0.0f, px2 - px1) * max(0.0f, py2 - py1);
    float targArea = max(0.0f, tx2 - tx1) * max(0.0f, ty2 - ty1);
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
    float predW=max(px2-px1, 1e-7f), predH=max(py2-py1, 1e-7f);
    float targW=max(tx2-tx1, 1e-7f), targH=max(ty2-ty1, 1e-7f);
    float rDiff = atan(targW/targH) - atan(predW/predH);
    float v = (4.0f / (3.14159265f * 3.14159265f)) * rDiff * rDiff;
    float alpha = v / (1.0f - iou + v + 1e-7f);
    loss[i] = 1.0f - (iou - centerDistSq/diagSq - alpha*v);
}

// Analytical IoU backward
inline void compute_iou_grad_ocl(float px1,float py1,float px2,float py2, float tx1,float ty1,float tx2,float ty2, float go, float* grad) {
    float ix1=max(px1,tx1),iy1=max(py1,ty1),ix2=min(px2,tx2),iy2=min(py2,ty2);
    float iw=max(0.0f,ix2-ix1),ih=max(0.0f,iy2-iy1),iA=iw*ih;
    float pw=max(0.0f,px2-px1),ph=max(0.0f,py2-py1),pA=pw*ph,tA=max(0.0f,tx2-tx1)*max(0.0f,ty2-ty1);
    float uA=pA+tA-iA+1e-7f; float hi=(iw>0.0f&&ih>0.0f)?1.0f:0.0f;
    float dI0=hi*(-(px1>tx1?1.0f:0.0f))*ih, dI1=hi*(-(py1>ty1?1.0f:0.0f))*iw;
    float dI2=hi*(px2<tx2?1.0f:0.0f)*ih, dI3=hi*(py2<ty2?1.0f:0.0f)*iw;
    float dU0=-ph-dI0, dU1=-pw-dI1, dU2=ph-dI2, dU3=pw-dI3;
    float uSq=uA*uA;
    grad[0]=go*(-(dI0*uA-iA*dU0)/uSq); grad[1]=go*(-(dI1*uA-iA*dU1)/uSq);
    grad[2]=go*(-(dI2*uA-iA*dU2)/uSq); grad[3]=go*(-(dI3*uA-iA*dU3)/uSq);
}

__kernel void iou_loss_backward(
    __global const float* gradOutput, __global const float* predicted,
    __global const float* target, __global float* gradPredicted, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float grad[4];
    compute_iou_grad_ocl(predicted[off],predicted[off+1],predicted[off+2],predicted[off+3],
        target[off],target[off+1],target[off+2],target[off+3], gradOutput[i], grad);
    gradPredicted[off]=grad[0]; gradPredicted[off+1]=grad[1];
    gradPredicted[off+2]=grad[2]; gradPredicted[off+3]=grad[3];
}

// Analytical GIoU backward
__kernel void giou_loss_backward(
    __global const float* gradOutput, __global const float* predicted,
    __global const float* target, __global float* gradPredicted, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float go = gradOutput[i];
    float px1=predicted[off],py1=predicted[off+1],px2=predicted[off+2],py2=predicted[off+3];
    float tx1=target[off],ty1=target[off+1],tx2=target[off+2],ty2=target[off+3];
    float iouGrad[4]; compute_iou_grad_ocl(px1,py1,px2,py2,tx1,ty1,tx2,ty2,1.0f,iouGrad);
    float encW=max(px2,tx2)-min(px1,tx1), encH=max(py2,ty2)-min(py1,ty1), encA=encW*encH+1e-7f;
    float ix1=max(px1,tx1),iy1=max(py1,ty1),ix2=min(px2,tx2),iy2=min(py2,ty2);
    float iw=max(0.0f,ix2-ix1),ih=max(0.0f,iy2-iy1),iA=iw*ih;
    float pw=px2-px1,ph=py2-py1,pA=pw*ph,tA=(tx2-tx1)*(ty2-ty1),uA=pA+tA-iA+1e-7f;
    float hi=(iw>0.0f&&ih>0.0f)?1.0f:0.0f;
    float dI[4]={hi*(-(px1>tx1?1.0f:0.0f))*ih, hi*(-(py1>ty1?1.0f:0.0f))*iw, hi*(px2<tx2?1.0f:0.0f)*ih, hi*(py2<ty2?1.0f:0.0f)*iw};
    float dU[4]={-ph-dI[0],-pw-dI[1],ph-dI[2],pw-dI[3]};
    float dEncA[4]={-(px1<tx1?1.0f:0.0f)*encH, -(py1<ty1?1.0f:0.0f)*encW, (px2>tx2?1.0f:0.0f)*encH, (py2>ty2?1.0f:0.0f)*encW};
    float encASq=encA*encA;
    for(int c=0;c<4;c++) {
        float dIoU=-iouGrad[c];
        float dP=-(dU[c]*encA-uA*dEncA[c])/encASq;
        gradPredicted[off+c]=go*(-dIoU/go+dP);  // Simplified: loss=1-IoU+P
    }
}

// Analytical DIoU backward
__kernel void diou_loss_backward(
    __global const float* gradOutput, __global const float* predicted,
    __global const float* target, __global float* gradPredicted, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float go = gradOutput[i];
    float px1=predicted[off],py1=predicted[off+1],px2=predicted[off+2],py2=predicted[off+3];
    float tx1=target[off],ty1=target[off+1],tx2=target[off+2],ty2=target[off+3];
    float iouGrad[4]; compute_iou_grad_ocl(px1,py1,px2,py2,tx1,ty1,tx2,ty2,1.0f,iouGrad);
    float pcx=0.5f*(px1+px2),pcy=0.5f*(py1+py2),tcx=0.5f*(tx1+tx2),tcy=0.5f*(ty1+ty2);
    float dx=pcx-tcx,dy=pcy-tcy,rhoSq=dx*dx+dy*dy;
    float dRho[4]={dx,dy,dx,dy};
    float encDx=max(px2,tx2)-min(px1,tx1),encDy=max(py2,ty2)-min(py1,ty1);
    float cSq=encDx*encDx+encDy*encDy+1e-7f,cSqSq=cSq*cSq;
    float dCSq[4]={2.0f*encDx*(-(px1<tx1?1.0f:0.0f)), 2.0f*encDy*(-(py1<ty1?1.0f:0.0f)), 2.0f*encDx*(px2>tx2?1.0f:0.0f), 2.0f*encDy*(py2>ty2?1.0f:0.0f)};
    for(int c=0;c<4;c++) {
        float dIoU=-iouGrad[c];
        float dP=(dRho[c]*cSq-rhoSq*dCSq[c])/cSqSq;
        gradPredicted[off+c]=go*(dIoU+dP);
    }
}

// Analytical CIoU backward
__kernel void ciou_loss_backward(
    __global const float* gradOutput, __global const float* predicted,
    __global const float* target, __global float* gradPredicted, int numBoxes)
{
    int i = get_global_id(0);
    if (i >= numBoxes) return;
    int off = i * 4;
    float go = gradOutput[i];
    float px1=predicted[off],py1=predicted[off+1],px2=predicted[off+2],py2=predicted[off+3];
    float tx1=target[off],ty1=target[off+1],tx2=target[off+2],ty2=target[off+3];
    float iouGrad[4]; compute_iou_grad_ocl(px1,py1,px2,py2,tx1,ty1,tx2,ty2,1.0f,iouGrad);
    float pcx=0.5f*(px1+px2),pcy=0.5f*(py1+py2),tcx=0.5f*(tx1+tx2),tcy=0.5f*(ty1+ty2);
    float dx=pcx-tcx,dy=pcy-tcy,rhoSq=dx*dx+dy*dy;
    float dRho[4]={dx,dy,dx,dy};
    float encDx=max(px2,tx2)-min(px1,tx1),encDy=max(py2,ty2)-min(py1,ty1);
    float cSq=encDx*encDx+encDy*encDy+1e-7f,cSqSq=cSq*cSq;
    float dCSq[4]={2.0f*encDx*(-(px1<tx1?1.0f:0.0f)), 2.0f*encDy*(-(py1<ty1?1.0f:0.0f)), 2.0f*encDx*(px2>tx2?1.0f:0.0f), 2.0f*encDy*(py2>ty2?1.0f:0.0f)};
    float pw=px2-px1+1e-7f,ph=py2-py1+1e-7f,tw=tx2-tx1+1e-7f,th=ty2-ty1+1e-7f;
    float atanDiff=atan(tw/th)-atan(pw/ph);
    float fourOverPiSq=4.0f/(3.14159265f*3.14159265f);
    float v=fourOverPiSq*atanDiff*atanDiff;
    float ix1=max(px1,tx1),iy1=max(py1,ty1),ix2=min(px2,tx2),iy2=min(py2,ty2);
    float iw=max(0.0f,ix2-ix1),ih=max(0.0f,iy2-iy1),iA=iw*ih;
    float pA=pw*ph,tA=tw*th,uA=pA+tA-iA+1e-7f,iou=iA/uA;
    float alpha=v/(1.0f-iou+v+1e-7f);
    float ratioSumSq=pw*pw+ph*ph;
    float dAtanPw=ph/ratioSumSq, dAtanPh=-pw/ratioSumSq;
    float dV[4]={2.0f*fourOverPiSq*atanDiff*dAtanPw, 2.0f*fourOverPiSq*atanDiff*dAtanPh, 2.0f*fourOverPiSq*atanDiff*(-dAtanPw), 2.0f*fourOverPiSq*atanDiff*(-dAtanPh)};
    for(int c=0;c<4;c++) {
        float dIoU=-iouGrad[c];
        float dDistP=(dRho[c]*cSq-rhoSq*dCSq[c])/cSqSq;
        float dAspectP=alpha*dV[c];
        gradPredicted[off+c]=go*(dIoU+dDistP+dAspectP);
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
