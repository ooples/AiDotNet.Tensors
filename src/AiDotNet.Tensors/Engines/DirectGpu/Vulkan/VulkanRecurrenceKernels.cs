// Copyright (c) AiDotNet. All rights reserved.
// Vulkan GLSL compute shaders for the fused recurrence / LM-head ops (#1464), compiled to SPIR-V
// at runtime (libshaderc). Real GPU kernels — mirror the HIP/CUDA/OpenCL/MSL kernels and the
// CpuEngine math exactly. Int params are passed as push constants. Forward only — the
// differentiable backward runs through the CpuEngine tape (engine override is forward-only).

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static class VulkanRecurrenceKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    public static string GlaScan => Header + @"
layout(set=0,binding=0) readonly buffer Qb { float Q[]; };
layout(set=0,binding=1) readonly buffer Kb { float K[]; };
layout(set=0,binding=2) readonly buffer Vb { float Vd[]; };
layout(set=0,binding=3) readonly buffer Gb { float G[]; };
layout(set=0,binding=4) writeonly buffer Ob { float outp[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint modelDim; uint numHeads; uint headDim; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int B=int(batch), SL=int(seqLen), MD=int(modelDim), NH=int(numHeads), HD=int(headDim);
    if (gid >= B*NH*HD) return;
    int di = gid % HD; int h = (gid / HD) % NH; int b = gid / (HD*NH); int hOff = h*HD;
    float Srow[256];
    for (int ki=0; ki<HD; ki++) Srow[ki]=0.0;
    for (int t=0; t<SL; t++) {
        int baseOff = (b*SL+t)*MD + hOff;
        float g = G[(b*SL+t)*NH + h];
        float vd = Vd[baseOff+di];
        float o = 0.0;
        for (int ki=0; ki<HD; ki++) { float s=g*Srow[ki]+vd*K[baseOff+ki]; Srow[ki]=s; o+=s*Q[baseOff+ki]; }
        outp[baseOff+di] = o;
    }
}";

    public static string XLstmScan => Header + @"
layout(set=0,binding=0) readonly buffer Qb { float Q[]; };
layout(set=0,binding=1) readonly buffer Kb { float K[]; };
layout(set=0,binding=2) readonly buffer Vb { float Vd[]; };
layout(set=0,binding=3) readonly buffer Ib { float Ig[]; };
layout(set=0,binding=4) readonly buffer Fb { float Fg[]; };
layout(set=0,binding=5) readonly buffer Ob2 { float Og[]; };
layout(set=0,binding=6) writeonly buffer Ob { float outp[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint modelDim; uint numHeads; uint headDim; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int B=int(batch), SL=int(seqLen), MD=int(modelDim), NH=int(numHeads), HD=int(headDim);
    if (gid >= B*NH) return;
    int h = gid % NH; int b = gid / NH; int hOff = h*HD; int hh = HD*HD;
    float kappa = 1.0/sqrt(float(HD));
    float C[4096]; float n[64];
    for (int i=0;i<hh;i++) C[i]=0.0;
    for (int i=0;i<HD;i++) n[i]=0.0;
    for (int t=0;t<SL;t++) {
        int baseOff=(b*SL+t)*MD+hOff; int gOff=(b*SL+t)*NH+h;
        float iv=Ig[gOff]; if (iv>4.85e8) iv=4.85e8;
        float f=Fg[gOff], o=Og[gOff];
        for (int di=0;di<HD;di++) {
            n[di]=f*n[di]+iv*(K[baseOff+di]*kappa);
            float vv=Vd[baseOff+di]; int srow=di*HD;
            for (int ki=0;ki<HD;ki++) C[srow+ki]=f*C[srow+ki]+iv*vv*(K[baseOff+ki]*kappa);
        }
        float nq=0.0; for (int j=0;j<HD;j++) nq+=n[j]*Q[baseOff+j];
        float nf=abs(nq); if (nf<1.0) nf=1.0;
        for (int di=0;di<HD;di++) {
            int srow=di*HD; float num=0.0;
            for (int ki=0;ki<HD;ki++) num+=C[srow+ki]*Q[baseOff+ki];
            outp[baseOff+di]=o*num/nf;
        }
    }
}";

    public static string GatedDeltaScan => Header + @"
layout(set=0,binding=0) readonly buffer Qb { float Q[]; };
layout(set=0,binding=1) readonly buffer Kb { float K[]; };
layout(set=0,binding=2) readonly buffer Vb { float Vd[]; };
layout(set=0,binding=3) readonly buffer Ab { float A[]; };
layout(set=0,binding=4) readonly buffer Bb { float Bt[]; };
layout(set=0,binding=5) writeonly buffer Ob { float outp[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint modelDim; uint numHeads; uint headDim; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int B=int(batch), SL=int(seqLen), MD=int(modelDim), NH=int(numHeads), HD=int(headDim);
    if (gid >= B*NH*HD) return;
    int di = gid % HD; int h = (gid / HD) % NH; int b = gid / (HD*NH); int hOff = h*HD;
    float kappa = 1.0/sqrt(float(HD));
    float Srow[256];
    for (int ki=0;ki<HD;ki++) Srow[ki]=0.0;
    for (int t=0;t<SL;t++) {
        int baseOff=(b*SL+t)*MD+hOff; int gIdx=(b*SL+t)*NH+h;
        float a=A[gIdx], bet=Bt[gIdx];
        float sK=0.0; for (int ki=0;ki<HD;ki++) sK+=Srow[ki]*(K[baseOff+ki]*kappa);
        float bd=bet*(Vd[baseOff+di]-sK);
        for (int ki=0;ki<HD;ki++) Srow[ki]=a*Srow[ki]+bd*(K[baseOff+ki]*kappa);
        float o=0.0; for (int ki=0;ki<HD;ki++) o+=Srow[ki]*Q[baseOff+ki];
        outp[baseOff+di]=o;
    }
}";

    public static string RgLruScan => Header + @"
layout(set=0,binding=0) readonly buffer Vb { float Vd[]; };
layout(set=0,binding=1) readonly buffer Rb { float R[]; };
layout(set=0,binding=2) readonly buffer Ib { float Ig[]; };
layout(set=0,binding=3) readonly buffer Db { float decayb[]; };
layout(set=0,binding=4) writeonly buffer Ob { float outp[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint recDim; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int B=int(batch), SL=int(seqLen), RD=int(recDim);
    if (gid >= B*RD) return;
    int c = gid % RD; int b = gid / RD;
    float base = 1.0/(1.0+exp(decayb[c]));
    float h = 0.0;
    for (int t=0;t<SL;t++) {
        int off=(b*SL+t)*RD+c;
        float a=R[off]*base; float om=1.0-a*a; float s = om>0.0 ? sqrt(om) : 0.0;
        h = a*h + s*(Ig[off]*Vd[off]); outp[off]=h;
    }
}";

    public static string Rwkv4Wkv => Header + @"
layout(set=0,binding=0) readonly buffer Rb { float R[]; };
layout(set=0,binding=1) readonly buffer Kb { float K[]; };
layout(set=0,binding=2) readonly buffer Vb { float Vd[]; };
layout(set=0,binding=3) readonly buffer Tdb { float timeDecay[]; };
layout(set=0,binding=4) readonly buffer Tfb { float timeFirst[]; };
layout(set=0,binding=5) writeonly buffer Ob { float outp[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint modelDim; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int B=int(batch), SL=int(seqLen), MD=int(modelDim);
    if (gid >= B*MD) return;
    int c = gid % MD; int b = gid / MD;
    float w = -exp(timeDecay[c]); float u = timeFirst[c];
    float aa=0.0, bb=0.0, pp=-1.0e38;
    for (int t=0;t<SL;t++) {
        int off=(b*SL+t)*MD+c; float k=K[off]; float v=Vd[off];
        float ww=u+k; float q=max(pp,ww); float e1=exp(pp-q); float e2=exp(ww-q);
        float wkv=(e1*aa+e2*v)/(e1*bb+e2);
        outp[off]=(1.0/(1.0+exp(-R[off])))*wkv;
        float ww2=pp+w; float q2=max(ww2,k); float e1b=exp(ww2-q2); float e2b=exp(k-q2);
        aa=e1b*aa+e2b*v; bb=e1b*bb+e2b; pp=q2;
    }
}";

    public static string MambaScan => Header + @"
layout(set=0,binding=0) readonly buffer Xb { float X[]; };
layout(set=0,binding=1) readonly buffer Dtb { float delta[]; };
layout(set=0,binding=2) readonly buffer Alb { float aLog[]; };
layout(set=0,binding=3) readonly buffer Bb { float Bt[]; };
layout(set=0,binding=4) readonly buffer Cb { float Ct[]; };
layout(set=0,binding=5) readonly buffer Db { float Dt[]; };
layout(set=0,binding=6) writeonly buffer Ob { float outp[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint innerDim; uint stateDim; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int B=int(batch), SL=int(seqLen), ID=int(innerDim), SD=int(stateDim);
    if (gid >= B*ID) return;
    int di = gid % ID; int b = gid / ID; int hrow = di*SD;
    float negA[256]; float h[256];
    for (int n=0;n<SD;n++) { negA[n]=-exp(aLog[hrow+n]); h[n]=0.0; }
    for (int t=0;t<SL;t++) {
        int baseID=(b*SL+t)*ID; int baseSD=(b*SL+t)*SD;
        float dt=delta[baseID+di]; float xv=X[baseID+di]; float y=0.0;
        for (int n=0;n<SD;n++) { float aBar=exp(dt*negA[n]); float hv=aBar*h[n]+dt*Bt[baseSD+n]*xv; h[n]=hv; y+=Ct[baseSD+n]*hv; }
        outp[baseID+di]=y+Dt[di]*xv;
    }
}";

    public static string Mamba2Ssd => Header + @"
layout(set=0,binding=0) readonly buffer Xb { float X[]; };
layout(set=0,binding=1) readonly buffer Dtb { float delta[]; };
layout(set=0,binding=2) readonly buffer Alb { float aLog[]; };
layout(set=0,binding=3) readonly buffer Bb { float Bt[]; };
layout(set=0,binding=4) readonly buffer Cb { float Ct[]; };
layout(set=0,binding=5) readonly buffer Db { float Dt[]; };
layout(set=0,binding=6) writeonly buffer Ob { float outp[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint innerDim; uint numHeads; uint headDim; uint sd; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int B=int(batch), SL=int(seqLen), ID=int(innerDim), NH=int(numHeads), HD=int(headDim), SD=int(sd);
    if (gid >= B*ID) return;
    int flatD = gid % ID; int b = gid / ID; int hi = flatD / HD;
    float negA=-exp(aLog[hi]); float dv=Dt[hi];
    float h[256];
    for (int n=0;n<SD;n++) h[n]=0.0;
    for (int t=0;t<SL;t++) {
        int btInner=(b*SL+t)*ID; int btState=(b*SL+t)*SD; int btHead=(b*SL+t)*NH;
        float dt=delta[btHead+hi]; float aBar=exp(dt*negA); float xv=X[btInner+flatD]; float y=0.0;
        for (int n=0;n<SD;n++) { float hNew=aBar*h[n]+dt*Bt[btState+n]*xv; h[n]=hNew; y+=Ct[btState+n]*hNew; }
        outp[btInner+flatD]=y+dv*xv;
    }
}";

    public static string FusedCeIndex => Header + @"
layout(set=0,binding=0) readonly buffer Hb { float hidden[]; };
layout(set=0,binding=1) readonly buffer Wb { float weight[]; };
layout(set=0,binding=2) readonly buffer Bb { float bias[]; };
layout(set=0,binding=3) readonly buffer Tb { float targetIds[]; };
layout(set=0,binding=4) writeonly buffer Lb { float rowLoss[]; };
layout(push_constant) uniform PC { uint N; uint d; uint vocab; };
void main() {
    int r = int(gl_GlobalInvocationID.x);
    int NN=int(N), D=int(d), VC=int(vocab);
    if (r >= NN) return;
    int hRow=r*D; int id=int(targetIds[r]+0.5);
    float mx=-1.0e38, logitTarget=0.0;
    for (int vv=0;vv<VC;vv++) { float s=bias[vv]; for (int j=0;j<D;j++) s+=hidden[hRow+j]*weight[j*VC+vv]; if (s>mx) mx=s; if (vv==id) logitTarget=s; }
    float sumExp=0.0;
    for (int vv=0;vv<VC;vv++) { float s=bias[vv]; for (int j=0;j<D;j++) s+=hidden[hRow+j]*weight[j*VC+vv]; sumExp+=exp(s-mx); }
    rowLoss[r]=(mx+log(sumExp))-logitTarget;
}";

    public static string FusedCeDense => Header + @"
layout(set=0,binding=0) readonly buffer Hb { float hidden[]; };
layout(set=0,binding=1) readonly buffer Wb { float weight[]; };
layout(set=0,binding=2) readonly buffer Bb { float bias[]; };
layout(set=0,binding=3) readonly buffer Tb { float target[]; };
layout(set=0,binding=4) writeonly buffer Lb { float rowLoss[]; };
layout(push_constant) uniform PC { uint N; uint d; uint vocab; };
void main() {
    int r = int(gl_GlobalInvocationID.x);
    int NN=int(N), D=int(d), VC=int(vocab);
    if (r >= NN) return;
    int hRow=r*D; int tRow=r*VC;
    float mx=-1.0e38;
    for (int vv=0;vv<VC;vv++) { float s=bias[vv]; for (int j=0;j<D;j++) s+=hidden[hRow+j]*weight[j*VC+vv]; if (s>mx) mx=s; }
    float sumExp=0.0;
    for (int vv=0;vv<VC;vv++) { float s=bias[vv]; for (int j=0;j<D;j++) s+=hidden[hRow+j]*weight[j*VC+vv]; sumExp+=exp(s-mx); }
    float lse=mx+log(sumExp); float rl=0.0;
    for (int vv=0;vv<VC;vv++) { float s=bias[vv]; for (int j=0;j<D;j++) s+=hidden[hRow+j]*weight[j*VC+vv]; rl+=target[tRow+vv]*(s-lse); }
    rowLoss[r]=-rl;
}";
}
