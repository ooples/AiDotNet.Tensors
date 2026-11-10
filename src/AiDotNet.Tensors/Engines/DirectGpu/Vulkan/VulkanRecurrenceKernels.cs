// Copyright (c) CrowdTrainer LLC. All rights reserved.
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

    public static string ComplexDiagonalSsmScan => Header + @"
layout(set=0,binding=0) readonly buffer Xb { float X[]; };
layout(set=0,binding=1) readonly buffer Arb { float Ar[]; };
layout(set=0,binding=2) readonly buffer Aib { float Ai[]; };
layout(set=0,binding=3) readonly buffer Brb { float Br[]; };
layout(set=0,binding=4) readonly buffer Bib { float Bi[]; };
layout(set=0,binding=5) readonly buffer Crb { float Cr[]; };
layout(set=0,binding=6) readonly buffer Cib { float Ci[]; };
layout(set=0,binding=7) readonly buffer Db { float D[]; };
layout(set=0,binding=8) writeonly buffer Ob { float outp[]; };
layout(push_constant) uniform PC { uint batch; uint time; uint groups; uint width; uint state; };
void main() {
    int gid=int(gl_GlobalInvocationID.x), G=int(groups), T=int(time), W=int(width), S=int(state);
    if (gid>=int(batch)*G) return; int b=gid/G, g=gid%G; float hr[256]; float hi[256];
    for(int n=0;n<S;n++){hr[n]=0.0;hi[n]=0.0;}
    for(int t=0;t<T;t++) { int xb=((b*T+t)*G+g)*W;
        for(int n=0;n<S;n++){int a=g*S+n,bm=(g*S+n)*W;float orr=hr[n],oi=hi[n];float nr=Ar[a]*orr-Ai[a]*oi,ni=Ar[a]*oi+Ai[a]*orr;
            for(int w=0;w<W;w++){float xv=X[xb+w];nr+=Br[bm+w]*xv;ni+=Bi[bm+w]*xv;}hr[n]=nr;hi[n]=ni;}
        for(int w=0;w<W;w++){int cm=(g*W+w)*S;float y=D[g*W+w]*X[xb+w];for(int n=0;n<S;n++)y+=Cr[cm+n]*hr[n]-Ci[cm+n]*hi[n];outp[xb+w]=y;}
    }
}";

    public static string MesaScan => Header + @"
layout(set=0,binding=0) readonly buffer Qb{float Q[];};
layout(set=0,binding=1) readonly buffer Kb{float K[];};
layout(set=0,binding=2) readonly buffer Vb{float V[];};
layout(set=0,binding=3) readonly buffer W0b{float W0[];};
layout(set=0,binding=4) readonly buffer Rb{float regularization[];};
layout(set=0,binding=5) buffer Ob{float outputValue[];};
layout(set=0,binding=6) buffer Wb{float workW[];};
layout(set=0,binding=7) buffer Pb{float covariance[];};
layout(push_constant) uniform PC{uint batch;uint time;uint model;uint heads;uint headDim;};
void main(){int bh=int(gl_GlobalInvocationID.x),D=int(headDim);if(bh>=int(batch*heads)||D>32)return;
 int b=bh/int(heads),h=bh%int(heads),matrix=D*D,base=bh*matrix,w0Base=h*matrix;float invLambda=1.0/regularization[0];
 for(int i=0;i<matrix;i++){workW[base+i]=W0[w0Base+i];covariance[base+i]=0.0;}for(int i=0;i<D;i++)covariance[base+i*D+i]=invLambda;
 float pk[32],error[32],row[32];for(int t=0;t<int(time);t++){int vb=(b*int(time)+t)*int(model)+h*D;
  for(int i=0;i<D;i++){float s=0.0;for(int j=0;j<D;j++)s+=covariance[base+i*D+j]*K[vb+j];pk[i]=s;}float denom=1.0;for(int i=0;i<D;i++)denom+=K[vb+i]*pk[i];
  for(int i=0;i<D;i++)for(int j=0;j<D;j++)covariance[base+i*D+j]-=pk[i]*pk[j]/denom;
  for(int i=0;i<D;i++){float s=0.0;for(int j=0;j<D;j++)s+=workW[base+i*D+j]*K[vb+j];error[i]=s-V[vb+i];}
  for(int j=0;j<D;j++){float s=0.0;for(int i=0;i<D;i++)s+=K[vb+i]*covariance[base+i*D+j];row[j]=s;}
  for(int i=0;i<D;i++)for(int j=0;j<D;j++)workW[base+i*D+j]-=error[i]*row[j];
  for(int i=0;i<D;i++){float s=0.0;for(int j=0;j<D;j++)s+=workW[base+i*D+j]*Q[vb+j];outputValue[vb+i]=s;}
 }}";

    public static string RoutedDiagonalSsmScan => Header + @"
layout(set=0,binding=0) readonly buffer Xb{float X[];};layout(set=0,binding=1) readonly buffer Mb{float mask[];};
layout(set=0,binding=2) readonly buffer Ab{float A[];};layout(set=0,binding=3) readonly buffer Bb{float B[];};
layout(set=0,binding=4) readonly buffer Cb{float C[];};layout(set=0,binding=5) readonly buffer Db{float D[];};
layout(set=0,binding=6) buffer Ob{float outputValue[];};layout(set=0,binding=7) buffer Hb{float hState[];};
layout(push_constant) uniform PC{uint batch;uint time;uint model;uint experts;uint state;};
void main(){int be=int(gl_GlobalInvocationID.x);if(be>=int(batch*experts))return;int b=be/int(experts),e=be%int(experts),hb=be*int(state);
 for(int s=0;s<int(state);s++)hState[hb+s]=0.0;for(int t=0;t<int(time);t++){int xb=(b*int(time)+t)*int(model),mi=(b*int(time)+t)*int(experts)+e;float isActive=mask[mi];
  for(int s=0;s<int(state);s++){float next=A[e*int(state)+s]*hState[hb+s];int bb=(e*int(state)+s)*int(model);for(int d=0;d<int(model);d++)next+=B[bb+d]*X[xb+d];hState[hb+s]=isActive*next;}
  int yb=((b*int(time)+t)*int(experts)+e)*int(model);for(int d=0;d<int(model);d++){float y=D[e*int(model)+d]*X[xb+d];int cb=(e*int(model)+d)*int(state);for(int s=0;s<int(state);s++)y+=C[cb+s]*hState[hb+s];outputValue[yb+d]=isActive*y;}}
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

    // GLA BPTT backward, checkpointed: pass 1 re-runs the forward scan saving the state entering each chunk of
    // `chunk` steps (Ckpt, B*H*nChunks*d^2); pass 2 walks chunks in reverse, recomputes the chunk's states into
    // Scratch (B*H*chunk*d^2) and reverse-sweeps it: O(sqrt(S)) state memory instead of the O(S) trajectory
    // (GlaScanCheckpoint). dQ/dK/dG are bound as uint SSBOs and accumulated via a CAS float-atomic add (GLSL has no
    // portable float atomicAdd); dV is a plain per-thread store. dQ/dK/dG must be pre-zeroed.
    public static string GlaBackwardCheckpointed => Header + @"
layout(set=0,binding=0) readonly buffer DOb { float dOut[]; };
layout(set=0,binding=1) readonly buffer Qb { float Q[]; };
layout(set=0,binding=2) readonly buffer Kb { float K[]; };
layout(set=0,binding=3) readonly buffer Vb { float Vd[]; };
layout(set=0,binding=4) readonly buffer Gb { float G[]; };
layout(set=0,binding=5) buffer Cb { float Ckpt[]; };
layout(set=0,binding=6) buffer Sb { float Scratch[]; };
layout(set=0,binding=7) buffer DQb { uint dQ[]; };
layout(set=0,binding=8) buffer DKb { uint dK[]; };
layout(set=0,binding=9) buffer DVb { float dV[]; };
layout(set=0,binding=10) buffer DGb { uint dG[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint modelDim; uint numHeads; uint headDim; uint chunk; };
void addDQ(int idx, float val){ uint old=dQ[idx]; uint assumed; do { assumed=old; old=atomicCompSwap(dQ[idx], assumed, floatBitsToUint(uintBitsToFloat(assumed)+val)); } while(old!=assumed); }
void addDK(int idx, float val){ uint old=dK[idx]; uint assumed; do { assumed=old; old=atomicCompSwap(dK[idx], assumed, floatBitsToUint(uintBitsToFloat(assumed)+val)); } while(old!=assumed); }
void addDG(int idx, float val){ uint old=dG[idx]; uint assumed; do { assumed=old; old=atomicCompSwap(dG[idx], assumed, floatBitsToUint(uintBitsToFloat(assumed)+val)); } while(old!=assumed); }
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int B=int(batch), SL=int(seqLen), MD=int(modelDim), NH=int(numHeads), HD=int(headDim), CH=int(chunk);
    if (gid >= B*NH*HD) return;
    int di = gid % HD; int h = (gid/HD) % NH; int b = gid/(HD*NH); int hOff = h*HD; int hh = HD*HD;
    int bh = b*NH + h; int rowOff = di*HD; int nChunks = (SL + CH - 1) / CH;
    float Srow[256];
    float dSrow[256];
    for (int ki=0; ki<HD; ki++) Srow[ki]=0.0;
    for (int t=0; t<SL; t++) {
        if (t % CH == 0) { int ck = (bh*nChunks + t/CH)*hh + rowOff; for (int ki=0; ki<HD; ki++) Ckpt[ck+ki]=Srow[ki]; }
        int baseOff = (b*SL+t)*MD + hOff;
        float g = G[(b*SL+t)*NH + h]; float vd = Vd[baseOff+di];
        for (int ki=0; ki<HD; ki++) Srow[ki]=g*Srow[ki]+vd*K[baseOff+ki];
    }
    for (int ki=0; ki<HD; ki++) dSrow[ki]=0.0;
    for (int c=nChunks-1; c>=0; c--) {
        int t0 = c*CH; int t1 = min(SL, t0+CH);
        int ck = (bh*nChunks + c)*hh + rowOff;
        for (int ki=0; ki<HD; ki++) Srow[ki]=Ckpt[ck+ki];
        for (int t=t0; t<t1; t++) {
            int baseOff = (b*SL+t)*MD + hOff;
            float g = G[(b*SL+t)*NH + h]; float vd = Vd[baseOff+di];
            int sc = (bh*CH + (t-t0))*hh + rowOff;
            for (int ki=0; ki<HD; ki++) { float s=g*Srow[ki]+vd*K[baseOff+ki]; Srow[ki]=s; Scratch[sc+ki]=s; }
        }
        for (int t=t1-1; t>=t0; t--) {
            int baseOff = (b*SL+t)*MD + hOff; int gOff = (b*SL+t)*NH + h; float g = G[gOff];
            int sc = (bh*CH + (t-t0))*hh + rowOff;
            int sp = (t>t0) ? (bh*CH + (t-1-t0))*hh + rowOff : -1;
            float dOutVal = dOut[baseOff+di];
            for (int ki=0; ki<HD; ki++) { addDQ(baseOff+ki, dOutVal*Scratch[sc+ki]); dSrow[ki]+=dOutVal*Q[baseOff+ki]; }
            float vd = Vd[baseOff+di]; float dg=0.0; float dVacc=0.0;
            for (int ki=0; ki<HD; ki++) { float dStv=dSrow[ki]; float sprev = (sp>=0) ? Scratch[sp+ki] : Ckpt[ck+ki]; dg+=dStv*sprev; addDK(baseOff+ki, dStv*vd); dVacc+=dStv*K[baseOff+ki]; }
            dV[baseOff+di] = dVacc;
            addDG(gOff, dg);
            for (int ki=0; ki<HD; ki++) dSrow[ki]*=g;
        }
    }
}";

    // Channels-last causal depthwise conv (ICausalDepthwiseConvBackend): x, y, dy, dx are [rows, seqLen, C]; w, dw
    // are [K, C]; tap j reads j steps back. Every output element is written and nothing is atomic: deterministic.
    public static string CausalConvForward => Header + @"
layout(set=0,binding=0) readonly buffer Xb { float x[]; };
layout(set=0,binding=1) readonly buffer Wb { float w[]; };
layout(set=0,binding=2) buffer Yb { float y[]; };
layout(push_constant) uniform PC { uint rows; uint seqLen; uint channels; uint taps; };
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int SL=int(seqLen), C=int(channels), K=int(taps);
    if (idx >= int(rows)*SL*C) return;
    int c = idx % C; int t = (idx / C) % SL;
    int maxJ = min(K - 1, t);
    float acc = 0.0;
    for (int j=0; j<=maxJ; j++) acc += w[j*C + c] * x[idx - j*C];
    y[idx] = acc;
}";

    public static string CausalConvBackwardInput => Header + @"
layout(set=0,binding=0) readonly buffer Gb { float dy[]; };
layout(set=0,binding=1) readonly buffer Wb { float w[]; };
layout(set=0,binding=2) buffer DXb { float dx[]; };
layout(push_constant) uniform PC { uint rows; uint seqLen; uint channels; uint taps; };
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int SL=int(seqLen), C=int(channels), K=int(taps);
    if (idx >= int(rows)*SL*C) return;
    int c = idx % C; int t = (idx / C) % SL;
    int maxJ = min(K - 1, SL - 1 - t);
    float acc = 0.0;
    for (int j=0; j<=maxJ; j++) acc += w[j*C + c] * dy[idx + j*C];
    dx[idx] = acc;
}";

    // One invocation per (tap j, channel c), summing every row and step in ascending order.
    public static string CausalConvBackwardKernel => Header + @"
layout(set=0,binding=0) readonly buffer Gb { float dy[]; };
layout(set=0,binding=1) readonly buffer Xb { float x[]; };
layout(set=0,binding=2) buffer DWb { float dw[]; };
layout(push_constant) uniform PC { uint rows; uint seqLen; uint channels; uint taps; };
void main() {
    int jc = int(gl_GlobalInvocationID.x);
    int R=int(rows), SL=int(seqLen), C=int(channels), K=int(taps);
    if (jc >= K*C) return;
    int j = jc / C; int c = jc % C;
    float acc = 0.0;
    for (int r=0; r<R; r++) {
        int rowBase = r*SL*C;
        for (int t=j; t<SL; t++) acc += dy[rowBase + t*C + c] * x[rowBase + (t-j)*C + c];
    }
    dw[jc] = acc;
}";

    public static string GatedDeltaRecompute => Header + @"
layout(set=0,binding=0) readonly buffer Kb { float K[]; };
layout(set=0,binding=1) readonly buffer Vb { float Vd[]; };
layout(set=0,binding=2) readonly buffer Ab { float A[]; };
layout(set=0,binding=3) readonly buffer Bb { float Bt[]; };
layout(set=0,binding=4) buffer Sb { float Straj[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint modelDim; uint numHeads; uint headDim; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int B=int(batch), SL=int(seqLen), MD=int(modelDim), NH=int(numHeads), HD=int(headDim);
    if (gid >= B*NH*HD) return;
    int di = gid % HD; int h = (gid/HD) % NH; int b = gid/(HD*NH); int hOff = h*HD; int hh = HD*HD;
    float kappa = 1.0/sqrt(float(HD));
    float Srow[256];
    for (int ki=0; ki<HD; ki++) Srow[ki]=0.0;
    for (int t=0; t<SL; t++) {
        int baseOff = (b*SL+t)*MD + hOff; int gIdx = (b*SL+t)*NH + h;
        float a = A[gIdx], bet = Bt[gIdx];
        float sK = 0.0; for (int ki=0; ki<HD; ki++) sK += Srow[ki]*(K[baseOff+ki]*kappa);
        float bd = bet*(Vd[baseOff+di]-sK);
        int trajBase = ((b*NH+h)*SL+t)*hh + di*HD;
        for (int ki=0; ki<HD; ki++) { float s = a*Srow[ki] + bd*(K[baseOff+ki]*kappa); Srow[ki]=s; Straj[trajBase+ki]=s; }
    }
}";

    public static string GatedDeltaBackward => GatedDeltaBackwardSource(deterministic: false);

    /// <summary>
    /// The backward without atomics: dQ comes from <see cref="GatedDeltaDq"/>, each row's dK contribution overwrites
    /// that row of the trajectory (read only by this invocation, and last read at step t + 1), and dA/dB go to per-row
    /// partials P = [dA | dB]. <see cref="GatedDeltaReduce"/> then sums the rows in a fixed order.
    /// </summary>
    public static string GatedDeltaBackwardDeterministic => GatedDeltaBackwardSource(deterministic: true);

    private static string GatedDeltaBackwardSource(bool deterministic) => Header + (deterministic ? @"
layout(set=0,binding=0) readonly buffer DOb { float dOut[]; };
layout(set=0,binding=1) readonly buffer Qb { float Q[]; };
layout(set=0,binding=2) readonly buffer Kb { float K[]; };
layout(set=0,binding=3) readonly buffer Vb { float Vd[]; };
layout(set=0,binding=4) readonly buffer Ab { float A[]; };
layout(set=0,binding=5) readonly buffer Bb { float Bt[]; };
layout(set=0,binding=6) buffer Sb { float Straj[]; };
layout(set=0,binding=7) writeonly buffer Pb { float P[]; };
layout(set=0,binding=8) writeonly buffer DVb { float dV[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint modelDim; uint numHeads; uint headDim; };
void addDQ(int idx, float val){ }
void addDK(int idx, float val){ }
" : @"
layout(set=0,binding=0) readonly buffer DOb { float dOut[]; };
layout(set=0,binding=1) readonly buffer Qb { float Q[]; };
layout(set=0,binding=2) readonly buffer Kb { float K[]; };
layout(set=0,binding=3) readonly buffer Vb { float Vd[]; };
layout(set=0,binding=4) readonly buffer Ab { float A[]; };
layout(set=0,binding=5) readonly buffer Bb { float Bt[]; };
layout(set=0,binding=6) readonly buffer Sb { float Straj[]; };
layout(set=0,binding=7) buffer DQb { uint dQ[]; };
layout(set=0,binding=8) buffer DKb { uint dK[]; };
layout(set=0,binding=9) buffer DVb { float dV[]; };
layout(set=0,binding=10) buffer DAb { uint dA[]; };
layout(set=0,binding=11) buffer DBb { uint dB[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint modelDim; uint numHeads; uint headDim; };
void addDQ(int idx, float val){ uint old=dQ[idx]; uint assumed; do { assumed=old; old=atomicCompSwap(dQ[idx], assumed, floatBitsToUint(uintBitsToFloat(assumed)+val)); } while(old!=assumed); }
void addDK(int idx, float val){ uint old=dK[idx]; uint assumed; do { assumed=old; old=atomicCompSwap(dK[idx], assumed, floatBitsToUint(uintBitsToFloat(assumed)+val)); } while(old!=assumed); }
void addDA(int idx, float val){ uint old=dA[idx]; uint assumed; do { assumed=old; old=atomicCompSwap(dA[idx], assumed, floatBitsToUint(uintBitsToFloat(assumed)+val)); } while(old!=assumed); }
void addDB(int idx, float val){ uint old=dB[idx]; uint assumed; do { assumed=old; old=atomicCompSwap(dB[idx], assumed, floatBitsToUint(uintBitsToFloat(assumed)+val)); } while(old!=assumed); }
") + (deterministic ? @"
#define DET 1
" : @"
#define DET 0
") + @"
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int B=int(batch), SL=int(seqLen), MD=int(modelDim), NH=int(numHeads), HD=int(headDim);
    if (gid >= B*NH*HD) return;
    int di = gid % HD; int h = (gid/HD) % NH; int b = gid/(HD*NH); int hOff = h*HD; int hh = HD*HD;
    float kappa = 1.0/sqrt(float(HD));
    float dSrow[256];
    for (int ki=0; ki<HD; ki++) dSrow[ki]=0.0;
    for (int t=SL-1; t>=0; t--) {
        int baseOff = (b*SL+t)*MD + hOff; int gIdx = (b*SL+t)*NH + h;
        float a = A[gIdx], bet = Bt[gIdx];
        int trajBase = ((b*NH+h)*SL+t)*hh + di*HD;
        int sprevBase = (t>0) ? (((b*NH+h)*SL+(t-1))*hh + di*HD) : 0;
        float sK = 0.0;
        for (int ki=0; ki<HD; ki++) { float sprev = (t>0) ? Straj[sprevBase+ki] : 0.0; sK += sprev*(K[baseOff+ki]*kappa); }
        float delta = Vd[baseOff+di] - sK;
        float dOutVal = dOut[baseOff+di];
        for (int ki=0; ki<HD; ki++) { if (DET == 0) addDQ(baseOff+ki, dOutVal*Straj[trajBase+ki]); dSrow[ki]+=dOutVal*Q[baseOff+ki]; }
        float dA_acc=0.0, dB_acc=0.0, dDelta=0.0;
        for (int ki=0; ki<HD; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t>0) ? Straj[sprevBase+ki] : 0.0;
            float kScaled = K[baseOff+ki]*kappa;
            dA_acc += dSt*sprev;
            dB_acc += dSt*delta*kScaled;
            dDelta += dSt*bet*kScaled;
        }
        dV[baseOff+di] = dDelta;
#if DET
        P[gIdx*HD + di] = dA_acc;
        P[B*SL*MD + gIdx*HD + di] = dB_acc;
#else
        addDA(gIdx, dA_acc);
        addDB(gIdx, dB_acc);
#endif
        float dSk = -dDelta;
        for (int ki=0; ki<HD; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t>0) ? Straj[sprevBase+ki] : 0.0;
            float kScaled = K[baseOff+ki]*kappa;
            float dKS = dSt*(bet*delta) + dSk*sprev;
#if DET
            Straj[trajBase+ki] = dKS*kappa;
#else
            addDK(baseOff+ki, dKS*kappa);
#endif
            dSrow[ki] = dSt*a + dSk*kScaled;
        }
    }
}";

    /// <summary>Deterministic dQ: dQ[b,t,h,ki] = sum over di (ascending) of dOut[b,t,h,di] * S_t[di,ki].</summary>
    public static string GatedDeltaDq => Header + @"
layout(set=0,binding=0) readonly buffer DOb { float dOut[]; };
layout(set=0,binding=1) readonly buffer Sb { float Straj[]; };
layout(set=0,binding=2) writeonly buffer DQb { float dQ[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint modelDim; uint numHeads; uint headDim; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int SL=int(seqLen), MD=int(modelDim), NH=int(numHeads), HD=int(headDim);
    if (gid >= int(batch)*SL*MD) return;
    int ki = gid % HD; int h = (gid/HD) % NH; int bt = gid / MD; int t = bt % SL; int b = bt / SL;
    int trajT = ((b*NH+h)*SL+t)*HD*HD;
    int base = bt*MD + h*HD;
    float acc = 0.0;
    for (int di=0; di<HD; di++) acc += dOut[base+di]*Straj[trajT + di*HD + ki];
    dQ[gid] = acc;
}";

    /// <summary>
    /// Deterministic reduction of <see cref="GatedDeltaBackwardDeterministic"/>'s partials: dK from the rows of PK (the
    /// overwritten trajectory) and dA/dB from P = [dA | dB], each summed over di in ascending order.
    /// </summary>
    public static string GatedDeltaReduce => Header + @"
layout(set=0,binding=0) readonly buffer PKb { float PK[]; };
layout(set=0,binding=1) readonly buffer Pb { float P[]; };
layout(set=0,binding=2) writeonly buffer DKb { float dK[]; };
layout(set=0,binding=3) writeonly buffer DAb { float dA[]; };
layout(set=0,binding=4) writeonly buffer DBb { float dB[]; };
layout(push_constant) uniform PC { uint batch; uint seqLen; uint modelDim; uint numHeads; uint headDim; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int SL=int(seqLen), MD=int(modelDim), NH=int(numHeads), HD=int(headDim);
    int n = int(batch)*SL*MD;
    if (gid >= n) return;
    int ki = gid % HD; int h = (gid/HD) % NH; int bt = gid / MD; int t = bt % SL; int b = bt / SL;
    int trajT = ((b*NH+h)*SL+t)*HD*HD;
    float acc = 0.0;
    for (int di=0; di<HD; di++) acc += PK[trajT + di*HD + ki];
    dK[gid] = acc;
    if (ki == 0) {
        int gIdx = bt*NH + h;
        float sa = 0.0, sb = 0.0;
        for (int di=0; di<HD; di++) { sa += P[gIdx*HD + di]; sb += P[n + gIdx*HD + di]; }
        dA[gIdx] = sa;
        dB[gIdx] = sb;
    }
}";
}
