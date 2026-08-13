// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernel for the fused Mamba selective scan forward (#1464). One thread per (batch, channel).

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

internal static class CudaMambaKernels
{
    public const int MaxStateDim = 256;

    public static string GetSource()
    {
        return @"
#include <math.h>

#define MAMBA_MAX_STATEDIM 256
#define MESA_MAX_HEADDIM 32

extern ""C"" __global__ void mamba_selective_scan_forward(
    const float* X, const float* delta, const float* aLog,
    const float* B, const float* C, const float* D,
    float* output,
    int batch, int seqLen, int innerDim, int stateDim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * innerDim;
    if (gid >= total) return;

    int di = gid % innerDim;
    int b = gid / innerDim;
    int hrow = di * stateDim;

    float negA[MAMBA_MAX_STATEDIM];
    float h[MAMBA_MAX_STATEDIM];
    for (int ni = 0; ni < stateDim; ni++) { negA[ni] = -expf(aLog[hrow + ni]); h[ni] = 0.0f; }

    for (int t = 0; t < seqLen; t++) {
        int baseID = (b * seqLen + t) * innerDim;
        int baseSD = (b * seqLen + t) * stateDim;
        float dt = delta[baseID + di];
        float xv = X[baseID + di];
        float y = 0.0f;
        for (int ni = 0; ni < stateDim; ni++) {
            float aBar = expf(dt * negA[ni]);
            float hv = aBar * h[ni] + dt * B[baseSD + ni] * xv;
            h[ni] = hv;
            y += C[baseSD + ni] * hv;
        }
        output[baseID + di] = y + D[di] * xv;
    }
}

extern ""C"" __global__ void complex_diagonal_ssm_scan_forward(
    const float* X, const float* Ar, const float* Ai, const float* Br, const float* Bi,
    const float* Cr, const float* Ci, const float* D, float* output,
    int batch, int time, int groups, int width, int state)
{
    int bg = blockIdx.x;
    int lane = threadIdx.x;
    if (bg >= batch * groups) return;
    int b = bg / groups;
    int g = bg % groups;
    __shared__ float hr[256];
    __shared__ float hi[256];
    if (lane < state) { hr[lane] = 0.0f; hi[lane] = 0.0f; }
    __syncthreads();
    for (int t = 0; t < time; t++) {
        int xBase = ((b * time + t) * groups + g) * width;
        if (lane < state) {
            int a = g * state + lane;
            int bm = (g * state + lane) * width;
            float oldR = hr[lane], oldI = hi[lane];
            float nextR = Ar[a] * oldR - Ai[a] * oldI;
            float nextI = Ar[a] * oldI + Ai[a] * oldR;
            for (int w = 0; w < width; w++) { float xv = X[xBase+w]; nextR += Br[bm+w]*xv; nextI += Bi[bm+w]*xv; }
            hr[lane] = nextR; hi[lane] = nextI;
        }
        __syncthreads();
        if (lane < width) {
            int cm = (g * width + lane) * state;
            float y = D[g * width + lane] * X[xBase + lane];
            for (int n = 0; n < state; n++) y += Cr[cm+n] * hr[n] - Ci[cm+n] * hi[n];
            output[xBase + lane] = y;
        }
        __syncthreads();
    }
}

extern ""C"" __global__ void mesa_scan_forward(
    const float* Q, const float* K, const float* V, const float* W0,
    const float* regularization, float* output, float* workW, float* covariance,
    int batch, int time, int model, int heads, int headDim)
{
    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= batch * heads || headDim > MESA_MAX_HEADDIM) return;
    int b = bh / heads, h = bh % heads;
    int matrix = headDim * headDim, base = bh * matrix, w0Base = h * matrix;
    float invLambda = 1.0f / regularization[0];
    for (int i=0;i<matrix;i++) { workW[base+i]=W0[w0Base+i]; covariance[base+i]=0.0f; }
    for (int i=0;i<headDim;i++) covariance[base+i*headDim+i]=invLambda;
    float pk[MESA_MAX_HEADDIM], error[MESA_MAX_HEADDIM], row[MESA_MAX_HEADDIM];
    for (int t=0;t<time;t++) {
        int vb=(b*time+t)*model+h*headDim;
        for (int i=0;i<headDim;i++) { float s=0.0f; for(int j=0;j<headDim;j++) s+=covariance[base+i*headDim+j]*K[vb+j]; pk[i]=s; }
        float denom=1.0f; for(int i=0;i<headDim;i++) denom+=K[vb+i]*pk[i];
        for(int i=0;i<headDim;i++) for(int j=0;j<headDim;j++) covariance[base+i*headDim+j]-=pk[i]*pk[j]/denom;
        for(int i=0;i<headDim;i++) { float s=0.0f; for(int j=0;j<headDim;j++) s+=workW[base+i*headDim+j]*K[vb+j]; error[i]=s-V[vb+i]; }
        for(int j=0;j<headDim;j++) { float s=0.0f; for(int i=0;i<headDim;i++) s+=K[vb+i]*covariance[base+i*headDim+j]; row[j]=s; }
        for(int i=0;i<headDim;i++) for(int j=0;j<headDim;j++) workW[base+i*headDim+j]-=error[i]*row[j];
        for(int i=0;i<headDim;i++) { float s=0.0f; for(int j=0;j<headDim;j++) s+=workW[base+i*headDim+j]*Q[vb+j]; output[vb+i]=s; }
    }
}

extern ""C"" __global__ void routed_diagonal_ssm_scan_forward(
 const float* X,const float* mask,const float* A,const float* B,const float* C,const float* D,
 float* output,float* hState,int batch,int time,int model,int experts,int state)
{
 int be=blockIdx.x*blockDim.x+threadIdx.x;if(be>=batch*experts)return;int b=be/experts,e=be%experts;
 int hb=be*state;for(int s=0;s<state;s++)hState[hb+s]=0.0f;
 for(int t=0;t<time;t++){int xb=(b*time+t)*model,mi=(b*time+t)*experts+e;float active=mask[mi];
  for(int s=0;s<state;s++){float next=A[e*state+s]*hState[hb+s];int bb=(e*state+s)*model;for(int d=0;d<model;d++)next+=B[bb+d]*X[xb+d];hState[hb+s]=active*next;}
  int yb=((b*time+t)*experts+e)*model;for(int d=0;d<model;d++){float y=D[e*model+d]*X[xb+d];int cb=(e*model+d)*state;for(int s=0;s<state;s++)y+=C[cb+s]*hState[hb+s];output[yb+d]=active*y;}
 }
}
";
    }

    public static string[] GetKernelNames() => new[] { "mamba_selective_scan_forward", "complex_diagonal_ssm_scan_forward", "mesa_scan_forward", "routed_diagonal_ssm_scan_forward" };
}
