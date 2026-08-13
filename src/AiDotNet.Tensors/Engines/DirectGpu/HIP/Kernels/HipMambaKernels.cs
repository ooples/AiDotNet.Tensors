// Copyright (c) AiDotNet. All rights reserved.
// HIP kernel for the fused Mamba selective scan forward (#1464).
// Mirrors CpuEngine.MambaScan / RecurrenceCpuKernels.MambaSelectiveScanForward. Per (channel di)
// state h[di, 0..stateDim) is independent, so one thread owns (batch b, channel di). Backward via tape.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipMambaKernels
{
    public const int MaxStateDim = 256;

    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

#define MAMBA_MAX_STATEDIM 256

extern ""C"" __global__ __launch_bounds__(1024) void mamba_selective_scan_forward(
    const float* X, const float* delta, const float* aLog,
    const float* B, const float* C, const float* D,
    float* output,
    int batch, int seqLen, int innerDim, int stateDim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * innerDim;
    if (gid >= total) return;
    if (stateDim < 0 || stateDim > MAMBA_MAX_STATEDIM) return;

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

extern ""C"" __global__ __launch_bounds__(256) void complex_diagonal_ssm_scan_forward(
    const float* X, const float* Ar, const float* Ai, const float* Br, const float* Bi,
    const float* Cr, const float* Ci, const float* D, float* output,
    int batch, int time, int groups, int width, int state)
{
    int bg = blockIdx.x;
    int lane = threadIdx.x;
    if (bg >= batch * groups) return;
    int b = bg / groups, g = bg % groups;
    __shared__ float hr[256], hi[256];
    if (lane < state) { hr[lane] = 0.0f; hi[lane] = 0.0f; }
    __syncthreads();
    for (int t = 0; t < time; t++) {
        int xBase = ((b * time + t) * groups + g) * width;
        if (lane < state) {
            int a = g * state + lane, bm = (g * state + lane) * width;
            float oldR = hr[lane], oldI = hi[lane];
            float nextR = Ar[a] * oldR - Ai[a] * oldI;
            float nextI = Ar[a] * oldI + Ai[a] * oldR;
            for (int w = 0; w < width; w++) { float xv=X[xBase+w]; nextR+=Br[bm+w]*xv; nextI+=Bi[bm+w]*xv; }
            hr[lane]=nextR; hi[lane]=nextI;
        }
        __syncthreads();
        if (lane < width) {
            int cm=(g*width+lane)*state; float y=D[g*width+lane]*X[xBase+lane];
            for (int n=0;n<state;n++) y += Cr[cm+n]*hr[n]-Ci[cm+n]*hi[n];
            output[xBase+lane]=y;
        }
        __syncthreads();
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { "mamba_selective_scan_forward", "complex_diagonal_ssm_scan_forward" };
}
