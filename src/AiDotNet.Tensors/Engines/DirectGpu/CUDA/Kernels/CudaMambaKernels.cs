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
";
    }

    public static string[] GetKernelNames() => new[] { "mamba_selective_scan_forward" };
}
