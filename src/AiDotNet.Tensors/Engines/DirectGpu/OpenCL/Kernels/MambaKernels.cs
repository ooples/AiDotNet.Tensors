// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernel for the fused Mamba selective scan forward (#1464). One work-item per (batch, channel).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class MambaKernels
{
    public const int MaxStateDim = 256;

    public static string GetSource()
    {
        return @"
#define MAMBA_MAX_STATEDIM 256

__kernel void mamba_selective_scan_forward(
    __global const float* X, __global const float* delta, __global const float* aLog,
    __global const float* B, __global const float* C, __global const float* D,
    __global float* output,
    int batch, int seqLen, int innerDim, int stateDim)
{
    int gid = get_global_id(0);
    int total = batch * innerDim;
    if (gid >= total) return;

    int di = gid % innerDim;
    int b = gid / innerDim;
    int hrow = di * stateDim;

    float negA[MAMBA_MAX_STATEDIM];
    float h[MAMBA_MAX_STATEDIM];
    for (int ni = 0; ni < stateDim; ni++) { negA[ni] = -exp(aLog[hrow + ni]); h[ni] = 0.0f; }

    for (int t = 0; t < seqLen; t++) {
        int baseID = (b * seqLen + t) * innerDim;
        int baseSD = (b * seqLen + t) * stateDim;
        float dt = delta[baseID + di];
        float xv = X[baseID + di];
        float y = 0.0f;
        for (int ni = 0; ni < stateDim; ni++) {
            float aBar = exp(dt * negA[ni]);
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
