// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernel for the fused Mamba-2 SSD scan forward (#1464). One work-item per (batch, channel flatD).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class Mamba2Kernels
{
    public const int MaxStateDim = 256;

    public static string GetSource()
    {
        return @"
#define MAMBA2_MAX_STATEDIM 256

__kernel void mamba2_ssd_scan_forward(
    __global const float* X, __global const float* delta, __global const float* aLog,
    __global const float* B, __global const float* C, __global const float* D,
    __global float* output,
    int batch, int seqLen, int innerDim, int numHeads, int headDim, int sd)
{
    int gid = get_global_id(0);
    int total = batch * innerDim;
    if (gid >= total) return;

    int flatD = gid % innerDim;
    int b = gid / innerDim;
    int hi = flatD / headDim;
    float negA = -exp(aLog[hi]);
    float dv = D[hi];

    float h[MAMBA2_MAX_STATEDIM];
    for (int n = 0; n < sd; n++) h[n] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int btInner = (b * seqLen + t) * innerDim;
        int btState = (b * seqLen + t) * sd;
        int btHead = (b * seqLen + t) * numHeads;
        float dt = delta[btHead + hi];
        float aBar = exp(dt * negA);
        float xv = X[btInner + flatD];
        float y = 0.0f;
        for (int n = 0; n < sd; n++) {
            float hNew = aBar * h[n] + dt * B[btState + n] * xv;
            h[n] = hNew;
            y += C[btState + n] * hNew;
        }
        output[btInner + flatD] = y + dv * xv;
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { "mamba2_ssd_scan_forward" };
}
