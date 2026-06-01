// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernel for the fused Gated DeltaNet scan forward (issue ooples/AiDotNet#1464).
// Same design as HipGatedDeltaNetKernels: one work-item per (batch, head, value-row).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class GatedDeltaNetKernels
{
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#define GDN_MAX_HEADDIM 256

__kernel void gated_delta_scan_forward(
    __global const float* Q, __global const float* K, __global const float* V,
    __global const float* A, __global const float* B,
    __global float* output,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    int gid = get_global_id(0);
    int total = batch * numHeads * headDim;
    if (gid >= total) return;

    int di = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int b  = gid / (headDim * numHeads);
    int hOff = h * headDim;
    float kappa = 1.0f / sqrt((float)headDim);

    float Srow[GDN_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        float sK = 0.0f;
        for (int ki = 0; ki < headDim; ki++) sK += Srow[ki] * (K[baseOff + ki] * kappa);
        float bd = bet * (V[baseOff + di] - sK);
        for (int ki = 0; ki < headDim; ki++)
            Srow[ki] = a * Srow[ki] + bd * (K[baseOff + ki] * kappa);
        float o = 0.0f;
        for (int ki = 0; ki < headDim; ki++) o += Srow[ki] * Q[baseOff + ki];
        output[baseOff + di] = o;
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { "gated_delta_scan_forward" };
}
