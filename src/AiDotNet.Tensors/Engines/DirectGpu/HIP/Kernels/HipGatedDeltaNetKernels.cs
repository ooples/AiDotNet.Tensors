// Copyright (c) AiDotNet. All rights reserved.
// HIP kernel for the fused Gated DeltaNet (delta-rule) scan forward (issue ooples/AiDotNet#1464).
// Mirrors CpuEngine.GatedDeltaNetScan / RecurrenceCpuKernels.GatedDeltaNetForward. Each value row
// is independent, so one thread owns (batch b, head h, value-row di) and carries that row of the
// state across time. The differentiable backward runs through the CpuEngine tape path.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipGatedDeltaNetKernels
{
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

#define GDN_MAX_HEADDIM 256

extern ""C"" __global__ __launch_bounds__(1024) void gated_delta_scan_forward(
    const float* Q, const float* K, const float* V, const float* A, const float* B,
    float* output,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * numHeads * headDim;
    if (gid >= total) return;

    int di = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int b  = gid / (headDim * numHeads);
    int hOff = h * headDim;
    float kappa = 1.0f / sqrtf((float)headDim);

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
