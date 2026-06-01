// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for the fused Gated Linear Attention (GLA) sequence scan (issue ooples/AiDotNet#1464).
// Mirrors CpuEngine.GlaScan.cs exactly (see HipGlaKernels for the shared design notes):
//   S_t[di,ki] = g_t * S_{t-1}[di,ki] + V_t[di] * K_t[ki]
//   O_t[di]    = sum_ki S_t[di,ki] * Q_t[ki]
// One thread owns (batch b, head h, value-row di) and carries that state row across time.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for the fused GLA scan (forward + BPTT backward) on NVIDIA GPUs.
/// </summary>
internal static class CudaGlaKernels
{
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#include <math.h>

#define GLA_MAX_HEADDIM 256

extern ""C"" __global__ void gla_scan_forward(
    const float* Q, const float* K, const float* V, const float* G,
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

    float Srow[GLA_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        float g  = G[(b * seqLen + t) * numHeads + h];
        float vd = V[baseOff + di];
        float o  = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float s = g * Srow[ki] + vd * K[baseOff + ki];
            Srow[ki] = s;
            o += s * Q[baseOff + ki];
        }
        output[baseOff + di] = o;
    }
}

extern ""C"" __global__ void gla_scan_recompute(
    const float* K, const float* V, const float* G,
    float* Straj,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * numHeads * headDim;
    if (gid >= total) return;

    int di = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int b  = gid / (headDim * numHeads);
    int hOff = h * headDim;
    int hh = headDim * headDim;

    float Srow[GLA_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        float g  = G[(b * seqLen + t) * numHeads + h];
        float vd = V[baseOff + di];
        long trajBase = ((long)(b * numHeads + h) * seqLen + t) * hh + (long)di * headDim;
        for (int ki = 0; ki < headDim; ki++) {
            float s = g * Srow[ki] + vd * K[baseOff + ki];
            Srow[ki] = s;
            Straj[trajBase + ki] = s;
        }
    }
}

extern ""C"" __global__ void gla_scan_backward(
    const float* dOut, const float* Q, const float* K, const float* V, const float* G,
    const float* Straj,
    float* dQ, float* dK, float* dV, float* dG,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * numHeads * headDim;
    if (gid >= total) return;

    int di = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int b  = gid / (headDim * numHeads);
    int hOff = h * headDim;
    int hh = headDim * headDim;

    float dSrow[GLA_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) dSrow[ki] = 0.0f;

    for (int t = seqLen - 1; t >= 0; t--) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gOff = (b * seqLen + t) * numHeads + h;
        float g = G[gOff];
        long trajBase = ((long)(b * numHeads + h) * seqLen + t) * hh + (long)di * headDim;
        long sprevBase = ((long)(b * numHeads + h) * seqLen + (t - 1)) * hh + (long)di * headDim;

        float dOutVal = dOut[baseOff + di];
        for (int ki = 0; ki < headDim; ki++) {
            float sval = Straj[trajBase + ki];
            atomicAdd(&dQ[baseOff + ki], dOutVal * sval);
            dSrow[ki] += dOutVal * Q[baseOff + ki];
        }

        float vd = V[baseOff + di];
        float dg = 0.0f;
        float dVacc = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float dStv = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            dg += dStv * sprev;
            atomicAdd(&dK[baseOff + ki], dStv * vd);
            dVacc += dStv * K[baseOff + ki];
        }
        atomicAdd(&dV[baseOff + di], dVacc);
        atomicAdd(&dG[gOff], dg);

        for (int ki = 0; ki < headDim; ki++) dSrow[ki] *= g;
    }
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "gla_scan_forward",
            "gla_scan_recompute",
            "gla_scan_backward"
        };
    }
}
