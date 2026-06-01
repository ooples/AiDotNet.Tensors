// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for the fused Gated Linear Attention (GLA) sequence scan (issue ooples/AiDotNet#1464).
// Mirrors CpuEngine.GlaScan.cs / HipGlaKernels exactly:
//   S_t[di,ki] = g_t * S_{t-1}[di,ki] + V_t[di] * K_t[ki]
//   O_t[di]    = sum_ki S_t[di,ki] * Q_t[ki]
// One work-item owns (batch b, head h, value-row di), carrying that state row across time.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for the fused GLA scan (forward + BPTT backward).
/// </summary>
internal static class GlaKernels
{
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#define GLA_MAX_HEADDIM 256

// CAS-based float atomic add (OpenCL 1.2 has no native float atomics).
inline void atomic_add_float(__global volatile float* addr, float val) {
    union { unsigned int u; float f; } oldval, newval;
    do {
        oldval.f = *addr;
        newval.f = oldval.f + val;
    } while (atomic_cmpxchg((__global volatile unsigned int*)addr, oldval.u, newval.u) != oldval.u);
}

__kernel void gla_scan_forward(
    __global const float* Q, __global const float* K, __global const float* V, __global const float* G,
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

__kernel void gla_scan_recompute(
    __global const float* K, __global const float* V, __global const float* G,
    __global float* Straj,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    int gid = get_global_id(0);
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

__kernel void gla_scan_backward(
    __global const float* dOut, __global const float* Q, __global const float* K,
    __global const float* V, __global const float* G, __global const float* Straj,
    __global volatile float* dQ, __global volatile float* dK,
    __global volatile float* dV, __global volatile float* dG,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    int gid = get_global_id(0);
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
            atomic_add_float(&dQ[baseOff + ki], dOutVal * sval);
            dSrow[ki] += dOutVal * Q[baseOff + ki];
        }

        float vd = V[baseOff + di];
        float dg = 0.0f;
        float dVacc = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float dStv = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            dg += dStv * sprev;
            atomic_add_float(&dK[baseOff + ki], dStv * vd);
            dVacc += dStv * K[baseOff + ki];
        }
        atomic_add_float(&dV[baseOff + di], dVacc);
        atomic_add_float(&dG[gOff], dg);

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
