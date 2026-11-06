// Copyright (c) CrowdTrainer LLC. All rights reserved.
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

inline void gdn_atomic_add_float(__global volatile float* addr, float val) {
    union { unsigned int u; float f; } oldval, newval;
    do {
        oldval.f = *addr;
        newval.f = oldval.f + val;
    } while (atomic_cmpxchg((__global volatile unsigned int*)addr, oldval.u, newval.u) != oldval.u);
}

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

__kernel void gated_delta_scan_recompute(
    __global const float* K, __global const float* V,
    __global const float* A, __global const float* B,
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
        long trajBase = ((long)(b * numHeads + h) * seqLen + t) * hh + (long)di * headDim;
        for (int ki = 0; ki < headDim; ki++) {
            float s = a * Srow[ki] + bd * (K[baseOff + ki] * kappa);
            Srow[ki] = s;
            Straj[trajBase + ki] = s;
        }
    }
}

__kernel void gated_delta_scan_backward(
    __global const float* dOut, __global const float* Q, __global const float* K,
    __global const float* V, __global const float* A, __global const float* B,
    __global const float* Straj,
    __global volatile float* dQ, __global volatile float* dK, __global float* dV,
    __global volatile float* dA, __global volatile float* dB,
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
    float kappa = 1.0f / sqrt((float)headDim);

    float dSrow[GDN_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) dSrow[ki] = 0.0f;

    for (int t = seqLen - 1; t >= 0; t--) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        long trajBase = ((long)(b * numHeads + h) * seqLen + t) * hh + (long)di * headDim;
        long sprevBase = (t > 0) ? (((long)(b * numHeads + h) * seqLen + (t - 1)) * hh + (long)di * headDim) : 0L;

        float sK = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            sK += sprev * (K[baseOff + ki] * kappa);
        }
        float delta = V[baseOff + di] - sK;

        float dOutVal = dOut[baseOff + di];
        for (int ki = 0; ki < headDim; ki++) {
            float sVal = Straj[trajBase + ki];
            gdn_atomic_add_float(&dQ[baseOff + ki], dOutVal * sVal);
            dSrow[ki] += dOutVal * Q[baseOff + ki];
        }

        float dA_acc = 0.0f;
        float dB_acc = 0.0f;
        float dDelta = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            float kScaled = K[baseOff + ki] * kappa;
            dA_acc += dSt * sprev;
            dB_acc += dSt * delta * kScaled;
            dDelta += dSt * bet * kScaled;
        }
        dV[baseOff + di] = dDelta;
        gdn_atomic_add_float(&dA[gIdx], dA_acc);
        gdn_atomic_add_float(&dB[gIdx], dB_acc);

        float dSk = -dDelta;
        for (int ki = 0; ki < headDim; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            float kScaled = K[baseOff + ki] * kappa;
            float dKS_ki = dSt * (bet * delta) + dSk * sprev;
            gdn_atomic_add_float(&dK[baseOff + ki], dKS_ki * kappa);
            dSrow[ki] = dSt * a + dSk * kScaled;
        }
    }
}
";
    }

    public static string[] GetKernelNames() => new[]
    {
        "gated_delta_scan_forward",
        "gated_delta_scan_recompute",
        "gated_delta_scan_backward"
    };
}
