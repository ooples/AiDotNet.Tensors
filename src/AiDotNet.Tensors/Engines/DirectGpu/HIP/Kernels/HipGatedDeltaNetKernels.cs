// Copyright (c) CrowdTrainer LLC. All rights reserved.
// HIP kernels for the fused Gated DeltaNet (delta-rule) scan forward + BPTT backward (issue ooples/AiDotNet#1464).
// Mirrors CpuEngine.GatedDeltaNetScan. Each value row is independent in both forward and backward, so one thread
// owns (batch b, head h, value-row di) and carries that row of the state/adjoint across time.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipGatedDeltaNetKernels
{
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>

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

    if (headDim < 0 || headDim > GDN_MAX_HEADDIM) return;
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

extern ""C"" __global__ __launch_bounds__(1024) void gated_delta_scan_recompute(
    const float* K, const float* V, const float* A, const float* B,
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
    float kappa = 1.0f / sqrtf((float)headDim);

    if (headDim < 0 || headDim > GDN_MAX_HEADDIM) return;
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

extern ""C"" __global__ __launch_bounds__(1024) void gated_delta_scan_backward(
    const float* dOut, const float* Q, const float* K, const float* V, const float* A, const float* B,
    const float* Straj,
    float* dQ, float* dK, float* dV, float* dA, float* dB,
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
    float kappa = 1.0f / sqrtf((float)headDim);

    if (headDim < 0 || headDim > GDN_MAX_HEADDIM) return;
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
            atomicAdd(&dQ[baseOff + ki], dOutVal * sVal);
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
        atomicAdd(&dA[gIdx], dA_acc);
        atomicAdd(&dB[gIdx], dB_acc);

        float dSk = -dDelta;
        for (int ki = 0; ki < headDim; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            float kScaled = K[baseOff + ki] * kappa;
            float dKS_ki = dSt * (bet * delta) + dSk * sprev;
            atomicAdd(&dK[baseOff + ki], dKS_ki * kappa);
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
