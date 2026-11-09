// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CUDA kernels for the fused Gated DeltaNet scan (forward + recompute BPTT backward, issue ooples/AiDotNet#1464).
// Same design as HipGatedDeltaNetKernels: each value row di is independent in both forward and BPTT backward,
// so one thread per (batch, head, value-row) carries that state/adjoint row in registers across time.
// The BPTT trajectory is stored [b, h, t, ki, di] (value row innermost): the threads of a warp are consecutive di, so
// each of their per-ki reads and writes is one coalesced transaction. Stored [.., di, ki] they were headDim floats
// apart (one transaction per thread), and recompute + backward were 26% of an LM training step's GPU time.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

internal static class CudaGatedDeltaNetKernels
{
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#include <math.h>

#define GDN_MAX_HEADDIM 256

extern ""C"" __global__ void gated_delta_scan_forward(
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

extern ""C"" __global__ void gated_delta_scan_recompute(
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

    float Srow[GDN_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        float sK = 0.0f;
        for (int ki = 0; ki < headDim; ki++) sK += Srow[ki] * (K[baseOff + ki] * kappa);
        float bd = bet * (V[baseOff + di] - sK);
        long trajBase = ((long)(b * numHeads + h) * seqLen + t) * hh + di;
        for (int ki = 0; ki < headDim; ki++) {
            float s = a * Srow[ki] + bd * (K[baseOff + ki] * kappa);
            Srow[ki] = s;
            Straj[trajBase + (long)ki * headDim] = s;
        }
    }
}

// det == 0: dQ/dK/dA/dB accumulate across the head's rows with atomicAdd (unordered).
// det != 0: nothing is accumulated here. dQ comes from gated_delta_scan_dq (run before this kernel), dQ's slot carries
// the per-row dA/dB partials ([dA | dB], batch*seqLen*modelDim floats each), and each row's dK contribution overwrites
// that row of Straj, which this thread alone reads and has finished with (row t was last read at step t + 1).
// gated_delta_scan_reduce then sums the rows in a fixed order.
extern ""C"" __global__ void gated_delta_scan_backward(
    const float* dOut, const float* Q, const float* K, const float* V, const float* A, const float* B,
    float* Straj,
    float* dQ, float* dK, float* dV, float* dA, float* dB,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int det)
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

    float dSrow[GDN_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) dSrow[ki] = 0.0f;

    for (int t = seqLen - 1; t >= 0; t--) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        long trajBase = ((long)(b * numHeads + h) * seqLen + t) * hh + di;
        long sprevBase = (t > 0) ? (((long)(b * numHeads + h) * seqLen + (t - 1)) * hh + di) : 0L;

        float sK = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float sprev = (t > 0) ? Straj[sprevBase + (long)ki * headDim] : 0.0f;
            sK += sprev * (K[baseOff + ki] * kappa);
        }
        float delta = V[baseOff + di] - sK;

        float dOutVal = dOut[baseOff + di];
        for (int ki = 0; ki < headDim; ki++) {
            if (!det) atomicAdd(&dQ[baseOff + ki], dOutVal * Straj[trajBase + (long)ki * headDim]);
            dSrow[ki] += dOutVal * Q[baseOff + ki];
        }

        float dA_acc = 0.0f;
        float dB_acc = 0.0f;
        float dDelta = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + (long)ki * headDim] : 0.0f;
            float kScaled = K[baseOff + ki] * kappa;
            dA_acc += dSt * sprev;
            dB_acc += dSt * delta * kScaled;
            dDelta += dSt * bet * kScaled;
        }
        dV[baseOff + di] = dDelta;
        if (det) {
            int n = batch * seqLen * modelDim;
            dQ[gIdx * headDim + di] = dA_acc;
            dQ[n + gIdx * headDim + di] = dB_acc;
        } else {
            atomicAdd(&dA[gIdx], dA_acc);
            atomicAdd(&dB[gIdx], dB_acc);
        }

        float dSk = -dDelta;
        for (int ki = 0; ki < headDim; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + (long)ki * headDim] : 0.0f;
            float kScaled = K[baseOff + ki] * kappa;
            float dKS_ki = dSt * (bet * delta) + dSk * sprev;
            if (det) Straj[trajBase + (long)ki * headDim] = dKS_ki * kappa;
            else atomicAdd(&dK[baseOff + ki], dKS_ki * kappa);
            dSrow[ki] = dSt * a + dSk * kScaled;
        }
    }
}

// Deterministic dQ: dQ[b,t,h,ki] = sum over di of dOut[b,t,h,di] * S_t[ki,di]. One warp per dQ element: lane l sums
// di = l, l+32, ... (consecutive lanes read consecutive di of the [ki, di] trajectory row, one transaction), then a
// fixed shuffle tree combines the lanes, so the order is the same on every run. Launched with gridDim*blockDim >= 32*n
// and blockDim a multiple of 32, so a warp leaves together.
__device__ __forceinline__ float gdn_warp_sum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

extern ""C"" __global__ void gated_delta_scan_dq(
    const float* dOut, const float* Straj, float* dQ,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long)batch * seqLen * modelDim) return;
    int gid = (int)gw;
    int ki = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int bt = gid / modelDim;
    int t = bt % seqLen, b = bt / seqLen;
    long row = ((long)(b * numHeads + h) * seqLen + t) * (long)(headDim * headDim) + (long)ki * headDim;
    int base = bt * modelDim + h * headDim;
    float acc = 0.0f;
    for (int di = lane; di < headDim; di += 32) acc += dOut[base + di] * Straj[row + di];
    acc = gdn_warp_sum(acc);
    if (lane == 0) dQ[gid] = acc;
}

// Deterministic reduction of the backward's per-row partials: dK from the rows of PK (the overwritten trajectory) and
// dA/dB from P = [dA | dB], each summed over di by one warp per dK element (same lane order and shuffle tree as dq).
extern ""C"" __global__ void gated_delta_scan_reduce(
    const float* PK, const float* P, float* dK, float* dA, float* dB,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int n = batch * seqLen * modelDim;
    if (gw >= n) return;
    int gid = (int)gw;
    int ki = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int bt = gid / modelDim;
    int t = bt % seqLen, b = bt / seqLen;
    long row = ((long)(b * numHeads + h) * seqLen + t) * (long)(headDim * headDim) + (long)ki * headDim;
    float acc = 0.0f;
    for (int di = lane; di < headDim; di += 32) acc += PK[row + di];
    acc = gdn_warp_sum(acc);
    if (lane == 0) dK[gid] = acc;
    if (ki == 0) {
        int gIdx = bt * numHeads + h;
        float sa = 0.0f, sb = 0.0f;
        for (int di = lane; di < headDim; di += 32) {
            sa += P[gIdx * headDim + di];
            sb += P[n + gIdx * headDim + di];
        }
        sa = gdn_warp_sum(sa);
        sb = gdn_warp_sum(sb);
        if (lane == 0) { dA[gIdx] = sa; dB[gIdx] = sb; }
    }
}
";
    }

    public static string[] GetKernelNames() => new[]
    {
        "gated_delta_scan_forward",
        "gated_delta_scan_recompute",
        "gated_delta_scan_backward",
        "gated_delta_scan_dq",
        "gated_delta_scan_reduce"
    };
}
