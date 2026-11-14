// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CUDA kernels for the fused Gated DeltaNet scan (forward + recompute BPTT backward, issue ooples/AiDotNet#1464).
// Each value row di of a (batch, head) state is independent in both forward and BPTT backward, so one WARP per
// (batch, head, value-row) carries that state/adjoint row across time: lane l holds entries ki = l, l + 32, ... in
// registers (R = ceil(headDim / 32) of them, a compile-time size), and the per-step dot products over ki are fixed
// shuffle trees, the same order on every run. A thread per row (the previous design) launched batch*heads*headDim
// threads -- 8 blocks for an 8 x 4 x 64 layer on a 68-SM GPU -- and kept a 256-float row in local memory: recompute +
// backward took 26% of an LM training step's GPU time.
// The BPTT trajectory is stored [b, h, t, di, ki] (ki innermost): a warp's lanes read and write consecutive ki of one
// row, one coalesced transaction.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

internal static class CudaGatedDeltaNetKernels
{
    public const int MaxHeadDim = 256;

    /// <summary>The kernel-name suffix for a head size: the register entries per lane, 1, 2, 4 or 8.</summary>
    internal static string Variant(int headDim) => headDim <= 32 ? "_r1" : headDim <= 64 ? "_r2" : headDim <= 128 ? "_r4" : "_r8";

    public static string GetSource()
    {
        return @"
#include <math.h>

#define GDN_FULL 0xffffffffu

// The sum over the warp, in every lane: a fixed xor tree.
__device__ __forceinline__ float gdn_allsum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(GDN_FULL, v, o);
    return v;
}

// Warp w of the launch is state row w = (b * numHeads + h) * headDim + di. The whole warp shares w, so it leaves (or
// stays) together and every shuffle has all 32 lanes.
#define GDN_ROW_PROLOGUE                                                                         \
    long w = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;                                 \
    int lane = threadIdx.x & 31;                                                                 \
    if (w >= (long)batch * numHeads * headDim) return;                                           \
    int di = (int)(w % headDim);                                                                 \
    int h = (int)((w / headDim) % numHeads);                                                     \
    int b = (int)(w / ((long)headDim * numHeads));                                               \
    int hOff = h * headDim;                                                                      \
    long bh = (long)b * numHeads + h;                                                            \
    float kappa = 1.0f / sqrtf((float)headDim);

template <int R>
__device__ __forceinline__ void gdn_forward(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ output, float* __restrict__ Straj,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    GDN_ROW_PROLOGUE
    float S[R];
#pragma unroll
    for (int j = 0; j < R; j++) S[j] = 0.0f;
    for (int t = 0; t < seqLen; t++)
    {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        float kk[R];
        float sK = 0.0f;
#pragma unroll
        for (int j = 0; j < R; j++)
        {
            int ki = lane + 32 * j;
            kk[j] = ki < headDim ? K[baseOff + ki] * kappa : 0.0f;
            sK += S[j] * kk[j];
        }
        sK = gdn_allsum(sK);
        float bd = bet * (V[baseOff + di] - sK);
        float o = 0.0f;
        long row = ((bh * seqLen + t) * headDim + di) * headDim;
#pragma unroll
        for (int j = 0; j < R; j++)
        {
            int ki = lane + 32 * j;
            S[j] = a * S[j] + bd * kk[j];
            if (ki < headDim)
            {
                if (Straj) Straj[row + ki] = S[j];
                else o += S[j] * Q[baseOff + ki];
            }
        }
        if (!Straj)
        {
            o = gdn_allsum(o);
            if (lane == 0) output[baseOff + di] = o;
        }
    }
}

// det == 0: dQ/dK/dA/dB accumulate across the head's rows with atomicAdd (unordered).
// det != 0: nothing is accumulated here. dQ comes from gated_delta_scan_dq (run before this kernel), dQ's slot carries
// the per-row dA/dB partials ([dA | dB], batch*seqLen*modelDim floats each), and each row's dK contribution overwrites
// that row of Straj, which only this warp reads and has finished with (row t was last read at step t + 1).
// gated_delta_scan_reduce then sums the rows in a fixed order.
template <int R>
__device__ __forceinline__ void gdn_backward(
    const float* __restrict__ dOut, const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ A, const float* __restrict__ B, float* Straj,
    float* dQ, float* dK, float* __restrict__ dV, float* dA, float* dB,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int det)
{
    GDN_ROW_PROLOGUE
    float dS[R];
#pragma unroll
    for (int j = 0; j < R; j++) dS[j] = 0.0f;
    int n = batch * seqLen * modelDim;
    for (int t = seqLen - 1; t >= 0; t--)
    {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        long row = ((bh * seqLen + t) * headDim + di) * headDim;
        long prev = row - (long)headDim * headDim;   // row t - 1, same di
        float kk[R], sp[R];
        float sK = 0.0f;
#pragma unroll
        for (int j = 0; j < R; j++)
        {
            int ki = lane + 32 * j;
            bool in = ki < headDim;
            kk[j] = in ? K[baseOff + ki] * kappa : 0.0f;
            sp[j] = in && t > 0 ? Straj[prev + ki] : 0.0f;
            sK += sp[j] * kk[j];
        }
        sK = gdn_allsum(sK);
        float delta = V[baseOff + di] - sK;
        float dOutVal = dOut[baseOff + di];

        float dAp = 0.0f, dBp = 0.0f, dDp = 0.0f;
#pragma unroll
        for (int j = 0; j < R; j++)
        {
            int ki = lane + 32 * j;
            if (ki < headDim)
            {
                if (!det) atomicAdd(&dQ[baseOff + ki], dOutVal * Straj[row + ki]);
                dS[j] += dOutVal * Q[baseOff + ki];
            }
            dAp += dS[j] * sp[j];
            dBp += dS[j] * delta * kk[j];
            dDp += dS[j] * bet * kk[j];
        }
        float dA_acc = gdn_allsum(dAp);
        float dB_acc = gdn_allsum(dBp);
        float dDelta = gdn_allsum(dDp);
        if (lane == 0)
        {
            dV[baseOff + di] = dDelta;
            if (det)
            {
                dQ[gIdx * headDim + di] = dA_acc;
                dQ[n + gIdx * headDim + di] = dB_acc;
            }
            else
            {
                atomicAdd(&dA[gIdx], dA_acc);
                atomicAdd(&dB[gIdx], dB_acc);
            }
        }

        float dSk = -dDelta;
#pragma unroll
        for (int j = 0; j < R; j++)
        {
            int ki = lane + 32 * j;
            float dKS = dS[j] * (bet * delta) + dSk * sp[j];
            if (ki < headDim)
            {
                if (det) Straj[row + ki] = dKS * kappa;
                else atomicAdd(&dK[baseOff + ki], dKS * kappa);
            }
            dS[j] = dS[j] * a + dSk * kk[j];
        }
    }
}

// Deterministic dQ: dQ[b,t,h,ki] = sum over di of dOut[b,t,h,di] * S_t[di,ki], in increasing di. One warp per
// (b, t, h): lane l owns ki = l, l + 32, ..., so each di step reads one coalesced trajectory row.
template <int R>
__device__ __forceinline__ void gdn_dq(
    const float* __restrict__ dOut, const float* __restrict__ Straj, float* __restrict__ dQ,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    long w = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= (long)batch * seqLen * numHeads) return;
    int h = (int)(w % numHeads);
    int bt = (int)(w / numHeads);
    int t = bt % seqLen, b = bt / seqLen;
    int base = bt * modelDim + h * headDim;
    long traj = (((long)b * numHeads + h) * seqLen + t) * (long)headDim * headDim;
    float acc[R];
#pragma unroll
    for (int j = 0; j < R; j++) acc[j] = 0.0f;
    for (int di = 0; di < headDim; di++)
    {
        float d = dOut[base + di];
        long row = traj + (long)di * headDim;
#pragma unroll
        for (int j = 0; j < R; j++)
        {
            int ki = lane + 32 * j;
            if (ki < headDim) acc[j] += d * Straj[row + ki];
        }
    }
#pragma unroll
    for (int j = 0; j < R; j++)
    {
        int ki = lane + 32 * j;
        if (ki < headDim) dQ[base + ki] = acc[j];
    }
}

// Deterministic reduction of the backward's per-row partials: dK[b,t,h,ki] = the sum over di of the rows of PK (the
// overwritten trajectory), in increasing di; dA/dB from P = [dA | dB], each summed over di by the fixed shuffle tree.
template <int R>
__device__ __forceinline__ void gdn_reduce(
    const float* __restrict__ PK, const float* __restrict__ P, float* __restrict__ dK, float* __restrict__ dA,
    float* __restrict__ dB, int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    long w = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= (long)batch * seqLen * numHeads) return;
    int h = (int)(w % numHeads);
    int bt = (int)(w / numHeads);
    int t = bt % seqLen, b = bt / seqLen;
    int base = bt * modelDim + h * headDim;
    long traj = (((long)b * numHeads + h) * seqLen + t) * (long)headDim * headDim;
    float acc[R];
#pragma unroll
    for (int j = 0; j < R; j++) acc[j] = 0.0f;
    for (int di = 0; di < headDim; di++)
    {
        long row = traj + (long)di * headDim;
#pragma unroll
        for (int j = 0; j < R; j++)
        {
            int ki = lane + 32 * j;
            if (ki < headDim) acc[j] += PK[row + ki];
        }
    }
#pragma unroll
    for (int j = 0; j < R; j++)
    {
        int ki = lane + 32 * j;
        if (ki < headDim) dK[base + ki] = acc[j];
    }
    int n = batch * seqLen * modelDim;
    int gIdx = bt * numHeads + h;
    float sa = 0.0f, sb = 0.0f;
    for (int di = lane; di < headDim; di += 32)
    {
        sa += P[gIdx * headDim + di];
        sb += P[n + gIdx * headDim + di];
    }
    sa = gdn_allsum(sa);
    sb = gdn_allsum(sb);
    if (lane == 0) { dA[gIdx] = sa; dB[gIdx] = sb; }
}

#define GDN_VARIANTS(R)                                                                                              \
extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_forward_r##R(                                   \
    const float* Q, const float* K, const float* V, const float* A, const float* B, float* output,                   \
    int batch, int seqLen, int modelDim, int numHeads, int headDim)                                                  \
{ gdn_forward<R>(Q, K, V, A, B, output, nullptr, batch, seqLen, modelDim, numHeads, headDim); }                     \
extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_recompute_r##R(                                 \
    const float* K, const float* V, const float* A, const float* B, float* Straj,                                   \
    int batch, int seqLen, int modelDim, int numHeads, int headDim)                                                  \
{ gdn_forward<R>(nullptr, K, V, A, B, nullptr, Straj, batch, seqLen, modelDim, numHeads, headDim); }                \
extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_backward_r##R(                                  \
    const float* dOut, const float* Q, const float* K, const float* V, const float* A, const float* B,              \
    float* Straj, float* dQ, float* dK, float* dV, float* dA, float* dB,                                            \
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int det)                                         \
{ gdn_backward<R>(dOut, Q, K, V, A, B, Straj, dQ, dK, dV, dA, dB, batch, seqLen, modelDim, numHeads, headDim, det); } \
extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_dq_r##R(                                        \
    const float* dOut, const float* Straj, float* dQ, int batch, int seqLen, int modelDim, int numHeads, int headDim) \
{ gdn_dq<R>(dOut, Straj, dQ, batch, seqLen, modelDim, numHeads, headDim); }                                         \
extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_reduce_r##R(                                    \
    const float* PK, const float* P, float* dK, float* dA, float* dB,                                               \
    int batch, int seqLen, int modelDim, int numHeads, int headDim)                                                  \
{ gdn_reduce<R>(PK, P, dK, dA, dB, batch, seqLen, modelDim, numHeads, headDim); }

GDN_VARIANTS(1)
GDN_VARIANTS(2)
GDN_VARIANTS(4)
GDN_VARIANTS(8)
";
    }

    public static string[] GetKernelNames()
    {
        var names = new System.Collections.Generic.List<string>();
        foreach (var r in new[] { "_r1", "_r2", "_r4", "_r8" })
            foreach (var kernel in new[] { "forward", "recompute", "backward", "dq", "reduce" })
                names.Add("gated_delta_scan_" + kernel + r);
        return names.ToArray();
    }
}
