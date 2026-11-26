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
    float kappa = keyScale;

template <int R>
__device__ __forceinline__ void gdn_forward(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ output, float* __restrict__ Straj,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)
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
        // decayFirst (the paper's rule): the prediction reads the decayed state a S_{t-1}.
        float bd = bet * (V[baseOff + di] - (decayFirst ? a : 1.0f) * sK);
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
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int det, float keyScale, int decayFirst)
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
        // pa scales the prediction: a under the paper's rule (a S_{t-1} k), 1 under the original one.
        float pa = decayFirst ? a : 1.0f;
        float delta = V[baseOff + di] - pa * sK;
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
        // The prediction a S_{t-1} k also depends on a: its adjoint -dDelta times the undecayed read.
        if (decayFirst) dA_acc -= dDelta * sK;
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
            float dKS = dS[j] * (bet * delta) + dSk * pa * sp[j];
            if (ki < headDim)
            {
                if (det) Straj[row + ki] = dKS * kappa;
                else atomicAdd(&dK[baseOff + ki], dKS * kappa);
            }
            dS[j] = dS[j] * a + dSk * pa * kk[j];
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
    int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)                  \
{ gdn_forward<R>(Q, K, V, A, B, output, nullptr, batch, seqLen, modelDim, numHeads, headDim, keyScale, decayFirst); } \
extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_recompute_r##R(                                 \
    const float* K, const float* V, const float* A, const float* B, float* Straj,                                   \
    int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)                  \
{ gdn_forward<R>(nullptr, K, V, A, B, nullptr, Straj, batch, seqLen, modelDim, numHeads, headDim, keyScale, decayFirst); } \
extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_backward_r##R(                                  \
    const float* dOut, const float* Q, const float* K, const float* V, const float* A, const float* B,              \
    float* Straj, float* dQ, float* dK, float* dV, float* dA, float* dB,                                            \
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int det, float keyScale, int decayFirst)          \
{ gdn_backward<R>(dOut, Q, K, V, A, B, Straj, dQ, dK, dV, dA, dB, batch, seqLen, modelDim, numHeads, headDim, det,    \
    keyScale, decayFirst); }                                                                                         \
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

template <int R, int CHUNK>
__device__ __forceinline__ void gdn_bwd_ckpt(
    const float* __restrict__ dOut, const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ Hckpt, float* __restrict__ dHckpt, float* __restrict__ dV,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)
{
    long w = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    long totalRows = (long)batch * numHeads * headDim;
    if (w >= 2 * totalRows) return;
    bool isReverse = (w >= totalRows);
    long rw = isReverse ? (w - totalRows) : w;
    int di = (int)(rw % headDim);
    int h  = (int)((rw / headDim) % numHeads);
    int b  = (int)(rw / ((long)headDim * numHeads));
    int hOff = h * headDim;
    long bh = (long)b * numHeads + h;
    int nChunks = (seqLen + CHUNK - 1) / CHUNK;
    long hh = (long)headDim * headDim;
    long rowOff = (long)di * headDim;

    if (!isReverse)
    {
        float S[R];
#pragma unroll
        for (int j = 0; j < R; j++) S[j] = 0.0f;
        for (int t = 0; t < seqLen; t++)
        {
            if (t % CHUNK == 0)
            {
                long ck = (bh * nChunks + (t / CHUNK)) * hh + rowOff;
#pragma unroll
                for (int j = 0; j < R; j++)
                {
                    int ki = lane + 32 * j;
                    if (ki < headDim) Hckpt[ck + ki] = S[j];
                }
            }
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            int gIdx = (b * seqLen + t) * numHeads + h;
            float a = A[gIdx], bet = B[gIdx];
            float kk[R];
            float sK = 0.0f;
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                kk[j] = ki < headDim ? K[baseOff + ki] * keyScale : 0.0f;
                sK += S[j] * kk[j];
            }
            sK = gdn_allsum(sK);
            float bd = bet * (V[baseOff + di] - (decayFirst ? a : 1.0f) * sK);
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                if (ki < headDim) S[j] = a * S[j] + bd * kk[j];
            }
        }
    }
    else
    {
        float dS[R];
#pragma unroll
        for (int j = 0; j < R; j++) dS[j] = 0.0f;
        for (int t = seqLen - 1; t >= 0; t--)
        {
            if (t == seqLen - 1 || ((t + 1) % CHUNK) == 0)
            {
                long ck = (bh * nChunks + (t / CHUNK)) * hh + rowOff;
#pragma unroll
                for (int j = 0; j < R; j++)
                {
                    int ki = lane + 32 * j;
                    if (ki < headDim) dHckpt[ck + ki] = dS[j];
                }
            }
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            int gIdx = (b * seqLen + t) * numHeads + h;
            float a = A[gIdx], bet = B[gIdx];
            float pa = decayFirst ? a : 1.0f;
            float dOutVal = dOut[baseOff + di];
            float kk[R];
            float dDp = 0.0f;
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                bool in = ki < headDim;
                kk[j] = in ? K[baseOff + ki] * keyScale : 0.0f;
                if (in) dS[j] += dOutVal * Q[baseOff + ki];
                dDp += dS[j] * bet * kk[j];
            }
            float dDelta = gdn_allsum(dDp);
            if (lane == 0) dV[baseOff + di] = dDelta;
            float dSk = -dDelta;
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                if (ki < headDim) dS[j] = dS[j] * a + dSk * pa * kk[j];
            }
        }
    }
}

template <int R, int CHUNK, int TILES>
__device__ __forceinline__ void gdn_bwd_chunk(
    const float* __restrict__ dOut, const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ A, const float* __restrict__ B,
    const float* __restrict__ Hckpt, const float* __restrict__ dHckpt,
    float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dA, float* __restrict__ dB,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)
{
    const int W = 32 * R;
    __shared__ float s_Sprev[(CHUNK - 1) * 32 * W];
    __shared__ float s_red[8 * W];
    __shared__ float s_ab[16];
    __shared__ float s_dQ[CHUNK * W];
    __shared__ float s_dK[CHUNK * W];
    __shared__ float s_dAB[CHUNK * 2];

    int nChunks = (seqLen + CHUNK - 1) / CHUNK;
    int bid = (int)blockIdx.x;
    if (bid >= batch * numHeads * nChunks) return;
    int c = bid % nChunks;
    int bh = bid / nChunks;
    int h = bh % numHeads;
    int b = bh / numHeads;
    int hOff = h * headDim;
    int t0 = c * CHUNK;
    int t1 = min(seqLen, t0 + CHUNK);
    int len = t1 - t0;
    int tid = (int)threadIdx.x;
    int warpId = tid >> 5;
    int lane = tid & 31;
    long hh = (long)headDim * headDim;
    long ckBase = ((long)bh * nChunks + c) * hh;

    for (int i = tid; i < CHUNK * W; i += 256) { s_dQ[i] = 0.0f; s_dK[i] = 0.0f; }
    for (int i = tid; i < CHUNK * 2; i += 256) { s_dAB[i] = 0.0f; }
    __syncthreads();

    for (int tile = 0; tile < TILES; tile++)
    {
        int tileRow0 = tile * 32;
        if (tileRow0 >= headDim) break;

        float S[4][R];
#pragma unroll
        for (int r = 0; r < 4; r++)
        {
            int di = tileRow0 + warpId * 4 + r;
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                S[r][j] = (di < headDim && ki < headDim) ? Hckpt[ckBase + (long)di * headDim + ki] : 0.0f;
            }
        }

        for (int lt = 0; lt < len; lt++)
        {
            int t = t0 + lt;
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            int gIdx = (b * seqLen + t) * numHeads + h;
            float a = A[gIdx], bet = B[gIdx];
            float kk[R];
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                kk[j] = ki < headDim ? K[baseOff + ki] * keyScale : 0.0f;
            }

            float dqPart[R];
#pragma unroll
            for (int j = 0; j < R; j++) dqPart[j] = 0.0f;

#pragma unroll
            for (int r = 0; r < 4; r++)
            {
                int localRow = warpId * 4 + r;
                int di = tileRow0 + localRow;
                if (di < headDim)
                {
                    if (lt > 0)
                    {
                        int spBase = ((lt - 1) * 32 + localRow) * W;
#pragma unroll
                        for (int j = 0; j < R; j++) s_Sprev[spBase + lane + 32 * j] = S[r][j];
                    }
                    float sK = 0.0f;
#pragma unroll
                    for (int j = 0; j < R; j++) sK += S[r][j] * kk[j];
                    sK = gdn_allsum(sK);
                    float bd = bet * (V[baseOff + di] - (decayFirst ? a : 1.0f) * sK);
                    float dOutVal = dOut[baseOff + di];
#pragma unroll
                    for (int j = 0; j < R; j++)
                    {
                        int ki = lane + 32 * j;
                        if (ki < headDim)
                        {
                            S[r][j] = a * S[r][j] + bd * kk[j];
                            dqPart[j] += dOutVal * S[r][j];
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < R; j++) s_red[warpId * W + lane + 32 * j] = dqPart[j];
            __syncthreads();
            if (warpId == 0)
            {
#pragma unroll
                for (int j = 0; j < R; j++)
                {
                    int col = lane + 32 * j;
                    float sum = 0.0f;
#pragma unroll
                    for (int w = 0; w < 8; w++) sum += s_red[w * W + col];
                    s_dQ[lt * W + col] += sum;
                }
            }
            __syncthreads();
        }

        float dS[4][R];
#pragma unroll
        for (int r = 0; r < 4; r++)
        {
            int di = tileRow0 + warpId * 4 + r;
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                dS[r][j] = (di < headDim && ki < headDim) ? dHckpt[ckBase + (long)di * headDim + ki] : 0.0f;
            }
        }

        for (int lt = len - 1; lt >= 0; lt--)
        {
            int t = t0 + lt;
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            int gIdx = (b * seqLen + t) * numHeads + h;
            float a = A[gIdx], bet = B[gIdx];
            float pa = decayFirst ? a : 1.0f;
            float kk[R], qq[R];
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                bool in = ki < headDim;
                kk[j] = in ? K[baseOff + ki] * keyScale : 0.0f;
                qq[j] = in ? Q[baseOff + ki] : 0.0f;
            }

            float dkPart[R];
#pragma unroll
            for (int j = 0; j < R; j++) dkPart[j] = 0.0f;
            float dAPart = 0.0f, dBPart = 0.0f;

#pragma unroll
            for (int r = 0; r < 4; r++)
            {
                int localRow = warpId * 4 + r;
                int di = tileRow0 + localRow;
                if (di < headDim)
                {
                    float sp[R];
                    float sK = 0.0f;
                    int spBase = ((lt - 1) * 32 + localRow) * W;
#pragma unroll
                    for (int j = 0; j < R; j++)
                    {
                        int ki = lane + 32 * j;
                        if (ki < headDim)
                            sp[j] = (lt > 0) ? s_Sprev[spBase + ki] : Hckpt[ckBase + (long)di * headDim + ki];
                        else
                            sp[j] = 0.0f;
                        sK += sp[j] * kk[j];
                    }
                    sK = gdn_allsum(sK);
                    float delta = V[baseOff + di] - pa * sK;
                    float dOutVal = dOut[baseOff + di];
                    float dAp = 0.0f, dBp = 0.0f, dDp = 0.0f;
#pragma unroll
                    for (int j = 0; j < R; j++)
                    {
                        int ki = lane + 32 * j;
                        if (ki < headDim) dS[r][j] += dOutVal * qq[j];
                        dAp += dS[r][j] * sp[j];
                        dBp += dS[r][j] * delta * kk[j];
                        dDp += dS[r][j] * bet * kk[j];
                    }
                    float dA_row = gdn_allsum(dAp);
                    float dB_row = gdn_allsum(dBp);
                    float dDelta = gdn_allsum(dDp);
                    if (decayFirst) dA_row -= dDelta * sK;
                    if (lane == 0)
                    {
                        dAPart += dA_row;
                        dBPart += dB_row;
                    }
                    float dSk = -dDelta;
#pragma unroll
                    for (int j = 0; j < R; j++)
                    {
                        int ki = lane + 32 * j;
                        if (ki < headDim)
                        {
                            float dKS = dS[r][j] * (bet * delta) + dSk * pa * sp[j];
                            dkPart[j] += dKS * keyScale;
                            dS[r][j] = dS[r][j] * a + dSk * pa * kk[j];
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < R; j++) s_red[warpId * W + lane + 32 * j] = dkPart[j];
            if (lane == 0) { s_ab[warpId * 2] = dAPart; s_ab[warpId * 2 + 1] = dBPart; }
            __syncthreads();
            if (warpId == 0)
            {
#pragma unroll
                for (int j = 0; j < R; j++)
                {
                    int col = lane + 32 * j;
                    float sum = 0.0f;
#pragma unroll
                    for (int w = 0; w < 8; w++) sum += s_red[w * W + col];
                    s_dK[lt * W + col] += sum;
                }
                if (lane == 0)
                {
                    float sa = 0.0f, sb = 0.0f;
#pragma unroll
                    for (int w = 0; w < 8; w++) { sa += s_ab[w * 2]; sb += s_ab[w * 2 + 1]; }
                    s_dAB[lt * 2] += sa;
                    s_dAB[lt * 2 + 1] += sb;
                }
            }
            __syncthreads();
        }
    }

    for (int lt = warpId; lt < len; lt += 8)
    {
        int t = t0 + lt;
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
#pragma unroll
        for (int j = 0; j < R; j++)
        {
            int ki = lane + 32 * j;
            if (ki < headDim)
            {
                dQ[baseOff + ki] = s_dQ[lt * W + ki];
                dK[baseOff + ki] = s_dK[lt * W + ki];
            }
        }
        if (lane == 0)
        {
            dA[gIdx] = s_dAB[lt * 2];
            dB[gIdx] = s_dAB[lt * 2 + 1];
        }
    }
}

extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_bwd_ckpt_r1(
    const float* dOut, const float* Q, const float* K, const float* V, const float* A, const float* B,
    float* Hckpt, float* dHckpt, float* dV, int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)
{ gdn_bwd_ckpt<1, 8>(dOut, Q, K, V, A, B, Hckpt, dHckpt, dV, batch, seqLen, modelDim, numHeads, headDim, keyScale, decayFirst); }

extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_bwd_chunk_r1(
    const float* dOut, const float* Q, const float* K, const float* V, const float* A, const float* B,
    const float* Hckpt, const float* dHckpt, float* dQ, float* dK, float* dA, float* dB,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)
{ gdn_bwd_chunk<1, 8, 1>(dOut, Q, K, V, A, B, Hckpt, dHckpt, dQ, dK, dA, dB, batch, seqLen, modelDim, numHeads, headDim, keyScale, decayFirst); }

extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_bwd_ckpt_r2(
    const float* dOut, const float* Q, const float* K, const float* V, const float* A, const float* B,
    float* Hckpt, float* dHckpt, float* dV, int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)
{ gdn_bwd_ckpt<2, 6>(dOut, Q, K, V, A, B, Hckpt, dHckpt, dV, batch, seqLen, modelDim, numHeads, headDim, keyScale, decayFirst); }

extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_bwd_chunk_r2(
    const float* dOut, const float* Q, const float* K, const float* V, const float* A, const float* B,
    const float* Hckpt, const float* dHckpt, float* dQ, float* dK, float* dA, float* dB,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)
{ gdn_bwd_chunk<2, 6, 2>(dOut, Q, K, V, A, B, Hckpt, dHckpt, dQ, dK, dA, dB, batch, seqLen, modelDim, numHeads, headDim, keyScale, decayFirst); }
";
    }

    public static string[] GetKernelNames()
    {
        var names = new System.Collections.Generic.List<string>();
        foreach (var r in new[] { "_r1", "_r2", "_r4", "_r8" })
            foreach (var kernel in new[] { "forward", "recompute", "backward", "dq", "reduce" })
                names.Add("gated_delta_scan_" + kernel + r);
        names.Add("gated_delta_scan_bwd_ckpt_r1");
        names.Add("gated_delta_scan_bwd_chunk_r1");
        names.Add("gated_delta_scan_bwd_ckpt_r2");
        names.Add("gated_delta_scan_bwd_chunk_r2");
        return names.ToArray();
    }
}
