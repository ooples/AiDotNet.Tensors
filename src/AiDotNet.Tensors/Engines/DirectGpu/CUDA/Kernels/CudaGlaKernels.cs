// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
    internal const int TileThreads = 1024;
    internal const int TileElementsPerThread = 12;
    internal const int TileWarpSize = 32;
    internal const int TileWarpPartials = TileThreads / TileWarpSize;
    internal const int TileSharedMemoryLimitBytes = 48 * 1024;

    /// <summary>The kernel-name suffix for a head size: register entries per warp lane (1, 2, 4 or 8).</summary>
    internal static string Variant(int headDim) => headDim <= 32 ? "_r1" : headDim <= 64 ? "_r2" : headDim <= 128 ? "_r4" : "_r8";

    public static string GetSource()
    {
        return "#define GLA_TILE_THREADS " + TileThreads + "\n"
            + "#define GLA_TILE_EPT " + TileElementsPerThread + "\n"
            + "#define GLA_TILE_WARP_SIZE " + TileWarpSize + "\n"
            + @"
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

#define GLA_WARP_FULL 0xffffffffu

__device__ __forceinline__ float gla_allsum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(GLA_WARP_FULL, v, o);
    return v;
}

template <int R>
__device__ __forceinline__ void gla_forward_warp(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, const float* __restrict__ G,
    float* __restrict__ output,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    long w = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= (long)batch * numHeads * headDim) return;
    int di = (int)(w % headDim);
    int h  = (int)((w / headDim) % numHeads);
    int b  = (int)(w / ((long)headDim * numHeads));
    int hOff = h * headDim;

    float S[R];
#pragma unroll
    for (int j = 0; j < R; j++) S[j] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        float g  = G[(b * seqLen + t) * numHeads + h];
        float vd = V[baseOff + di];
        float o  = 0.0f;
#pragma unroll
        for (int j = 0; j < R; j++) {
            int ki = lane + 32 * j;
            if (ki < headDim) {
                float s = g * S[j] + vd * K[baseOff + ki];
                S[j] = s;
                o += s * Q[baseOff + ki];
            }
        }
        o = gla_allsum(o);
        if (lane == 0) output[baseOff + di] = o;
    }
}

template <int R>
__device__ __forceinline__ void gla_backward_warp(
    const float* __restrict__ dOut, const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ G,
    float* __restrict__ Ckpt, float* __restrict__ Scratch,
    float* dQ, float* dK, float* __restrict__ dV, float* dG,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int chunk)
{
    long w = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= (long)batch * numHeads * headDim) return;
    int di = (int)(w % headDim);
    int h  = (int)((w / headDim) % numHeads);
    int b  = (int)(w / ((long)headDim * numHeads));
    int hOff = h * headDim;
    long hh = (long)headDim * headDim;
    long bh = (long)b * numHeads + h;
    long rowOff = (long)di * headDim;
    int nChunks = (seqLen + chunk - 1) / chunk;

    float S[R], dS[R];
#pragma unroll
    for (int j = 0; j < R; j++) { S[j] = 0.0f; dS[j] = 0.0f; }

    for (int t = 0; t < seqLen; t++) {
        if (t % chunk == 0) {
            long ck = (bh * nChunks + t / chunk) * hh + rowOff;
#pragma unroll
            for (int j = 0; j < R; j++) {
                int ki = lane + 32 * j;
                if (ki < headDim) Ckpt[ck + ki] = S[j];
            }
        }
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        float g  = G[(b * seqLen + t) * numHeads + h];
        float vd = V[baseOff + di];
#pragma unroll
        for (int j = 0; j < R; j++) {
            int ki = lane + 32 * j;
            if (ki < headDim) S[j] = g * S[j] + vd * K[baseOff + ki];
        }
    }

    for (int c = nChunks - 1; c >= 0; c--) {
        int t0 = c * chunk;
        int t1 = min(seqLen, t0 + chunk);
        long ck = (bh * nChunks + c) * hh + rowOff;
#pragma unroll
        for (int j = 0; j < R; j++) {
            int ki = lane + 32 * j;
            S[j] = ki < headDim ? Ckpt[ck + ki] : 0.0f;
        }
        for (int t = t0; t < t1; t++) {
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            float g  = G[(b * seqLen + t) * numHeads + h];
            float vd = V[baseOff + di];
            long sc = (bh * chunk + (t - t0)) * hh + rowOff;
#pragma unroll
            for (int j = 0; j < R; j++) {
                int ki = lane + 32 * j;
                if (ki < headDim) {
                    float s = g * S[j] + vd * K[baseOff + ki];
                    S[j] = s;
                    if (t + 1 < t1) Scratch[sc + ki] = s;
                }
            }
        }
        for (int t = t1 - 1; t >= t0; t--) {
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            int gOff = (b * seqLen + t) * numHeads + h;
            float g = G[gOff];
            const float* sprevRow = (t > t0) ? (Scratch + (bh * chunk + (t - 1 - t0)) * hh + rowOff)
                                             : (Ckpt + ck);
            float dOutVal = dOut[baseOff + di];
            float vd = V[baseOff + di];
            float dg = 0.0f;
            float dVacc = 0.0f;
#pragma unroll
            for (int j = 0; j < R; j++) {
                int ki = lane + 32 * j;
                if (ki < headDim) {
                    if (dOutVal != 0.0f) {
                        atomicAdd(&dQ[baseOff + ki], dOutVal * S[j]);
                        dS[j] += dOutVal * Q[baseOff + ki];
                    }
                    float sp = sprevRow[ki];
                    S[j] = sp;
                    float dStv = dS[j];
                    dg += dStv * sp;
                    if (vd != 0.0f && dStv != 0.0f) atomicAdd(&dK[baseOff + ki], dStv * vd);
                    dVacc += dStv * K[baseOff + ki];
                    dS[j] = dStv * g;
                }
            }
            dg = gla_allsum(dg);
            dVacc = gla_allsum(dVacc);
            if (lane == 0) {
                dV[baseOff + di] = dVacc;
                if (dg != 0.0f) atomicAdd(&dG[gOff], dg);
            }
        }
    }
}

#define GLA_WARP_KERNELS(SFX, R) \
extern ""C"" __global__ void gla_scan_forward##SFX( \
    const float* Q, const float* K, const float* V, const float* G, float* output, \
    int batch, int seqLen, int modelDim, int numHeads, int headDim) \
{ gla_forward_warp<R>(Q, K, V, G, output, batch, seqLen, modelDim, numHeads, headDim); } \
extern ""C"" __global__ void gla_scan_backward_warp##SFX( \
    const float* dOut, const float* Q, const float* K, const float* V, const float* G, \
    float* Ckpt, float* Scratch, float* dQ, float* dK, float* dV, float* dG, \
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int chunk) \
{ gla_backward_warp<R>(dOut, Q, K, V, G, Ckpt, Scratch, dQ, dK, dV, dG, batch, seqLen, modelDim, numHeads, headDim, chunk); }

GLA_WARP_KERNELS(_r1, 1)
GLA_WARP_KERNELS(_r2, 2)
GLA_WARP_KERNELS(_r4, 4)
GLA_WARP_KERNELS(_r8, 8)

template <int R, int CHUNK>
__device__ __forceinline__ void gla_bwd_ckpt(
    const float* __restrict__ dOut, const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ G,
    float* __restrict__ Hckpt, float* __restrict__ dHckpt, float* __restrict__ dV,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
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
            float g = G[(b * seqLen + t) * numHeads + h];
            float vd = V[baseOff + di];
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                if (ki < headDim) S[j] = g * S[j] + vd * K[baseOff + ki];
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
            float g = G[(b * seqLen + t) * numHeads + h];
            float dOutVal = dOut[baseOff + di];
            float dVacc = 0.0f;
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                if (ki < headDim)
                {
                    dS[j] += dOutVal * Q[baseOff + ki];
                    dVacc += dS[j] * K[baseOff + ki];
                    dS[j] *= g;
                }
            }
            dVacc = gla_allsum(dVacc);
            if (lane == 0) dV[baseOff + di] = dVacc;
        }
    }
}

template <int R, int CHUNK, int TILES>
__device__ __forceinline__ void gla_bwd_chunk(
    const float* __restrict__ dOut, const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ G,
    const float* __restrict__ Hckpt, const float* __restrict__ dHckpt,
    float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dG,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    const int W = 32 * R;
    __shared__ float s_Sprev[(CHUNK - 1) * 32 * W];
    __shared__ float s_red[8 * W];
    __shared__ float s_g[8];
    __shared__ float s_dQ[CHUNK * W];
    __shared__ float s_dK[CHUNK * W];
    __shared__ float s_dG[CHUNK];

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
    for (int i = tid; i < CHUNK; i += 256) { s_dG[i] = 0.0f; }
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
            float g = G[(b * seqLen + t) * numHeads + h];
            float kk[R];
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                kk[j] = ki < headDim ? K[baseOff + ki] : 0.0f;
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
                    float vd = V[baseOff + di];
                    float dOutVal = dOut[baseOff + di];
#pragma unroll
                    for (int j = 0; j < R; j++)
                    {
                        int ki = lane + 32 * j;
                        if (ki < headDim)
                        {
                            S[r][j] = g * S[r][j] + vd * kk[j];
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
            float g = G[(b * seqLen + t) * numHeads + h];
            float qq[R];
#pragma unroll
            for (int j = 0; j < R; j++)
            {
                int ki = lane + 32 * j;
                qq[j] = ki < headDim ? Q[baseOff + ki] : 0.0f;
            }

            float dkPart[R];
#pragma unroll
            for (int j = 0; j < R; j++) dkPart[j] = 0.0f;
            float dgPart = 0.0f;

#pragma unroll
            for (int r = 0; r < 4; r++)
            {
                int localRow = warpId * 4 + r;
                int di = tileRow0 + localRow;
                if (di < headDim)
                {
                    float vd = V[baseOff + di];
                    float dOutVal = dOut[baseOff + di];
                    int spBase = ((lt - 1) * 32 + localRow) * W;
                    float dgRow = 0.0f;
#pragma unroll
                    for (int j = 0; j < R; j++)
                    {
                        int ki = lane + 32 * j;
                        if (ki < headDim)
                        {
                            float sp = (lt > 0) ? s_Sprev[spBase + ki] : Hckpt[ckBase + (long)di * headDim + ki];
                            dS[r][j] += dOutVal * qq[j];
                            float dStv = dS[r][j];
                            dgRow += dStv * sp;
                            dkPart[j] += dStv * vd;
                            dS[r][j] = dStv * g;
                        }
                    }
                    dgRow = gla_allsum(dgRow);
                    if (lane == 0) dgPart += dgRow;
                }
            }

#pragma unroll
            for (int j = 0; j < R; j++) s_red[warpId * W + lane + 32 * j] = dkPart[j];
            if (lane == 0) s_g[warpId] = dgPart;
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
                    float sg = 0.0f;
#pragma unroll
                    for (int w = 0; w < 8; w++) sg += s_g[w];
                    s_dG[lt] += sg;
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
        if (lane == 0) dG[gIdx] = s_dG[lt];
    }
}

extern ""C"" __global__ __launch_bounds__(256) void gla_scan_bwd_ckpt_r1(
    const float* dOut, const float* Q, const float* K, const float* V, const float* G,
    float* Hckpt, float* dHckpt, float* dV, int batch, int seqLen, int modelDim, int numHeads, int headDim)
{ gla_bwd_ckpt<1, 8>(dOut, Q, K, V, G, Hckpt, dHckpt, dV, batch, seqLen, modelDim, numHeads, headDim); }

extern ""C"" __global__ __launch_bounds__(256) void gla_scan_bwd_chunk_r1(
    const float* dOut, const float* Q, const float* K, const float* V, const float* G,
    const float* Hckpt, const float* dHckpt, float* dQ, float* dK, float* dG,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{ gla_bwd_chunk<1, 8, 1>(dOut, Q, K, V, G, Hckpt, dHckpt, dQ, dK, dG, batch, seqLen, modelDim, numHeads, headDim); }

extern ""C"" __global__ __launch_bounds__(256) void gla_scan_bwd_ckpt_r2(
    const float* dOut, const float* Q, const float* K, const float* V, const float* G,
    float* Hckpt, float* dHckpt, float* dV, int batch, int seqLen, int modelDim, int numHeads, int headDim)
{ gla_bwd_ckpt<2, 6>(dOut, Q, K, V, G, Hckpt, dHckpt, dV, batch, seqLen, modelDim, numHeads, headDim); }

extern ""C"" __global__ __launch_bounds__(256) void gla_scan_bwd_chunk_r2(
    const float* dOut, const float* Q, const float* K, const float* V, const float* G,
    const float* Hckpt, const float* dHckpt, float* dQ, float* dK, float* dG,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{ gla_bwd_chunk<2, 6, 2>(dOut, Q, K, V, G, Hckpt, dHckpt, dQ, dK, dG, batch, seqLen, modelDim, numHeads, headDim); }

// Checkpointed BPTT backward. The previous recompute + backward pair kept the whole state
// trajectory, batch*heads*seqLen*headDim^2 floats (2.1 GB at batch 16, seqLen 256, 8 heads of 128).
// This kernel stores the state only at chunk boundaries (Ckpt, batch*heads*nChunks*headDim^2) and
// recomputes one chunk at a time into Scratch (batch*heads*chunk*headDim^2), so memory is
// O(seqLen/chunk + chunk) instead of O(seqLen); chunk = ceil(sqrt(seqLen)) gives O(sqrt(seqLen)).
// Extra work: one forward pass, the same as the recompute kernel it replaces. Each thread reads and
// writes only its own state row, so Ckpt/Scratch need no synchronization.
extern ""C"" __global__ void gla_scan_backward_ckpt(
    const float* dOut, const float* Q, const float* K, const float* V, const float* G,
    float* Ckpt, float* Scratch,
    float* dQ, float* dK, float* dV, float* dG,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int chunk)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * numHeads * headDim;
    if (gid >= total) return;

    int di = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int b  = gid / (headDim * numHeads);
    int hOff = h * headDim;
    long hh = (long)headDim * headDim;
    long bh = (long)b * numHeads + h;
    long rowOff = (long)di * headDim;
    int nChunks = (seqLen + chunk - 1) / chunk;

    float Srow[GLA_MAX_HEADDIM];
    float dSrow[GLA_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;

    // Pass 1: forward scan, saving S_{t0-1} (the state entering each chunk) at every chunk start.
    for (int t = 0; t < seqLen; t++) {
        if (t % chunk == 0) {
            long ck = (bh * nChunks + t / chunk) * hh + rowOff;
            for (int ki = 0; ki < headDim; ki++) Ckpt[ck + ki] = Srow[ki];
        }
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        float g  = G[(b * seqLen + t) * numHeads + h];
        float vd = V[baseOff + di];
        for (int ki = 0; ki < headDim; ki++) Srow[ki] = g * Srow[ki] + vd * K[baseOff + ki];
    }

    // Pass 2: chunks in reverse; recompute the chunk's states, then run BPTT over it.
    for (int ki = 0; ki < headDim; ki++) dSrow[ki] = 0.0f;
    for (int c = nChunks - 1; c >= 0; c--) {
        int t0 = c * chunk;
        int t1 = min(seqLen, t0 + chunk);
        long ck = (bh * nChunks + c) * hh + rowOff;
        for (int ki = 0; ki < headDim; ki++) Srow[ki] = Ckpt[ck + ki];
        for (int t = t0; t < t1; t++) {
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            float g  = G[(b * seqLen + t) * numHeads + h];
            float vd = V[baseOff + di];
            long sc = (bh * chunk + (t - t0)) * hh + rowOff;
            for (int ki = 0; ki < headDim; ki++) {
                float s = g * Srow[ki] + vd * K[baseOff + ki];
                Srow[ki] = s;
                Scratch[sc + ki] = s;
            }
        }
        for (int t = t1 - 1; t >= t0; t--) {
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            int gOff = (b * seqLen + t) * numHeads + h;
            float g = G[gOff];
            long sc = (bh * chunk + (t - t0)) * hh + rowOff;
            // S_{t-1}: the previous scratch row inside the chunk, else the chunk's checkpoint.
            const float* sprevRow = (t > t0) ? (Scratch + (bh * chunk + (t - 1 - t0)) * hh + rowOff)
                                             : (Ckpt + ck);

            float dOutVal = dOut[baseOff + di];
            for (int ki = 0; ki < headDim; ki++) {
                atomicAdd(&dQ[baseOff + ki], dOutVal * Scratch[sc + ki]);
                dSrow[ki] += dOutVal * Q[baseOff + ki];
            }

            float vd = V[baseOff + di];
            float dg = 0.0f;
            float dVacc = 0.0f;
            for (int ki = 0; ki < headDim; ki++) {
                float dStv = dSrow[ki];
                dg += dStv * sprevRow[ki];
                atomicAdd(&dK[baseOff + ki], dStv * vd);
                dVacc += dStv * K[baseOff + ki];
            }
            atomicAdd(&dV[baseOff + di], dVacc);
            atomicAdd(&dG[gOff], dg);

            for (int ki = 0; ki < headDim; ki++) dSrow[ki] *= g;
        }
    }
}

// Checkpointed BPTT backward, one 1024-thread block per (batch, head), each thread owning up to GLA_TILE_EPT elements of
// the headDim x headDim state in registers (element e = di*headDim + ki, e = threadIdx.x + j*1024). Same recurrence,
// checkpoints and chunk recompute as gla_scan_backward_ckpt, whose threads each walked a whole state row serially and
// atomically added into the same dQ[ki] / dK[ki] in lockstep (14.5 ms per call at batch 16, seqLen 256, 4 heads of 65;
// the forward is 0.9 ms). Here every state update is elementwise and parallel, and the per-position reductions run
// through a padded shared [headDim][headDim|1] tile in a fixed order: dQ and dK are column sums, dV a row sum, dG a
// warp-shuffle tree. All are plain stores, so the result is deterministic and needs no zero-fill. Checkpoint and
// scratch states are stored element-contiguous, so their reads and writes are coalesced.
// Dynamic shared memory: headDim*(headDim|1) + 4*headDim + 32 floats.
// Sums the tile's columns (COLS=1: out[c] = sum_r tile[r][c]) or rows (COLS=0: out[r] = sum_c tile[r][c]) in index
// order, one line per thread. Ends with every thread synced.
#define GLA_TILE_SUM(COLS, OUT) { if (tid < headDim) { float acc = 0.0f; for (int r = 0; r < headDim; r++) acc += (COLS) ? tile[r * stride + tid] : tile[tid * stride + r]; OUT = acc; } __syncthreads(); }
// Walks this thread's state elements e = tid + j*1024 with their (row d_, column k_) kept incrementally (no index arrays,
// so S and dS stay in registers at 1024 threads per block).
#define GLA_TILE_FOR(BODY) { int d_ = di0, k_ = ki0;     _Pragma(""unroll"") for (int j = 0; j < GLA_TILE_EPT; j++) { int e = tid + j * GLA_TILE_THREADS; if (e >= hh) break;     BODY; k_ += stepK; d_ += stepD; if (k_ >= headDim) { k_ -= headDim; d_++; } } }
extern ""C"" __global__ __launch_bounds__(GLA_TILE_THREADS) void gla_scan_backward_tiled(
    const float* dOut, const float* Q, const float* K, const float* V, const float* G,
    float* Ckpt, float* Scratch,
    float* dQ, float* dK, float* dV, float* dG,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int chunk)
{
    extern __shared__ float smem[];
    int stride = headDim | 1;                 // odd row stride: row and column walks hit distinct banks
    float* tile = smem;                       // [headDim][stride]
    float* qs = tile + headDim * stride;
    float* ks = qs + headDim;
    float* vs = ks + headDim;
    float* dos = vs + headDim;
    float* warpPart = dos + headDim;          // [32]

    long bh = blockIdx.x;
    if (bh >= (long)batch * numHeads) return;
    int b = (int)(bh / numHeads), h = (int)(bh % numHeads);
    int tid = threadIdx.x;
    int hOff = h * headDim;
    int hh = headDim * headDim;
    int nChunks = (seqLen + chunk - 1) / chunk;

    float S[GLA_TILE_EPT], dS[GLA_TILE_EPT];
#pragma unroll
    for (int j = 0; j < GLA_TILE_EPT; j++) { S[j] = 0.0f; dS[j] = 0.0f; }
    int di0 = tid / headDim, ki0 = tid % headDim;
    int stepD = GLA_TILE_THREADS / headDim, stepK = GLA_TILE_THREADS % headDim;

    // Pass 1: forward scan, saving S_{t0-1} at every chunk start.
    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        __syncthreads();
        if (tid < headDim) { ks[tid] = K[baseOff + tid]; vs[tid] = V[baseOff + tid]; }
        __syncthreads();
        float g = G[(b * seqLen + t) * numHeads + h];
        long ck = (bh * nChunks + t / chunk) * hh;
        GLA_TILE_FOR(
            if (t % chunk == 0) Ckpt[ck + e] = S[j];
            S[j] = g * S[j] + vs[d_] * ks[k_];
        )
    }

    // Pass 2: chunks in reverse; recompute the chunk's states into Scratch, then BPTT over it.
    for (int c = nChunks - 1; c >= 0; c--) {
        int t0 = c * chunk;
        int t1 = min(seqLen, t0 + chunk);
        long ck = (bh * nChunks + c) * hh;
#pragma unroll
        for (int j = 0; j < GLA_TILE_EPT; j++) {
            int e = tid + j * GLA_TILE_THREADS;
            if (e < hh) S[j] = Ckpt[ck + e];
        }
        for (int t = t0; t < t1; t++) {
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            __syncthreads();
            if (tid < headDim) { ks[tid] = K[baseOff + tid]; vs[tid] = V[baseOff + tid]; }
            __syncthreads();
            float g = G[(b * seqLen + t) * numHeads + h];
            long sc = (bh * chunk + (t - t0)) * hh;
            GLA_TILE_FOR(
                S[j] = g * S[j] + vs[d_] * ks[k_];
                Scratch[sc + e] = S[j];
            )
        }
        for (int t = t1 - 1; t >= t0; t--) {
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            int gOff = (b * seqLen + t) * numHeads + h;
            __syncthreads();
            if (tid < headDim) {
                qs[tid] = Q[baseOff + tid]; ks[tid] = K[baseOff + tid];
                vs[tid] = V[baseOff + tid]; dos[tid] = dOut[baseOff + tid];
            }
            __syncthreads();
            float g = G[gOff];
            long sp = (t > t0) ? (bh * chunk + (t - 1 - t0)) * hh : ck;
            const float* prev = (t > t0) ? Scratch : Ckpt;

            // dQ[ki] = sum_di dOut[di] S_t[di][ki], then dS += dOut q^T. S holds S_t: after the recompute it is S_{t1-1},
            // and each position below leaves S_{t-1} in it for the next (earlier) one, so Scratch is read once per state.
            GLA_TILE_FOR(
                tile[d_ * stride + k_] = dos[d_] * S[j];
                dS[j] += dos[d_] * qs[k_];
            )
            __syncthreads();
            GLA_TILE_SUM(1, dQ[baseOff + tid])

            // dK[ki] = sum_di dS[di][ki] v[di];  dG = sum dS . S_{t-1}.
            float pg = 0.0f;
            GLA_TILE_FOR(
                float sPrev = prev[sp + e];
                pg += dS[j] * sPrev;
                S[j] = sPrev;
                tile[d_ * stride + k_] = dS[j] * vs[d_];
            )
            for (int off = 16; off > 0; off >>= 1) pg += __shfl_down_sync(0xffffffffu, pg, off);
            if ((tid & 31) == 0) warpPart[tid >> 5] = pg;
            __syncthreads();
            if (tid == 0) {
                float acc = 0.0f;
                for (int w = 0; w < GLA_TILE_THREADS / GLA_TILE_WARP_SIZE; w++) acc += warpPart[w];
                dG[gOff] = acc;
            }
            GLA_TILE_SUM(1, dK[baseOff + tid])

            // dV[di] = sum_ki dS[di][ki] k[ki]; then carry dS back through the gate.
            GLA_TILE_FOR(
                tile[d_ * stride + k_] = dS[j] * ks[k_];
                dS[j] *= g;
            )
            __syncthreads();
            GLA_TILE_SUM(0, dV[baseOff + tid])
        }
    }
}

// Channels-last causal depthwise conv (ICausalDepthwiseConvBackend): x, y, dy, dx are [rows, seqLen, C]; w, dw are
// [K, C]; tap j reads j steps back. Every output element is written; no atomics, so all three are deterministic.
extern ""C"" __global__ void causal_dwconv1d_forward(
    const float* x, const float* w, float* y, int rows, int seqLen, int channels, int taps)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * seqLen * channels;
    if (idx >= total) return;
    int c = (int)(idx % channels);
    int t = (int)((idx / channels) % seqLen);
    int maxJ = min(taps - 1, t);
    float acc = 0.0f;
    for (int j = 0; j <= maxJ; j++) acc += w[j * channels + c] * x[idx - (long)j * channels];
    y[idx] = acc;
}

extern ""C"" __global__ void causal_dwconv1d_backward_input(
    const float* dy, const float* w, float* dx, int rows, int seqLen, int channels, int taps)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * seqLen * channels;
    if (idx >= total) return;
    int c = (int)(idx % channels);
    int t = (int)((idx / channels) % seqLen);
    int maxJ = min(taps - 1, seqLen - 1 - t);
    float acc = 0.0f;
    for (int j = 0; j <= maxJ; j++) acc += w[j * channels + c] * dy[idx + (long)j * channels];
    dx[idx] = acc;
}

// Block (32 channel lanes x 8 groups) per (32-channel tile, tap): group g sums steps g, g + 8, ... of the flattened
// [rows * seqLen] grid (lanes read consecutive channels, so loads coalesce), then lane 0..31 of group 0 adds the 8
// partials in a fixed order. Launch: grid (ceil(C / 32), K), block (32, 8).
extern ""C"" __global__ void causal_dwconv1d_backward_kernel(
    const float* dy, const float* x, float* dw, int rows, int seqLen, int channels, int taps)
{
    __shared__ float partial[8][32];
    int lane = threadIdx.x, group = threadIdx.y;
    int c = blockIdx.x * 32 + lane;
    int j = blockIdx.y;
    float acc = 0.0f;
    if (c < channels) {
        long steps = (long)rows * seqLen;
        for (long rt = group; rt < steps; rt += 8) {
            int t = (int)(rt % seqLen);
            if (t >= j) acc += dy[rt * channels + c] * x[(rt - j) * channels + c];
        }
    }
    partial[group][lane] = acc;
    __syncthreads();
    if (group == 0 && c < channels) {
        float s = 0.0f;
        for (int g = 0; g < 8; g++) s += partial[g][lane];
        dw[j * channels + c] = s;
    }
}

// Sparse routing & codebook primitives (ISparseRoutingBackend): zero atomics, deterministic ascending reductions.
extern ""C"" __global__ void gather_row_values_forward(
    const float* values, const int* indices, float* output, int rows, int cols, int k)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * k;
    if (gid >= total) return;
    int r = (int)(gid / k);
    int col = indices[gid];
    output[gid] = (col >= 0 && col < cols) ? values[(long)r * cols + col] : 0.0f;
}

extern ""C"" __global__ void gather_row_values_backward(
    const float* gradOut, const int* indices, float* gradValues, int rows, int cols, int k)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * cols;
    if (gid >= total) return;
    int r = (int)(gid / cols);
    int c = (int)(gid % cols);
    long kBase = (long)r * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        if (indices[kBase + j] == c) acc += gradOut[kBase + j];
    }
    gradValues[gid] = acc;
}

extern ""C"" __global__ void topk_weighted_row_mix_forward(
    const float* table, const int* indices, const float* weights, float* output,
    int rows, int numEntries, int k, int channels)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * channels;
    if (gid >= total) return;
    int r = (int)(gid / channels);
    int c = (int)(gid % channels);
    long kBase = (long)r * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        int entry = indices[kBase + j];
        if (entry >= 0 && entry < numEntries)
            acc += weights[kBase + j] * table[(long)entry * channels + c];
    }
    output[gid] = acc;
}

extern ""C"" __global__ void topk_weighted_row_mix_backward_weights(
    const float* gradOut, const float* table, const int* indices, float* gradWeights,
    int rows, int numEntries, int k, int channels)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * k;
    if (gid >= total) return;
    int r = (int)(gid / k);
    int entry = indices[gid];
    float acc = 0.0f;
    if (entry >= 0 && entry < numEntries) {
        long goBase = (long)r * channels;
        long tbBase = (long)entry * channels;
        for (int c = 0; c < channels; c++)
            acc += gradOut[goBase + c] * table[tbBase + c];
    }
    gradWeights[gid] = acc;
}

extern ""C"" __global__ void sparse_zero_int32(int* buf, int offset, int count)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) buf[(long)offset + gid] = 0;
}

extern ""C"" __global__ void topk_csr_tile_counts(
    const int* indices, int* csr, int totalSlots, int numEntries, int numTiles, int offTile)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numTiles || totalSlots <= 0) return;
    int tileSpan = ((totalSlots - 1) / numTiles) + 1;
    if (t > totalSlots / tileSpan) return;
    int start = t * tileSpan;
    int rem = totalSlots - start;
    int end = start + (rem < tileSpan ? rem : tileSpan);
    long rowBase = (long)offTile + (long)t * numEntries;
    for (int s = start; s < end; s++) {
        int e = indices[s];
        if (e >= 0 && e < numEntries)
            csr[rowBase + e] += 1;
    }
}

extern ""C"" __global__ void topk_csr_entry_prefix(
    int* csr, int numEntries, int numTiles, int offTile)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= numEntries) return;
    int acc = 0;
    for (int t = 0; t < numTiles; t++) {
        long idx = (long)offTile + (long)t * numEntries + g;
        int c = csr[idx];
        csr[idx] = acc;
        acc += c;
    }
    csr[g] = acc;
}

extern ""C"" __global__ void topk_csr_chunk_sums(
    int* csr, int numEntries, int offChunk)
{
    int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cIdx >= 256) return;
    int acc = 0;
    if (numEntries > 0) {
        int chunkSpan = ((numEntries - 1) / 256) + 1;
        if (cIdx <= numEntries / chunkSpan) {
            int start = cIdx * chunkSpan;
            int rem = numEntries - start;
            int end = start + (rem < chunkSpan ? rem : chunkSpan);
            for (int g = start; g < end; g++) {
                int c = csr[g];
                csr[g] = acc;
                acc += c;
            }
        }
    }
    csr[offChunk + cIdx] = acc;
}

extern ""C"" __global__ void topk_csr_chunk_finalize(
    int* csr, int numEntries, int offChunk)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g > numEntries) return;
    if (g == numEntries) {
        int total = 0;
        for (int m = 0; m < 256; m++) total += csr[offChunk + m];
        csr[numEntries] = total;
        return;
    }
    int chunkSpan = ((numEntries - 1) / 256) + 1;
    int chunkIdx = g / chunkSpan;
    int baseOff = 0;
    for (int m = 0; m < chunkIdx; m++) baseOff += csr[offChunk + m];
    csr[g] += baseOff;
}

extern ""C"" __global__ void topk_csr_scatter_slots(
    const int* indices, int* csr, int totalSlots, int numEntries, int numTiles, int offSorted, int offTile)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numTiles || totalSlots <= 0) return;
    int tileSpan = ((totalSlots - 1) / numTiles) + 1;
    if (t > totalSlots / tileSpan) return;
    int start = t * tileSpan;
    int rem = totalSlots - start;
    int end = start + (rem < tileSpan ? rem : tileSpan);
    long rowBase = (long)offTile + (long)t * numEntries;
    for (int s = start; s < end; s++) {
        int e = indices[s];
        if (e >= 0 && e < numEntries) {
            int localOff = csr[rowBase + e];
            csr[rowBase + e] = localOff + 1;
            int pos = csr[e] + localOff;
            csr[offSorted + pos] = s;
        }
    }
}

// 1D flattened over (numEntries * channels): supports numEntries > 65535 and sums CSR buckets in exact ascending slot order in O(rows * k * C).
extern ""C"" __global__ void topk_weighted_row_mix_backward_table(
    const float* gradOut, const float* weights, const int* csr, float* gradTable,
    int numEntries, int k, int channels, int offSorted)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)numEntries * channels;
    if (gid >= total) return;
    int g = (int)(gid / channels);
    int c = (int)(gid % channels);
    int start = csr[g];
    int end = csr[g + 1];
    float acc = 0.0f;
    for (int p = start; p < end; p++) {
        int slot = csr[offSorted + p];
        int r = slot / k;
        acc += weights[slot] * gradOut[(long)r * channels + c];
    }
    gradTable[gid] = acc;
}

extern ""C"" __global__ void count_indices_accumulate_int32(
    const int* indices, int* countsInt, int length, int numBins, int totalThreads)
{
    __shared__ int s_counts[256];
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    long gid = (long)blockIdx.x * bdim + tid;
    unsigned int uLen = (unsigned int)length;
    unsigned int uStride = (unsigned int)totalThreads;
    if (numBins <= 256) {
        for (int b = tid; b < numBins; b += bdim) s_counts[b] = 0;
        __syncthreads();
        if (gid < totalThreads) {
            for (unsigned int i = (unsigned int)gid; i < uLen; i += uStride) {
                int bin = indices[i];
                if (bin >= 0 && bin < numBins) atomicAdd(&s_counts[bin], 1);
            }
        }
        __syncthreads();
        for (int b = tid; b < numBins; b += bdim) {
            int c = s_counts[b];
            if (c != 0) atomicAdd(&countsInt[b], c);
        }
    } else {
        if (gid < totalThreads) {
            for (unsigned int i = (unsigned int)gid; i < uLen; i += uStride) {
                int bin = indices[i];
                if (bin >= 0 && bin < numBins) atomicAdd(&countsInt[bin], 1);
            }
        }
    }
}

extern ""C"" __global__ void count_indices_finalize_float(int* countsBuf, int numBins)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < numBins) {
        int c = countsBuf[gid];
        ((float*)countsBuf)[gid] = (float)c;
    }
}

extern ""C"" __global__ void minmax_init_int32(int* outMinMax)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0) outMinMax[0] = 2147483647;
    else if (gid == 1) outMinMax[1] = (-2147483647 - 1);
}

extern ""C"" __global__ void minmax_reduce_int32(
    const int* data, int* outMinMax, int length, int totalThreads)
{
    __shared__ int s_min;
    __shared__ int s_max;
    int tid = threadIdx.x;
    long gid = (long)blockIdx.x * blockDim.x + tid;
    if (tid == 0) {
        s_min = 2147483647;
        s_max = (-2147483647 - 1);
    }
    __syncthreads();
    if (gid < totalThreads) {
        unsigned int uLen = (unsigned int)length;
        unsigned int uStride = (unsigned int)totalThreads;
        int lo = 2147483647;
        int hi = (-2147483647 - 1);
        int any = 0;
        for (unsigned int i = (unsigned int)gid; i < uLen; i += uStride) {
            int v = data[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
            any = 1;
        }
        if (any != 0) {
            atomicMin(&s_min, lo);
            atomicMax(&s_max, hi);
        }
    }
    __syncthreads();
    if (tid == 0 && s_min <= s_max) {
        atomicMin(&outMinMax[0], s_min);
        atomicMax(&outMinMax[1], s_max);
    }
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "gla_scan_forward",
            "gla_scan_forward_r1",
            "gla_scan_forward_r2",
            "gla_scan_forward_r4",
            "gla_scan_forward_r8",
            "gla_scan_backward_ckpt",
            "gla_scan_backward_tiled",
            "gla_scan_backward_warp_r1",
            "gla_scan_backward_warp_r2",
            "gla_scan_backward_warp_r4",
            "gla_scan_backward_warp_r8",
            "gla_scan_bwd_ckpt_r1",
            "gla_scan_bwd_chunk_r1",
            "gla_scan_bwd_ckpt_r2",
            "gla_scan_bwd_chunk_r2",
            "causal_dwconv1d_forward",
            "causal_dwconv1d_backward_input",
            "causal_dwconv1d_backward_kernel",
            "gather_row_values_forward",
            "gather_row_values_backward",
            "topk_weighted_row_mix_forward",
            "topk_weighted_row_mix_backward_weights",
            "sparse_zero_int32",
            "topk_csr_tile_counts",
            "topk_csr_entry_prefix",
            "topk_csr_chunk_sums",
            "topk_csr_chunk_finalize",
            "topk_csr_scatter_slots",
            "topk_weighted_row_mix_backward_table",
            "count_indices_accumulate_int32",
            "count_indices_finalize_float",
            "minmax_init_int32",
            "minmax_reduce_int32"
        };
    }
}
