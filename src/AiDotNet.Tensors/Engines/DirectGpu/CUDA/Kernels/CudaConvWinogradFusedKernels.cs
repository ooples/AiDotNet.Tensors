// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Fused Winograd F(2x2, 3x3) (Lavin and Gray, CVPR 2016) for 3x3 stride-1 dilation-1: one kernel per convolution after
/// a filter transform. The unfused path (<see cref="CudaConvWinogradKernels"/>) stores the input transform V[16][C][P]
/// and the products M[16][K][P] in global memory between three launches; here each block transforms its input patches
/// into shared memory, runs the sixteen tile GEMMs from there, and applies the output transform in its epilogue, so
/// V and M never leave the SM (cuDNN's input gradient for ResNet's 32x32 and 16x16 layers is this algorithm). The
/// arithmetic is the unfused path's: the same transforms, fp32, the reduction in channel order, so the rounding is
/// F(2x2)'s (about 1e-7 of sum|terms|) and runs are bit-reproducible (no atomics, fixed order).
/// </summary>
internal static class CudaConvWinogradFusedKernels
{
    public static string GetSource() => @"
#define WFZ_BO 32   // output channels per block
#define WFZ_BP 32   // 2x2 output tiles per block
#define WFZ_CK 8    // reduction channels per main-loop step

// F(2x2,3x3) filter transform U[x][r][o] = (G g G^T)[x], G = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]], with the
// reduction channel r outermost and the output channel o contiguous (the fused kernel's block reads 32 o per r).
__device__ __forceinline__ void wfz_transform_filter(const float g[3][3], float* __restrict__ u, long long plane, int idx)
{
    float t[4][3];
    #pragma unroll
    for (int j = 0; j < 3; j++)
    {
        float g0 = g[0][j], g1 = g[1][j], g2 = g[2][j];
        t[0][j] = g0;
        t[1][j] = 0.5f * (g0 + g1 + g2);
        t[2][j] = 0.5f * (g0 - g1 + g2);
        t[3][j] = g2;
    }
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float a = t[i][0], b = t[i][1], c = t[i][2];
        u[(i * 4 + 0) * plane + idx] = a;
        u[(i * 4 + 1) * plane + idx] = 0.5f * (a + b + c);
        u[(i * 4 + 2) * plane + idx] = 0.5f * (a - b + c);
        u[(i * 4 + 3) * plane + idx] = c;
    }
}

// Forward: y = conv(x, w), w[K][C][3][3]; reduction r = c, output o = k: U[x][c][k].
extern ""C"" __global__ void wfz_filter(const float* __restrict__ w, float* __restrict__ u, int K, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = c*K + k
    if (idx >= K * C) return;
    int c = idx / K, k = idx - c * K;
    const float* src = w + ((long long)k * C + c) * 9;
    float g[3][3];
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) g[i][j] = src[i * 3 + j];
    wfz_transform_filter(g, u, (long long)K * C, idx);
}

// Input gradient: dX = conv(dY, w') with w'[c][k] = w[k][c] rotated 180 degrees; reduction r = k (dY's channel), output
// o = c: U[x][k][c], so idx = k*C + c is also w's [k][c] index.
extern ""C"" __global__ void wfz_filter_flip(const float* __restrict__ w, float* __restrict__ u, int K, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * C) return;
    const float* src = w + (long long)idx * 9;
    float g[3][3];
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) g[i][j] = src[(2 - i) * 3 + (2 - j)];
    wfz_transform_filter(g, u, (long long)K * C, idx);
}

// out[n][O][OH][OW] = conv3x3_s1(in[n][R][IH][IW], padding pad) through F(2x2,3x3) with u = U[16][R][O].
// Block: WFZ_BO output channels x WFZ_BP tiles x all 16 Winograd components. 256 threads.
//  - Loader role: thread (lp = tid & 31, lc = tid >> 5) loads the 4x4 input patch of tile p0 + lp, channel r0 + lc, and
//    the U values (x, r0 + (tid >> 5), o0 + (tid & 31)) for all 16 x; the next step's loads are issued before this
//    step's arithmetic, so they are in flight during it.
//  - Compute role: thread (x = tid >> 4, s = tid & 15) accumulates component x of an 8 (o) x 8 (p) sub-tile.
//  - Epilogue: the products go through shared memory in two halves of 16 output channels; each (o, tile) pair's 16
//    components are reduced by A^T m A, A^T = [[1,1,1,0],[0,1,-1,-1]], into its in-bounds 2x2 outputs.
// STAGES = 1: one U | V buffer, stored between two barriers per step. STAGES = 2: two buffers, the next step's stored
// while this step's is read, one barrier per step. sm holds STAGES * 2 * 16 * WFZ_CK * 32 floats; the epilogue reuses
// the first 16 * 16 * 32 of them.
#define WFZ_STAGE_FLOATS (2 * 16 * WFZ_CK * 32)
template <int STAGES>
__device__ __forceinline__ void wfz_conv_body(float* __restrict__ sm, const float* __restrict__ in, const float* __restrict__ u,
    float* __restrict__ out, int N, int R, int IH, int IW, int O, int OH, int OW, int pad, int tilesH, int tilesW)
{
    const int tid = threadIdx.x;
    const int tilesPerImage = tilesH * tilesW;
    const int P = N * tilesPerImage;
    const int p0 = blockIdx.x * WFZ_BP, o0 = blockIdx.y * WFZ_BO;

    const int lp = tid & 31, lc = tid >> 5;
    const int gp = p0 + lp;
    unsigned rowOk = 0, colOk = 0;
    long long inOff = 0;
    if (gp < P)
    {
        int n = gp / tilesPerImage, t = gp - n * tilesPerImage;
        int th = t / tilesW, tw = t - th * tilesW;
        int h0 = th * 2 - pad, w0 = tw * 2 - pad;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            if (h0 + i >= 0 && h0 + i < IH) rowOk |= 1u << i;
            if (w0 + i >= 0 && w0 + i < IW) colOk |= 1u << i;
        }
        inOff = (long long)n * R * IH * IW + (long long)h0 * IW + w0;
    }
    const long long chanStride = (long long)IH * IW;
    const long long uPlane = (long long)R * O;
    const int uo = o0 + (tid & 31);

    const int x = tid >> 4, s = tid & 15;
    const int oo = (s >> 2) * 8, pp = (s & 3) * 8;

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    float d[16], ur[16];
    auto load = [&](int r0)
    {
        const int r = r0 + lc;
        const bool rv = r < R;
        const float* src = in + inOff + (long long)r * chanStride;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++)
                d[i * 4 + j] = (rv && ((rowOk >> i) & 1u) && ((colOk >> j) & 1u)) ? __ldg(src + i * IW + j) : 0.0f;
        const bool uv = rv && uo < O;   // the U row is r0 + (tid >> 5) = r too
        const float* usrc = u + (long long)r * O + uo;
        #pragma unroll
        for (int q = 0; q < 16; q++) ur[q] = uv ? __ldg(usrc + q * uPlane) : 0.0f;
    };
    auto store = [&](int stage)
    {
        float* us = sm + stage * WFZ_STAGE_FLOATS;
        float* vs = us + 16 * WFZ_CK * 32;
        float t[4][4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            t[0][j] = d[0 * 4 + j] - d[2 * 4 + j];
            t[1][j] = d[1 * 4 + j] + d[2 * 4 + j];
            t[2][j] = d[2 * 4 + j] - d[1 * 4 + j];
            t[3][j] = d[1 * 4 + j] - d[3 * 4 + j];
        }
        float* v = vs + lc * 32 + lp;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            v[(i * 4 + 0) * WFZ_CK * 32] = t[i][0] - t[i][2];
            v[(i * 4 + 1) * WFZ_CK * 32] = t[i][1] + t[i][2];
            v[(i * 4 + 2) * WFZ_CK * 32] = t[i][2] - t[i][1];
            v[(i * 4 + 3) * WFZ_CK * 32] = t[i][1] - t[i][3];
        }
        float* uu = us + tid;   // (r = tid >> 5, o = tid & 31) within the step
        #pragma unroll
        for (int q = 0; q < 16; q++) uu[q * WFZ_CK * 32] = ur[q];
    };

    const int steps = (R + WFZ_CK - 1) / WFZ_CK;
    load(0);
    if (STAGES == 2) { store(0); __syncthreads(); }
    for (int step = 0; step < steps; step++)
    {
        const int stage = STAGES == 2 ? (step & 1) : 0;
        if (STAGES == 1)
        {
            __syncthreads();
            store(0);
            __syncthreads();
        }
        if (step + 1 < steps) load((step + 1) * WFZ_CK);
        const float* ub = sm + stage * WFZ_STAGE_FLOATS + x * WFZ_CK * 32 + oo;
        const float* vb = sm + stage * WFZ_STAGE_FLOATS + 16 * WFZ_CK * 32 + x * WFZ_CK * 32 + pp;
        #pragma unroll
        for (int r = 0; r < WFZ_CK; r++)
        {
            float4 a0 = *reinterpret_cast<const float4*>(ub + r * 32);
            float4 a1 = *reinterpret_cast<const float4*>(ub + r * 32 + 4);
            float4 b0 = *reinterpret_cast<const float4*>(vb + r * 32);
            float4 b1 = *reinterpret_cast<const float4*>(vb + r * 32 + 4);
            float a[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
            float b[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (STAGES == 2 && step + 1 < steps)
        {
            store(stage ^ 1);   // last read during step - 1, before that step's closing barrier
            __syncthreads();
        }
    }

    // Epilogue: ms[x][ol][pl] for 16 output channels at a time (16 * 16 * 32 floats = the whole shared buffer).
    #pragma unroll
    for (int half = 0; half < 2; half++)
    {
        __syncthreads();
        if ((oo >> 4) == half)
        {
            #pragma unroll
            for (int i = 0; i < 8; i++)
            {
                float* row = sm + (x * 16 + oo - 16 * half + i) * 32 + pp;
                *reinterpret_cast<float4*>(row) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                *reinterpret_cast<float4*>(row + 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < 2; q++)
        {
            const int pair = tid + q * 256, ol = pair >> 5, pl = pair & 31;
            const int o = o0 + 16 * half + ol, p = p0 + pl;
            if (o >= O || p >= P) continue;
            float m[16];
            #pragma unroll
            for (int c = 0; c < 16; c++) m[c] = sm[(c * 16 + ol) * 32 + pl];
            float t0[4], t1[4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                t0[j] = m[0 * 4 + j] + m[1 * 4 + j] + m[2 * 4 + j];
                t1[j] = m[1 * 4 + j] - m[2 * 4 + j] - m[3 * 4 + j];
            }
            int n = p / tilesPerImage, t = p - n * tilesPerImage;
            int th = t / tilesW, tw = t - th * tilesW;
            int h = th * 2, w = tw * 2;
            float* dst = out + (((long long)n * O + o) * OH + h) * OW + w;
            dst[0] = t0[0] + t0[1] + t0[2];
            if (w + 1 < OW) dst[1] = t0[1] - t0[2] - t0[3];
            if (h + 1 < OH)
            {
                dst[OW] = t1[0] + t1[1] + t1[2];
                if (w + 1 < OW) dst[OW + 1] = t1[1] - t1[2] - t1[3];
            }
        }
    }
}

// Two blocks per SM (128 registers, a small spill) and one (no spill): the tuned slot measures both.
extern ""C"" __global__ __launch_bounds__(256, 2) void wfz_conv(const float* __restrict__ in, const float* __restrict__ u,
    float* __restrict__ out, int N, int R, int IH, int IW, int O, int OH, int OW, int pad, int tilesH, int tilesW)
{
    __shared__ __align__(16) float sm[WFZ_STAGE_FLOATS];
    wfz_conv_body<1>(sm, in, u, out, N, R, IH, IW, O, OH, OW, pad, tilesH, tilesW);
}
extern ""C"" __global__ __launch_bounds__(256, 1) void wfz_conv1(const float* __restrict__ in, const float* __restrict__ u,
    float* __restrict__ out, int N, int R, int IH, int IW, int O, int OH, int OW, int pad, int tilesH, int tilesW)
{
    __shared__ __align__(16) float sm[WFZ_STAGE_FLOATS];
    wfz_conv_body<1>(sm, in, u, out, N, R, IH, IW, O, OH, OW, pad, tilesH, tilesW);
}
// Double-buffered: 64 KB of dynamic shared memory (the host raises the function's limit), one block per SM.
extern ""C"" __global__ __launch_bounds__(256, 1) void wfz_conv_db(const float* __restrict__ in, const float* __restrict__ u,
    float* __restrict__ out, int N, int R, int IH, int IW, int O, int OH, int OW, int pad, int tilesH, int tilesW)
{
    extern __shared__ __align__(16) float smd[];
    wfz_conv_body<2>(smd, in, u, out, N, R, IH, IW, O, OH, OW, pad, tilesH, tilesW);
}
// The same tile (32 output channels x 32 tiles x 16 components, 8 reduction channels per step) on 512 threads: warp x
// owns component x, lane (oq = lane >> 3, pq = lane & 7) an 8 (o) x 4 (p) sub-tile, so a warp's shared reads per
// reduction channel are 32 U and 32 V floats (all broadcast) and sixteen warps per SM hide the global loads. Loaders:
// warps 0-7 load one 4x4 patch each (tile lp, channel lc), every thread 8 of the 16 U planes (x = (tid >> 8) + 2q).
template <int STAGES>
__device__ __forceinline__ void wfz512_body(float* __restrict__ sm, const float* __restrict__ in, const float* __restrict__ u,
    float* __restrict__ out, int N, int R, int IH, int IW, int O, int OH, int OW, int pad, int tilesH, int tilesW)
{
    const int tid = threadIdx.x;
    const int tilesPerImage = tilesH * tilesW;
    const int P = N * tilesPerImage;
    const int p0 = blockIdx.x * WFZ_BP, o0 = blockIdx.y * WFZ_BO;

    const bool patchLoader = tid < 256;
    const int lp = tid & 31, lc = (tid >> 5) & 7;
    const int gp = p0 + lp;
    unsigned rowOk = 0, colOk = 0;
    long long inOff = 0;
    if (patchLoader && gp < P)
    {
        int n = gp / tilesPerImage, t = gp - n * tilesPerImage;
        int th = t / tilesW, tw = t - th * tilesW;
        int h0 = th * 2 - pad, w0 = tw * 2 - pad;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            if (h0 + i >= 0 && h0 + i < IH) rowOk |= 1u << i;
            if (w0 + i >= 0 && w0 + i < IW) colOk |= 1u << i;
        }
        inOff = (long long)n * R * IH * IW + (long long)h0 * IW + w0;
    }
    const long long chanStride = (long long)IH * IW;
    const long long uPlane = (long long)R * O;
    const int uo = o0 + (tid & 31), ux0 = tid >> 8;

    const int x = tid >> 5, lane = tid & 31;
    const int oo = (lane >> 3) * 8, pp = (lane & 7) * 4;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    float d[16], ur[8];
    auto load = [&](int r0)
    {
        const int r = r0 + lc;
        const bool rv = r < R;
        if (patchLoader)
        {
            const float* src = in + inOff + (long long)r * chanStride;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    d[i * 4 + j] = (rv && ((rowOk >> i) & 1u) && ((colOk >> j) & 1u)) ? __ldg(src + i * IW + j) : 0.0f;
        }
        const bool uv = rv && uo < O;
        const float* usrc = u + (long long)(ux0 * R + r) * O + uo;
        #pragma unroll
        for (int q = 0; q < 8; q++) ur[q] = uv ? __ldg(usrc + 2 * q * uPlane) : 0.0f;
    };
    auto store = [&](int stage)
    {
        float* us = sm + stage * WFZ_STAGE_FLOATS;
        float* vs = us + 16 * WFZ_CK * 32;
        if (patchLoader)
        {
            float t[4][4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                t[0][j] = d[0 * 4 + j] - d[2 * 4 + j];
                t[1][j] = d[1 * 4 + j] + d[2 * 4 + j];
                t[2][j] = d[2 * 4 + j] - d[1 * 4 + j];
                t[3][j] = d[1 * 4 + j] - d[3 * 4 + j];
            }
            float* v = vs + lc * 32 + lp;
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                v[(i * 4 + 0) * WFZ_CK * 32] = t[i][0] - t[i][2];
                v[(i * 4 + 1) * WFZ_CK * 32] = t[i][1] + t[i][2];
                v[(i * 4 + 2) * WFZ_CK * 32] = t[i][2] - t[i][1];
                v[(i * 4 + 3) * WFZ_CK * 32] = t[i][1] - t[i][3];
            }
        }
        float* uu = us + ux0 * WFZ_CK * 32 + (tid & 255);   // (x = ux0 + 2q, r = lc, o = tid & 31)
        #pragma unroll
        for (int q = 0; q < 8; q++) uu[2 * q * WFZ_CK * 32] = ur[q];
    };

    const int steps = (R + WFZ_CK - 1) / WFZ_CK;
    load(0);
    if (STAGES == 2) { store(0); __syncthreads(); }
    for (int step = 0; step < steps; step++)
    {
        const int stage = STAGES == 2 ? (step & 1) : 0;
        if (STAGES == 1)
        {
            __syncthreads();
            store(0);
            __syncthreads();
        }
        if (step + 1 < steps) load((step + 1) * WFZ_CK);
        const float* ub = sm + stage * WFZ_STAGE_FLOATS + x * WFZ_CK * 32 + oo;
        const float* vb = sm + stage * WFZ_STAGE_FLOATS + 16 * WFZ_CK * 32 + x * WFZ_CK * 32 + pp;
        #pragma unroll
        for (int r = 0; r < WFZ_CK; r++)
        {
            float4 a0 = *reinterpret_cast<const float4*>(ub + r * 32);
            float4 a1 = *reinterpret_cast<const float4*>(ub + r * 32 + 4);
            float4 b0 = *reinterpret_cast<const float4*>(vb + r * 32);
            float a[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
            float b[4] = { b0.x, b0.y, b0.z, b0.w };
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (STAGES == 2 && step + 1 < steps)
        {
            store(stage ^ 1);
            __syncthreads();
        }
    }

    // Epilogue as in wfz_conv_body: two halves of 16 output channels, one (o, tile) pair per thread per half.
    #pragma unroll
    for (int half = 0; half < 2; half++)
    {
        __syncthreads();
        if ((oo >> 4) == half)
        {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                *reinterpret_cast<float4*>(sm + (x * 16 + oo - 16 * half + i) * 32 + pp) =
                    make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        }
        __syncthreads();
        const int ol = tid >> 5, pl = tid & 31;
        const int o = o0 + 16 * half + ol, p = p0 + pl;
        if (o < O && p < P)
        {
            float m[16];
            #pragma unroll
            for (int c = 0; c < 16; c++) m[c] = sm[(c * 16 + ol) * 32 + pl];
            float t0[4], t1[4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                t0[j] = m[0 * 4 + j] + m[1 * 4 + j] + m[2 * 4 + j];
                t1[j] = m[1 * 4 + j] - m[2 * 4 + j] - m[3 * 4 + j];
            }
            int n = p / tilesPerImage, t = p - n * tilesPerImage;
            int th = t / tilesW, tw = t - th * tilesW;
            int h = th * 2, w = tw * 2;
            float* dst = out + (((long long)n * O + o) * OH + h) * OW + w;
            dst[0] = t0[0] + t0[1] + t0[2];
            if (w + 1 < OW) dst[1] = t0[1] - t0[2] - t0[3];
            if (h + 1 < OH)
            {
                dst[OW] = t1[0] + t1[1] + t1[2];
                if (w + 1 < OW) dst[OW + 1] = t1[1] - t1[2] - t1[3];
            }
        }
    }
}

extern ""C"" __global__ __launch_bounds__(512, 1) void wfz512_db(const float* __restrict__ in, const float* __restrict__ u,
    float* __restrict__ out, int N, int R, int IH, int IW, int O, int OH, int OW, int pad, int tilesH, int tilesW)
{
    extern __shared__ __align__(16) float smd[];
    wfz512_body<2>(smd, in, u, out, N, R, IH, IW, O, OH, OW, pad, tilesH, tilesW);
}
extern ""C"" __global__ __launch_bounds__(512, 1) void wfz512(const float* __restrict__ in, const float* __restrict__ u,
    float* __restrict__ out, int N, int R, int IH, int IW, int O, int OH, int OW, int pad, int tilesH, int tilesW)
{
    __shared__ __align__(16) float sm[WFZ_STAGE_FLOATS];
    wfz512_body<1>(sm, in, u, out, N, R, IH, IW, O, OH, OW, pad, tilesH, tilesW);
}
";

    public static string[] GetKernelNames() => new[] { "wfz_filter", "wfz_filter_flip", "wfz_conv", "wfz_conv1", "wfz_conv_db", "wfz512_db", "wfz512" };
}
