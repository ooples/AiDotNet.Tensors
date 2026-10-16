// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Fused Winograd F(2x2, 3x3) (Lavin and Gray, CVPR 2016) for 3x3 stride-1 dilation-1: one kernel per convolution,
/// filter transform included. The unfused path (<see cref="CudaConvWinogradKernels"/>) stores the input transform V[16][C][P]
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

// out[n][O][OH][OW] = conv3x3_s1(in[n][R][IH][IW], padding pad) through F(2x2,3x3).
// Block: WFZ_BO output channels x WFZ_BP tiles x all 16 Winograd components, 512 threads (sixteen warps per SM: one
// block's ~115 registers per thread fill the register file; measured faster than 256 threads with an 8 x 8 sub-tile
// at one or two blocks per SM, and than a double-buffered or a 16-channel step).
//  - Loaders: warps 0-7 each load the 4x4 input patches of the block's 32 tiles for one channel (tile lp = tid & 31,
//    channel r0 + lc) and transform them into shared V[x][lc][lp]; warps 8-15 load the step's 8 x 32 filters and
//    transform them into shared U[x][r][o] (below). The next step's loads are issued before this step's arithmetic.
//  - Compute: warp x owns component x, lane (lane >> 3, lane & 7) an 8 (o) x 4 (p) sub-tile, so a warp's shared reads
//    per reduction channel are 32 U and 32 V floats, all broadcasts.
//  - Epilogue: the products go through shared memory in two halves of 16 output channels; each (o, tile) pair's 16
//    components are reduced by A^T m A, A^T = [[1,1,1,0],[0,1,-1,-1]], into its in-bounds 2x2 outputs.
// The filters come raw: WMODE 1 the forward layout w[O][R][3][3], WMODE 2 the input gradient's w[R][O][3][3] read rotated
// 180 degrees. Warps 8-15 load and transform each step's 8 x 32 (r, o) filters themselves (U = G g G^T): no separate
// transform launch, and 9 floats read per filter instead of a pre-transformed U's 16 written and re-read (on 512@4x4
// that U was 16.8 MB, past L2; measured: dX 512@4 82 -> 39 us, 256@8 47.5 -> 34.7 us).
template <int WMODE>
__device__ __forceinline__ void wfz_conv_body(const float* __restrict__ in, const float* __restrict__ wt,
    float* __restrict__ out, int N, int R, int IH, int IW, int O, int OH, int OW, int pad, int tilesH, int tilesW,
    int rPerSplit)
{
    __shared__ __align__(16) float sm[2 * 16 * WFZ_CK * 32];   // U | V; the epilogue reuses all of it
    float* us = sm;
    float* vs = sm + 16 * WFZ_CK * 32;
    const int tid = threadIdx.x;
    const int tilesPerImage = tilesH * tilesW;
    const int P = N * tilesPerImage;
    const int p0 = blockIdx.x * WFZ_BP, o0 = blockIdx.y * WFZ_BO;
    // Reduction split: blockIdx.z takes channels [rBegin, rEnd) (rPerSplit a multiple of WFZ_CK) and writes its own
    // partial output slice; with one split the slice is the output itself.
    const int rBegin = blockIdx.z * rPerSplit, rEnd = min(R, rBegin + rPerSplit);

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

    const int x = tid >> 5, lane = tid & 31;
    const int oo = (lane >> 3) * 8, pp = (lane & 7) * 4;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    // Filter loader: thread tid - 256 owns one (r, o) filter of the step, lanes along r for the forward
    // layout (8 consecutive r are 72 contiguous floats) and along o for the input-gradient layout.
    const int ft = tid - 256;
    const int fr = WMODE == 1 ? (ft & 7) : (ft >> 5), fo = WMODE == 1 ? (ft >> 3) : (ft & 31);
    float d[16], wv[9];
    auto load = [&](int r0)
    {
        const int r = r0 + lc;
        const bool rv = r < rEnd;
        if (patchLoader)
        {
            const float* src = in + inOff + (long long)r * chanStride;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    d[i * 4 + j] = (rv && ((rowOk >> i) & 1u) && ((colOk >> j) & 1u)) ? __ldg(src + i * IW + j) : 0.0f;
        }
        if (!patchLoader)
        {
            const int wr = r0 + fr, wo = o0 + fo;
            const bool wvOk = wr < rEnd && wo < O;
            const float* src = wt + (WMODE == 1 ? (long long)wo * R + wr : (long long)wr * O + wo) * 9;
            #pragma unroll
            for (int q = 0; q < 9; q++) wv[q] = wvOk ? __ldg(src + q) : 0.0f;
        }
    };
    auto store = [&]()
    {
        if (patchLoader)
        {
            // V = B^T d B, B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]].
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
        if (!patchLoader)
        {
            // U = G g G^T, G = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]], the unfused path's filter transform.
            float g[3][3];
            #pragma unroll
            for (int i = 0; i < 3; i++)
                #pragma unroll
                for (int j = 0; j < 3; j++) g[i][j] = WMODE == 1 ? wv[i * 3 + j] : wv[(2 - i) * 3 + (2 - j)];
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
            float* uu = us + fr * 32 + fo;
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float a = t[i][0], b = t[i][1], c = t[i][2];
                uu[(i * 4 + 0) * WFZ_CK * 32] = a;
                uu[(i * 4 + 1) * WFZ_CK * 32] = 0.5f * (a + b + c);
                uu[(i * 4 + 2) * WFZ_CK * 32] = 0.5f * (a - b + c);
                uu[(i * 4 + 3) * WFZ_CK * 32] = c;
            }
        }
    };

    const int steps = rBegin < rEnd ? (rEnd - rBegin + WFZ_CK - 1) / WFZ_CK : 0;
    out += (long long)blockIdx.z * N * O * OH * OW;
    load(rBegin);
    for (int step = 0; step < steps; step++)
    {
        __syncthreads();
        store();
        __syncthreads();
        if (step + 1 < steps) load(rBegin + (step + 1) * WFZ_CK);
        const float* ub = us + x * WFZ_CK * 32 + oo;
        const float* vb = vs + x * WFZ_CK * 32 + pp;
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
    }

    // Epilogue: ms[x][ol][pl] for 16 output channels at a time (16 * 16 * 32 floats = the whole shared buffer), one
    // (o, tile) pair per thread per half.
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

#define WFZ_ENTRY(NAME, MODE) \
extern ""C"" __global__ __launch_bounds__(512, 1) void NAME(const float* __restrict__ in, const float* __restrict__ wt, \
    float* __restrict__ out, int N, int R, int IH, int IW, int O, int OH, int OW, int pad, int tilesH, int tilesW, \
    int rPerSplit) \
{ wfz_conv_body<MODE>(in, wt, out, N, R, IH, IW, O, OH, OW, pad, tilesH, tilesW, rPerSplit); }
WFZ_ENTRY(wfz_conv_w, 1)
WFZ_ENTRY(wfz_conv_wf, 2)
";

    public static string[] GetKernelNames() => new[] { "wfz_conv_w", "wfz_conv_wf" };
}
