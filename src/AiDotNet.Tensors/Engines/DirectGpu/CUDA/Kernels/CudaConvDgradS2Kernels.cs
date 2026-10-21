// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Input gradient of a 3 x 3, stride-2, padding-1 convolution whose input is exactly twice its output (ResNet's
/// downsampling convolutions), all four sub-pixel phases in one kernel. With x = 2a + ph, y = 2b + pw, the forward
/// y[o] = sum x[2o - 1 + k] gives
/// <code>
/// dX[2a    ][2b    ] = dY[a][b] w11
/// dX[2a    ][2b + 1] = dY[a][b + 1] w10 + dY[a][b] w12
/// dX[2a + 1][2b    ] = dY[a + 1][b] w01 + dY[a][b] w21
/// dX[2a + 1][2b + 1] = dY[a + 1][b + 1] w00 + dY[a + 1][b] w02 + dY[a][b + 1] w20 + dY[a][b] w22
/// </code>
/// summed over the output channels co, so every dY cell (a, b) owns a 2 x 2 block of dX fed by the 2 x 2 dY window at
/// (a, b): nine FMAs per (cell, co, c). The phase GEMM (conv2d_dgrad_ce64_phase) runs the phases as separate
/// GEMMs of 1, 2, 2 and 4 taps: unequal blocks, a dY element read per tap, and on the deep layers a split-K reduce to
/// fill the device. Here every block does the same work: 32 cells (whole dY rows) x 32 input channels, 64 threads of
/// 4 cells x 4 channels x 4 phases; per co a thread reads its cells' two 5-wide dY rows and the nine taps of its four
/// channels (four and nine 16-byte shared loads) for 144 FMAs. The co reduction may be split across blockIdx.z into
/// partial slices of dX's layout, summed in split order by conv2d_wgrad_splitk_reduce (deterministic).
/// </summary>
/// <remarks>
/// CUDA only for now, as a specialization of an operation every backend already serves rather than a CUDA-only
/// capability: the other five keep computing this gradient through their own tuned input-gradient candidates, so
/// nothing is unavailable there. The follow-ups for the ports are HIP #1479, Metal #1480, OpenCL #1481,
/// Vulkan #1482 and WebGPU #1483.
/// </remarks>
internal static class CudaConvDgradS2Kernels
{
    /// <summary>Output channels staged per step; the host rounds split slices to it.</summary>
    internal const int CoTile = 8;

    /// <summary>dY cells per block (whole rows: the host admits output widths that divide it).</summary>
    internal const int CellTile = 32;

    /// <summary>Input-gradient channels per block.</summary>
    internal const int ChannelTile = 32;

    /// <summary>
    /// Threads per in-block output-channel group. The host multiplies it by the group count for the block size and
    /// the kernel derives its launch bounds, thread count and thread roles from the same value, so the two geometries
    /// cannot drift apart.
    /// </summary>
    internal const int GroupThreads = 64;

    // log2(GroupThreads), so the thread roles stay a shift and a mask rather than a signed division.
    private const int GroupThreadsLog2 = 6;

    public static string GetSource() => $@"
#define S2_GT {GroupThreads}
#define S2_GT_LOG2 {GroupThreadsLog2}
" + @"
static_assert((1 << S2_GT_LOG2) == S2_GT, ""S2_GT_LOG2 must be log2(S2_GT)."");
#define S2_CO 8
#define S2_CELLS 32
#define S2_C 32
#define S2_YLD 72   // a staged co's dY rows: (S2_CELLS / OW + 1) x (OW + 4) <= 72 for OW in {4, 8, 16, 32}

// G groups of S2_GT threads share each staged chunk: group g accumulates co0 + g, co0 + g + G, ... of it, and the groups'
// sums are added in group order through shared memory at the end (deterministic).
// SC (the projection-shortcut pair): dX also receives the input gradient of a 1 x 1, stride-2, padding-0 convolution of
// the same input, from dy2 [N][K2][OH][OW] and ws [K2][C]. That convolution reads x[2a][2b], this one's centre tap, so
// its gradient lands in phase (0, 0) alone. Each step stages a chunk of both reductions (the shortcut's into its own
// buffers; co2PerSplit per blockIdx.z) and adds both, so the pair costs one loop of max(K, K2) / S2_CO steps, and dX is
// written once (no second launch, no accumulation add).
template <int G, bool SC = false>
__device__ __forceinline__ void conv2d_dgrad_s2k3_body(
    const float* __restrict__ dy, const float* __restrict__ w, float* __restrict__ dx,
    int N, int C, int K, int OH, int OW, int coPerSplit,
    const float* __restrict__ dy2 = nullptr, const float* __restrict__ ws = nullptr, int K2 = 0, int co2PerSplit = 0)
{
    constexpr int NT = S2_GT * G, YL = (128 + NT - 1) / NT, WL = (576 + NT - 1) / NT;
    __shared__ __align__(16) float Ys[2][S2_CO][S2_YLD];
    __shared__ __align__(16) float Ws[2][S2_CO][9][S2_C];
    __shared__ __align__(16) float Ys2[SC ? 2 : 1][S2_CO][S2_YLD];   // the shortcut's dY chunk
    __shared__ __align__(16) float Ws2[SC ? 2 : 1][S2_CO][S2_C];     // the shortcut's weights ws[co2][c0 ..]
    const int tid = threadIdx.x, g = tid >> S2_GT_LOG2, ct = tid & (S2_GT - 1);
    // The block's dY rows (n, a) are R0 .. R0 + RB - 1 of the N * OH; it stages one more (the row below its last) and
    // a zero column OW, at a 16-byte-aligned row stride.
    const int RB = S2_CELLS / OW, YS = OW + 4;
    const int rows = N * OH, H = 2 * OH, W = 2 * OW;
    const int R0 = blockIdx.x * RB, c0 = blockIdx.y * S2_C;
    const int coBegin = blockIdx.z * coPerSplit, coEnd = min(K, coBegin + coPerSplit);
    const int co2Begin = SC ? blockIdx.z * co2PerSplit : 0, co2End = SC ? min(K2, co2Begin + co2PerSplit) : 0;
    dx += (long long)blockIdx.z * N * C * H * W;

    // Compute role: cells cg * 4 .. + 3 of the block (one dY row l, columns b0 ..), channels c0 + chg * 4 ...
    const int cg = ct & 7, chg = ct >> 3;
    const int l = (cg * 4) / OW, b0 = cg * 4 - l * OW;
    const int R = R0 + l;
    const int aRow = R % OH;
    const bool lastRow = aRow == OH - 1;   // its row below is outside the image: zero

    // Staging roles, fixed for the whole reduction. dY: float4 f = tid + NT i (i < YL) of a chunk's
    // S2_CO x (RB + 1) x OW / 4 (at most 128); its global offset is yOff[i] + co0 * OH * OW. Weights: float4
    // f = tid + NT i (i < WL)
    // of a chunk's S2_CO x 72 (a co's 32 channels x 9 taps are contiguous); element 4f' + e of a co is (channel
    // (4f' + e) / 9, tap (4f' + e) % 9).
    const int q4 = OW / 4, yPerCo = (RB + 1) * q4, yCount = S2_CO * yPerCo;
    int yOff[YL], yDst[YL];
    bool yOk[YL];
    #pragma unroll
    for (int i = 0; i < YL; i++)
    {
        const int f = tid + NT * i;
        const int kl = f / yPerCo, rem = f - kl * yPerCo, r = rem / q4, c4 = rem - r * q4;
        const int rr = R0 + r;
        yOk[i] = f < yCount && rr < rows;
        const int n = rr / OH, a = rr - n * OH;
        yOff[i] = yOk[i] ? ((n * K + kl) * OH + a) * OW + c4 * 4 : 0;
        yDst[i] = f < yCount ? kl * S2_YLD + r * YS + c4 * 4 : -1;
    }
    const int cRem = (C - c0) * 9;   // a co's valid floats in this channel slice (a multiple of 4: C % 4 == 0)
    // The zero column OW (and the pad up to the row stride) of every staged row in both buffers.
    for (int z = tid; z < 2 * S2_CO * (RB + 1); z += NT)
    {
        const int bsel = z / (S2_CO * (RB + 1)), t = z - bsel * (S2_CO * (RB + 1));
        const int kl = t / (RB + 1), r = t - kl * (RB + 1);
        *reinterpret_cast<float4*>(&Ys[bsel][kl][r * YS + OW]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    // The shortcut's staging: dY offsets over K2; weights as float4 f = tid + NT i (i < WL2) of S2_CO x 8.
    constexpr int WL2 = SC ? (S2_CO * 8 + NT - 1) / NT : 1;
    int yOff2[YL];
    #pragma unroll
    for (int i = 0; i < YL; i++)
    {
        const int f = tid + NT * i;
        const int kl = f / yPerCo, rem = f - kl * yPerCo, r = rem / q4;
        const int rr = R0 + r, n = rr / OH, a = rr - n * OH;
        yOff2[i] = (SC && yOk[i]) ? ((n * K2 + kl) * OH + a) * OW + (rem - r * q4) * 4 : 0;
    }

    float4 py[YL], pw[WL], py2[SC ? YL : 1], pw2[WL2];
    auto load = [&](int co0)
    {
        #pragma unroll
        for (int i = 0; i < YL; i++)
        {
            const int kl = (tid + NT * i) / yPerCo;
            py[i] = (yOk[i] && co0 + kl < coEnd)
                ? __ldg(reinterpret_cast<const float4*>(dy + yOff[i] + (long long)co0 * OH * OW)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        #pragma unroll
        for (int i = 0; i < WL; i++)
        {
            const int f = tid + NT * i, kl = f / 72, f4 = f - kl * 72;
            const int co = co0 + kl;
            pw[i] = (f < 576 && co < coEnd && f4 * 4 < cRem)
                ? __ldg(reinterpret_cast<const float4*>(w + ((long long)co * C + c0) * 9) + f4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto store = [&](int buf)
    {
        #pragma unroll
        for (int i = 0; i < YL; i++)
            if (yDst[i] >= 0) *reinterpret_cast<float4*>(&Ys[buf][0][0] + yDst[i]) = py[i];
        #pragma unroll
        for (int i = 0; i < WL; i++)
        {
            const int f = tid + NT * i, kl = f / 72, f4 = f - kl * 72;
            if (f >= 576) break;
            const float v[4] = { pw[i].x, pw[i].y, pw[i].z, pw[i].w };
            #pragma unroll
            for (int e = 0; e < 4; e++)
            {
                const int fe = f4 * 4 + e, cl = fe / 9, tap = fe - cl * 9;
                Ws[buf][kl][tap][cl] = v[e];
            }
        }
    };

    auto load2 = [&](int co0)
    {
        #pragma unroll
        for (int i = 0; i < YL; i++)
        {
            const int kl = (tid + NT * i) / yPerCo;
            py2[i] = (yOk[i] && co0 + kl < co2End)
                ? __ldg(reinterpret_cast<const float4*>(dy2 + yOff2[i] + (long long)co0 * OH * OW)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        #pragma unroll
        for (int i = 0; i < WL2; i++)
        {
            const int f = tid + NT * i, kl = f >> 3, c4 = f & 7;
            pw2[i] = (f < S2_CO * 8 && co0 + kl < co2End && c0 + 4 * c4 < C)
                ? __ldg(reinterpret_cast<const float4*>(ws + (long long)(co0 + kl) * C + c0 + 4 * c4)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto store2 = [&](int b)
    {
        #pragma unroll
        for (int i = 0; i < YL; i++)
            if (yDst[i] >= 0) *reinterpret_cast<float4*>(&Ys2[b][0][0] + yDst[i]) = py2[i];
        #pragma unroll
        for (int i = 0; i < WL2; i++)
        {
            const int f = tid + NT * i;
            if (f >= S2_CO * 8) break;
            *reinterpret_cast<float4*>(&Ws2[b][f >> 3][4 * (f & 7)]) = pw2[i];
        }
    };

    float acc[4][4][4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int q = 0; q < 4; q++)
            #pragma unroll
            for (int p = 0; p < 4; p++) acc[j][q][p] = 0.0f;

    // Steps of both reductions (a chunk past a range's end loads zeros and adds nothing).
    const int steps1 = coBegin < coEnd ? (coEnd - coBegin + S2_CO - 1) / S2_CO : 0;
    const int steps2 = co2Begin < co2End ? (co2End - co2Begin + S2_CO - 1) / S2_CO : 0;
    const int steps = SC ? max(steps1, steps2) : steps1;
    int buf = 0;
    if (steps > 0)
    {
        load(coBegin); store(0);
        if constexpr (SC) { load2(co2Begin); store2(0); }
    }
    __syncthreads();
    for (int step = 0; step < steps; step++)
    {
        const int co0 = coBegin + step * S2_CO;
        const bool more = step + 1 < steps;
        if (more)
        {
            load(co0 + S2_CO);
            if constexpr (SC) load2(co2Begin + (step + 1) * S2_CO);
        }
        #pragma unroll
        for (int kk = 0; kk < S2_CO / G; kk++)
        {
            const int k = g + G * kk;
            const float* yp = &Ys[buf][k][l * YS + b0];
            const float4 ya = *reinterpret_cast<const float4*>(yp);
            float4 yb = *reinterpret_cast<const float4*>(yp + YS);
            const float ya4 = yp[4];
            float yb4 = yp[YS + 4];
            if (lastRow) { yb = make_float4(0.0f, 0.0f, 0.0f, 0.0f); yb4 = 0.0f; }
            const float y0[5] = { ya.x, ya.y, ya.z, ya.w, ya4 };
            const float y1[5] = { yb.x, yb.y, yb.z, yb.w, yb4 };
            float4 wt[9];
            #pragma unroll
            for (int t = 0; t < 9; t++) wt[t] = *reinterpret_cast<const float4*>(&Ws[buf][k][t][chg * 4]);
            #pragma unroll
            for (int q = 0; q < 4; q++)
            {
                float wq[9];
                #pragma unroll
                for (int t = 0; t < 9; t++) wq[t] = q == 0 ? wt[t].x : q == 1 ? wt[t].y : q == 2 ? wt[t].z : wt[t].w;
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    acc[j][q][0] = fmaf(y0[j], wq[4], acc[j][q][0]);
                    acc[j][q][1] = fmaf(y0[j + 1], wq[3], fmaf(y0[j], wq[5], acc[j][q][1]));
                    acc[j][q][2] = fmaf(y1[j], wq[1], fmaf(y0[j], wq[7], acc[j][q][2]));
                    acc[j][q][3] = fmaf(y1[j + 1], wq[0], fmaf(y1[j], wq[2], fmaf(y0[j + 1], wq[6], fmaf(y0[j], wq[8], acc[j][q][3]))));
                }
            }
        }
        if constexpr (SC)
        {
            #pragma unroll
            for (int kk = 0; kk < S2_CO / G; kk++)
            {
                const int k = g + G * kk;
                const float4 ya = *reinterpret_cast<const float4*>(&Ys2[buf][k][l * YS + b0]);
                const float4 w4 = *reinterpret_cast<const float4*>(&Ws2[buf][k][chg * 4]);
                const float y0[4] = { ya.x, ya.y, ya.z, ya.w }, wq[4] = { w4.x, w4.y, w4.z, w4.w };
                #pragma unroll
                for (int q = 0; q < 4; q++)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) acc[j][q][0] = fmaf(y0[j], wq[q], acc[j][q][0]);
            }
        }
        if (more)
        {
            store(buf ^ 1);
            if constexpr (SC) store2(buf ^ 1);
        }
        __syncthreads();
        buf ^= 1;
    }

    // The groups' sums in group order: group gg's accumulators go through red[v][ct] (the idle weight buffers), group 0
    // adds them.
    if (G > 1)
    {
        float* red = &Ws[0][0][0][0];
        #pragma unroll
        for (int gg = 1; gg < G; gg++)
        {
            if (g == gg)
            {
                #pragma unroll
                for (int v = 0; v < 64; v++) red[v * S2_GT + ct] = acc[v >> 4][(v >> 2) & 3][v & 3];
            }
            __syncthreads();
            if (g == 0)
            {
                #pragma unroll
                for (int v = 0; v < 64; v++) acc[v >> 4][(v >> 2) & 3][v & 3] += red[v * S2_GT + ct];
            }
            __syncthreads();
        }
        if (g != 0) return;
    }
    if (R >= rows) return;
    const int n = R / OH;
    #pragma unroll
    for (int q = 0; q < 4; q++)
    {
        const int c = c0 + chg * 4 + q;
        if (c >= C) break;
        float* row0 = dx + (((long long)n * C + c) * H + 2 * aRow) * W + 2 * b0;
        float* row1 = row0 + W;
        *reinterpret_cast<float4*>(row0) = make_float4(acc[0][q][0], acc[0][q][1], acc[1][q][0], acc[1][q][1]);
        *reinterpret_cast<float4*>(row0 + 4) = make_float4(acc[2][q][0], acc[2][q][1], acc[3][q][0], acc[3][q][1]);
        *reinterpret_cast<float4*>(row1) = make_float4(acc[0][q][2], acc[0][q][3], acc[1][q][2], acc[1][q][3]);
        *reinterpret_cast<float4*>(row1 + 4) = make_float4(acc[2][q][2], acc[2][q][3], acc[3][q][2], acc[3][q][3]);
    }
}

#define S2_ENTRY(G) \
extern ""C"" __global__ __launch_bounds__(S2_GT * G) void conv2d_dgrad_s2k3_g##G( \
    const float* __restrict__ dy, const float* __restrict__ w, float* __restrict__ dx, \
    int N, int C, int K, int OH, int OW, int coPerSplit) \
{ conv2d_dgrad_s2k3_body<G>(dy, w, dx, N, C, K, OH, OW, coPerSplit); }
S2_ENTRY(1)
S2_ENTRY(2)
S2_ENTRY(4)
#define S2SC_ENTRY(G) \
extern ""C"" __global__ __launch_bounds__(S2_GT * G) void conv2d_dgrad_s2k3sc_g##G( \
    const float* __restrict__ dy, const float* __restrict__ w, float* __restrict__ dx, \
    int N, int C, int K, int OH, int OW, int coPerSplit, \
    const float* __restrict__ dy2, const float* __restrict__ ws, int K2, int co2PerSplit) \
{ conv2d_dgrad_s2k3_body<G, true>(dy, w, dx, N, C, K, OH, OW, coPerSplit, dy2, ws, K2, co2PerSplit); }
S2SC_ENTRY(1)
S2SC_ENTRY(2)
S2SC_ENTRY(4)

// The 1 x 1, stride-2, padding-0 input gradient (ResNet's downsampling shortcuts): dX[c][2a][2b] = sum over co of
// dY[co][a][b] w[co][c], the other three sub-pixel phases zero. A GEMM over the dY cells, written straight into dX's
// layout with its zero phases (the phase kernel ran three tapless phase layers and, on the deep layers, a phase
// reduce). A block owns 32 consecutive dY cells (the host admits output widths that are multiples of four, so a
// thread's four cells share a row) x 32 channels; per co a thread reads its four cells' dY and its four channels'
// weights (one 16-byte shared load each) for 16 FMAs. Groups and split slices as in conv2d_dgrad_s2k3.
template <int G>
__device__ __forceinline__ void conv2d_dgrad_s2k1_body(
    const float* __restrict__ dy, const float* __restrict__ w, float* __restrict__ dx,
    int N, int C, int K, int OH, int OW, int coPerSplit)
{
    constexpr int NT = S2_GT * G, SL = (S2_CO * S2_CELLS / 4 + NT - 1) / NT;   // float4s per chunk, per operand: 64
    static_assert(S2_C == S2_CELLS, ""the two operands share one array shape"");
    __shared__ __align__(16) float s2k1[2][2][S2_CO][S2_CELLS];   // [dY | weights][buffer][co][cell | channel]
    float (*Ys)[S2_CO][S2_CELLS] = s2k1[0];
    float (*Ws)[S2_CO][S2_C] = s2k1[1];
    const int tid = threadIdx.x, g = tid >> S2_GT_LOG2, ct = tid & (S2_GT - 1);
    const int HW = OH * OW, M = N * HW, H = 2 * OH, W = 2 * OW;
    const int m0 = blockIdx.x * S2_CELLS, c0 = blockIdx.y * S2_C;
    const int coBegin = blockIdx.z * coPerSplit, coEnd = min(K, coBegin + coPerSplit);
    dx += (long long)blockIdx.z * N * C * H * W;

    // Compute role: cells m0 + cg * 4 .. + 3 (one dY row), channels c0 + chg * 4 ...
    const int cg = ct & 7, chg = ct >> 3;

    // Staging roles, fixed for the whole reduction: float4 f = tid + NT i (i < SL) of a chunk's S2_CO x 8 per operand.
    // dY: co f >> 3, cells m0 + 4 (f & 7) .. (four consecutive cells of one row: contiguous in dY); weights: co f >> 3,
    // channels c0 + 4 (f & 7) .. (w[co][c], contiguous in c; C % 4 == 0).
    int yOff[SL];
    bool yOk[SL], wOk[SL];
    #pragma unroll
    for (int i = 0; i < SL; i++)
    {
        const int f = tid + NT * i, m = m0 + 4 * (f & 7);
        yOk[i] = f < S2_CO * 8 && m < M;
        const int n = m / HW, rem = m - n * HW;
        yOff[i] = yOk[i] ? (n * K + (f >> 3)) * HW + rem : 0;
        wOk[i] = f < S2_CO * 8 && c0 + 4 * (f & 7) < C;
    }

    float4 py[SL], pw[SL];
    auto load = [&](int co0)
    {
        #pragma unroll
        for (int i = 0; i < SL; i++)
        {
            const int f = tid + NT * i, co = co0 + (f >> 3);
            const bool live = co < coEnd;
            py[i] = (yOk[i] && live) ? __ldg(reinterpret_cast<const float4*>(dy + yOff[i] + (long long)co0 * HW))
                                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            pw[i] = (wOk[i] && live) ? __ldg(reinterpret_cast<const float4*>(w + (long long)co * C + c0 + 4 * (f & 7)))
                                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto store = [&](int buf)
    {
        #pragma unroll
        for (int i = 0; i < SL; i++)
        {
            const int f = tid + NT * i;
            if (f >= S2_CO * 8) break;
            *reinterpret_cast<float4*>(&Ys[buf][f >> 3][4 * (f & 7)]) = py[i];
            *reinterpret_cast<float4*>(&Ws[buf][f >> 3][4 * (f & 7)]) = pw[i];
        }
    };

    float acc[4][4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int q = 0; q < 4; q++) acc[j][q] = 0.0f;

    int buf = 0;
    if (coBegin < coEnd) { load(coBegin); store(0); }
    __syncthreads();
    for (int co0 = coBegin; co0 < coEnd; co0 += S2_CO)
    {
        const bool more = co0 + S2_CO < coEnd;
        if (more) load(co0 + S2_CO);
        #pragma unroll
        for (int kk = 0; kk < S2_CO / G; kk++)
        {
            const int k = g + G * kk;
            const float4 y4 = *reinterpret_cast<const float4*>(&Ys[buf][k][cg * 4]);
            const float4 w4 = *reinterpret_cast<const float4*>(&Ws[buf][k][chg * 4]);
            const float y[4] = { y4.x, y4.y, y4.z, y4.w }, wq[4] = { w4.x, w4.y, w4.z, w4.w };
            #pragma unroll
            for (int j = 0; j < 4; j++)
                #pragma unroll
                for (int q = 0; q < 4; q++) acc[j][q] = fmaf(y[j], wq[q], acc[j][q]);
        }
        if (more) store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }

    if (G > 1)
    {
        float* red = &s2k1[0][0][0][0];   // 16 x S2_GT floats: all four staging buffers, idle now
        #pragma unroll
        for (int gg = 1; gg < G; gg++)
        {
            if (g == gg)
            {
                #pragma unroll
                for (int v = 0; v < 16; v++) red[v * S2_GT + ct] = acc[v >> 2][v & 3];
            }
            __syncthreads();
            if (g == 0)
            {
                #pragma unroll
                for (int v = 0; v < 16; v++) acc[v >> 2][v & 3] += red[v * S2_GT + ct];
            }
            __syncthreads();
        }
        if (g != 0) return;
    }
    const int m = m0 + cg * 4;
    if (m >= M) return;
    const int n = m / HW, rem = m - n * HW, a = rem / OW, b0 = rem - a * OW;
    const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int q = 0; q < 4; q++)
    {
        const int c = c0 + chg * 4 + q;
        if (c >= C) break;
        float* row0 = dx + (((long long)n * C + c) * H + 2 * a) * W + 2 * b0;
        float* row1 = row0 + W;
        *reinterpret_cast<float4*>(row0) = make_float4(acc[0][q], 0.0f, acc[1][q], 0.0f);
        *reinterpret_cast<float4*>(row0 + 4) = make_float4(acc[2][q], 0.0f, acc[3][q], 0.0f);
        *reinterpret_cast<float4*>(row1) = z;
        *reinterpret_cast<float4*>(row1 + 4) = z;
    }
}

#define S2K1_ENTRY(G) \
extern ""C"" __global__ __launch_bounds__(S2_GT * G) void conv2d_dgrad_s2k1_g##G( \
    const float* __restrict__ dy, const float* __restrict__ w, float* __restrict__ dx, \
    int N, int C, int K, int OH, int OW, int coPerSplit) \
{ conv2d_dgrad_s2k1_body<G>(dy, w, dx, N, C, K, OH, OW, coPerSplit); }
S2K1_ENTRY(1)
S2K1_ENTRY(2)
S2K1_ENTRY(4)
";

    /// <summary>The in-block output-channel group counts compiled.</summary>
    internal static readonly int[] Groups = { 1, 2, 4 };

    public static string[] GetKernelNames() =>
        System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Concat(System.Array.ConvertAll(Groups, KernelName),
            System.Linq.Enumerable.Concat(System.Array.ConvertAll(Groups, KernelNameK1),
                System.Array.ConvertAll(Groups, KernelNamePair))));

    internal static string KernelName(int groups) => $"conv2d_dgrad_s2k3_g{groups}";

    internal static string KernelNameK1(int groups) => $"conv2d_dgrad_s2k1_g{groups}";

    internal static string KernelNamePair(int groups) => $"conv2d_dgrad_s2k3sc_g{groups}";
}
