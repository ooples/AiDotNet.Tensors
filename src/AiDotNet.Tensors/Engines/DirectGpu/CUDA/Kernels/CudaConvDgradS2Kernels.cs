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
template <int G>
__device__ __forceinline__ void conv2d_dgrad_s2k3_body(
    const float* __restrict__ dy, const float* __restrict__ w, float* __restrict__ dx,
    int N, int C, int K, int OH, int OW, int coPerSplit)
{
    constexpr int NT = S2_GT * G, YL = (128 + NT - 1) / NT, WL = (576 + NT - 1) / NT;
    __shared__ __align__(16) float Ys[2][S2_CO][S2_YLD];
    __shared__ __align__(16) float Ws[2][S2_CO][9][S2_C];
    const int tid = threadIdx.x, g = tid >> S2_GT_LOG2, ct = tid & (S2_GT - 1);
    // The block's dY rows (n, a) are R0 .. R0 + RB - 1 of the N * OH; it stages one more (the row below its last) and
    // a zero column OW, at a 16-byte-aligned row stride.
    const int RB = S2_CELLS / OW, YS = OW + 4;
    const int rows = N * OH, H = 2 * OH, W = 2 * OW;
    const int R0 = blockIdx.x * RB, c0 = blockIdx.y * S2_C;
    const int coBegin = blockIdx.z * coPerSplit, coEnd = min(K, coBegin + coPerSplit);
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

    float4 py[YL], pw[WL];
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

    float acc[4][4][4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int q = 0; q < 4; q++)
            #pragma unroll
            for (int p = 0; p < 4; p++) acc[j][q][p] = 0.0f;

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
        if (more) store(buf ^ 1);
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
";

    /// <summary>The in-block output-channel group counts compiled.</summary>
    internal static readonly int[] Groups = { 1, 2, 4 };

    public static string[] GetKernelNames() => System.Array.ConvertAll(Groups, KernelName);

    internal static string KernelName(int groups) => $"conv2d_dgrad_s2k3_g{groups}";
}
