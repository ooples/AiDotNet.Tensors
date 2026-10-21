using System.Linq;
// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Convolution forward as an implicit GEMM with 8 x 8 per-thread tiles (the weight-gradient r8 kernels' structure,
/// <see cref="CudaConvWgradKernels"/>, applied to the forward): y[b, co, oh, ow] = sum over k = (ci, kh, kw) of
/// w[co, k] * x[b, ci, oh*sH - pH + kh*dH, ow*sW - pW + kw*dW]. Rows are output channels, columns output pixels
/// (b, oh, ow), K the filter taps. Nsight Compute on the engine's 64 x 64 / 256-thread forward (RTX 3080, ResNet-18's
/// stride-2 layers) found it waiting on shared-memory reads (short-scoreboard the top stall, issue active ~50%) with 4 x 4
/// per-thread tiles and 2-3 blocks per SM. Here
/// <list type="bullet">
/// <item>each thread owns an 8 x 8 block of the tile: 64 FMAs per K step against four 16-byte shared loads;</item>
/// <item>the weights' K tile is read as 16-byte loads (a row of w is contiguous in K; the host admits input channel
/// counts that are multiples of four, so every K slice and row start is 16-byte aligned);</item>
/// <item>a block computes every column's input offset and window origin once, into shared memory, and each thread keeps
/// its K lane's (ci, tap) incrementally, so a gathered input element costs a table read, an add, two unsigned range
/// checks and a load;</item>
/// <item>split-K partial slices have the output's layout and are summed in split order by conv2d_wgrad_splitk_reduce
/// (deterministic).</item>
/// </list>
/// </summary>
internal static class CudaConvFpropR8Kernels
{
    /// <summary>The K depth of every tile; the host rounds split-K slices to it.</summary>
    internal const int KTile = 16;

    /// <summary>The (rows, columns) output tiles compiled and offered as forward candidates.</summary>
    internal static readonly (int M, int N)[] TileShapes = { (64, 128), (128, 64) };

    public static string GetSource() => @"
// Column table entry: the column's input offset (image base + window origin), its window origin, and validity.
struct FrCol { int off; int ih0; int iw0; int valid; };

template <int BM, int BN, int FR_BK>
__device__ __forceinline__ void conv2d_fprop_r8_body(
    const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit)
{
    constexpr int THREADS = (BM / 8) * (BN / 8);
    constexpr int GROUPS = THREADS / FR_BK;           // loader lanes per K position
    constexpr int RB = BN / GROUPS;                   // input elements each thread loads per tile
    constexpr int VA = BM * FR_BK / 4 / THREADS;      // 16-byte weight loads each thread issues per tile
    __shared__ __align__(16) float As[2][FR_BK][BM + 4];
    __shared__ __align__(16) float Bs[2][FR_BK][BN + 4];
    __shared__ FrCol cols[BN];

    const int L = outH * outW, khw = kH * kW, K = inC * khw, HW = H * W, Ncols = batch * L;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int kBegin = blockIdx.z * kPerSplit, kEnd = min(K, kBegin + kPerSplit);
    const int tid = threadIdx.x;

    for (int c = tid; c < BN; c += THREADS)
    {
        int n = n0 + c;
        FrCol e;
        e.valid = n < Ncols;
        int nn = e.valid ? n : 0;
        int b = nn / L, l = nn - b * L, oh = l / outW, ow = l - oh * outW;
        e.ih0 = oh * sH - pH;
        e.iw0 = ow * sW - pW;
        e.off = b * inC * HW + e.ih0 * W + e.iw0;
        cols[c] = e;
    }

    // This thread's K lane: k = kBegin + tile * FR_BK + klane, kept as (ci, kh, kw).
    const int klane = tid % FR_BK, group = tid / FR_BK;
    int kAbs = kBegin + klane;
    int kci = kAbs / khw, ktap = kAbs - kci * khw;
    int kkh = ktap / kW, kkw = ktap - kkh * kW;

    float rb[RB];
    float4 rav[VA];
    int tileK = kBegin;
    auto load = [&]()
    {
        // Weights: thread tid + v * THREADS loads row mm = idx / (FR_BK / 4), K quad qq = idx % (FR_BK / 4).
        #pragma unroll
        for (int v = 0; v < VA; v++)
        {
            int idx = tid + v * THREADS, mm = idx / (FR_BK / 4), qq = idx - mm * (FR_BK / 4);
            int m = m0 + mm, k = tileK + qq * 4;
            rav[v] = (m < outC && k < kEnd)   // kEnd and k are multiples of four: the quad is all in or all out
                ? __ldg(reinterpret_cast<const float4*>(weight + (long long)m * K + k))
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        bool kIn = kAbs < kEnd;
        int khd = kkh * dH, kwd = kkw * dW;
        int kOff = kci * HW + khd * W + kwd;
        #pragma unroll
        for (int j = 0; j < RB; j++)
        {
            FrCol e = cols[group + GROUPS * j];
            bool ok = kIn && e.valid && (unsigned)(e.ih0 + khd) < (unsigned)H && (unsigned)(e.iw0 + kwd) < (unsigned)W;
            rb[j] = ok ? __ldg(input + e.off + kOff) : 0.0f;
        }
    };
    auto store = [&](int buf)
    {
        #pragma unroll
        for (int v = 0; v < VA; v++)
        {
            int idx = tid + v * THREADS, mm = idx / (FR_BK / 4), qq = idx - mm * (FR_BK / 4);
            As[buf][qq * 4][mm] = rav[v].x;
            As[buf][qq * 4 + 1][mm] = rav[v].y;
            As[buf][qq * 4 + 2][mm] = rav[v].z;
            As[buf][qq * 4 + 3][mm] = rav[v].w;
        }
        #pragma unroll
        for (int j = 0; j < RB; j++) Bs[buf][klane][group + GROUPS * j] = rb[j];
    };
    auto advance = [&]()
    {
        tileK += FR_BK;
        kAbs += FR_BK;
        kkw += FR_BK;
        while (kkw >= kW) { kkw -= kW; if (++kkh == kH) { kkh = 0; ++kci; } }
    };

    const int tx = tid % (BN / 8), ty = tid / (BN / 8);
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    __syncthreads();   // the column table
    const int tiles = kBegin < kEnd ? (kEnd - kBegin + FR_BK - 1) / FR_BK : 0;
    int buf = 0;
    if (tiles > 0)
    {
        load();
        store(0);
        __syncthreads();
    }
    for (int tIdx = 0; tIdx < tiles; tIdx++)
    {
        bool more = tIdx + 1 < tiles;
        if (more) { advance(); load(); }
        #pragma unroll
        for (int kk = 0; kk < FR_BK; kk++)
        {
            float4 a0 = *reinterpret_cast<const float4*>(&As[buf][kk][ty * 4]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[buf][kk][BM / 2 + ty * 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[buf][kk][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[buf][kk][BN / 2 + tx * 4]);
            float a[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
            float b[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] += a[i] * b[j];
        }
        if (more)
        {
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }

    // Rows ty*4 + {0..3} and BM/2 + ty*4 + {0..3}; columns tx*4 + {0..3} and BN/2 + tx*4 + {0..3}. Column n is pixel
    // (b, l): y[b][co][l]. A quad of columns stays in one image when L is a multiple of four (n0 and the quad offsets
    // are), so it is one 16-byte store.
    float* slice = out + (long long)blockIdx.z * outC * Ncols;
    const bool vec = (L & 3) == 0;
    #pragma unroll
    for (int h = 0; h < 2; h++)
    {
        int n = n0 + h * (BN / 2) + tx * 4;
        int b = n / L, l = n - b * L;
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            int m = m0 + (i >> 2) * (BM / 2) + ty * 4 + (i & 3);
            if (m >= outC) continue;
            if (vec && n + 3 < Ncols)
            {
                *reinterpret_cast<float4*>(slice + ((long long)b * outC + m) * L + l) =
                    make_float4(acc[i][4 * h], acc[i][4 * h + 1], acc[i][4 * h + 2], acc[i][4 * h + 3]);
            }
            else
            {
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    int nj = n + j;
                    if (nj >= Ncols) break;
                    int bj = nj / L, lj = nj - bj * L;
                    slice[((long long)bj * outC + m) * L + lj] = acc[i][4 * h + j];
                }
            }
        }
    }
}

#define FR_KERNEL(NAME, BM, BN, BK) \
extern ""C"" __global__ __launch_bounds__((BM / 8) * (BN / 8)) void NAME( \
    const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ out, \
    int batch, int inC, int H, int W, int outC, int outH, int outW, \
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit) \
{ \
    conv2d_fprop_r8_body<BM, BN, BK>(input, weight, out, batch, inC, H, W, outC, outH, outW, \
        kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit); \
}
" + string.Concat(TileShapes.Select(t => $"FR_KERNEL({KernelName(t.M, t.N)}, {t.M}, {t.N}, {KTile})\n"));

    public static string[] GetKernelNames() => TileShapes.Select(t => KernelName(t.M, t.N)).ToArray();

    internal static string KernelName(int tileM, int tileN) => $"conv2d_fprop_r8_{tileM}x{tileN}";
}
