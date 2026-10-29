using System.Linq;
// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Convolution weight gradient as an implicit GEMM built to keep the FMA pipe busy:
/// dW[co, p] = sum over k = (b, oh, ow) of dY[b, co, oh, ow] * X[b, ci, oh*sH - pH + kh*dH, ow*sW - pW + kw*dW], with
/// p = (ci, kh, kw). Nsight Compute on the earlier kernels (RTX 3080, ResNet-18 layers) found them instruction-overhead
/// bound: only 40-46% of executed instructions were FMAs, the rest per-element index arithmetic and bounds checks. Here
/// <list type="bullet">
/// <item>each thread owns an 8 x 8 block of the output tile (64 FMAs per K step against four 16-byte shared loads, in the
/// split-quad layout that keeps those loads conflict-free);</item>
/// <item>a block computes every column's input offset and kernel shifts once, into shared memory, and each thread decodes
/// its K position (b, oh, ow) incrementally per tile instead of dividing, so a gathered input element costs a table
/// read, an add, two unsigned range checks and a load;</item>
/// <item>split-K partial rows are written as 16-byte stores and summed in split order by conv2d_wgrad_splitk_reduce
/// (deterministic);</item>
/// <item>the r8v variants read dY's K tile as 16-byte loads: for one output channel, consecutive K positions of one image
/// are consecutive in dY, so where an image's K positions (outH * outW) are a multiple of the K tile, a tile never crosses
/// an image and each thread fetches four positions of one row with one aligned load instead of four indexed scalar loads
/// (ResNet-18, batch 8: 64@32 66.8 -> 54.9 us, 128@16 65.7 -> 60.2, 256@8 69.2 -> 63.3, 512@4 56.2 -> 52.4).</item>
/// </list>
/// </summary>
internal static class CudaConvWgradKernels
{
    /// <summary>The K depth of every r8 tile (WR_KERNEL's BK); the host rounds split-K slices to it.</summary>
    internal const int KTile = 16;

    /// <summary>The (rows, columns) output tiles compiled and offered as dW candidates: 64 x 128 for the common short
    /// Cout rows, 128 x 64 for few, long rows.</summary>
    internal static readonly (int M, int N)[] TileShapes = { (64, 128), (128, 64) };

    public static string GetSource() => @"
// Column table entry: x offset of column (ci, kh, kw) relative to the K position's window origin, and its shifts.
struct WrCol { int off; int khd; int kwd; int valid; };

// VECA: dY's K tile read as float4s (the r8v kernels; the host's condition: outH * outW a multiple of WR_BK).
template <int BM, int BN, int WR_BK, bool VECA = false>
__device__ __forceinline__ void conv2d_wgrad_r8_body(
    const float* __restrict__ input, const float* __restrict__ gradOutput, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit)
{
    constexpr int THREADS = (BM / 8) * (BN / 8);
    constexpr int GROUPS = THREADS / WR_BK;          // loader lanes per K position
    constexpr int RA = BM / GROUPS, RB = BN / GROUPS; // elements each thread loads per tile
    __shared__ __align__(16) float As[2][WR_BK][BM + 4];
    __shared__ __align__(16) float Bs[2][WR_BK][BN + 4];
    __shared__ WrCol cols[BN];

    const int L = outH * outW, khw = kH * kW, P = inC * khw, K = batch * L, HW = H * W;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int kBegin = blockIdx.z * kPerSplit, kEnd = min(K, kBegin + kPerSplit);
    const int tid = threadIdx.x;

    for (int c = tid; c < BN; c += THREADS)
    {
        int p = n0 + c;
        WrCol e;
        e.valid = p < P;
        int pp = e.valid ? p : 0;
        int ci = pp / khw, t = pp - ci * khw, kh = t / kW, kw = t - kh * kW;
        e.khd = kh * dH;
        e.kwd = kw * dW;
        e.off = ci * HW + e.khd * W + e.kwd;
        cols[c] = e;
    }

    // This thread's K lane and its incrementally maintained (b, oh, ow).
    const int klane = tid % WR_BK, group = tid / WR_BK;
    int kAbs = kBegin + klane;
    int kb = 0, koh = 0, kow = 0;
    if (kAbs < K) { kb = kAbs / L; int l = kAbs - kb * L; koh = l / outW; kow = l - koh * outW; }

    float ra[RA], rb[RB];
    // VECA: the block's K-tile base as (image, position in image), advanced with the tile; thread tid + v * THREADS
    // loads row (idx / (WR_BK / 4)), K positions 4 * (idx % (WR_BK / 4)) .. + 3 of the tile.
    constexpr int VA = BM * WR_BK / 4 / THREADS > 0 ? BM * WR_BK / 4 / THREADS : 1;
    float4 rav[VA];
    int tileK = kBegin, vImage = kBegin / L, vPos = kBegin - (kBegin / L) * L;
    auto load = [&]()
    {
        bool kIn = kAbs < kEnd;
        if constexpr (VECA)
        {
            bool tileIn = tileK < kEnd;
            #pragma unroll
            for (int v = 0; v < VA; v++)
            {
                int idx = tid + v * THREADS, mm = idx / (WR_BK / 4), qq = idx - mm * (WR_BK / 4);
                int m = m0 + mm;
                rav[v] = (tileIn && m < outC)
                    ? *reinterpret_cast<const float4*>(gradOutput + ((long long)vImage * outC + m) * L + vPos + qq * 4)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        else
        {
            const float* dyK = gradOutput + ((long long)kb * outC + m0 + group) * L + koh * outW + kow;
            #pragma unroll
            for (int i = 0; i < RA; i++)
            {
                int m = m0 + group + GROUPS * i;
                ra[i] = (kIn && m < outC) ? dyK[(long long)(GROUPS * i) * L] : 0.0f;
            }
        }
        int ihBase = koh * sH - pH, iwBase = kow * sW - pW;
        const float* xK = input + (long long)kb * inC * HW + (long long)ihBase * W + iwBase;
        #pragma unroll
        for (int j = 0; j < RB; j++)
        {
            WrCol e = cols[group + GROUPS * j];
            bool ok = kIn && e.valid && (unsigned)(ihBase + e.khd) < (unsigned)H && (unsigned)(iwBase + e.kwd) < (unsigned)W;
            rb[j] = ok ? xK[e.off] : 0.0f;
        }
    };
    auto store = [&](int buf)
    {
        if constexpr (VECA)
        {
            #pragma unroll
            for (int v = 0; v < VA; v++)
            {
                int idx = tid + v * THREADS, mm = idx / (WR_BK / 4), qq = idx - mm * (WR_BK / 4);
                As[buf][qq * 4][mm] = rav[v].x;
                As[buf][qq * 4 + 1][mm] = rav[v].y;
                As[buf][qq * 4 + 2][mm] = rav[v].z;
                As[buf][qq * 4 + 3][mm] = rav[v].w;
            }
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < RA; i++) As[buf][klane][group + GROUPS * i] = ra[i];
        }
        #pragma unroll
        for (int j = 0; j < RB; j++) Bs[buf][klane][group + GROUPS * j] = rb[j];
    };
    auto advance = [&]()
    {
        tileK += WR_BK;
        vPos += WR_BK;
        if (vPos >= L) { vPos -= L; ++vImage; }
        kAbs += WR_BK;
        kow += WR_BK;
        while (kow >= outW) { kow -= outW; if (++koh == outH) { koh = 0; ++kb; } }
    };

    const int tx = tid % (BN / 8), ty = tid / (BN / 8);
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    __syncthreads();   // the column table
    const int tiles = kBegin < kEnd ? (kEnd - kBegin + WR_BK - 1) / WR_BK : 0;
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
        for (int kk = 0; kk < WR_BK; kk++)
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

    // Rows ty*4 + {0..3} and BM/2 + ty*4 + {0..3}; columns tx*4 + {0..3} and BN/2 + tx*4 + {0..3}.
    float* slice = out + (long long)blockIdx.z * outC * P;
    const bool vec = (P & 3) == 0;
    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int m = m0 + (i >> 2) * (BM / 2) + ty * 4 + (i & 3);
        if (m >= outC) continue;
        #pragma unroll
        for (int h = 0; h < 2; h++)
        {
            int n = n0 + h * (BN / 2) + tx * 4;
            float* row = slice + (long long)m * P + n;
            if (vec && n + 3 < P)
                *reinterpret_cast<float4*>(row) = make_float4(acc[i][4 * h], acc[i][4 * h + 1], acc[i][4 * h + 2], acc[i][4 * h + 3]);
            else
            {
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    if (n + j < P) row[j] = acc[i][4 * h + j];
            }
        }
    }
}

#define WR_KERNEL(NAME, BM, BN, BK) WR_KERNEL_V(NAME, BM, BN, BK, false)
#define WR_KERNEL_V(NAME, BM, BN, BK, VECA) \
extern ""C"" __global__ __launch_bounds__((BM / 8) * (BN / 8)) void NAME( \
    const float* __restrict__ input, const float* __restrict__ gradOutput, float* __restrict__ out, \
    int batch, int inC, int H, int W, int outC, int outH, int outW, \
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit) \
{ \
    conv2d_wgrad_r8_body<BM, BN, BK, VECA>(input, gradOutput, out, batch, inC, H, W, outC, outH, outW, \
        kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit); \
}
" + string.Concat(TileShapes.Select(t => $"WR_KERNEL({KernelName(t.M, t.N)}, {t.M}, {t.N}, {KTile})\n"))
     + string.Concat(TileShapes.Select(t => $"WR_KERNEL_V({KernelName(t.M, t.N, vectorDy: true)}, {t.M}, {t.N}, {KTile}, true)\n"));

    public static string[] GetKernelNames() => TileShapes.Select(t => KernelName(t.M, t.N))
        .Concat(TileShapes.Select(t => KernelName(t.M, t.N, vectorDy: true))).ToArray();

    internal static string KernelName(int tileM, int tileN, bool vectorDy = false) =>
        $"conv2d_wgrad_r8{(vectorDy ? "v" : "")}_{tileM}x{tileN}";
}
