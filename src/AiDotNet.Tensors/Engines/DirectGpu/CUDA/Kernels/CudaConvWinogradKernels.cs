// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Winograd F(2x2, 3x3) forward convolution for 3x3 stride-1 dilation-1 (Lavin and Gray, "Fast Algorithms for
/// Convolutional Neural Networks", CVPR 2016): Y = A^T [ (G g G^T) . (B^T d B) ] A per 2x2 output tile, computed as a
/// filter transform U[16][K][C], an input transform V[16][C][P] (P = batch * tile rows * tile columns), sixteen GEMMs
/// M[x] = U[x] V[x] on the #1325 engine main loop (<see cref="CudaConvolutionKernels.ConvEngineCore"/>), and an output
/// transform. 2.25x fewer multiplies than the direct sum, with fp32 rounding close to it (all transform coefficients are
/// 0, +-1 and +-1/2).
/// </summary>
/// <remarks>
/// CUDA only for now. The other backends' follow-ups: HIP #1368, Metal #1369, OpenCL #1370, Vulkan #1371,
/// WebGPU #1372. Until each lands, those backends serve their existing conv routes for these shapes.
/// </remarks>
internal static class CudaConvWinogradKernels
{
    /// <summary>
    /// The engine main loop (<see cref="CudaConvolutionKernels.ConvEngineCore"/>) plus the batched Winograd GEMM template
    /// <c>wino_gemm_body&lt;BM, BN&gt;</c>, with no <c>extern "C"</c> entry point: every Winograd module (this one, and any
    /// other tile size) prepends it and instantiates its own entry points, so there is one GEMM definition.
    /// </summary>
    public const string WinogradGemmCore = CudaConvolutionKernels.ConvEngineCore + @"

// Batched GEMM on the engine main loop: Cm[x][s] = A[x] (M x Kd, row-major) * B[x] (Kd x Nc, row-major) over the
// reduction slice s of kPerSplit, x = blockIdx.z / splits, s = blockIdx.z % splits. One split: the plain batched GEMM.
template <int BM, int BN>
__device__ __forceinline__ void wino_gemm_body(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ Cm, int M, int Nc, int Kd, int splits, int kPerSplit)
{
    __shared__ __align__(16) float smem[2 * CE_BK * (BM + 4) + 2 * CE_BK * (BN + 4)];
    constexpr int TM = BM / 16, TN = BN / 16, RA = BM * CE_BK / 256, RB = BN * CE_BK / 256, BSTEP = 256 / BN;
    const long long x = blockIdx.z / splits, s = blockIdx.z - x * splits;
    A += x * M * Kd; B += x * Kd * Nc; Cm += (x * splits + s) * M * Nc;
    const int kBegin = (int)s * kPerSplit, kEnd = min(Kd, kBegin + kPerSplit);
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, tid = threadIdx.x;
    const int ak = tid & 7, am = tid >> 3, bn = tid % BN, bk = tid / BN;
    const int tiles = kBegin < kEnd ? (kEnd - kBegin + CE_BK - 1) / CE_BK : 0;
    const int kLimit = kEnd;   // loads past this slice's end read as zero (Kd stays the row stride)
    int kc = kBegin;
    float ra[RA], rb[RB];
    auto load = [&]()
    {
        #pragma unroll
        for (int r = 0; r < RA; r++)
        {
            int m = m0 + am + 32 * r, k = kc + ak;
            ra[r] = (m < M && k < kLimit) ? A[(long long)m * Kd + k] : 0.0f;
        }
        #pragma unroll
        for (int r = 0; r < RB; r++)
        {
            int k = kc + bk + BSTEP * r, n = n0 + bn;
            rb[r] = (k < kLimit && n < Nc) ? B[(long long)k * Nc + n] : 0.0f;
        }
    };
    auto store = [&](float (*as)[BM + 4], float (*bs)[BN + 4])
    {
        #pragma unroll
        for (int r = 0; r < RA; r++) as[ak][am + 32 * r] = ra[r];
        #pragma unroll
        for (int r = 0; r < RB; r++) bs[bk + BSTEP * r][bn] = rb[r];
    };
    auto advance = [&]() { kc += CE_BK; };
    float acc[TM][TN];
    ce_mainloop<BM, BN>(smem, tiles, load, store, advance, acc);
    const int tx = tid & 15, ty = tid >> 4;
    #pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int m = m0 + (i >> 2) * 64 + ty * 4 + (i & 3);
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++)
        {
            int n = n0 + (j >> 2) * 64 + tx * 4 + (j & 3);
            if (n < Nc) Cm[(long long)m * Nc + n] = acc[i][j];
        }
    }
}
";

    public static string GetSource() => CudaOptimizerKernels.PingPongDevice + WinogradGemmCore + @"

// U[x][k][c] = (G g G^T)[x], G = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]].
extern ""C"" __global__ void wino_f23_filter(const float* __restrict__ w, float* __restrict__ u, int K, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * C) return;
    const float* g = w + (long long)idx * 9;   // w[k][c][3][3], idx = k*C + c
    float t[4][3];
    #pragma unroll
    for (int j = 0; j < 3; j++)
    {
        float g0 = g[j], g1 = g[3 + j], g2 = g[6 + j];
        t[0][j] = g0;
        t[1][j] = 0.5f * (g0 + g1 + g2);
        t[2][j] = 0.5f * (g0 - g1 + g2);
        t[3][j] = g2;
    }
    long long plane = (long long)K * C;
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

// Backward data: dX = conv(dY, w') with w'[c][k] = w[k][c] rotated 180 degrees (and padding 2 - pad for 3x3 stride 1),
// so U'[x][c][k] = (G w'[c][k] G^T)[x]: the forward transform of the flipped, channel-transposed filter.
extern ""C"" __global__ void wino_f23_filter_flip(const float* __restrict__ w, float* __restrict__ u, int K, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = c*K + k
    if (idx >= K * C) return;
    int c = idx / K, k = idx - c * K;
    const float* src = w + ((long long)k * C + c) * 9;
    float g[3][3];
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) g[i][j] = src[(2 - i) * 3 + (2 - j)];
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
    long long plane = (long long)K * C;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float a = t[i][0], b = t[i][1], cc = t[i][2];
        u[(i * 4 + 0) * plane + idx] = a;
        u[(i * 4 + 1) * plane + idx] = 0.5f * (a + b + cc);
        u[(i * 4 + 2) * plane + idx] = 0.5f * (a - b + cc);
        u[(i * 4 + 3) * plane + idx] = cc;
    }
}

// V[x][c][p] = (B^T d B)[x] for the 4x4 input patch of tile p (zero outside the image), B^T = [[1,0,-1,0],[0,1,1,0],
// [0,-1,1,0],[0,1,0,-1]].
extern ""C"" __global__ void wino_f23_input(const float* __restrict__ x, float* __restrict__ v,
    int N, int C, int H, int W, int pad, int tilesH, int tilesW)
{
    int P = N * tilesH * tilesW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * P) return;
    int c = idx / P, p = idx - c * P;
    int n = p / (tilesH * tilesW), t = p - n * (tilesH * tilesW);
    int th = t / tilesW, tw = t - th * tilesW;
    int h0 = th * 2 - pad, w0 = tw * 2 - pad;
    const float* src = x + ((long long)n * C + c) * H * W;
    float d[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int h = h0 + i;
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int ww = w0 + j;
            d[i][j] = (h >= 0 && h < H && ww >= 0 && ww < W) ? __ldg(&src[h * W + ww]) : 0.0f;
        }
    }
    float s[4][4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        s[0][j] = d[0][j] - d[2][j];
        s[1][j] = d[1][j] + d[2][j];
        s[2][j] = d[2][j] - d[1][j];
        s[3][j] = d[1][j] - d[3][j];
    }
    long long plane = (long long)C * P;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        v[(i * 4 + 0) * plane + idx] = s[i][0] - s[i][2];
        v[(i * 4 + 1) * plane + idx] = s[i][1] + s[i][2];
        v[(i * 4 + 2) * plane + idx] = s[i][2] - s[i][1];
        v[(i * 4 + 3) * plane + idx] = s[i][1] - s[i][3];
    }
}

extern ""C"" __global__ __launch_bounds__(256) void wino_gemm128(const float* A, const float* B, float* Cm, int M, int Nc, int Kd, int splits, int kPerSplit)
{ wino_gemm_body<128, 128>(A, B, Cm, M, Nc, Kd, splits, kPerSplit); }
extern ""C"" __global__ __launch_bounds__(256) void wino_gemm64(const float* A, const float* B, float* Cm, int M, int Nc, int Kd, int splits, int kPerSplit)
{ wino_gemm_body<64, 64>(A, B, Cm, M, Nc, Kd, splits, kPerSplit); }

// ---- Weight gradient: F(3x3, 2x2), the transpose dual of F(2x2, 3x3) (A and G trade places, B is unchanged). ----
// dW[k][c][i][j] = sum over n, h, w of dY[n][k][h][w] * x[n][c][h + i - pad][w + j - pad]: a correlation of x with dY.
// dY is cut into 2x2 tiles g (the 'filter'), each paired with the 4x4 x window it touches; per tile
// U = A g A^T with A = [[1,0],[1,1],[1,-1],[0,-1]] (F(2,3)'s A), V = B^T d B, M[x] = sum over tiles U[x] V[x], and the
// 3x3 gradient is G^T M G with F(2,3)'s G.

// U[x][k][p] = (A g A^T)[x] for dY tile p of channel k (zero past the edge).
extern ""C"" __global__ void wino_f32w_dy(const float* __restrict__ dy, float* __restrict__ u,
    int N, int K, int outH, int outW, int tilesH, int tilesW)
{
    int P = N * tilesH * tilesW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = k*P + p
    if (idx >= K * P) return;
    int k = idx / P, p = idx - k * P;
    int n = p / (tilesH * tilesW), t = p - n * (tilesH * tilesW);
    int th = t / tilesW, tw = t - th * tilesW;
    const float* src = dy + ((long long)n * K + k) * outH * outW;
    int h = th * 2, w = tw * 2;
    float g00 = __ldg(&src[h * outW + w]);
    float g01 = (w + 1 < outW) ? __ldg(&src[h * outW + w + 1]) : 0.0f;
    float g10 = (h + 1 < outH) ? __ldg(&src[(h + 1) * outW + w]) : 0.0f;
    float g11 = (h + 1 < outH && w + 1 < outW) ? __ldg(&src[(h + 1) * outW + w + 1]) : 0.0f;
    // t = A g (4x2): rows g0, g0+g1, g0-g1, -g1 over the two columns.
    float t0[2] = { g00, g01 }, t1[2] = { g00 + g10, g01 + g11 }, t2[2] = { g00 - g10, g01 - g11 }, t3[2] = { -g10, -g11 };
    float* rows[4] = { t0, t1, t2, t3 };
    long long plane = (long long)K * P;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float a = rows[i][0], b = rows[i][1];
        u[(i * 4 + 0) * plane + idx] = a;
        u[(i * 4 + 1) * plane + idx] = a + b;
        u[(i * 4 + 2) * plane + idx] = a - b;
        u[(i * 4 + 3) * plane + idx] = -b;
    }
}

// V[x][p][c] = (B^T d B)[x] for the 4x4 x window of dY tile p (rows th*2 - pad .. +3), stored tile-major so the GEMM
// reads it as its row-major (P x C) operand.
extern ""C"" __global__ void wino_f32w_input(const float* __restrict__ x, float* __restrict__ v,
    int N, int C, int H, int W, int pad, int tilesH, int tilesW)
{
    int P = N * tilesH * tilesW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = p*C + c
    if (idx >= C * P) return;
    int p = idx / C, c = idx - p * C;
    int n = p / (tilesH * tilesW), t = p - n * (tilesH * tilesW);
    int th = t / tilesW, tw = t - th * tilesW;
    int h0 = th * 2 - pad, w0 = tw * 2 - pad;
    const float* src = x + ((long long)n * C + c) * H * W;
    float d[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int hh = h0 + i;
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int ww = w0 + j;
            d[i][j] = (hh >= 0 && hh < H && ww >= 0 && ww < W) ? __ldg(&src[hh * W + ww]) : 0.0f;
        }
    }
    float s[4][4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        s[0][j] = d[0][j] - d[2][j];
        s[1][j] = d[1][j] + d[2][j];
        s[2][j] = d[2][j] - d[1][j];
        s[3][j] = d[1][j] - d[3][j];
    }
    long long plane = (long long)C * P;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        v[(i * 4 + 0) * plane + idx] = s[i][0] - s[i][2];
        v[(i * 4 + 1) * plane + idx] = s[i][1] + s[i][2];
        v[(i * 4 + 2) * plane + idx] = s[i][2] - s[i][1];
        v[(i * 4 + 3) * plane + idx] = s[i][1] - s[i][3];
    }
}

// dW[k][c] = G^T M G (3x3) from M[x][split][k][c], summing the split-K partials in order. G^T = [[1,1/2,1/2,0],
// [0,1/2,-1/2,0],[0,1/2,1/2,1]].
extern ""C"" __global__ void wino_f32w_output(const float* __restrict__ mm, float* __restrict__ dw, int K, int C, int splits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = k*C + c
    if (idx >= K * C) return;
    long long plane = (long long)K * C;
    float m[16];
    #pragma unroll
    for (int x = 0; x < 16; x++)
    {
        float acc = 0.0f;
        for (int s = 0; s < splits; s++) acc += mm[((long long)x * splits + s) * plane + idx];
        m[x] = acc;
    }
    float t[3][4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        float m0 = m[0 * 4 + j], m1 = m[1 * 4 + j], m2 = m[2 * 4 + j], m3 = m[3 * 4 + j];
        t[0][j] = m0 + 0.5f * (m1 + m2);
        t[1][j] = 0.5f * (m1 - m2);
        t[2][j] = 0.5f * (m1 + m2) + m3;
    }
    float* dst = dw + (long long)idx * 9;
    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float a = t[i][0], b = t[i][1], cc = t[i][2], d = t[i][3];
        dst[i * 3 + 0] = a + 0.5f * (b + cc);
        dst[i * 3 + 1] = 0.5f * (b - cc);
        dst[i * 3 + 2] = 0.5f * (b + cc) + d;
    }
}

// Y = A^T m A for tile p of output channel k, A^T = [[1,1,1,0],[0,1,-1,-1]]; writes the in-bounds 2x2 outputs.
// The convolution's epilogue: epi bit 0 adds bias[k], bit 1 applies ReLU (v < 0 is false for NaN, which passes).
extern ""C"" __global__ void wino_f23_output(const float* __restrict__ mm, float* __restrict__ y,
    const float* __restrict__ bias, int N, int K, int outH, int outW, int tilesH, int tilesW, int epi)
{
    int P = N * tilesH * tilesW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * P) return;
    int k = idx / P, p = idx - k * P;
    int n = p / (tilesH * tilesW), t = p - n * (tilesH * tilesW);
    int th = t / tilesW, tw = t - th * tilesW;
    long long plane = (long long)K * P;
    float m[16];
    #pragma unroll
    for (int x = 0; x < 16; x++) m[x] = mm[x * plane + idx];
    float s[2][4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        s[0][j] = m[0 * 4 + j] + m[1 * 4 + j] + m[2 * 4 + j];
        s[1][j] = m[1 * 4 + j] - m[2 * 4 + j] - m[3 * 4 + j];
    }
    float* dst = y + ((long long)n * K + k) * outH * outW;
    int h = th * 2, w = tw * 2;
    float bk = (epi & 1) ? bias[k] : 0.0f;
    bool relu = (epi & 2) != 0;
    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        if (h + i >= outH) break;
        float o0 = s[i][0] + s[i][1] + s[i][2] + bk;
        float o1 = s[i][1] - s[i][2] - s[i][3] + bk;
        if (relu && o0 < 0.0f) o0 = 0.0f;
        if (relu && o1 < 0.0f) o1 = 0.0f;
        dst[(h + i) * outW + w] = o0;
        if (w + 1 < outW) dst[(h + i) * outW + w + 1] = o1;
    }
}
// Fused Winograd F(3x3, 2x2) weight gradient: one launch (plus conv2d_wgrad_winograd_finish when the tiles are split),
// in place of wino_f32w_dy + wino_f32w_input + wino_gemm + wino_f32w_output (whose 16-plane U and V scratch alone were
// 4 MB each at ResNet-18's 128-channel layers, mostly DRAM traffic). Block (blockIdx.x, blockIdx.y, blockIdx.z) owns input channels
// c0 .. c0 + 31, output channels k0 .. k0 + 31 and the dY tiles [z * tilesPerSplit, ...). Per chunk of WF_PC tiles it
// transforms dY (U = A g A^T) and x (V = B^T d B) into shared memory, then thread (tx, tk, tc) accumulates
// M[x][k][c] += U[x][p][k] V[x][p][c] for components 4tx .. 4tx + 3, channels 4tk .., 4tc .. (64 accumulators). The
// epilogue gathers each (k, c)'s 16 components through shared memory and writes G^T M G (3x3): to dW when the tiles are
// not split (DIRECT: staged through shared memory so the stores are contiguous dW rows), else component-major to
// out[z][q][k][c] for conv2d_wgrad_winograd_finish.
// HIP, OpenCL, Vulkan, Metal and WebGPU run the same algorithm from WinogradF23KernelTemplate (FusedFilterGradient*,
// restated for 16 KB of shared memory) through WinogradF23Pipeline.TryBackwardFilterFused.
#define WF_TK 32
#define WF_TC 32
#define WF_PC 8
#define WF_LD 36   // padded row: a warp's 8 tiles x 4 channels of transform stores hit 32 distinct banks
// SGD (direct only): the ping-pong SGD step fused into the epilogue (PpFused): besides dW, each element's updated weight
// goes to f.dst, and the block's sum of dW^2 to f.partials (sgd_pingpong_update then skips this tensor).
template <bool DIRECT, bool SGD = false>
__device__ __forceinline__ void wino_f32w_fused_body(
    const float* __restrict__ x, const float* __restrict__ dy, float* __restrict__ out,
    int N, int C, int H, int W, int K, int outH, int outW, int pad, int tilesH, int tilesW, int tilesPerSplit,
    PpFused f = PpFused())
{
    __shared__ __align__(16) float smem[2 * 16 * WF_PC * WF_LD];
    float (*Us)[WF_PC][WF_LD] = reinterpret_cast<float (*)[WF_PC][WF_LD]>(smem);
    float (*Vs)[WF_PC][WF_LD] = reinterpret_cast<float (*)[WF_PC][WF_LD]>(smem + 16 * WF_PC * WF_LD);
    const int tid = threadIdx.x;
    const int c0 = blockIdx.x * WF_TC, k0 = blockIdx.y * WF_TK;
    const int TT = tilesH * tilesW, P = N * TT;
    const int pBegin = blockIdx.z * tilesPerSplit, pEnd = min(P, pBegin + tilesPerSplit);
    const int tc = tid & 7, tk = (tid >> 3) & 7, tx = tid >> 6;
    // Transform role: tile j (fastest, so a warp's loads walk along image rows) of the chunk, channel row r.
    const int tj = tid & (WF_PC - 1), tr = tid / WF_PC;

    float acc[4][4][4];
    #pragma unroll
    for (int a = 0; a < 4; a++)
        #pragma unroll
        for (int b = 0; b < 4; b++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[a][b][e] = 0.0f;

    for (int pc = pBegin; pc < pEnd; pc += WF_PC)
    {
        const int p = pc + tj;
        const bool pin = p < pEnd;
        int n = 0, th = 0, tw = 0;
        if (pin) { n = p / TT; int t = p - n * TT; th = t / tilesW; tw = t - th * tilesW; }
        {
            const int k = k0 + tr;
            float g00 = 0.0f, g01 = 0.0f, g10 = 0.0f, g11 = 0.0f;
            if (pin && k < K)
            {
                const float* src = dy + ((long long)n * K + k) * outH * outW;
                const int h = th * 2, w = tw * 2;
                g00 = __ldg(src + h * outW + w);
                g01 = w + 1 < outW ? __ldg(src + h * outW + w + 1) : 0.0f;
                g10 = h + 1 < outH ? __ldg(src + (h + 1) * outW + w) : 0.0f;
                g11 = (h + 1 < outH && w + 1 < outW) ? __ldg(src + (h + 1) * outW + w + 1) : 0.0f;
            }
            const float ra[4] = { g00, g00 + g10, g00 - g10, -g10 };
            const float rb[4] = { g01, g01 + g11, g01 - g11, -g11 };
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                Us[i * 4 + 0][tj][tr] = ra[i];
                Us[i * 4 + 1][tj][tr] = ra[i] + rb[i];
                Us[i * 4 + 2][tj][tr] = ra[i] - rb[i];
                Us[i * 4 + 3][tj][tr] = -rb[i];
            }
        }
        {
            const int c = c0 + tr;
            const int h0 = th * 2 - pad, w0 = tw * 2 - pad;
            const float* src = x + ((long long)n * C + (c < C ? c : 0)) * H * W;
            const bool cin = pin && c < C;
            float d[4][4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                const int hh = h0 + i;
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    const int ww = w0 + j;
                    d[i][j] = (cin && (unsigned)hh < (unsigned)H && (unsigned)ww < (unsigned)W) ? __ldg(src + hh * W + ww) : 0.0f;
                }
            }
            float s[4][4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                s[0][j] = d[0][j] - d[2][j];
                s[1][j] = d[1][j] + d[2][j];
                s[2][j] = d[2][j] - d[1][j];
                s[3][j] = d[1][j] - d[3][j];
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                Vs[i * 4 + 0][tj][tr] = s[i][0] - s[i][2];
                Vs[i * 4 + 1][tj][tr] = s[i][1] + s[i][2];
                Vs[i * 4 + 2][tj][tr] = s[i][2] - s[i][1];
                Vs[i * 4 + 3][tj][tr] = s[i][1] - s[i][3];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < WF_PC; j++)
        {
            #pragma unroll
            for (int xi = 0; xi < 4; xi++)
            {
                const int xx = tx * 4 + xi;
                const float4 u = *reinterpret_cast<const float4*>(&Us[xx][j][tk * 4]);
                const float4 v = *reinterpret_cast<const float4*>(&Vs[xx][j][tc * 4]);
                const float uu[4] = { u.x, u.y, u.z, u.w }, vv[4] = { v.x, v.y, v.z, v.w };
                #pragma unroll
                for (int a = 0; a < 4; a++)
                    #pragma unroll
                    for (int b = 0; b < 4; b++) acc[xi][a][b] = fmaf(uu[a], vv[b], acc[xi][a][b]);
            }
        }
        __syncthreads();
    }

    if constexpr (DIRECT)
    {
    // Epilogue, in quarters of the block's k rows. Ms[x][8 k][33] gathers each (k, c)'s 16 components (row stride 33: the
    // writers' 4 k rows x 8 column quads hit 32 distinct banks); each (k, c)'s G^T M G goes to Os[8 k][32 c * 9], whose
    // rows are the contiguous dW[k][c0 .. c0 + 31][3][3] spans, copied out coalesced. The host launches this entry
    // (wino_f32w_fused_direct) with one split only, so out is dW itself; blockIdx.z's dW-sized offset would address a
    // separate per-split dW slice. The split path is the component-major one below (conv2d_wgrad_winograd_finish).
    float* Ms = smem;                      // [16][8][33]
    float* Os = smem + 16 * 8 * 33;        // [8][32 * 9]
    float* dst = out + (long long)blockIdx.z * K * C * 9;
    const int cw = min(WF_TC, C - c0);     // this block's columns (ragged last tile)
    double sumSquares = 0.0;
    const float learningRate = SGD ? f.dyn[0] : 0.0f;
    #pragma unroll
    for (int quarter = 0; quarter < 4; quarter++)
    {
        // SGD: this quarter's old weights, the 9 elements each thread updates in the copy loop below, loaded now so the
        // loads overlap the gather and the transform rather than stalling each store in turn.
        float oldWeights[9];
        if constexpr (SGD)
        {
            #pragma unroll
            for (int j = 0; j < 9; j++)
            {
                const int i = tid + j * 256, kl = i / 288, r = i - kl * 288;
                const int k = k0 + quarter * 8 + kl;
                oldWeights[j] = (k < K && r < cw * 9) ? __ldg(f.src + ((long long)k * C + c0) * 9 + r) : 0.0f;
            }
        }
        if ((tk >> 1) == quarter)
        {
            #pragma unroll
            for (int xi = 0; xi < 4; xi++)
                #pragma unroll
                for (int a = 0; a < 4; a++)
                    #pragma unroll
                    for (int b = 0; b < 4; b++)
                        Ms[(tx * 4 + xi) * 264 + ((tk & 1) * 4 + a) * 33 + tc * 4 + b] = acc[xi][a][b];
        }
        __syncthreads();
        {
            const int pair = tid, kl = pair >> 5, cl = pair & 31;   // 256 pairs: 8 k x 32 c
            float m[16];
            #pragma unroll
            for (int q = 0; q < 16; q++) m[q] = Ms[q * 264 + kl * 33 + cl];
            float t[3][4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                const float m0 = m[j], m1 = m[4 + j], m2 = m[8 + j], m3 = m[12 + j];
                t[0][j] = m0 + 0.5f * (m1 + m2);
                t[1][j] = 0.5f * (m1 - m2);
                t[2][j] = 0.5f * (m1 + m2) + m3;
            }
            float* o = Os + kl * 288 + cl * 9;
            #pragma unroll
            for (int i = 0; i < 3; i++)
            {
                const float a = t[i][0], b = t[i][1], cc = t[i][2], dd = t[i][3];
                o[i * 3 + 0] = a + 0.5f * (b + cc);
                o[i * 3 + 1] = 0.5f * (b - cc);
                o[i * 3 + 2] = 0.5f * (b + cc) + dd;
            }
        }
        __syncthreads();
        // Copy the 8 rows of cw * 9 contiguous floats out (8 * 288 = 9 * 256: nine elements per thread).
        #pragma unroll
        for (int j = 0; j < 9; j++)
        {
            const int i = tid + j * 256, kl = i / 288, r = i - kl * 288;
            const int k = k0 + quarter * 8 + kl;
            if (k < K && r < cw * 9)
            {
                const long long e = ((long long)k * C + c0) * 9 + r;
                const float g = Os[i];
                dst[e] = g;
                if constexpr (SGD)
                {
                    sumSquares += (double)g * (double)g;
                    f.dst[e] = pp_sgd_step(oldWeights[j], g, learningRate, f.weightDecay);
                }
            }
        }
        __syncthreads();
    }
    if constexpr (SGD)
    {
        __shared__ double warpSums[8];
        const double blockSum = pp_block_sum(sumSquares, warpSums);
        if (tid == 0) f.partials[f.slot + blockIdx.y * gridDim.x + blockIdx.x] = blockSum;
    }
    }
    else
    {
    // Epilogue, in two halves of the k rows: Ms[x][16 k][33] gathers each (k, c)'s 16 components (row stride 33: the
    // writers' 4 k rows x 8 column quads hit 32 distinct banks). Each (k, c)'s G^T M G goes out component-major,
    // out[z][q][k][c] (q = 3i + j), so a warp's stores are contiguous; conv2d_wgrad_winograd_finish sums the splits and
    // writes dW[k][c][3][3].
    float* Ms = smem;
    const long long plane = (long long)K * C;
    float* dst = out + (long long)blockIdx.z * 9 * plane;
    #pragma unroll
    for (int half = 0; half < 2; half++)
    {
        if ((tk >> 2) == half)
        {
            #pragma unroll
            for (int xi = 0; xi < 4; xi++)
                #pragma unroll
                for (int a = 0; a < 4; a++)
                    #pragma unroll
                    for (int b = 0; b < 4; b++)
                        Ms[(tx * 4 + xi) * 528 + ((tk & 3) * 4 + a) * 33 + tc * 4 + b] = acc[xi][a][b];
        }
        __syncthreads();
        for (int pair = tid; pair < 512; pair += 256)
        {
            const int kl = pair >> 5, cl = pair & 31;
            const int k = k0 + half * 16 + kl, c = c0 + cl;
            if (k < K && c < C)
            {
                float m[16];
                #pragma unroll
                for (int q = 0; q < 16; q++) m[q] = Ms[q * 528 + kl * 33 + cl];
                float t[3][4];
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    const float m0 = m[j], m1 = m[4 + j], m2 = m[8 + j], m3 = m[12 + j];
                    t[0][j] = m0 + 0.5f * (m1 + m2);
                    t[1][j] = 0.5f * (m1 - m2);
                    t[2][j] = 0.5f * (m1 + m2) + m3;
                }
                float* o = dst + (long long)k * C + c;
                #pragma unroll
                for (int i = 0; i < 3; i++)
                {
                    const float a = t[i][0], b = t[i][1], cc = t[i][2], dd = t[i][3];
                    o[(i * 3 + 0) * plane] = a + 0.5f * (b + cc);
                    o[(i * 3 + 1) * plane] = 0.5f * (b - cc);
                    o[(i * 3 + 2) * plane] = 0.5f * (b + cc) + dd;
                }
            }
        }
        __syncthreads();
    }
    }
}

// One split of the tiles: writes dW directly.
extern ""C"" __global__ __launch_bounds__(256) void wino_f32w_fused_direct(
    const float* __restrict__ x, const float* __restrict__ dy, float* __restrict__ out,
    int N, int C, int H, int W, int K, int outH, int outW, int pad, int tilesH, int tilesW, int tilesPerSplit)
{
    wino_f32w_fused_body<true>(x, dy, out, N, C, H, W, K, outH, outW, pad, tilesH, tilesW, tilesPerSplit);
}

// wino_f32w_fused_direct with the ping-pong SGD step fused into its epilogue (one split; grid z = 1).
extern ""C"" __global__ __launch_bounds__(256) void wino_f32w_fused_direct_sgd(
    const float* __restrict__ x, const float* __restrict__ dy, float* __restrict__ out,
    int N, int C, int H, int W, int K, int outH, int outW, int pad, int tilesH, int tilesW, int tilesPerSplit, PpFused f)
{
    wino_f32w_fused_body<true, true>(x, dy, out, N, C, H, W, K, outH, outW, pad, tilesH, tilesW, tilesPerSplit, f);
}

// Split tiles: component-major partials for conv2d_wgrad_winograd_finish.
extern ""C"" __global__ __launch_bounds__(256) void wino_f32w_fused(
    const float* __restrict__ x, const float* __restrict__ dy, float* __restrict__ out,
    int N, int C, int H, int W, int K, int outH, int outW, int pad, int tilesH, int tilesW, int tilesPerSplit)
{
    wino_f32w_fused_body<false>(x, dy, out, N, C, H, W, K, outH, outW, pad, tilesH, tilesW, tilesPerSplit);
}

// dW[k][c][3][3] = sum over the splits of partials[z][q][k][c] (q = 3i + j), in split order.
extern ""C"" __global__ __launch_bounds__(256) void conv2d_wgrad_winograd_finish(
    const float* __restrict__ partials, float* __restrict__ dw, int KC, int splits)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // i = (k * C + c) * 9 + q, the dW element
    if (i >= KC * 9) return;
    const int kc = i / 9, q = i - kc * 9;
    float sum = 0.0f;
    for (int z = 0; z < splits; z++) sum += partials[((long long)z * 9 + q) * KC + kc];
    dw[i] = sum;
}

// conv2d_wgrad_winograd_finish with the ping-pong SGD step fused (PpFused): dW, the updated weights, and each block's
// sum of dW^2 in f.partials[f.slot + blockIdx.x].
extern ""C"" __global__ __launch_bounds__(256) void conv2d_wgrad_winograd_finish_sgd(
    const float* __restrict__ partials, float* __restrict__ dw, int KC, int splits, PpFused f)
{
    __shared__ double warpSums[8];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sumSquares = 0.0;
    if (i < KC * 9)
    {
        const float oldWeight = __ldg(f.src + i);   // issued first: its latency overlaps the split loads
        const int kc = i / 9, q = i - kc * 9;
        float sum = 0.0f;
        for (int z = 0; z < splits; z++) sum += partials[((long long)z * 9 + q) * KC + kc];
        dw[i] = sum;
        sumSquares = (double)sum * (double)sum;
        f.dst[i] = pp_sgd_step(oldWeight, sum, f.dyn[0], f.weightDecay);
    }
    const double blockSum = pp_block_sum(sumSquares, warpSums);
    if (threadIdx.x == 0) f.partials[f.slot + blockIdx.x] = blockSum;
}
";

    public static string[] GetKernelNames() => new[]
    {
        "wino_f23_filter", "wino_f23_filter_flip", "wino_f23_input", "wino_gemm128", "wino_gemm64", "wino_f23_output", "wino_f32w_dy", "wino_f32w_input", "wino_f32w_output",
        "wino_f32w_fused", "wino_f32w_fused_direct", "conv2d_wgrad_winograd_finish",
        "wino_f32w_fused_direct_sgd", "conv2d_wgrad_winograd_finish_sgd",
    };
}
