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
internal static class CudaConvWinogradKernels
{
    public static string GetSource() => CudaConvolutionKernels.ConvEngineCore + @"

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

// Batched GEMM on the engine main loop: Cm[z] = A[z] (M x Kd, row-major) * B[z] (Kd x Nc, row-major), z = blockIdx.z.
template <int BM, int BN>
__device__ __forceinline__ void wino_gemm_body(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ Cm, int M, int Nc, int Kd)
{
    __shared__ __align__(16) float smem[2 * CE_BK * (BM + 4) + 2 * CE_BK * (BN + 4)];
    constexpr int TM = BM / 16, TN = BN / 16, RA = BM * CE_BK / 256, RB = BN * CE_BK / 256, BSTEP = 256 / BN;
    const long long z = blockIdx.z;
    A += z * M * Kd; B += z * Kd * Nc; Cm += z * M * Nc;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, tid = threadIdx.x;
    const int ak = tid & 7, am = tid >> 3, bn = tid % BN, bk = tid / BN;
    const int tiles = (Kd + CE_BK - 1) / CE_BK;
    int kc = 0;
    float ra[RA], rb[RB];
    auto load = [&]()
    {
        #pragma unroll
        for (int r = 0; r < RA; r++)
        {
            int m = m0 + am + 32 * r, k = kc + ak;
            ra[r] = (m < M && k < Kd) ? A[(long long)m * Kd + k] : 0.0f;
        }
        #pragma unroll
        for (int r = 0; r < RB; r++)
        {
            int k = kc + bk + BSTEP * r, n = n0 + bn;
            rb[r] = (k < Kd && n < Nc) ? B[(long long)k * Nc + n] : 0.0f;
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
extern ""C"" __global__ __launch_bounds__(256) void wino_gemm128(const float* A, const float* B, float* Cm, int M, int Nc, int Kd)
{ wino_gemm_body<128, 128>(A, B, Cm, M, Nc, Kd); }
extern ""C"" __global__ __launch_bounds__(256) void wino_gemm64(const float* A, const float* B, float* Cm, int M, int Nc, int Kd)
{ wino_gemm_body<64, 64>(A, B, Cm, M, Nc, Kd); }

// Y = A^T m A for tile p of output channel k, A^T = [[1,1,1,0],[0,1,-1,-1]]; writes the in-bounds 2x2 outputs.
extern ""C"" __global__ void wino_f23_output(const float* __restrict__ mm, float* __restrict__ y,
    int N, int K, int outH, int outW, int tilesH, int tilesW)
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
    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        if (h + i >= outH) break;
        float o0 = s[i][0] + s[i][1] + s[i][2];
        float o1 = s[i][1] - s[i][2] - s[i][3];
        dst[(h + i) * outW + w] = o0;
        if (w + 1 < outW) dst[(h + i) * outW + w + 1] = o1;
    }
}
";

    public static string[] GetKernelNames() => new[]
    {
        "wino_f23_filter", "wino_f23_filter_flip", "wino_f23_input", "wino_gemm128", "wino_gemm64", "wino_f23_output",
    };
}
