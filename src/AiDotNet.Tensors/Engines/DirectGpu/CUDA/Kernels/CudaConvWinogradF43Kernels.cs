// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Winograd F(4x4, 3x3) for 3x3 stride-1 dilation-1 convolution (Lavin and Gray, CVPR 2016), the reduced-precision
/// sibling of <see cref="CudaConvWinogradKernels"/>' F(2x2, 3x3): Y = A^T [ (G g G^T) . (B^T d B) ] A per 4x4 output
/// tile, as a filter transform U[36][K][C], an input transform V[36][C][P], thirty-six GEMMs M[x] = U[x] V[x] on the
/// shared Winograd GEMM body, and an output transform. 4x fewer multiplies than the direct sum (F(2x2): 2.25x), but the
/// transform constants (up to 8, 1/24) cost accuracy: about 2e-6 of sum|terms| in fp32 against F(2x2)'s 1e-7, so these
/// kernels run only under the explicit opt-in CudaDispatchPolicy.ConvWinogradF4 (#1335 owner decision).
/// </summary>
internal static class CudaConvWinogradF43Kernels
{
    public static string GetSource() => CudaConvWinogradKernels.WinogradGemmCore + @"

// One-dimensional transforms (applied to rows, then columns).
// B^T (6x6): [[4,0,-5,0,1,0],[0,-4,-4,1,1,0],[0,4,-4,-1,1,0],[0,-2,-1,2,1,0],[0,2,-1,-2,1,0],[0,4,0,-5,0,1]].
__device__ __forceinline__ void wino4_bt(const float* d, float* o)
{
    o[0] = 4.0f * d[0] - 5.0f * d[2] + d[4];
    o[1] = -4.0f * d[1] - 4.0f * d[2] + d[3] + d[4];
    o[2] = 4.0f * d[1] - 4.0f * d[2] - d[3] + d[4];
    o[3] = -2.0f * d[1] - d[2] + 2.0f * d[3] + d[4];
    o[4] = 2.0f * d[1] - d[2] - 2.0f * d[3] + d[4];
    o[5] = 4.0f * d[1] - 5.0f * d[3] + d[5];
}

// G (6x3): [[1/4,0,0],[-1/6,-1/6,-1/6],[-1/6,1/6,-1/6],[1/24,1/12,1/6],[1/24,-1/12,1/6],[0,0,1]].
__device__ __forceinline__ void wino4_g(const float* g, float* o)
{
    o[0] = 0.25f * g[0];
    o[1] = -(g[0] + g[1] + g[2]) * (1.0f / 6.0f);
    o[2] = -(g[0] - g[1] + g[2]) * (1.0f / 6.0f);
    o[3] = g[0] * (1.0f / 24.0f) + g[1] * (1.0f / 12.0f) + g[2] * (1.0f / 6.0f);
    o[4] = g[0] * (1.0f / 24.0f) - g[1] * (1.0f / 12.0f) + g[2] * (1.0f / 6.0f);
    o[5] = g[2];
}

// A^T (4x6): [[1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]].
__device__ __forceinline__ void wino4_at(const float* m, float* o)
{
    o[0] = m[0] + m[1] + m[2] + m[3] + m[4];
    o[1] = m[1] - m[2] + 2.0f * (m[3] - m[4]);
    o[2] = m[1] + m[2] + 4.0f * (m[3] + m[4]);
    o[3] = m[1] - m[2] + 8.0f * (m[3] - m[4]) + m[5];
}

// U[x][row][col] = (G g G^T)[x] for the 3x3 filter g; plane = rows * cols, idx = row * cols + col.
__device__ __forceinline__ void wino4_filter_store(const float g[3][3], float* __restrict__ u, long long plane, int idx)
{
    float t[6][3];
    #pragma unroll
    for (int j = 0; j < 3; j++)
    {
        float col[3] = { g[0][j], g[1][j], g[2][j] }, o[6];
        wino4_g(col, o);
        #pragma unroll
        for (int i = 0; i < 6; i++) t[i][j] = o[i];
    }
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float o[6];
        wino4_g(t[i], o);
        #pragma unroll
        for (int j = 0; j < 6; j++) u[(i * 6 + j) * plane + idx] = o[j];
    }
}

// Forward: U[x][k][c] from w[k][c][3][3].
extern ""C"" __global__ void wino_f43_filter(const float* __restrict__ w, float* __restrict__ u, int K, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = k*C + c
    if (idx >= K * C) return;
    const float* src = w + (long long)idx * 9;
    float g[3][3];
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) g[i][j] = src[i * 3 + j];
    wino4_filter_store(g, u, (long long)K * C, idx);
}

// Backward data: U'[x][c][k] = (G w'[c][k] G^T)[x] for w' = w[k][c] rotated 180 degrees.
extern ""C"" __global__ void wino_f43_filter_flip(const float* __restrict__ w, float* __restrict__ u, int K, int C)
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
    wino4_filter_store(g, u, (long long)K * C, idx);
}

// V[x][c][p] = (B^T d B)[x] for the 6x6 input window of 4x4 output tile p (rows th*4 - pad .., zero outside the image).
extern ""C"" __global__ void wino_f43_input(const float* __restrict__ x, float* __restrict__ v,
    int N, int C, int H, int W, int pad, int tilesH, int tilesW)
{
    int P = N * tilesH * tilesW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = c*P + p
    if (idx >= C * P) return;
    int c = idx / P, p = idx - c * P;
    int n = p / (tilesH * tilesW), t = p - n * (tilesH * tilesW);
    int th = t / tilesW, tw = t - th * tilesW;
    int h0 = th * 4 - pad, w0 = tw * 4 - pad;
    const float* src = x + ((long long)n * C + c) * H * W;
    float d[6][6];
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        int h = h0 + i;
        #pragma unroll
        for (int j = 0; j < 6; j++)
        {
            int ww = w0 + j;
            d[i][j] = (h >= 0 && h < H && ww >= 0 && ww < W) ? __ldg(&src[h * W + ww]) : 0.0f;
        }
    }
    float s[6][6];
    #pragma unroll
    for (int j = 0; j < 6; j++)
    {
        float col[6] = { d[0][j], d[1][j], d[2][j], d[3][j], d[4][j], d[5][j] }, o[6];
        wino4_bt(col, o);
        #pragma unroll
        for (int i = 0; i < 6; i++) s[i][j] = o[i];
    }
    long long plane = (long long)C * P;
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float o[6];
        wino4_bt(s[i], o);
        #pragma unroll
        for (int j = 0; j < 6; j++) v[(i * 6 + j) * plane + idx] = o[j];
    }
}

// y[n][k] 4x4 tile = A^T M A from M[x][k][p] (one GEMM split).
extern ""C"" __global__ void wino_f43_output(const float* __restrict__ mm, float* __restrict__ y,
    int N, int K, int outH, int outW, int tilesH, int tilesW)
{
    int P = N * tilesH * tilesW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = k*P + p
    if (idx >= K * P) return;
    int k = idx / P, p = idx - k * P;
    int n = p / (tilesH * tilesW), t = p - n * (tilesH * tilesW);
    int th = t / tilesW, tw = t - th * tilesW;
    long long plane = (long long)K * P;
    float m[6][6];
    #pragma unroll
    for (int i = 0; i < 6; i++)
        #pragma unroll
        for (int j = 0; j < 6; j++) m[i][j] = mm[(i * 6 + j) * plane + idx];
    float r[4][6];
    #pragma unroll
    for (int j = 0; j < 6; j++)
    {
        float col[6] = { m[0][j], m[1][j], m[2][j], m[3][j], m[4][j], m[5][j] }, o[4];
        wino4_at(col, o);
        #pragma unroll
        for (int i = 0; i < 4; i++) r[i][j] = o[i];
    }
    float* dst = y + ((long long)n * K + k) * outH * outW;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float o[4];
        wino4_at(r[i], o);
        int h = th * 4 + i;
        if (h >= outH) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int ww = tw * 4 + j;
            if (ww < outW) dst[h * outW + ww] = o[j];
        }
    }
}

// ---- Weight gradient: F(3x3, 4x4), the transpose dual of F(4x4, 3x3) (A and G trade places, B is unchanged). ----
// dW[k][c][i][j] = sum over n, h, w of dY[n][k][h][w] * x[n][c][h + i - pad][w + j - pad]. dY is cut into 4x4 tiles g, each
// paired with the 6x6 x window it touches; U = A g A^T with A = F(4,3)'s A (6x4), V = B^T d B, M[x] = sum over tiles of
// U[x] V[x] (the GEMM), and the 3x3 gradient is G^T M G with F(4,3)'s G (6x3).
// A (6x4) rows: [1,0,0,0], [1,1,1,1], [1,-1,1,-1], [1,2,4,8], [1,-2,4,-8], [0,0,0,1].
__device__ __forceinline__ void wino4_a(const float* g, float* o)
{
    o[0] = g[0];
    o[1] = g[0] + g[1] + g[2] + g[3];
    o[2] = g[0] - g[1] + g[2] - g[3];
    o[3] = g[0] + 2.0f * g[1] + 4.0f * g[2] + 8.0f * g[3];
    o[4] = g[0] - 2.0f * g[1] + 4.0f * g[2] - 8.0f * g[3];
    o[5] = g[3];
}

// G^T (3x6) rows: [1/4,-1/6,-1/6,1/24,1/24,0], [0,-1/6,1/6,1/12,-1/12,0], [0,-1/6,-1/6,1/6,1/6,1].
__device__ __forceinline__ void wino4_gt(const float* m, float* o)
{
    o[0] = 0.25f * m[0] - (m[1] + m[2]) * (1.0f / 6.0f) + (m[3] + m[4]) * (1.0f / 24.0f);
    o[1] = (m[2] - m[1]) * (1.0f / 6.0f) + (m[3] - m[4]) * (1.0f / 12.0f);
    o[2] = (m[3] + m[4] - m[1] - m[2]) * (1.0f / 6.0f) + m[5];
}

// U[x][k][p] = (A g A^T)[x] for dY tile p of channel k (zero past the edge).
extern ""C"" __global__ void wino_f34w_dy(const float* __restrict__ dy, float* __restrict__ u,
    int N, int K, int outH, int outW, int tilesH, int tilesW)
{
    int P = N * tilesH * tilesW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = k*P + p
    if (idx >= K * P) return;
    int k = idx / P, p = idx - k * P;
    int n = p / (tilesH * tilesW), t = p - n * (tilesH * tilesW);
    int th = t / tilesW, tw = t - th * tilesW;
    const float* src = dy + ((long long)n * K + k) * outH * outW;
    float g[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int h = th * 4 + i;
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int w = tw * 4 + j;
            g[i][j] = (h < outH && w < outW) ? __ldg(&src[h * outW + w]) : 0.0f;
        }
    }
    float s[6][4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        float col[4] = { g[0][j], g[1][j], g[2][j], g[3][j] }, o[6];
        wino4_a(col, o);
        #pragma unroll
        for (int i = 0; i < 6; i++) s[i][j] = o[i];
    }
    long long plane = (long long)K * P;
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float o[6];
        wino4_a(s[i], o);
        #pragma unroll
        for (int j = 0; j < 6; j++) u[(i * 6 + j) * plane + idx] = o[j];
    }
}

// V[x][p][c] = (B^T d B)[x] for the 6x6 x window of dY tile p (rows th*4 - pad .. +5), stored tile-major so the GEMM
// reads it as its row-major (P x C) operand.
extern ""C"" __global__ void wino_f34w_input(const float* __restrict__ x, float* __restrict__ v,
    int N, int C, int H, int W, int pad, int tilesH, int tilesW)
{
    int P = N * tilesH * tilesW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = p*C + c
    if (idx >= C * P) return;
    int p = idx / C, c = idx - p * C;
    int n = p / (tilesH * tilesW), t = p - n * (tilesH * tilesW);
    int th = t / tilesW, tw = t - th * tilesW;
    int h0 = th * 4 - pad, w0 = tw * 4 - pad;
    const float* src = x + ((long long)n * C + c) * H * W;
    float d[6][6];
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        int hh = h0 + i;
        #pragma unroll
        for (int j = 0; j < 6; j++)
        {
            int ww = w0 + j;
            d[i][j] = (hh >= 0 && hh < H && ww >= 0 && ww < W) ? __ldg(&src[hh * W + ww]) : 0.0f;
        }
    }
    float s[6][6];
    #pragma unroll
    for (int j = 0; j < 6; j++)
    {
        float col[6] = { d[0][j], d[1][j], d[2][j], d[3][j], d[4][j], d[5][j] }, o[6];
        wino4_bt(col, o);
        #pragma unroll
        for (int i = 0; i < 6; i++) s[i][j] = o[i];
    }
    long long plane = (long long)C * P;
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float o[6];
        wino4_bt(s[i], o);
        #pragma unroll
        for (int j = 0; j < 6; j++) v[(i * 6 + j) * plane + idx] = o[j];
    }
}

// dW[k][c] = G^T M G (3x3) from M[x][split][k][c], summing the split-K partials in order.
extern ""C"" __global__ void wino_f34w_output(const float* __restrict__ mm, float* __restrict__ dw, int K, int C, int splits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = k*C + c
    if (idx >= K * C) return;
    long long plane = (long long)K * C;
    float m[6][6];
    #pragma unroll
    for (int x = 0; x < 36; x++)
    {
        float acc = 0.0f;
        for (int s = 0; s < splits; s++) acc += mm[((long long)x * splits + s) * plane + idx];
        m[x / 6][x % 6] = acc;
    }
    float t[3][6];
    #pragma unroll
    for (int j = 0; j < 6; j++)
    {
        float col[6] = { m[0][j], m[1][j], m[2][j], m[3][j], m[4][j], m[5][j] }, o[3];
        wino4_gt(col, o);
        #pragma unroll
        for (int i = 0; i < 3; i++) t[i][j] = o[i];
    }
    float* dst = dw + (long long)idx * 9;
    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float o[3];
        wino4_gt(t[i], o);
        #pragma unroll
        for (int j = 0; j < 3; j++) dst[i * 3 + j] = o[j];
    }
}

extern ""C"" __global__ __launch_bounds__(256) void wino4_gemm128(const float* A, const float* B, float* Cm, int M, int Nc, int Kd, int splits, int kPerSplit)
{ wino_gemm_body<128, 128>(A, B, Cm, M, Nc, Kd, splits, kPerSplit); }
extern ""C"" __global__ __launch_bounds__(256) void wino4_gemm64(const float* A, const float* B, float* Cm, int M, int Nc, int Kd, int splits, int kPerSplit)
{ wino_gemm_body<64, 64>(A, B, Cm, M, Nc, Kd, splits, kPerSplit); }
";

    public static string[] GetKernelNames() => new[]
    {
        "wino_f43_filter", "wino_f43_filter_flip", "wino_f43_input", "wino_f43_output",
        "wino_f34w_dy", "wino_f34w_input", "wino_f34w_output", "wino4_gemm128", "wino4_gemm64",
    };
}
