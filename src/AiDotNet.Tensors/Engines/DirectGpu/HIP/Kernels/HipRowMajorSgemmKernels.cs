// Copyright (c) AiDotNet. All rights reserved.
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>One macro-tile shape of the row-major SGEMM: a BM x BN block of C per 256-thread block, TM x TN per thread.</summary>
internal readonly record struct HipRowMajorSgemmTile(int BlockM, int BlockN, int ThreadM, int ThreadN)
{
    internal const int BlockK = 16;
    internal const int Threads = 256;

    internal string KernelName =>
        string.Create(CultureInfo.InvariantCulture, $"rm_sgemm_{BlockM}x{BlockN}_{ThreadM}x{ThreadN}");

    internal bool IsValid =>
        BlockM > 0 && BlockN > 0 && ThreadM > 0 && ThreadN > 0 &&
        BlockM % ThreadM == 0 && BlockN % ThreadN == 0 &&
        (BlockM / ThreadM) * (BlockN / ThreadN) == Threads &&
        (BlockM * BlockK) % Threads == 0 && (BlockK * BlockN) % Threads == 0;
}

/// <summary>
/// Row-major SGEMM for HIP (#377): C[M,N] = alpha * A[M,K] * B[K,N] + beta * C, reading A and B where they are. The
/// CLBlast indirect kernel needs A transposed and both operands padded first, which on skinny shapes (4096x64x1024) is
/// a full extra pass over A per call. Each block stages a BK-deep slice of A (transposed into LDS) and of B, double
/// buffered: the next slice is fetched into registers while the current one is multiplied. Bounds are checked on every
/// global access, so no padding is needed.
/// </summary>
internal static class HipRowMajorSgemmKernels
{
    internal static readonly HipRowMajorSgemmTile[] Tiles =
    {
        new(64, 64, 4, 4),
        new(128, 64, 8, 4),
        new(64, 128, 4, 8),
        new(128, 128, 8, 8),
        new(32, 128, 2, 8),
        new(128, 32, 8, 2),
    };

    private const string Body = @"
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void rm_sgemm_body(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta)
{
    constexpr int BK = 16;
    constexpr int NT = (BM / TM) * (BN / TN);
    constexpr int AL = (BM * BK) / NT;
    constexpr int BL = (BK * BN) / NT;
    __shared__ float As[2][BK][BM + 4];
    __shared__ float Bs[2][BK][BN + 4];
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    float ra[AL], rb[BL];
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    auto fetch = [&](int k0)
    {
#pragma unroll
        for (int i = 0; i < AL; i++)
        {
            int idx = tid + i * NT, r = idx / BK, c = idx % BK, gm = bm + r, gk = k0 + c;
            ra[i] = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < BL; i++)
        {
            int idx = tid + i * NT, r = idx / BN, c = idx % BN, gk = k0 + r, gn = bn + c;
            rb[i] = (gk < K && gn < N) ? B[(size_t)gk * N + gn] : 0.0f;
        }
    };
    auto stash = [&](int buf)
    {
#pragma unroll
        for (int i = 0; i < AL; i++) { int idx = tid + i * NT; As[buf][idx % BK][idx / BK] = ra[i]; }
#pragma unroll
        for (int i = 0; i < BL; i++) { int idx = tid + i * NT; Bs[buf][idx / BN][idx % BN] = rb[i]; }
    };

    fetch(0);
    stash(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK)
    {
        bool more = k0 + BK < K;
        if (more) fetch(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk++)
        {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[buf][kk][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[buf][kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (more) stash(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }

#pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int row = bm + ty * TM + i;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; j++)
        {
            int col = bn + tx * TN + j;
            if (col >= N) continue;
            size_t o = (size_t)row * N + col;
            // beta == 0 never reads C: a fresh output buffer can hold NaN, and 0 * NaN is NaN.
            C[o] = beta == 0.0f ? alpha * acc[i][j] : alpha * acc[i][j] + beta * C[o];
        }
    }
}
";

    internal static string BuildSource()
    {
        var sb = new StringBuilder(Body);
        foreach (var t in Tiles)
        {
            sb.Append(CultureInfo.InvariantCulture,
                $"extern \"C\" __global__ __launch_bounds__(256) void {t.KernelName}(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta)\n");
            sb.Append(CultureInfo.InvariantCulture,
                $"{{ rm_sgemm_body<{t.BlockM}, {t.BlockN}, {t.ThreadM}, {t.ThreadN}>(A, B, C, M, N, K, alpha, beta); }}\n");
        }
        return sb.ToString();
    }

    internal static string[] KernelNames()
    {
        var names = new string[Tiles.Length];
        for (int i = 0; i < Tiles.Length; i++) names[i] = Tiles[i].KernelName;
        return names;
    }
}
