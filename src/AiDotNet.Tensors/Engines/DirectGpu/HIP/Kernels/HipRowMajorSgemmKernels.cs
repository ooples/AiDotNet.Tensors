// Copyright (c) AiDotNet. All rights reserved.
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>One macro-tile shape of the row-major SGEMM: a BM x BN block of C per 256-thread block, TM x TN per thread.</summary>
internal readonly record struct HipRowMajorSgemmTile(int BlockM, int BlockN, int ThreadM, int ThreadN, bool Vectorized = false)
{
    internal const int BlockK = 16;
    internal const int Threads = 256;

    internal string KernelName =>
        string.Format(CultureInfo.InvariantCulture, "rm_sgemm{0}_{1}x{2}_{3}x{4}",
            Vectorized ? "_v4" : string.Empty, BlockM, BlockN, ThreadM, ThreadN);

    internal bool IsValid =>
        BlockM > 0 && BlockN > 0 && ThreadM > 0 && ThreadN > 0 &&
        BlockM % ThreadM == 0 && BlockN % ThreadN == 0 &&
        (BlockM / ThreadM) * (BlockN / ThreadN) == Threads &&
        (BlockM * BlockK) % Threads == 0 && (BlockK * BlockN) % Threads == 0 &&
        (!Vectorized || (ThreadM % 4 == 0 && ThreadN % 4 == 0 && (BlockM * BlockK) % (4 * Threads) == 0 && (BlockK * BlockN) % (4 * Threads) == 0));

    /// <summary>Whether this tile can run shape (M, N, K): the float4 variant needs K and N to be multiples of 4.</summary>
    internal bool Supports(int m, int n, int k) => m > 0 && n > 0 && k > 0 && (!Vectorized || (k % 4 == 0 && n % 4 == 0));
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
        new(64, 64, 4, 4, true),
        new(128, 64, 8, 4, true),
        new(64, 128, 4, 8, true),
        new(128, 128, 8, 8, true),
    };

    private const string Body = @"
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void rm_sgemm_body(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta, int kChunk, size_t cStride)
{
    // Split-K: block z covers K range [z * kChunk, min(K, (z + 1) * kChunk)) and writes its own C slice.
    const int kBeg = blockIdx.z * kChunk;
    const int kEnd = min(K, kBeg + kChunk);
    C += blockIdx.z * cStride;
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
            ra[i] = (gm < M && gk < kEnd) ? A[(size_t)gm * K + gk] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < BL; i++)
        {
            int idx = tid + i * NT, r = idx / BN, c = idx % BN, gk = k0 + r, gn = bn + c;
            rb[i] = (gk < kEnd && gn < N) ? B[(size_t)gk * N + gn] : 0.0f;
        }
    };
    auto stash = [&](int buf)
    {
#pragma unroll
        for (int i = 0; i < AL; i++) { int idx = tid + i * NT; As[buf][idx % BK][idx / BK] = ra[i]; }
#pragma unroll
        for (int i = 0; i < BL; i++) { int idx = tid + i * NT; Bs[buf][idx / BN][idx % BN] = rb[i]; }
    };

    fetch(kBeg);
    stash(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = kBeg; k0 < kEnd; k0 += BK)
    {
        bool more = k0 + BK < kEnd;
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
// Vectorized variant: K and N multiples of 4 (so every float4 global access is aligned and whole), TM and TN multiples
// of 4. Each thread's TM rows are split into TM/4 groups of four strided BM/(TM/4) apart (likewise columns), so the
// float4 LDS reads of a wavefront hit consecutive 16-byte words instead of 2-way conflicting at a 32-byte stride.
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void rm_sgemm_vec_body(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta, int kChunk, size_t cStride)
{
    // Split-K: block z covers K range [z * kChunk, min(K, (z + 1) * kChunk)) and writes its own C slice.
    const int kBeg = blockIdx.z * kChunk;
    const int kEnd = min(K, kBeg + kChunk);
    C += blockIdx.z * cStride;
    constexpr int BK = 16;
    constexpr int NT = (BM / TM) * (BN / TN);
    constexpr int AL = (BM * BK) / (4 * NT);
    constexpr int BL = (BK * BN) / (4 * NT);
    constexpr int GM = TM / 4, GN = TN / 4;
    constexpr int SM = BM / GM, SN = BN / GN;
    __shared__ __align__(16) float As[2][BK][BM + 4];
    __shared__ __align__(16) float Bs[2][BK][BN + 4];
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    float4 ra[AL], rb[BL];
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
            int idx = tid + i * NT, r = idx / (BK / 4), c = (idx % (BK / 4)) * 4, gm = bm + r, gk = k0 + c;
            ra[i] = (gm < M && gk < kEnd) ? *reinterpret_cast<const float4*>(A + (size_t)gm * K + gk) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int i = 0; i < BL; i++)
        {
            int idx = tid + i * NT, r = idx / (BN / 4), c = (idx % (BN / 4)) * 4, gk = k0 + r, gn = bn + c;
            rb[i] = (gk < kEnd && gn < N) ? *reinterpret_cast<const float4*>(B + (size_t)gk * N + gn) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto stash = [&](int buf)
    {
#pragma unroll
        for (int i = 0; i < AL; i++)
        {
            int idx = tid + i * NT, r = idx / (BK / 4), c = (idx % (BK / 4)) * 4;
            As[buf][c][r] = ra[i].x; As[buf][c + 1][r] = ra[i].y; As[buf][c + 2][r] = ra[i].z; As[buf][c + 3][r] = ra[i].w;
        }
#pragma unroll
        for (int i = 0; i < BL; i++)
        {
            int idx = tid + i * NT, r = idx / (BN / 4), c = (idx % (BN / 4)) * 4;
            *reinterpret_cast<float4*>(&Bs[buf][r][c]) = rb[i];
        }
    };

    fetch(kBeg);
    stash(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = kBeg; k0 < kEnd; k0 += BK)
    {
        bool more = k0 + BK < kEnd;
        if (more) fetch(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk++)
        {
            float a[TM], b[TN];
#pragma unroll
            for (int g = 0; g < GM; g++)
            {
                float4 v = *reinterpret_cast<const float4*>(&As[buf][kk][g * SM + ty * 4]);
                a[g * 4] = v.x; a[g * 4 + 1] = v.y; a[g * 4 + 2] = v.z; a[g * 4 + 3] = v.w;
            }
#pragma unroll
            for (int g = 0; g < GN; g++)
            {
                float4 v = *reinterpret_cast<const float4*>(&Bs[buf][kk][g * SN + tx * 4]);
                b[g * 4] = v.x; b[g * 4 + 1] = v.y; b[g * 4 + 2] = v.z; b[g * 4 + 3] = v.w;
            }
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
        int row = bm + (i / 4) * SM + ty * 4 + (i % 4);
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; j++)
        {
            int col = bn + (j / 4) * SN + tx * 4 + (j % 4);
            if (col >= N) continue;
            size_t o = (size_t)row * N + col;
            C[o] = beta == 0.0f ? alpha * acc[i][j] : alpha * acc[i][j] + beta * C[o];
        }
    }
}
// Split-K reduction: C = alpha * sum_z partial[z] + beta * C, summed in z order (deterministic).
extern ""C"" __global__ __launch_bounds__(256) void rm_sgemm_splitk_reduce(const float* __restrict__ partial, float* __restrict__ C,
    int count, int splits, float alpha, float beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float s = 0.0f;
    for (int z = 0; z < splits; z++) s += partial[(size_t)z * count + i];
    C[i] = beta == 0.0f ? alpha * s : alpha * s + beta * C[i];
}
";

    internal const string SplitKReduceName = "rm_sgemm_splitk_reduce";

    internal static string BuildSource()
    {
        var sb = new StringBuilder(Body);
        foreach (var t in Tiles)
        {
            sb.Append("extern \"C\" __global__ __launch_bounds__(256) void ").Append(t.KernelName)
                .Append("(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta, int kChunk, size_t cStride)\n");
            sb.Append(string.Format(CultureInfo.InvariantCulture,
                "{{ {0}<{1}, {2}, {3}, {4}>(A, B, C, M, N, K, alpha, beta, kChunk, cStride); }}\n",
                t.Vectorized ? "rm_sgemm_vec_body" : "rm_sgemm_body", t.BlockM, t.BlockN, t.ThreadM, t.ThreadN));
        }
        return sb.ToString();
    }

    internal static string[] KernelNames()
    {
        var names = new string[Tiles.Length + 1];
        for (int i = 0; i < Tiles.Length; i++) names[i] = Tiles[i].KernelName;
        names[Tiles.Length] = SplitKReduceName;
        return names;
    }
}
