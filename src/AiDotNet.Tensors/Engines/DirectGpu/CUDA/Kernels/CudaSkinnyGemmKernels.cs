// Copyright (c) AiDotNet. All rights reserved.
// CUDA GEMM kernels for narrow row-major products (a small model's projections and their weight gradients).

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// Row-major FP32 GEMMs for the shapes cuBLAS serves worst: a tall operand times a narrow one
    /// (<c>[2048,64] x [64,64]</c>, a d_model-64 Transformer's projections) and the weight gradient of the same layer,
    /// a long reduction into a small output (<c>[2048,64]^T x [2048,64]</c>). On a GTX 1660 Ti cuBLAS ran the first
    /// as 32 blocks of 64 threads (~17 us) and the second as split-K plus a reduction kernel (~22 us); every algorithm
    /// cuBLASLt offers for them timed the same.
    /// </summary>
    internal static class CudaSkinnyGemmKernels
    {
        public static string GetSource()
        {
            return @"
#define SG_BM 32
#define SG_BN 64
#define SG_BK 32
#define SG_THREADS 64

// acc[i][j] += As[k][ty*4+i] * Bs[k][col j] over one SG_BK-deep chunk. Each of the block's 64 threads owns 4 rows x 8
// columns of the 32x64 tile -- columns tx*4..+3 and 32+tx*4..+3, so eight neighbouring threads read 128 contiguous
// bytes of B (no bank conflicts) -- and does 32 FMAs per 48 bytes it reads from shared memory. A 2x4 piece per thread
// (8 FMAs per 24 bytes) left the FMA pipes waiting on shared memory at ~30% of peak.
// A macro, not a function taking acc: an array parameter can put the accumulators in local memory.
#define SG_ACCUMULATE(DEPTH)                                                                                  \
    _Pragma(""unroll"")                                                                                     \
    for (int kk = 0; kk < (DEPTH); kk++)                                                                    \
    {                                                                                                       \
        float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);                                       \
        float4 b0 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);                                      \
        float4 b1 = *reinterpret_cast<const float4*>(&Bs[kk][32 + tx * 4]);                                 \
        float av[4] = { a.x, a.y, a.z, a.w };                                                               \
        _Pragma(""unroll"")                                                                                 \
        for (int i = 0; i < 4; i++)                                                                         \
        {                                                                                                   \
            acc[i][0] += av[i] * b0.x; acc[i][1] += av[i] * b0.y; acc[i][2] += av[i] * b0.z;                \
            acc[i][3] += av[i] * b0.w; acc[i][4] += av[i] * b1.x; acc[i][5] += av[i] * b1.y;                \
            acc[i][6] += av[i] * b1.z; acc[i][7] += av[i] * b1.w;                                           \
        }                                                                                                   \
    }

// Bs[k][n] = B[k0 + k][col0 + n] for a row-major B[rows, N], rows >= kEnd zero.
__device__ __forceinline__ void sg_load_b_rows(const float* __restrict__ B, float (*Bs)[SG_BN + 4],
    int k0, int kEnd, int col0, int N)
{
    for (int idx = threadIdx.x; idx < SG_BK * SG_BN / 4; idx += SG_THREADS)
    {
        int k = idx >> 4, c = (idx & 15) * 4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (k0 + k < kEnd && col0 + c < N)
            v = __ldg(reinterpret_cast<const float4*>(B + (long long)(k0 + k) * N + col0 + c));
        *reinterpret_cast<float4*>(&Bs[k][c]) = v;
    }
}

// The ReLU kernel's operation (relu_nan_safe): NaN passes through, so the fused epilogue equals BiasAdd then Relu.
__device__ __forceinline__ float sg_relu(float x)
{
    unsigned int bits = __float_as_uint(x) & 0x7fffffffu;
    return (bits > 0x7f800000u) ? x : (x > 0.0f ? x : 0.0f);
}

// Writes act(alpha * value + beta * C + bias) for one float4 of a thread's 4x8 piece, skipping what lies outside
// [M, N]. beta == 0 never reads C; bias may be null; relu applies sg_relu.
__device__ __forceinline__ void sg_store(float* __restrict__ C, int M, int N, int row, int col, float4 v,
    float alpha, float beta, const float* __restrict__ bias, int relu)
{
    if (row >= M || col >= N) return;
    float4* out = reinterpret_cast<float4*>(C + (long long)row * N + col);
    float4 r = make_float4(alpha * v.x, alpha * v.y, alpha * v.z, alpha * v.w);
    if (beta != 0.0f)
    {
        float4 c = *out;
        r.x += beta * c.x; r.y += beta * c.y; r.z += beta * c.z; r.w += beta * c.w;
    }
    if (bias)
    {
        float4 b = *reinterpret_cast<const float4*>(bias + col);
        r.x += b.x; r.y += b.y; r.z += b.z; r.w += b.w;
    }
    if (relu) { r.x = sg_relu(r.x); r.y = sg_relu(r.y); r.z = sg_relu(r.z); r.w = sg_relu(r.w); }
    *out = r;
}

// C[M,N] = act(alpha * A[M,K] * op(B) + beta * C + bias), row-major; op(B) is B[K,N], or B[N,K] transposed when
// transB; bias[N] is optional (null) and act is ReLU when relu, else identity.
// One 32x64 output tile per block of four 64-thread groups. K is swept in SG_BK-deep chunks staged in shared memory;
// each group accumulates the whole tile (4x8 per thread) over its quarter of every chunk, and the four partial tiles
// are added in group order at the end (deterministic). The next chunk's operands are loaded into registers before the
// current chunk's FMAs. One 64-thread block per tile (two warps, the chunk's load then its FMAs in sequence) left a
// d_model-64 Transformer's projections latency-bound: the in-projection [256,64]x[64,192] took 5.9 us on 24 blocks.
// Requires N and K multiples of four and 16-byte-aligned operands (the host checks).
#define SG_GROUPS 4
#define SG_BLOCK (SG_GROUPS * SG_THREADS)
// BM: the tile's rows (32, 16 or 8; each thread holds BM / 8 rows x 8 columns). The host takes the smallest that still
// gives the grid enough blocks: a 256-row product with N = 64 has 8 tiles of 32 rows, 8 SMs' worth.
template <int BM>
__device__ __forceinline__ void skinny_gemm_body(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int transB, float alpha, float beta, const float* __restrict__ bias, int relu)
{
    constexpr int R = BM / 8;
    constexpr int STAGE = SG_BK * (BM + 4) + SG_BK * (SG_BN + 4);
    constexpr int PART = SG_GROUPS * BM * SG_BN;
    // As / Bs for one chunk; reused for the four groups' partial tiles (SG_GROUPS x BM x 64) at the end.
    __shared__ __align__(16) float smem[STAGE > PART ? STAGE : PART];
    float (*As)[BM + 4] = reinterpret_cast<float (*)[BM + 4]>(smem);
    float (*Bs)[SG_BN + 4] = reinterpret_cast<float (*)[SG_BN + 4]>(smem + SG_BK * (BM + 4));
    const int tid = threadIdx.x, g = tid >> 6, lt = tid & 63;
    const int tx = lt & 7, ty = lt >> 3;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * SG_BN;
    float acc[R][8];
    _Pragma(""unroll"") for (int i = 0; i < R; i++)
        _Pragma(""unroll"") for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    // Per chunk: A is BM x 32 floats (one float4 for each of the first BM * 8 threads), B is 32 x 64 (two per thread).
    float4 pa = make_float4(0.0f, 0.0f, 0.0f, 0.0f), pb[2];
    auto fetch = [&](int k0) {
        if (tid < BM * 8)
        {
            int r = tid >> 3, c = (tid & 7) * 4;   // A[row0 + r][k0 + c .. +3]
            pa = (row0 + r < M && k0 + c < K)
                ? __ldg(reinterpret_cast<const float4*>(A + (long long)(row0 + r) * K + k0 + c))
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        _Pragma(""unroll"")
        for (int u = 0; u < 2; u++)
        {
            int idx = tid + u * SG_BLOCK;
            if (transB)
            {
                int n = idx >> 3, c = (idx & 7) * 4;   // B[col0 + n][k0 + c .. +3], B stored [N, K]
                pb[u] = (col0 + n < N && k0 + c < K)
                    ? __ldg(reinterpret_cast<const float4*>(B + (long long)(col0 + n) * K + k0 + c))
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            else
            {
                int k = idx >> 4, c = (idx & 15) * 4;   // B[k0 + k][col0 + c .. +3]
                pb[u] = (k0 + k < K && col0 + c < N)
                    ? __ldg(reinterpret_cast<const float4*>(B + (long long)(k0 + k) * N + col0 + c))
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    };
    fetch(0);
    for (int k0 = 0; k0 < K; k0 += SG_BK)
    {
        if (tid < BM * 8)
        {
            int r = tid >> 3, c = (tid & 7) * 4;
            As[c][r] = pa.x; As[c + 1][r] = pa.y; As[c + 2][r] = pa.z; As[c + 3][r] = pa.w;
        }
        _Pragma(""unroll"")
        for (int u = 0; u < 2; u++)
        {
            int idx = tid + u * SG_BLOCK;
            if (transB)
            {
                int n = idx >> 3, c = (idx & 7) * 4;
                Bs[c][n] = pb[u].x; Bs[c + 1][n] = pb[u].y; Bs[c + 2][n] = pb[u].z; Bs[c + 3][n] = pb[u].w;
            }
            else
            {
                int k = idx >> 4, c = (idx & 15) * 4;
                *reinterpret_cast<float4*>(&Bs[k][c]) = pb[u];
            }
        }
        __syncthreads();
        if (k0 + SG_BK < K) fetch(k0 + SG_BK);
        _Pragma(""unroll"")
        for (int q = 0; q < SG_BK / SG_GROUPS; q++)
        {
            int kk = g * (SG_BK / SG_GROUPS) + q;
            float av[R];
            _Pragma(""unroll"") for (int i = 0; i < R; i++) av[i] = As[kk][ty * R + i];
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[kk][32 + tx * 4]);
            _Pragma(""unroll"")
            for (int i = 0; i < R; i++)
            {
                acc[i][0] += av[i] * b0.x; acc[i][1] += av[i] * b0.y; acc[i][2] += av[i] * b0.z;
                acc[i][3] += av[i] * b0.w; acc[i][4] += av[i] * b1.x; acc[i][5] += av[i] * b1.y;
                acc[i][6] += av[i] * b1.z; acc[i][7] += av[i] * b1.w;
            }
        }
        __syncthreads();
    }

    // Partial tiles: group g writes part[g][BM][64]; then the block finishes the tile a float4 at a time.
    float* part = smem;
    _Pragma(""unroll"")
    for (int i = 0; i < R; i++)
    {
        float* row = part + (g * BM + ty * R + i) * SG_BN;
        *reinterpret_cast<float4*>(row + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(row + 32 + tx * 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
    __syncthreads();
    for (int e4 = tid; e4 < BM * SG_BN / 4; e4 += SG_BLOCK)
    {
        int e = e4 * 4, rowL = e / SG_BN, colL = e % SG_BN;
        float4 v = *reinterpret_cast<const float4*>(part + rowL * SG_BN + colL);
        _Pragma(""unroll"")
        for (int gg = 1; gg < SG_GROUPS; gg++)
        {
            float4 w = *reinterpret_cast<const float4*>(part + (gg * BM + rowL) * SG_BN + colL);
            v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w;
        }
        sg_store(C, M, N, row0 + rowL, col0 + colL, v, alpha, beta, bias, relu);
    }
}

#define SG_KERNEL(NAME, BM)                                                                                 \
extern ""C"" __global__ __launch_bounds__(SG_BLOCK) void NAME(                                              \
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,                        \
    int M, int N, int K, int transB, float alpha, float beta, const float* __restrict__ bias, int relu)     \
{                                                                                                           \
    skinny_gemm_body<BM>(A, B, C, M, N, K, transB, alpha, beta, bias, relu);                                \
}
SG_KERNEL(skinny_gemm, 32)
SG_KERNEL(skinny_gemm_m16, 16)
SG_KERNEL(skinny_gemm_m8, 8)

// Small products cuBLAS runs as one long-latency kernel each: a classifier head over a flattened sequence
// ([16,1024]x[1024,10], ~9 us as gemmSN) and its gradients ([16,10]x[10,1024] and [1024,16]x[16,10], ~3 us each).
// A(m, k) = A[m * aRow + k * aCol] and B(k, n) = B[k * bRow + n * bCol], so one kernel serves A or B transposed.

// Few outputs, long reduction: block (column group, row) computes C[row][16 * group .. +15] with its threads striding
// over k, then reduces each column's partial sums through warp shuffles and one shared-memory step (fixed order).
extern ""C"" __global__ __launch_bounds__(256) void small_gemm_rowreduce(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, long long aRow, long long aCol, long long bRow, long long bCol,
    float alpha, float beta, const float* __restrict__ bias, int relu)
{
    __shared__ float warpSums[8][16];
    int row = blockIdx.y, col0 = blockIdx.x * 16;
    float acc[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float a = A[row * aRow + k * aCol];
        #pragma unroll
        for (int j = 0; j < 16; j++)
            if (col0 + j < N) acc[j] = fmaf(a, B[k * bRow + (col0 + j) * bCol], acc[j]);
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        float v = acc[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        if (lane == 0) warpSums[warp][j] = v;
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        int j = threadIdx.x, col = col0 + j;
        if (col < N) {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; w++) v += warpSums[w][j];
            float r = alpha * v;
            if (beta != 0.0f) r += beta * C[(long long)row * N + col];
            if (bias) r += bias[col];
            C[(long long)row * N + col] = relu ? sg_relu(r) : r;
        }
    }
}

// Short reduction (K <= 32): one thread per output, its K products summed in order.
extern ""C"" __global__ __launch_bounds__(256) void small_gemm_tinyk(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, long long aRow, long long aCol, long long bRow, long long bCol,
    float alpha, float beta, const float* __restrict__ bias, int relu)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)M * N) return;
    int row = (int)(idx / N), col = (int)(idx - (long long)row * N);
    float v = 0.0f;
    for (int k = 0; k < K; k++) v = fmaf(A[row * aRow + k * aCol], B[k * bRow + col * bCol], v);
    float r = alpha * v;
    if (beta != 0.0f) r += beta * C[idx];
    if (bias) r += bias[col];
    C[idx] = relu ? sg_relu(r) : r;
}

// C[M,N] = alpha * A[K,M]^T * B[K,N] + beta * C, row-major, for a long reduction K (a weight gradient, K = rows of
// the batch). Grid (tilesN, tilesM, splits), 256 threads: block z reduces rows [z * rowsPerSplit, +rowsPerSplit) of its
// 32x64 tile, its four 64-thread groups taking interleaved 16-row chunks; the groups' sums are added in group order in
// shared memory and written to partials[z][tile]. The last block to arrive at a tile (counters[tile]) adds the tile's
// partials in split order 0, 1, ... -- so the bits do not depend on which block finished first -- writes C and resets
// the counter for the next launch. One kernel where cuBLAS ran split-K plus a reduction kernel; the groups give a
// 64x64 output (two tiles) enough warps without a split count whose reduction one block could not finish quickly.
#define SG_TN_GROUPS 4
#define SG_TN_BK 16
#define SG_TN_STAGE (SG_TN_BK * (SG_BM + 4) + SG_TN_BK * (SG_BN + 4))
extern ""C"" __global__ __launch_bounds__(SG_TN_GROUPS * SG_THREADS) void skinny_gemm_tn(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    float* __restrict__ partials, unsigned int* __restrict__ counters,
    int M, int N, int K, int rowsPerSplit, float alpha, float beta)
{
    __shared__ __align__(16) float smem[SG_TN_GROUPS * SG_TN_STAGE];
    __shared__ unsigned int isLast;
    int g = threadIdx.x / SG_THREADS, lt = threadIdx.x % SG_THREADS;
    int tx = lt & 7, ty = lt >> 3;
    float (*As)[SG_BM + 4] = reinterpret_cast<float (*)[SG_BM + 4]>(smem + g * SG_TN_STAGE);
    float (*Bs)[SG_BN + 4] = reinterpret_cast<float (*)[SG_BN + 4]>(smem + g * SG_TN_STAGE + SG_TN_BK * (SG_BM + 4));
    int row0 = blockIdx.y * SG_BM, col0 = blockIdx.x * SG_BN;
    int kBegin = blockIdx.z * rowsPerSplit;
    int kEnd = min(K, kBegin + rowsPerSplit);
    float acc[4][8];
    _Pragma(""unroll"") for (int i = 0; i < 4; i++)
        _Pragma(""unroll"") for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;
    // Every group runs the same number of passes (a pass past kEnd loads zeros), so the block-wide barriers line up.
    // Each thread holds the next pass's two float4s of A and four of B in registers, loaded before the current pass
    // computes: the DRAM round trip then overlaps the FMAs instead of preceding them (each pass waited a full round
    // trip, and a 64x64 output has too few blocks to hide it).
    float4 pa[2], pb[4];
#define SG_TN_FETCH(BASE)                                                                                   \
    {                                                                                                       \
        int k0 = (BASE) + g * SG_TN_BK;                                                                     \
        _Pragma(""unroll"")                                                                                 \
        for (int r = 0; r < 2; r++)                                                                         \
        {                                                                                                   \
            int idx = lt + r * SG_THREADS, k = idx >> 3, c = (idx & 7) * 4;                                 \
            pa[r] = (k0 + k < kEnd && row0 + c < M)                                                         \
                ? __ldg(reinterpret_cast<const float4*>(A + (long long)(k0 + k) * M + row0 + c))            \
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);                                                      \
        }                                                                                                   \
        _Pragma(""unroll"")                                                                                 \
        for (int r = 0; r < 4; r++)                                                                         \
        {                                                                                                   \
            int idx = lt + r * SG_THREADS, k = idx >> 4, c = (idx & 15) * 4;                                \
            pb[r] = (k0 + k < kEnd && col0 + c < N)                                                         \
                ? __ldg(reinterpret_cast<const float4*>(B + (long long)(k0 + k) * N + col0 + c))            \
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);                                                      \
        }                                                                                                   \
    }
    SG_TN_FETCH(kBegin)
    for (int base = kBegin; base < kEnd; base += SG_TN_GROUPS * SG_TN_BK)
    {
        // As[k][m] = A[k0 + k][row0 + m] (A is stored [K, M]); Bs[k][n] = B[k0 + k][col0 + n].
        _Pragma(""unroll"")
        for (int r = 0; r < 2; r++)
        {
            int idx = lt + r * SG_THREADS;
            *reinterpret_cast<float4*>(&As[idx >> 3][(idx & 7) * 4]) = pa[r];
        }
        _Pragma(""unroll"")
        for (int r = 0; r < 4; r++)
        {
            int idx = lt + r * SG_THREADS;
            *reinterpret_cast<float4*>(&Bs[idx >> 4][(idx & 15) * 4]) = pb[r];
        }
        __syncthreads();
        if (base + SG_TN_GROUPS * SG_TN_BK < kEnd) SG_TN_FETCH(base + SG_TN_GROUPS * SG_TN_BK)
        SG_ACCUMULATE(SG_TN_BK)
        __syncthreads();
    }
#undef SG_TN_FETCH
    // Groups 1..3 hand their sums to group 0 through shared memory (the staging area is free now), which adds them
    // in group order. Partials keep each thread's register order (4 rows x 8 values) so the tail reads float4s.
    if (g > 0)
    {
        float* handoff = smem + (g - 1) * (SG_THREADS * 32) + lt * 32;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++) handoff[i * 8 + j] = acc[i][j];
    }
    __syncthreads();
    int tiles = gridDim.x * gridDim.y;
    int tile = blockIdx.y * gridDim.x + blockIdx.x;
    if (g == 0)
    {
        for (int h = 0; h < SG_TN_GROUPS - 1; h++)
        {
            const float* other = smem + h * (SG_THREADS * 32) + lt * 32;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 8; j++) acc[i][j] += other[i * 8 + j];
        }
        float4* mine = reinterpret_cast<float4*>(partials + ((long long)blockIdx.z * tiles + tile) * (SG_BM * SG_BN))
            + lt * 8;
        for (int i = 0; i < 4; i++)
        {
            mine[2 * i] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            mine[2 * i + 1] = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        isLast = atomicAdd(&counters[tile], 1u) == gridDim.z - 1;
    __syncthreads();
    if (!isLast) return;
    __threadfence();

    // The tile's 512 float4s, two per thread: float4 p belongs to group-0 thread p / 8, value pair p % 8.
    for (int p = threadIdx.x; p < SG_BM * SG_BN / 4; p += SG_TN_GROUPS * SG_THREADS)
    {
        float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int z = 0; z < (int)gridDim.z; z++)
        {
            float4 v = __ldcg(reinterpret_cast<const float4*>(partials + ((long long)z * tiles + tile) * (SG_BM * SG_BN))
                + p);
            s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
        }
        int owner = p >> 3, q = p & 7;
        sg_store(C, M, N, row0 + (owner >> 3) * 4 + (q >> 1), col0 + (q & 1) * 32 + (owner & 7) * 4, s, alpha, beta, nullptr, 0);
    }
    if (threadIdx.x == 0) counters[tile] = 0u;
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "skinny_gemm",
                "skinny_gemm_m16",
                "skinny_gemm_m8",
                "small_gemm_rowreduce",
                "small_gemm_tinyk",
                "skinny_gemm_tn"
            };
        }
    }
}
