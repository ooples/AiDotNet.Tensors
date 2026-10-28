// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// Register-blocked strided-batched SGEMM, C[b] = A[b] * B[b] (row-major, batches end to end), for the Winograd
    /// convolutions' per-point products: M and K are channel counts, N the tile count, which is small on late layers
    /// (49 for a 28x28 F(4x4, 3x3) layer). A 16x16 work group computes a 64 x 64 block of C, each work item a 4 x 4
    /// sub-block strided by 16 so its LDS reads and C stores stay conflict-free and coalesced; K advances 16 at a time
    /// through LDS.
    /// </summary>
    internal static class BatchedGemmKernels
    {
        internal const string Rb64 = "gemm_batched_rb64";
        internal const string Rb64SplitK = "gemm_rb64_splitk";
        internal const string SplitKReduce = "gemm_splitk_reduce";
        internal const string Im2ColWide = "im2col_wide";
        internal const string Rb128V = "gemm_batched_rb128v";
        internal const string Rb64Prefetch = "gemm_batched_rb64p";

        internal static string[] GetKernelNames() => new[] { Rb64, Rb64SplitK, SplitKReduce, Im2ColWide, Rb128V, Rb64Prefetch };

        internal const string Source = @"
#define RB_BM 64
#define RB_BN 64
#define RB_BK 16
// A convolution's epilogue: epi bit 0 adds bias[channel], bit 1 applies ReLU (NaN passes through, as the relu kernel).
// The same arithmetic as the conv2d_bias_add and relu passes it replaces.
float conv_epilogue(float v, __global const float* bias, int channel, int epi)
{
    if (epi & 1) v += bias[channel];
    if (epi & 2) v = isnan(v) ? v : fmax(0.0f, v);
    return v;
}
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_batched_rb64(__global const float* restrict A, __global const float* restrict B, __global float* restrict C,
    const int M, const int N, const int K)
{
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bn0 = get_group_id(0) * RB_BN, bm0 = get_group_id(1) * RB_BM;
    const long b = get_group_id(2);
    A += b * M * K; B += b * K * N; C += b * M * N;
    __local float As[RB_BK][RB_BM + 1];
    __local float Bs[RB_BK][RB_BN + 1];
    float acc[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    const int tid = ty * 16 + tx;
    for (int k0 = 0; k0 < K; k0 += RB_BK)
    {
        for (int i = 0; i < 4; i++)
        {
            int e = tid + 256 * i;
            int r = e >> 4, c = e & 15;
            int gr = bm0 + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[(long)gr * K + gc] : 0.0f;
        }
        for (int i = 0; i < 4; i++)
        {
            int e = tid + 256 * i;
            int r = e >> 6, c = e & 63;
            int gr = k0 + r, gc = bn0 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[(long)gr * N + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < RB_BK; kk++)
        {
            float a[4], bb[4];
            for (int i = 0; i < 4; i++) a[i] = As[kk][ty + 16 * i];
            for (int j = 0; j < 4; j++) bb[j] = Bs[kk][tx + 16 * j];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) acc[i][j] = mad(a[i], bb[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < 4; i++)
    {
        int r = bm0 + ty + 16 * i;
        if (r >= M) continue;
        for (int j = 0; j < 4; j++)
        {
            int c = bn0 + tx + 16 * j;
            if (c < N) C[(long)r * N + c] = acc[i][j];
        }
    }
}
// One matrix, C = A * B (A is M x K with row stride lda, B is K x N with row stride ldb), with K split into slices: work
// group z sums K range [z*kChunk, min(K, (z+1)*kChunk)) into partial plane z of C (M x N each). A 64 x 64 tile grid
// gives a small product (512 x 49) only eight work groups on an eleven-CU device; the slices put every CU to work.
// With one slice the plane is C itself.
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_rb64_splitk(__global const float* restrict A, __global const float* restrict B, __global float* restrict C,
    const int M, const int N, const int K, const int lda, const int ldb, const int kChunk, __global const float* bias,
    const int epi)
{
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bn0 = get_group_id(0) * RB_BN, bm0 = get_group_id(1) * RB_BM;
    const int z = get_group_id(2);
    const int kBegin = z * kChunk, kEnd = min(K, kBegin + kChunk);
    C += (long)z * M * N;
    __local float As[RB_BK][RB_BM + 1];
    __local float Bs[RB_BK][RB_BN + 1];
    float acc[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    const int tid = ty * 16 + tx;
    for (int k0 = kBegin; k0 < kEnd; k0 += RB_BK)
    {
        for (int i = 0; i < 4; i++)
        {
            int e = tid + 256 * i;
            int r = e >> 4, c = e & 15;
            int gr = bm0 + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < kEnd) ? A[(long)gr * lda + gc] : 0.0f;
        }
        for (int i = 0; i < 4; i++)
        {
            int e = tid + 256 * i;
            int r = e >> 6, c = e & 63;
            int gr = k0 + r, gc = bn0 + c;
            Bs[r][c] = (gr < kEnd && gc < N) ? B[(long)gr * ldb + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < RB_BK; kk++)
        {
            float a[4], bb[4];
            for (int i = 0; i < 4; i++) a[i] = As[kk][ty + 16 * i];
            for (int j = 0; j < 4; j++) bb[j] = Bs[kk][tx + 16 * j];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) acc[i][j] = mad(a[i], bb[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < 4; i++)
    {
        int r = bm0 + ty + 16 * i;
        if (r >= M) continue;
        for (int j = 0; j < 4; j++)
        {
            int c = bn0 + tx + 16 * j;
            if (c < N) C[(long)r * N + c] = conv_epilogue(acc[i][j], bias, r, epi);
        }
    }
}

// y[i] = sum over s of partial[s][i], i < count; row i / n of y gets the epilogue (a convolution's output channel).
__kernel void gemm_splitk_reduce(__global const float* restrict partial, __global float* restrict y, const int count,
    const int splits, __global const float* bias, const int epi, const int n)
{
    int i = get_global_id(0);
    if (i >= count) return;
    float sum = 0.0f;
    for (int s = 0; s < splits; s++) sum += partial[(long)s * count + i];
    y[i] = conv_epilogue(sum, bias, i / n, epi);
}
// col[P x L] for one image (P = C*kH*kW, L = outH*outW), one work item per element: row q = (c, kh, kw), column s. The
// per-pixel im2col loops a work item over all of P, which leaves a late layer (14 x 14 output, P = 2304) 196 work items.
__kernel void im2col_wide(__global const float* restrict input, __global float* restrict col, const int channels,
    const int height, const int width, const int kernelH, const int kernelW, const int strideH, const int strideW,
    const int padH, const int padW, const int dilationH, const int dilationW, const int outH, const int outW)
{
    const int L = outH * outW;
    const long idx = get_global_id(0);
    const long total = (long)channels * kernelH * kernelW * L;
    if (idx >= total) return;
    const int s = (int)(idx % L);
    const int q = (int)(idx / L);
    const int kw = q % kernelW, kh = (q / kernelW) % kernelH, c = q / (kernelW * kernelH);
    const int oh = s / outW, ow = s - oh * outW;
    const int ih = oh * strideH - padH + kh * dilationH, iw = ow * strideW - padW + kw * dilationW;
    col[idx] = (ih >= 0 && ih < height && iw >= 0 && iw < width) ? input[((long)c * height + ih) * width + iw] : 0.0f;
}
// The large-product form of gemm_batched_rb64: a 128 x 128 block of C per 16x16 work group, each work item an 8 x 8
// sub-block strided by 16, K through LDS 8 at a time, with four-wide global loads and the next step's tiles held in
// registers while this step's products run.
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_batched_rb128v(__global const float* restrict A, __global const float* restrict B, __global float* restrict C,
    const int M, const int N, const int K)
{
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bn0 = get_group_id(0) * 128, bm0 = get_group_id(1) * 128;
    const long b = get_group_id(2);
    A += b * M * K; B += b * K * N; C += b * M * N;
    __local float As3[8][128 + 4];
    __local float Bs3[8][128 + 4];
    float acc[8][8];
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;
    const int tid = ty * 16 + tx;
    // A tile 128 x 8: row ar, columns ac..ac+3. B tile 8 x 128: row br, columns bc..bc+3.
    const int ar = tid >> 1, ac = (tid & 1) * 4;
    const int br = tid >> 5, bc = (tid & 31) * 4;
    float4 av, bv;
    #define RBV_LOAD(K0) {         const int gr = bm0 + ar, gc = (K0) + ac;         if (gr < M && gc + 3 < K) av = vload4(0, A + (long)gr * K + gc);         else { av.x = (gr < M && gc < K) ? A[(long)gr * K + gc] : 0.0f; av.y = (gr < M && gc + 1 < K) ? A[(long)gr * K + gc + 1] : 0.0f;                av.z = (gr < M && gc + 2 < K) ? A[(long)gr * K + gc + 2] : 0.0f; av.w = 0.0f; }         const int hr = (K0) + br, hc = bn0 + bc;         if (hr < K && hc + 3 < N) bv = vload4(0, B + (long)hr * N + hc);         else { bv.x = (hr < K && hc < N) ? B[(long)hr * N + hc] : 0.0f; bv.y = (hr < K && hc + 1 < N) ? B[(long)hr * N + hc + 1] : 0.0f;                bv.z = (hr < K && hc + 2 < N) ? B[(long)hr * N + hc + 2] : 0.0f; bv.w = (hr < K && hc + 3 < N) ? B[(long)hr * N + hc + 3] : 0.0f; } }
    RBV_LOAD(0)
    for (int k0 = 0; k0 < K; k0 += 8)
    {
        As3[ac][ar] = av.x; As3[ac + 1][ar] = av.y; As3[ac + 2][ar] = av.z; As3[ac + 3][ar] = av.w;
        vstore4(bv, 0, &Bs3[br][bc]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (k0 + 8 < K) RBV_LOAD(k0 + 8)
        for (int kk = 0; kk < 8; kk++)
        {
            float a[8], bb[8];
            for (int i = 0; i < 8; i++) a[i] = As3[kk][ty + 16 * i];
            for (int j = 0; j < 8; j++) bb[j] = Bs3[kk][tx + 16 * j];
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++) acc[i][j] = mad(a[i], bb[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #undef RBV_LOAD
    for (int i = 0; i < 8; i++)
    {
        int r = bm0 + ty + 16 * i;
        if (r >= M) continue;
        for (int j = 0; j < 8; j++)
        {
            int c = bn0 + tx + 16 * j;
            if (c < N) C[(long)r * N + c] = acc[i][j];
        }
    }
}
// #1432: rb64 for the Winograd per-point products of late layers (M = K = 512, N = 16-49 tiles), which stream A once and
// were latency-bound in rb64 (about 30 GB/s): K advances 32 at a time, A rows load as float4, and the next tile's loads
// are issued into registers before the current tile is multiplied, so they are in flight during the products. LDS is
// single-buffered: a double-buffered version halved occupancy and ran 2x slower.
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_batched_rb64p(__global const float* restrict A, __global const float* restrict B, __global float* restrict C,
    const int M, const int N, const int K)
{
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bn0 = get_group_id(0) * 64, bm0 = get_group_id(1) * 64;
    const long b = get_group_id(2);
    A += b * M * K; B += b * K * N; C += b * M * N;
    __local float As[32][65];
    __local float Bs[32][65];
    const int tid = ty * 16 + tx;
    // A tile 64 x 32: work item (row tid >> 2, columns (tid & 3) * 8 .. + 8). B tile 32 x 64: (row tid >> 3, columns (tid & 7) * 8 .. + 8).
    const int ar = tid >> 2, ac = (tid & 3) * 8, br = tid >> 3, bc = (tid & 7) * 8;
    const bool vecA = (K & 3) == 0;
    float pa[8], pb[8];
    float acc[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    #define RBP_LOAD(k0) { \
        int gr = bm0 + ar; \
        if (vecA && gr < M && (k0) + ac + 8 <= K) { \
            float4 v0 = vload4(0, A + (long)gr * K + (k0) + ac); float4 v1 = vload4(0, A + (long)gr * K + (k0) + ac + 4); \
            pa[0] = v0.x; pa[1] = v0.y; pa[2] = v0.z; pa[3] = v0.w; pa[4] = v1.x; pa[5] = v1.y; pa[6] = v1.z; pa[7] = v1.w; \
        } else { \
            for (int q = 0; q < 8; q++) { int gc = (k0) + ac + q; pa[q] = (gr < M && gc < K) ? A[(long)gr * K + gc] : 0.0f; } \
        } \
        int kr = (k0) + br; \
        for (int q = 0; q < 8; q++) { int gc = bn0 + bc + q; pb[q] = (kr < K && gc < N) ? B[(long)kr * N + gc] : 0.0f; } \
    }
    #define RBP_STORE() { \
        for (int q = 0; q < 8; q++) As[ac + q][ar] = pa[q]; \
        for (int q = 0; q < 8; q++) Bs[br][bc + q] = pb[q]; \
    }
    RBP_LOAD(0);
    for (int k0 = 0; k0 < K; k0 += 32)
    {
        RBP_STORE();
        barrier(CLK_LOCAL_MEM_FENCE);
        // The next tile's global loads are in flight while this one is multiplied.
        if (k0 + 32 < K) RBP_LOAD(k0 + 32);
        for (int kk = 0; kk < 32; kk++)
        {
            float a[4], bb[4];
            for (int i = 0; i < 4; i++) a[i] = As[kk][ty + 16 * i];
            for (int j = 0; j < 4; j++) bb[j] = Bs[kk][tx + 16 * j];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) acc[i][j] = mad(a[i], bb[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #undef RBP_LOAD
    #undef RBP_STORE
    for (int i = 0; i < 4; i++)
    {
        int r = bm0 + ty + 16 * i;
        if (r >= M) continue;
        for (int j = 0; j < 4; j++)
        {
            int c = bn0 + tx + 16 * j;
            if (c < N) C[(long)r * N + c] = acc[i][j];
        }
    }
}";
    }
}
