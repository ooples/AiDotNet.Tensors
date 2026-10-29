// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CUDA convolution as an implicit GEMM for narrow layers (few output channels): no im2col buffer.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// Conv2D forward for layers with few output channels -- a small CNN's convolutions -- as a GEMM whose left operand
    /// (the im2col matrix) is gathered from the NCHW input on the fly. GEMM view: M = batch * outH * outW output pixels,
    /// N = outChannels, K = inChannels * kH * kW. The im2col route wrote the whole [K, M] matrix (1.6 MB for the parity
    /// CNN's second layer) and then ran cuBLAS on a [12544 x 144] x [144 x 32] product at ~1.3 TFLOPS.
    /// </summary>
    internal static class CudaImplicitConvKernels
    {
        public static string GetSource()
        {
            return @"
#define IC_BM 64
#define IC_BN 32
#define IC_BK 16
#define IC_THREADS 64

// output[n, oc, oh, ow] = sum over (ic, r, s) of input[n, ic, oh*sH - pH + r*dH, ow*sW - pW + s*dW] * weight[oc, ic, r, s]
// (zero outside the input). One 64-pixel x 32-channel tile per 64-thread block; each thread owns 8 pixels x 4 output
// channels (32 accumulators), reading two float4s of A and one float4 of B from shared memory per k: 32 FMAs per 48
// bytes. A K-chunk of A is gathered straight from the input, so no im2col matrix is written.
extern ""C"" __global__ __launch_bounds__(IC_THREADS) void conv2d_implicit_gemm(
    const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ output,
    int batch, int inC, int inH, int inW, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW)
{
    __shared__ __align__(16) float As[IC_BK][IC_BM + 4];
    __shared__ __align__(16) float Bs[IC_BK][IC_BN + 4];
    int tx = threadIdx.x & 7, ty = threadIdx.x >> 3;
    int m0 = blockIdx.x * IC_BM, n0 = blockIdx.y * IC_BN;
    int P = outH * outW;
    long long M = (long long)batch * P;
    int K = inC * kH * kW, kHW = kH * kW;
    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    // Each thread gathers column threadIdx.x of A (one output pixel) for every k: its pixel's decomposition is done
    // once. Per K-chunk, 16 threads decode the chunk's (channel, r, s) into a table, so a gathered element costs adds
    // and a bounds check instead of eight integer divisions (which made the kernel slower than im2col + GEMM).
    __shared__ int kChannelOffset[IC_BK], kRow[IC_BK], kCol[IC_BK];
    long long myM = m0 + threadIdx.x;
    bool pixelValid = myM < M;
    int myN = 0, ihBase = 0, iwBase = 0;
    if (pixelValid)
    {
        myN = (int)(myM / P);
        int p = (int)(myM % P);
        ihBase = (p / outW) * sH - pH;
        iwBase = (p % outW) * sW - pW;
    }
    const float* imageBase = input + (long long)myN * inC * inH * inW;
    long long plane = (long long)inH * inW;

    for (int k0 = 0; k0 < K; k0 += IC_BK)
    {
        if (threadIdx.x < IC_BK)
        {
            int k = k0 + threadIdx.x;
            if (k < K)
            {
                int ic = k / kHW, rs = k % kHW;
                kChannelOffset[threadIdx.x] = ic;
                kRow[threadIdx.x] = (rs / kW) * dH;
                kCol[threadIdx.x] = (rs % kW) * dW;
            }
            else
            {
                kChannelOffset[threadIdx.x] = -1;
            }
        }
        __syncthreads();
        // As[kk][m]: column threadIdx.x, all 16 rows. Consecutive threads take consecutive pixels, so a warp reads
        // runs of neighbouring input columns.
        #pragma unroll
        for (int kk = 0; kk < IC_BK; kk++)
        {
            float v = 0.0f;
            int ic = kChannelOffset[kk];
            if (pixelValid && ic >= 0)
            {
                int ih = ihBase + kRow[kk], iw = iwBase + kCol[kk];
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                    v = __ldg(imageBase + ic * plane + (long long)ih * inW + iw);
            }
            As[kk][threadIdx.x] = v;
        }
        // Bs[kk][oc] = weight[n0 + oc][k0 + kk] (weight stored [outC, K]).
        for (int idx = threadIdx.x; idx < IC_BK * IC_BN; idx += IC_THREADS)
        {
            int oc = idx / IC_BK, kk = idx % IC_BK;
            int k = k0 + kk;
            Bs[kk][oc] = (n0 + oc < outC && k < K) ? __ldg(weight + (long long)(n0 + oc) * K + k) : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < IC_BK; kk++)
        {
            float4 a0 = *reinterpret_cast<const float4*>(&As[kk][ty * 8]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[kk][ty * 8 + 4]);
            float4 b = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
            float av[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
            #pragma unroll
            for (int i = 0; i < 8; i++)
            {
                acc[i][0] += av[i] * b.x; acc[i][1] += av[i] * b.y;
                acc[i][2] += av[i] * b.z; acc[i][3] += av[i] * b.w;
            }
        }
        __syncthreads();
    }

    // NCHW store: for each output channel the thread's 8 pixels are consecutive positions (within one image).
    for (int i = 0; i < 8; i++)
    {
        long long m = m0 + ty * 8 + i;
        if (m >= M) break;
        int n = (int)(m / P), p = (int)(m % P);
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int oc = n0 + tx * 4 + j;
            if (oc < outC) output[((long long)n * outC + oc) * P + p] = acc[i][j];
        }
    }
}

// A convolution's weight gradient without an im2col matrix: dW[oc][k], k = (ic, r, s), is the sum over every output
// pixel row (n, p) of dY[n, oc, p] * X[n, ic, oh*sH - pH + r*dH, ow*sW - pW + s*dW]. Grid (tilesK, tilesOC, splits),
// 256 threads: block z reduces rows [z * rowsPerSplit, +rowsPerSplit) of its 32(oc) x 64(k) tile, its four 64-thread
// groups taking interleaved 16-row chunks; the groups' sums are added in group order and written to
// partials[z][tile], and the last block to reach a tile adds the partials in split order, writes dW (overwriting) and
// resets the tile's counter. Each pass decodes its rows (image, base offsets) and each block its 64 columns once, in
// shared memory, so a gathered element costs adds and a bounds check. The im2col route wrote a [K, rows] matrix and
// ran one GEMM per image plus a batch sum.
#define WG_GROUPS 4
#define WG_BK 16
#define WG_STAGE (WG_BK * (32 + 4) + WG_BK * (64 + 4))
extern ""C"" __global__ __launch_bounds__(WG_GROUPS * 64) void conv2d_weight_grad_implicit(
    const float* __restrict__ input, const float* __restrict__ gradOutput, float* __restrict__ gradWeight,
    float* __restrict__ partials, unsigned int* __restrict__ counters,
    int batch, int inC, int inH, int inW, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int rowsPerSplit)
{
    __shared__ __align__(16) float smem[WG_GROUPS * WG_STAGE];
    __shared__ int colChannel[64], colRow[64], colCol[64];
    __shared__ long long rowDy[WG_GROUPS][WG_BK], rowX[WG_GROUPS][WG_BK];
    __shared__ int rowIh[WG_GROUPS][WG_BK], rowIw[WG_GROUPS][WG_BK];
    __shared__ unsigned int isLast;
    int g = threadIdx.x / 64, lt = threadIdx.x % 64;
    int tx = lt & 7, ty = lt >> 3;
    float (*As)[32 + 4] = reinterpret_cast<float (*)[32 + 4]>(smem + g * WG_STAGE);
    float (*Bs)[64 + 4] = reinterpret_cast<float (*)[64 + 4]>(smem + g * WG_STAGE + WG_BK * (32 + 4));
    int oc0 = blockIdx.y * 32, kc0 = blockIdx.x * 64;
    int K = inC * kH * kW, kHW = kH * kW, P = outH * outW;
    long long R = (long long)batch * P;
    long long plane = (long long)inH * inW;
    if (threadIdx.x < 64)
    {
        int k = kc0 + threadIdx.x;
        if (k < K)
        {
            int rs = k % kHW;
            colChannel[threadIdx.x] = k / kHW;
            colRow[threadIdx.x] = (rs / kW) * dH;
            colCol[threadIdx.x] = (rs % kW) * dW;
        }
        else colChannel[threadIdx.x] = -1;
    }
    float acc[4][8];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;
    long long rBegin = (long long)blockIdx.z * rowsPerSplit;
    long long rEnd = min(R, rBegin + rowsPerSplit);
    for (long long base = rBegin; base < rEnd; base += WG_GROUPS * WG_BK)
    {
        long long r0 = base + g * WG_BK;
        if (lt < WG_BK)
        {
            long long r = r0 + lt;
            if (r < rEnd)
            {
                int n = (int)(r / P), p = (int)(r % P);
                rowDy[g][lt] = (long long)n * outC * P + p;
                rowX[g][lt] = (long long)n * inC * plane;
                rowIh[g][lt] = (p / outW) * sH - pH;
                rowIw[g][lt] = (p % outW) * sW - pW;
            }
            else rowDy[g][lt] = -1;
        }
        __syncthreads();
        // As[kk][m] = dY[n, oc0 + m, p]: consecutive threads take consecutive rows (pixels), contiguous in dY.
        for (int idx = lt; idx < WG_BK * 32; idx += 64)
        {
            int kk = idx % WG_BK, m = idx / WG_BK;
            long long d = rowDy[g][kk];
            As[kk][m] = (d >= 0 && oc0 + m < outC) ? __ldg(gradOutput + d + (long long)(oc0 + m) * P) : 0.0f;
        }
        // Bs[kk][col] = X[n, ic, ih, iw] for the tile's column col.
        for (int idx = lt; idx < WG_BK * 64; idx += 64)
        {
            int kk = idx % WG_BK, col = idx / WG_BK;
            float v = 0.0f;
            int ic = colChannel[col];
            if (rowDy[g][kk] >= 0 && ic >= 0)
            {
                int ih = rowIh[g][kk] + colRow[col], iw = rowIw[g][kk] + colCol[col];
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                    v = __ldg(input + rowX[g][kk] + ic * plane + (long long)ih * inW + iw);
            }
            Bs[kk][col] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < WG_BK; kk++)
        {
            float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[kk][32 + tx * 4]);
            float av[4] = { a.x, a.y, a.z, a.w };
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                acc[i][0] += av[i] * b0.x; acc[i][1] += av[i] * b0.y; acc[i][2] += av[i] * b0.z; acc[i][3] += av[i] * b0.w;
                acc[i][4] += av[i] * b1.x; acc[i][5] += av[i] * b1.y; acc[i][6] += av[i] * b1.z; acc[i][7] += av[i] * b1.w;
            }
        }
        __syncthreads();
    }

    // Groups 1..3 hand their sums to group 0, which adds them in group order and writes the block's partial.
    if (g > 0)
    {
        float* handoff = smem + (g - 1) * (64 * 32) + lt * 32;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++) handoff[i * 8 + j] = acc[i][j];
    }
    __syncthreads();
    int tiles = gridDim.x * gridDim.y;
    int tile = blockIdx.y * gridDim.x + blockIdx.x;
    if (g == 0)
    {
        for (int h = 0; h < WG_GROUPS - 1; h++)
        {
            const float* other = smem + h * (64 * 32) + lt * 32;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 8; j++) acc[i][j] += other[i * 8 + j];
        }
        float4* mine = reinterpret_cast<float4*>(partials + ((long long)blockIdx.z * tiles + tile) * 2048) + lt * 8;
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
    // The tile's 512 float4s, two per thread: float4 q belongs to group-0 thread q / 8, value pair q % 8.
    for (int q = threadIdx.x; q < 512; q += WG_GROUPS * 64)
    {
        float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int z = 0; z < (int)gridDim.z; z++)
        {
            float4 v = __ldcg(reinterpret_cast<const float4*>(partials + ((long long)z * tiles + tile) * 2048) + q);
            s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
        }
        int owner = q >> 3, pair = q & 7;
        int oc = oc0 + (owner >> 3) * 4 + (pair >> 1);
        int col = kc0 + (pair & 1) * 32 + (owner & 7) * 4;
        if (oc >= outC) continue;
        float vals[4] = { s.x, s.y, s.z, s.w };
        for (int e = 0; e < 4; e++)
            if (col + e < K) gradWeight[(long long)oc * K + col + e] = vals[e];
    }
    if (threadIdx.x == 0) counters[tile] = 0u;
}";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "conv2d_implicit_gemm",
                "conv2d_weight_grad_implicit"
            };
        }
    }
}
