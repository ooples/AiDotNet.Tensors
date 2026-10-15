// Copyright (c) AiDotNet. All rights reserved.
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
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "conv2d_implicit_gemm"
            };
        }
    }
}
