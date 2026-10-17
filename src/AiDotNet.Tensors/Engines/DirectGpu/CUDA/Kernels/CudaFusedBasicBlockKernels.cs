namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// #1353 prototype: one persistent launch runs a ResNet basic block's training forward,
/// y = relu(bn2(conv2(relu(bn1(conv1(x))))) + x), with batch statistics (biased variance, as a training-mode
/// BatchNorm normalizes), 3x3 / stride 1 / pad 1 convolutions with as many output as input channels, and the identity
/// shortcut. FP32, NCHW.
/// <para>
/// Blocks stay resident (cooperative launch) and meet at a software grid barrier between the phases:
/// conv1 -> barrier -> BN1 statistics -> barrier -> conv2, with BN1 + ReLU applied while its input tile is gathered
/// -> barrier -> BN2 statistics -> barrier -> BN2 + residual + ReLU. The convolutions are implicit GEMMs over
/// 64 output channels x 64 pixels per tile (K = C * 9 in chunks of 16; 256 threads, 4 x 4 outputs each). Each channel's
/// statistics are reduced by one block in a fixed order (two passes: mean, then the squared deviations), so the result
/// does not depend on how many blocks run or how they are scheduled.
/// </para>
/// <para>
/// The barrier is a counter plus a generation word in global memory (the caller zeroes both before each launch):
/// cooperative_groups' grid.sync() needs cooperative_groups.h, which NVRTC cannot find without a full toolkit.
/// Cooperative launch guarantees every block is co-resident, which is what makes the spin wait safe.
/// </para>
/// </summary>
internal static class CudaFusedBasicBlockKernels
{
    internal const string ForwardKernel = "fbb_forward";

    /// <summary>Output channels and pixels per conv tile; the admitted shapes are multiples of both.</summary>
    internal const int Tile = 64;

    internal const int BlockThreads = 256;

    internal static string[] GetKernelNames() => new[] { ForwardKernel };

    internal static string GetSource() => @"
struct FbbParams {
    const float* x; const float* w1; const float* g1; const float* b1;
    const float* w2; const float* g2; const float* b2;
    float* c1; float* c2; float* stats; float* y; unsigned int* bar;
    int N; int C; int H; int W; float eps;
};

#define FBB_TILE 64
#define FBB_BK 16

__device__ __forceinline__ void fbb_grid_barrier(unsigned int* bar)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned int* gen = bar + 1;
        unsigned int g = *gen;
        __threadfence();
        unsigned int arrived = atomicAdd(bar, 1u);
        if (arrived == gridDim.x - 1) {
            bar[0] = 0u;
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            while (*gen == g) { }
        }
        __threadfence();
    }
    __syncthreads();
}

// One 64 x 64 tile of a 3x3 / stride 1 / pad 1 convolution as an implicit GEMM: rows are output channels, columns
// are pixels (n, h, w) flattened. pre != 0 applies relu(bn(v)) to every in-bounds input value as it is gathered;
// padding stays 0, as it is after the activation.
__device__ void fbb_conv_tile(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ out,
    int N, int C, int H, int W, int tile, int pre, const float* __restrict__ mean, const float* __restrict__ inv,
    const float* __restrict__ gamma, const float* __restrict__ beta, float* smA, float* smB)
{
    int HW = H * W, P = N * HW, K = C * 9;
    int tilesP = P / FBB_TILE;
    int co0 = (tile / tilesP) * FBB_TILE, p0 = (tile % tilesP) * FBB_TILE;
    int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += FBB_BK) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int e = tid + r * 256;              // 0 .. 1023
            int kk = e >> 6, m = e & 63;
            smA[kk * FBB_TILE + m] = w[(co0 + m) * K + k0 + kk];
            int k = k0 + kk;
            int ci = k / 9, rem = k - ci * 9, kh = rem / 3, kw = rem - kh * 3;
            int p = p0 + m;
            int n = p / HW, hw = p - n * HW, h = hw / W, x0 = hw - h * W;
            int ih = h + kh - 1, iw = x0 + kw - 1;
            float v = 0.0f;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                v = in[((n * C + ci) * H + ih) * W + iw];
                if (pre) v = fmaxf(0.0f, (v - mean[ci]) * inv[ci] * gamma[ci] + beta[ci]);
            }
            smB[kk * FBB_TILE + m] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < FBB_BK; kk++) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) a[i] = smA[kk * FBB_TILE + ty * 4 + i];
            #pragma unroll
            for (int j = 0; j < 4; j++) b[j] = smB[kk * FBB_TILE + tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int p = p0 + tx * 4 + j;
        int n = p / HW, hw = p - n * HW;
        #pragma unroll
        for (int i = 0; i < 4; i++) out[(n * C + co0 + ty * 4 + i) * HW + hw] = acc[i][j];
    }
}

// Per channel: mean, then 1 / sqrt(biased variance + eps), over all N * H * W values; one block per channel, a fixed
// per-thread stride and a fixed tree, so the result does not depend on the grid.
__device__ void fbb_channel_stats(const float* __restrict__ v, float* mean, float* inv, int N, int C, int HW, float eps,
    float* red)
{
    int P = N * HW;
    for (int c = blockIdx.x; c < C; c += gridDim.x) {
        float s = 0.0f;
        for (int i = threadIdx.x; i < P; i += 256) { int n = i / HW; s += v[(n * C + c) * HW + (i - n * HW)]; }
        red[threadIdx.x] = s;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) { if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st]; __syncthreads(); }
        float m = red[0] / (float)P;
        __syncthreads();
        float q = 0.0f;
        for (int i = threadIdx.x; i < P; i += 256) { int n = i / HW; float d = v[(n * C + c) * HW + (i - n * HW)] - m; q += d * d; }
        red[threadIdx.x] = q;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) { if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st]; __syncthreads(); }
        if (threadIdx.x == 0) { mean[c] = m; inv[c] = 1.0f / sqrtf(red[0] / (float)P + eps); }
        __syncthreads();
    }
}

extern ""C"" __global__ __launch_bounds__(256) void fbb_forward(FbbParams p)
{
    __shared__ float smA[FBB_BK * FBB_TILE];
    __shared__ float smB[FBB_BK * FBB_TILE];
    __shared__ float red[256];
    int HW = p.H * p.W, total = p.N * p.C * HW;
    int tiles = (p.C / FBB_TILE) * (p.N * HW / FBB_TILE);
    float* mean1 = p.stats; float* inv1 = p.stats + p.C; float* mean2 = p.stats + 2 * p.C; float* inv2 = p.stats + 3 * p.C;

    for (int t = blockIdx.x; t < tiles; t += gridDim.x)
        fbb_conv_tile(p.x, p.w1, p.c1, p.N, p.C, p.H, p.W, t, 0, 0, 0, 0, 0, smA, smB);
    fbb_grid_barrier(p.bar);
    fbb_channel_stats(p.c1, mean1, inv1, p.N, p.C, HW, p.eps, red);
    fbb_grid_barrier(p.bar);
    for (int t = blockIdx.x; t < tiles; t += gridDim.x)
        fbb_conv_tile(p.c1, p.w2, p.c2, p.N, p.C, p.H, p.W, t, 1, mean1, inv1, p.g1, p.b1, smA, smB);
    fbb_grid_barrier(p.bar);
    fbb_channel_stats(p.c2, mean2, inv2, p.N, p.C, HW, p.eps, red);
    fbb_grid_barrier(p.bar);
    for (int i = blockIdx.x * 256 + threadIdx.x; i < total; i += gridDim.x * 256) {
        int c = (i / HW) % p.C;
        p.y[i] = fmaxf(0.0f, (p.c2[i] - mean2[c]) * inv2[c] * p.g2[c] + p.b2[c] + p.x[i]);
    }
}
";
}
