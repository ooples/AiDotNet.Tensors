// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Fused training-forward kernels for a ResNet basic block (conv -> BN -> ReLU -> conv -> BN -> add -> ReLU), built on the
/// #1325 implicit-GEMM conv engine (<see cref="CudaConvolutionKernels.ConvEngineCore"/>) through its prologue/epilogue
/// hooks. The epilogue reduces each output channel's batch statistics while the tile is stored, so no separate statistics
/// pass reads the convolution output; the prologue applies the previous BatchNorm and ReLU as each input element is
/// loaded, so that normalized activation is never written to memory. Spike: forward only.
/// </summary>
internal static class CudaConvFusedKernels
{
    public static string GetSource() => CudaConvolutionKernels.ConvEngineCore + @"

// Per-channel sum and sum of squares of the stored outputs, reduced in shared memory per block (one slot per output
// channel of the block's BM rows) and flushed with one double atomic per channel per block.
struct FbStatsEpi
{
    float* s; float* q; int m0;
    __device__ __forceinline__ float operator()(int, int co, int, float y) const
    {
        atomicAdd(&s[co - m0], y);
        atomicAdd(&q[co - m0], y * y);
        return y;
    }
};

// The previous BatchNorm and ReLU applied to each in-bounds input element as it is loaded: relu(x * scale + shift) with
// scale = gamma * invStd and shift = beta - mean * scale per input channel. Padding stays 0 (the engine never calls the
// prologue for it), which is the zero-padded activation the unfused network convolves.
struct FbBnReluPro
{
    const float* scale; const float* shift;
    __device__ __forceinline__ float operator()(int, int ci, int, float x) const
    {
        return fmaxf(fmaf(x, __ldg(&scale[ci]), __ldg(&shift[ci])), 0.0f);
    }
};

template <int BM, bool TAPU, typename Pro>
__device__ __forceinline__ void fb_conv_stats(
    const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW,
    double* __restrict__ sum, double* __restrict__ sumSq, Pro pro)
{
    __shared__ float s[BM], q[BM];
    for (int i = threadIdx.x; i < BM; i += blockDim.x) { s[i] = 0.0f; q[i] = 0.0f; }
    __syncthreads();
    const int m0 = blockIdx.y * BM;
    FbStatsEpi epi{ s, q, m0 };
    // One K split: the epilogue sees final sums, never split-K partials.
    conv2d_fprop_ig_body<BM, BM, TAPU, Pro, FbStatsEpi>(input, kernel, out, batch, inC, H, W, outC, outH, outW,
        kH, kW, sH, sW, pH, pW, dH, dW, inC * kH * kW, pro, epi);
    __syncthreads();
    for (int i = threadIdx.x; i < BM; i += blockDim.x)
    {
        int c = m0 + i;
        if (c < outC) { atomicAdd(&sum[c], (double)s[i]); atomicAdd(&sumSq[c], (double)q[i]); }
    }
}

#define FB_ARGS const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ out, \
    int batch, int inC, int H, int W, int outC, int outH, int outW, \
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, \
    double* __restrict__ sum, double* __restrict__ sumSq
#define FB_PASS input, kernel, out, batch, inC, H, W, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW, sum, sumSq

// Plain input (the block's first convolution reads the previous block's output as is).
extern ""C"" __global__ __launch_bounds__(256) void fb_conv_stats128_tap(FB_ARGS) { fb_conv_stats<128, true >(FB_PASS, CeIdentityPro()); }
extern ""C"" __global__ __launch_bounds__(256) void fb_conv_stats128_gen(FB_ARGS) { fb_conv_stats<128, false>(FB_PASS, CeIdentityPro()); }
extern ""C"" __global__ __launch_bounds__(256) void fb_conv_stats64_tap(FB_ARGS)  { fb_conv_stats<64,  true >(FB_PASS, CeIdentityPro()); }
extern ""C"" __global__ __launch_bounds__(256) void fb_conv_stats64_gen(FB_ARGS)  { fb_conv_stats<64,  false>(FB_PASS, CeIdentityPro()); }

// Input through the previous BatchNorm + ReLU (the block's second convolution).
#define FB_PRO_ARGS FB_ARGS, const float* __restrict__ proScale, const float* __restrict__ proShift
extern ""C"" __global__ __launch_bounds__(256) void fb_bnrelu_conv_stats128_tap(FB_PRO_ARGS) { fb_conv_stats<128, true >(FB_PASS, FbBnReluPro{ proScale, proShift }); }
extern ""C"" __global__ __launch_bounds__(256) void fb_bnrelu_conv_stats128_gen(FB_PRO_ARGS) { fb_conv_stats<128, false>(FB_PASS, FbBnReluPro{ proScale, proShift }); }
extern ""C"" __global__ __launch_bounds__(256) void fb_bnrelu_conv_stats64_tap(FB_PRO_ARGS)  { fb_conv_stats<64,  true >(FB_PASS, FbBnReluPro{ proScale, proShift }); }
extern ""C"" __global__ __launch_bounds__(256) void fb_bnrelu_conv_stats64_gen(FB_PRO_ARGS)  { fb_conv_stats<64,  false>(FB_PASS, FbBnReluPro{ proScale, proShift }); }

// Small grids (the deep stages: 8-16 output tiles for 68 SMs) need split-K for occupancy, and a split launch stores
// partials, which a statistics epilogue cannot see. There the convolution keeps its BN+ReLU prologue and writes the
// split partials; fb_splitk_reduce_stats sums them in split order (as conv2d_wgrad_splitk_reduce does) and reduces the
// batch statistics of the final values in the same pass.
template <int BM, bool TAPU, typename Pro>
__device__ __forceinline__ void fb_conv_split(
    const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ out,
    int batch, int inC, int H, int W, int outC, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit, Pro pro)
{
    conv2d_fprop_ig_body<BM, BM, TAPU, Pro, CeIdentityEpi>(input, kernel, out, batch, inC, H, W, outC, outH, outW,
        kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit, pro, CeIdentityEpi());
}
#define FB_SPLIT_ARGS const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ out, \
    int batch, int inC, int H, int W, int outC, int outH, int outW, \
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int kPerSplit, \
    const float* __restrict__ proScale, const float* __restrict__ proShift
#define FB_SPLIT_PASS input, kernel, out, batch, inC, H, W, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW, kPerSplit, \
    FbBnReluPro{ proScale, proShift }
extern ""C"" __global__ __launch_bounds__(256) void fb_bnrelu_conv_split128_tap(FB_SPLIT_ARGS) { fb_conv_split<128, true >(FB_SPLIT_PASS); }
extern ""C"" __global__ __launch_bounds__(256) void fb_bnrelu_conv_split128_gen(FB_SPLIT_ARGS) { fb_conv_split<128, false>(FB_SPLIT_PASS); }
extern ""C"" __global__ __launch_bounds__(256) void fb_bnrelu_conv_split64_tap(FB_SPLIT_ARGS)  { fb_conv_split<64,  true >(FB_SPLIT_PASS); }
extern ""C"" __global__ __launch_bounds__(256) void fb_bnrelu_conv_split64_gen(FB_SPLIT_ARGS)  { fb_conv_split<64,  false>(FB_SPLIT_PASS); }

// y = sum of the split partials (in split order), with each channel's sum and sum of squares reduced in shared memory
// (2 * channels floats of dynamic shared memory) and flushed with one double atomic per channel the block touched.
extern ""C"" __global__ void fb_splitk_reduce_stats(
    const float* __restrict__ partials, float* __restrict__ y, int count, int splits, int channels, int spatial,
    double* __restrict__ sum, double* __restrict__ sumSq)
{
    extern __shared__ float fbStats[];
    for (int i = threadIdx.x; i < 2 * channels; i += blockDim.x) fbStats[i] = 0.0f;
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        float v = 0.0f;
        for (int z = 0; z < splits; z++) v += partials[(long long)z * count + idx];
        y[idx] = v;
        int c = (idx / spatial) % channels;
        atomicAdd(&fbStats[c], v);
        atomicAdd(&fbStats[channels + c], v * v);
    }
    __syncthreads();
    int first = (blockIdx.x * blockDim.x / spatial) % channels;
    int last = (min(count - 1, (blockIdx.x + 1) * blockDim.x - 1) / spatial) % channels;
    bool all = blockDim.x / spatial + 2 >= channels;   // the block's range can wrap past every channel
    for (int i = threadIdx.x; i < channels; i += blockDim.x)
    {
        bool touched = all || (first <= last ? (i >= first && i <= last) : (i >= first || i <= last));
        if (touched) { atomicAdd(&sum[i], (double)fbStats[i]); atomicAdd(&sumSq[i], (double)fbStats[channels + i]); }
    }
}

// Batch statistics -> mean, 1/sqrt(var + eps) (biased variance, what BatchNorm normalizes with), and the folded
// scale/shift the next prologue or the final apply uses. Re-zeroes the sums for the next step.
extern ""C"" __global__ void fb_bn_finalize(
    double* __restrict__ sum, double* __restrict__ sumSq, const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ mean, float* __restrict__ invStd, float* __restrict__ scale, float* __restrict__ shift,
    int channels, double count, float epsilon)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    double m = sum[c] / count;
    double var = sumSq[c] / count - m * m;
    if (var < 0.0) var = 0.0;
    float inv = (float)(1.0 / sqrt(var + (double)epsilon));
    float sc = gamma[c] * inv;
    mean[c] = (float)m;
    invStd[c] = inv;
    scale[c] = sc;
    shift[c] = beta[c] - (float)m * sc;
    sum[c] = 0.0;
    sumSq[c] = 0.0;
}

// Block output: relu(y * scale + shift + residual), NCHW, per-channel scale/shift.
extern ""C"" __global__ void fb_bn_add_relu(
    const float* __restrict__ y, const float* __restrict__ scale, const float* __restrict__ shift,
    const float* __restrict__ residual, float* __restrict__ out, int total, int channels, int spatial)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (i / spatial) % channels;
    out[i] = fmaxf(fmaf(y[i], scale[c], shift[c]) + residual[i], 0.0f);
}
";

    public static string[] GetKernelNames() => new[]
    {
        "fb_conv_stats128_tap", "fb_conv_stats128_gen", "fb_conv_stats64_tap", "fb_conv_stats64_gen",
        "fb_bnrelu_conv_stats128_tap", "fb_bnrelu_conv_stats128_gen", "fb_bnrelu_conv_stats64_tap", "fb_bnrelu_conv_stats64_gen",
        "fb_bnrelu_conv_split128_tap", "fb_bnrelu_conv_split128_gen", "fb_bnrelu_conv_split64_tap", "fb_bnrelu_conv_split64_gen",
        "fb_splitk_reduce_stats", "fb_bn_finalize", "fb_bn_add_relu",
    };
}
