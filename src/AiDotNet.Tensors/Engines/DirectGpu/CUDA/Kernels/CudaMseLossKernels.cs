// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>The fused mean-squared-error loss and its gradient (<see cref="IFusedMseLossBackend"/>).</summary>
internal static class CudaMseLossKernels
{
    public static string GetSource() => @"
// One block of 1024 threads: each thread sums (p - y)^2 over a fixed stride of the elements, then the block reduces
// through warp shuffles and one shared-memory step (a fixed order: deterministic).
extern ""C"" __global__ __launch_bounds__(1024) void mse_loss_mean(
    const float* __restrict__ p, const float* __restrict__ y, float* __restrict__ loss, int n)
{
    __shared__ float warpSums[32];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float d = p[i] - y[i];
        sum = fmaf(d, d, sum);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    if ((threadIdx.x & 31) == 0) warpSums[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < (blockDim.x >> 5) ? warpSums[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (threadIdx.x == 0) loss[0] = sum / (float)n;
    }
}

// dp = 2 (p - y) g / n with g the loss's gradient (gradLoss[0]); dy = -dp when requested.
extern ""C"" __global__ __launch_bounds__(256) void mse_loss_backward(
    const float* __restrict__ p, const float* __restrict__ y, const float* __restrict__ gradLoss,
    float* __restrict__ dp, float* __restrict__ dy, int n)
{
    float scale = 2.0f * gradLoss[0] / (float)n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float g = scale * (p[i] - y[i]);
        dp[i] = g;
        if (dy != 0) dy[i] = -g;
    }
}
";

    public static string[] GetKernelNames() => new[] { "mse_loss_mean", "mse_loss_backward" };
}
