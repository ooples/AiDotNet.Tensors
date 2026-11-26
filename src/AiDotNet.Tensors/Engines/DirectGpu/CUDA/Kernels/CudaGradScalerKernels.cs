// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Dynamic loss scaling on the device (PyTorch's <c>GradScaler</c>: <c>_amp_foreach_non_finite_check_and_unscale_</c>
/// then <c>_amp_update_scale_</c>), so a scaled step needs no host read: one multi-tensor launch unscales every gradient
/// and records whether any came out NaN or infinite, and a one-block launch updates the scale from that verdict and writes
/// it into the loss-gradient seed the next backward starts from. Both are plain kernels over device state, so a captured
/// step graph replays them.
/// </summary>
/// <remarks>
/// State words (<c>float*</c>/<c>unsigned int*</c> views of one buffer): [0] scale, [1] growth tracker (consecutive
/// finite steps), [2] found-non-finite flag of the current step, [3] overflowed steps, [4] steps.
/// </remarks>
internal static class CudaGradScalerKernels
{
    public const int StateWords = 8;

    public static string GetSource() => @"
extern ""C"" __global__ __launch_bounds__(256) void gs_unscale(
    const unsigned long long* __restrict__ ptrs, const int* __restrict__ sizes, const int* __restrict__ chunkTensor,
    const int* __restrict__ chunkStart, int totalChunks, int chunk, float* __restrict__ state)
{
    float inv = 1.0f / state[0];
    int bad = 0;
    for (int c = blockIdx.x; c < totalChunks; c += gridDim.x)
    {
        int t = chunkTensor[c], start = chunkStart[c];
        int end = min(start + chunk, sizes[t]);
        float* p = (float*)ptrs[t];
        for (int i = start + (int)threadIdx.x; i < end; i += blockDim.x)
        {
            float v = p[i] * inv;
            p[i] = v;
            bad |= (__float_as_uint(v) & 0x7F800000u) == 0x7F800000u;
        }
    }
    if (__syncthreads_or(bad) && threadIdx.x == 0) ((volatile unsigned int*)state)[2] = 1u;
}

extern ""C"" __global__ void gs_update(float* __restrict__ state, float* __restrict__ seed, int seedLength,
    float growthFactor, float backoffFactor, int growthInterval, float minScale, float maxScale, int dynamic)
{
    __shared__ float s;
    if (threadIdx.x == 0)
    {
        unsigned int* u = (unsigned int*)state;
        float scale = state[0];
        if (u[2] != 0u)
        {
            u[3] += 1u;
            u[1] = 0u;
            if (dynamic) scale = fmaxf(scale * backoffFactor, minScale);
        }
        else
        {
            unsigned int grown = u[1] + 1u;
            if (dynamic && grown >= (unsigned int)growthInterval)
            {
                scale = fminf(scale * growthFactor, maxScale);
                grown = 0u;
            }
            u[1] = grown;
        }
        u[2] = 0u;
        u[4] += 1u;
        state[0] = scale;
        s = scale;
    }
    __syncthreads();
    for (int i = (int)threadIdx.x; i < seedLength; i += blockDim.x) seed[i] = s;
}

extern ""C"" __global__ void gs_seed(const float* __restrict__ state, float* __restrict__ seed, int seedLength)
{
    float s = state[0];
    for (int i = (int)threadIdx.x; i < seedLength; i += blockDim.x) seed[i] = s;
}
";

    public static string[] GetKernelNames() => new[] { "gs_unscale", "gs_update", "gs_seed" };
}
