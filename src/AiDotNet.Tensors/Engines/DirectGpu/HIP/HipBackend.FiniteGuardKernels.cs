// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// The global-norm clip's sum of squares and clip scale, which also serve the device finiteness guard (#1519). Their own
// module, built without -ffast-math: finite-math-only lets the compiler assume no float is NaN or infinite, and a NaN
// gradient then left the guard finite.
public sealed partial class HipBackend
{
    private IntPtr _finiteGuardModule;

    private const string FiniteGuardKernelSource = @"
// The HIP twin of CUDA's multi_tensor_sum_squares. ptrs holds one device address per tensor; chunk c covers tensor
// chunkTensor[c] from chunkStart[c], one 256-thread block wide. The sum accumulates in double, so every finite gradient
// gives a finite sum; a non-finite element adds NaN, so the guard reads NaN exactly then (an infinity alone would sum to
// +inf). *out must be zeroed first. A shared-memory tree rather than warp shuffles: the wavefront is 32 or 64 wide.
extern ""C"" __global__ __launch_bounds__(256) void multi_tensor_sum_squares(
    const unsigned long long* ptrs, const int* sizes, const int* chunkTensor, const int* chunkStart,
    int totalChunks, double* out)
{
    __shared__ double partial[256];
    double v = 0.0;
    for (int c = blockIdx.x; c < totalChunks; c += gridDim.x)
    {
        int t = chunkTensor[c];
        int i = chunkStart[c] + threadIdx.x;
        if (i < sizes[t])
        {
            float f = ((const float*)ptrs[t])[i];
            if ((__float_as_uint(f) & 0x7f800000u) == 0x7f800000u) v += __longlong_as_double(0x7ff8000000000000LL);
            else
            {
                double x = (double)f;
                v += x * x;
            }
        }
    }
    partial[threadIdx.x] = v;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1)
    {
        if ((int)threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, partial[0]);
}

// min(1, maxNorm / (sqrt(sumSq) + 1e-6)); a non-finite norm leaves the gradients unscaled (1).
extern ""C"" __global__ void clip_scale_from_sum_squares(const double* sumSquares, float maxNorm, float* scale)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double norm = sqrt(*sumSquares);
    double c = (__double_as_longlong(norm) & 0x7ff0000000000000LL) != 0x7ff0000000000000LL ? (double)maxNorm / (norm + 1e-6) : 1.0;
    *scale = (float)(c < 1.0 ? c : 1.0);
}
";

    private void CompileFiniteGuardKernels()
    {
        var failure = CompileKernelModule(FiniteGuardKernelSource, "finite_guard", ref _finiteGuardModule,
            new[] { "multi_tensor_sum_squares", "clip_scale_from_sum_squares" }, useFastMath: false);
        if (failure is not null)
            System.Diagnostics.Trace.TraceWarning($"HIP global-norm clip kernels unavailable: {failure.Message}");
    }
}
