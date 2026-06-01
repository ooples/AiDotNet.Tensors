// Copyright (c) AiDotNet. All rights reserved.
// HIP kernel for the fused RG-LRU (real-gated linear recurrent unit) scan forward (#1464).
// Mirrors CpuEngine.RgLruScan / RecurrenceCpuKernels.RgLruForward. Per-channel recurrence, so
// one thread owns (batch b, channel c) and carries h[c] across time. Backward via CpuEngine tape.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipRgLruKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

extern ""C"" __global__ __launch_bounds__(1024) void rglru_scan_forward(
    const float* V, const float* R, const float* I, const float* decay,
    float* output,
    int batch, int seqLen, int recDim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * recDim;
    if (gid >= total) return;

    int c = gid % recDim;
    int b = gid / recDim;
    float base = 1.0f / (1.0f + expf(decay[c])); // sigmoid(-decay)
    float h = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int off = (b * seqLen + t) * recDim + c;
        float a = R[off] * base;
        float om = 1.0f - a * a;
        float s = om > 0.0f ? sqrtf(om) : 0.0f;
        h = a * h + s * (I[off] * V[off]);
        output[off] = h;
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { "rglru_scan_forward" };
}
