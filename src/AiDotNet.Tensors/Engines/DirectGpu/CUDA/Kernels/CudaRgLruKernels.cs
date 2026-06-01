// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernel for the fused RG-LRU scan forward (#1464). One thread per (batch, channel).

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

internal static class CudaRgLruKernels
{
    public static string GetSource()
    {
        return @"
#include <math.h>

extern ""C"" __global__ void rglru_scan_forward(
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
