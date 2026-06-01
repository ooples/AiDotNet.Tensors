// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernel for the fused RG-LRU scan forward (#1464). One work-item per (batch, channel).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class RgLruKernels
{
    public static string GetSource()
    {
        return @"
__kernel void rglru_scan_forward(
    __global const float* V, __global const float* R, __global const float* I, __global const float* decay,
    __global float* output,
    int batch, int seqLen, int recDim)
{
    int gid = get_global_id(0);
    int total = batch * recDim;
    if (gid >= total) return;

    int c = gid % recDim;
    int b = gid / recDim;
    float base = 1.0f / (1.0f + exp(decay[c])); // sigmoid(-decay)
    float h = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int off = (b * seqLen + t) * recDim + c;
        float a = R[off] * base;
        float om = 1.0f - a * a;
        float s = om > 0.0f ? sqrt(om) : 0.0f;
        h = a * h + s * (I[off] * V[off]);
        output[off] = h;
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { "rglru_scan_forward" };
}
