// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernel for the fused RWKV-4 time-mixing WKV scan forward (#1464). One work-item per (batch, channel).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class Rwkv4Kernels
{
    public static string GetSource()
    {
        return @"
#define RWKV4_NEG_INF (-INFINITY)

__kernel void rwkv4_wkv_forward(
    __global const float* R, __global const float* K, __global const float* V,
    __global const float* timeDecay, __global const float* timeFirst,
    __global float* output,
    int batch, int seqLen, int modelDim)
{
    int gid = get_global_id(0);
    int total = batch * modelDim;
    if (gid >= total) return;

    int c = gid % modelDim;
    int b = gid / modelDim;
    float w = -exp(timeDecay[c]);
    float u = timeFirst[c];

    float aa = 0.0f, bb = 0.0f, pp = RWKV4_NEG_INF;
    for (int t = 0; t < seqLen; t++) {
        int off = (b * seqLen + t) * modelDim + c;
        float k = K[off];
        float v = V[off];

        float ww = u + k;
        float q = fmax(pp, ww);
        float e1 = exp(pp - q);
        float e2 = exp(ww - q);
        float wkv = (e1 * aa + e2 * v) / (e1 * bb + e2);
        output[off] = (1.0f / (1.0f + exp(-R[off]))) * wkv;

        float ww2 = pp + w;
        float q2 = fmax(ww2, k);
        float e1b = exp(ww2 - q2);
        float e2b = exp(k - q2);
        aa = e1b * aa + e2b * v;
        bb = e1b * bb + e2b;
        pp = q2;
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { "rwkv4_wkv_forward" };
}
