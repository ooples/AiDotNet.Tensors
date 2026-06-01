// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernel for the fused RWKV-4 time-mixing WKV scan forward (#1464). One thread per (batch, channel).

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

internal static class CudaRwkv4Kernels
{
    public static string GetSource()
    {
        return @"
#include <math.h>

#define RWKV4_NEG_INF (-1.0e38f)

extern ""C"" __global__ void rwkv4_wkv_forward(
    const float* R, const float* K, const float* V,
    const float* timeDecay, const float* timeFirst,
    float* output,
    int batch, int seqLen, int modelDim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * modelDim;
    if (gid >= total) return;

    int c = gid % modelDim;
    int b = gid / modelDim;
    float w = -expf(timeDecay[c]);
    float u = timeFirst[c];

    float aa = 0.0f, bb = 0.0f, pp = RWKV4_NEG_INF;
    for (int t = 0; t < seqLen; t++) {
        int off = (b * seqLen + t) * modelDim + c;
        float k = K[off];
        float v = V[off];

        float ww = u + k;
        float q = fmaxf(pp, ww);
        float e1 = expf(pp - q);
        float e2 = expf(ww - q);
        float wkv = (e1 * aa + e2 * v) / (e1 * bb + e2);
        output[off] = (1.0f / (1.0f + expf(-R[off]))) * wkv;

        float ww2 = pp + w;
        float q2 = fmaxf(ww2, k);
        float e1b = expf(ww2 - q2);
        float e2b = expf(k - q2);
        aa = e1b * aa + e2b * v;
        bb = e1b * bb + e2b;
        pp = q2;
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { "rwkv4_wkv_forward" };
}
