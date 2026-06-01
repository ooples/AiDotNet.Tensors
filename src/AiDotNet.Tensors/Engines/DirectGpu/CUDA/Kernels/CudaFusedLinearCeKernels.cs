// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for the fused linear + cross-entropy loss (#1464). One thread per row; loss
// atomic-added to a scalar accumulator (caller divides by N). Logits are never materialized.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

internal static class CudaFusedLinearCeKernels
{
    public static string GetSource()
    {
        return @"
#include <math.h>

extern ""C"" __global__ void fused_linear_ce_index(
    const float* hidden, const float* weight, const float* bias, const float* targetIds,
    float* totalLoss, int N, int d, int vocab)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    int hRow = r * d;
    int id = (int)(targetIds[r] + 0.5f);

    float mx = -1.0e38f, logitTarget = 0.0f;
    for (int vv = 0; vv < vocab; vv++) {
        float s = bias[vv];
        for (int j = 0; j < d; j++) s += hidden[hRow + j] * weight[j * vocab + vv];
        if (s > mx) mx = s;
        if (vv == id) logitTarget = s;
    }
    float sumExp = 0.0f;
    for (int vv = 0; vv < vocab; vv++) {
        float s = bias[vv];
        for (int j = 0; j < d; j++) s += hidden[hRow + j] * weight[j * vocab + vv];
        sumExp += expf(s - mx);
    }
    float loss = (mx + logf(sumExp)) - logitTarget;
    atomicAdd(totalLoss, loss);
}

extern ""C"" __global__ void fused_linear_ce_dense(
    const float* hidden, const float* weight, const float* bias, const float* target,
    float* totalLoss, int N, int d, int vocab)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    int hRow = r * d;
    int tRow = r * vocab;

    float mx = -1.0e38f;
    for (int vv = 0; vv < vocab; vv++) {
        float s = bias[vv];
        for (int j = 0; j < d; j++) s += hidden[hRow + j] * weight[j * vocab + vv];
        if (s > mx) mx = s;
    }
    float sumExp = 0.0f;
    for (int vv = 0; vv < vocab; vv++) {
        float s = bias[vv];
        for (int j = 0; j < d; j++) s += hidden[hRow + j] * weight[j * vocab + vv];
        sumExp += expf(s - mx);
    }
    float lse = mx + logf(sumExp);
    float rowLoss = 0.0f;
    for (int vv = 0; vv < vocab; vv++) {
        float s = bias[vv];
        for (int j = 0; j < d; j++) s += hidden[hRow + j] * weight[j * vocab + vv];
        rowLoss += target[tRow + vv] * (s - lse);
    }
    atomicAdd(totalLoss, -rowLoss);
}
";
    }

    public static string[] GetKernelNames() => new[] { "fused_linear_ce_index", "fused_linear_ce_dense" };
}
