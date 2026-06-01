// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for the fused linear (LM head) + cross-entropy loss (#1464), WITHOUT materializing
// the [N, vocab] logits. One thread per row recomputes its logits on the fly (the whole point of
// the fused op) and atomic-adds its loss to a scalar accumulator; the caller divides by N.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipFusedLinearCeKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

// Index targets: loss_r = logsumexp(logits_r) - logits_r[target_r].
extern ""C"" __global__ __launch_bounds__(256) void fused_linear_ce_index(
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

// Dense soft targets: loss_r = -sum_v target[r,v]*(logits_r[v] - logsumexp).
extern ""C"" __global__ __launch_bounds__(256) void fused_linear_ce_dense(
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
