// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for the fused linear + cross-entropy loss (#1464). One work-item per row; loss
// atomic-added to a scalar accumulator via a CAS float-add (OpenCL 1.2 has no native float atomics).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class FusedLinearCeKernels
{
    public static string GetSource()
    {
        return @"
inline void atomic_add_float(__global volatile float* addr, float val) {
    union { unsigned int u; float f; } oldval, newval;
    do {
        oldval.f = *addr;
        newval.f = oldval.f + val;
    } while (atomic_cmpxchg((__global volatile unsigned int*)addr, oldval.u, newval.u) != oldval.u);
}

__kernel void fused_linear_ce_index(
    __global const float* hidden, __global const float* weight, __global const float* bias,
    __global const float* targetIds, __global volatile float* totalLoss, int N, int d, int vocab)
{
    int r = get_global_id(0);
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
        sumExp += exp(s - mx);
    }
    float loss = (mx + log(sumExp)) - logitTarget;
    atomic_add_float(totalLoss, loss);
}

__kernel void fused_linear_ce_dense(
    __global const float* hidden, __global const float* weight, __global const float* bias,
    __global const float* target, __global volatile float* totalLoss, int N, int d, int vocab)
{
    int r = get_global_id(0);
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
        sumExp += exp(s - mx);
    }
    float lse = mx + log(sumExp);
    float rowLoss = 0.0f;
    for (int vv = 0; vv < vocab; vv++) {
        float s = bias[vv];
        for (int j = 0; j < d; j++) s += hidden[hRow + j] * weight[j * vocab + vv];
        rowLoss += target[tRow + vv] * (s - lse);
    }
    atomic_add_float(totalLoss, -rowLoss);
}
";
    }

    public static string[] GetKernelNames() => new[] { "fused_linear_ce_index", "fused_linear_ce_dense" };
}
