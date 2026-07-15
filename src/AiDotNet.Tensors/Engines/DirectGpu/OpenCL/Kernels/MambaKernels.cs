// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernel for the fused Mamba selective scan forward (#1464). One work-item per (batch, channel).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class MambaKernels
{
    public const int MaxStateDim = 256;

    public static string GetSource()
    {
        // The scan state `h` lives in a __global scratch buffer (one stateDim-slice per work-item),
        // NOT a per-work-item private float[MAX_STATEDIM]. The old kernel declared two 2KB private
        // arrays (negA[256] + h[256]); on register-constrained GPUs (e.g. AMD RX 5500) that spilled and
        // triggered an async command-queue fault that crashed the host on a LATER op (op-parity #775
        // GpuUnsafe). Moving h to global scratch and recomputing negA inline bounds private memory to
        // O(1) and, as a bonus, removes the compile-time state-dim ceiling (works for any stateDim).
        return @"
__kernel void mamba_selective_scan_forward(
    __global const float* X, __global const float* delta, __global const float* aLog,
    __global const float* B, __global const float* C, __global const float* D,
    __global float* output, __global float* hState,
    int batch, int seqLen, int innerDim, int stateDim)
{
    int gid = get_global_id(0);
    int total = batch * innerDim;
    if (gid >= total) return;

    int di = gid % innerDim;
    int b = gid / innerDim;
    int hrow = di * stateDim;
    int hbase = gid * stateDim;   // this work-item's slice of the global state scratch

    for (int ni = 0; ni < stateDim; ni++) hState[hbase + ni] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int baseID = (b * seqLen + t) * innerDim;
        int baseSD = (b * seqLen + t) * stateDim;
        float dt = delta[baseID + di];
        float xv = X[baseID + di];
        float y = 0.0f;
        for (int ni = 0; ni < stateDim; ni++) {
            float negA_ni = -exp(aLog[hrow + ni]);
            float aBar = exp(dt * negA_ni);
            float hv = aBar * hState[hbase + ni] + dt * B[baseSD + ni] * xv;
            hState[hbase + ni] = hv;
            y += C[baseSD + ni] * hv;
        }
        output[baseID + di] = y + D[di] * xv;
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { "mamba_selective_scan_forward" };
}
