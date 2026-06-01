// Copyright (c) AiDotNet. All rights reserved.
// HIP kernel for the fused Mamba-2 SSD scan forward (#1464).
// Mirrors CpuEngine.Mamba2SsdScan / RecurrenceCpuKernels.Mamba2SsdScanForward. Per channel
// flatD = head*headDim + di carries an independent state h[flatD, 0..stateDim), so one thread owns
// (batch b, channel flatD). delta/aLog/D are per-head. Backward via tape.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipMamba2Kernels
{
    public const int MaxStateDim = 256;

    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

#define MAMBA2_MAX_STATEDIM 256

extern ""C"" __global__ __launch_bounds__(1024) void mamba2_ssd_scan_forward(
    const float* X, const float* delta, const float* aLog,
    const float* B, const float* C, const float* D,
    float* output,
    int batch, int seqLen, int innerDim, int numHeads, int headDim, int sd)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * innerDim;
    if (gid >= total) return;

    int flatD = gid % innerDim;
    int b = gid / innerDim;
    int hi = flatD / headDim;
    float negA = -expf(aLog[hi]);
    float dv = D[hi];

    if (sd < 0 || sd > MAMBA2_MAX_STATEDIM) return;
    float h[MAMBA2_MAX_STATEDIM];
    for (int n = 0; n < sd; n++) h[n] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int btInner = (b * seqLen + t) * innerDim;
        int btState = (b * seqLen + t) * sd;
        int btHead = (b * seqLen + t) * numHeads;
        float dt = delta[btHead + hi];
        float aBar = expf(dt * negA);
        float xv = X[btInner + flatD];
        float y = 0.0f;
        for (int n = 0; n < sd; n++) {
            float hNew = aBar * h[n] + dt * B[btState + n] * xv;
            h[n] = hNew;
            y += C[btState + n] * hNew;
        }
        output[btInner + flatD] = y + dv * xv;
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { "mamba2_ssd_scan_forward" };
}
