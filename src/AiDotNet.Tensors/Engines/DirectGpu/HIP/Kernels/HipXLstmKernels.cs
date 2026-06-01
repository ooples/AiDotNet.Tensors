// Copyright (c) AiDotNet. All rights reserved.
// HIP kernel for the fused xLSTM (mLSTM) matrix-memory scan forward (issue ooples/AiDotNet#1464).
// Mirrors CpuEngine.XLstmScan.cs / RecurrenceCpuKernels.XLstmForward. One thread owns (batch b,
// head h) and carries the full C[headDim*headDim] cell + n[headDim] normalizer in private memory,
// because nf = max(|sum_di n[di]*Q[di]|, 1) couples all value rows of a head. headDim is capped so
// the private state fits. The differentiable backward runs through the CpuEngine tape path.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipXLstmKernels
{
    public const int MaxHeadDim = 64;

    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

#define XLSTM_MAX_HEADDIM 64
#define XLSTM_MAX_HH 4096
#define XLSTM_IGATE_CLAMP 4.85e8f

extern ""C"" __global__ __launch_bounds__(256) void xlstm_scan_forward(
    const float* Q, const float* K, const float* V,
    const float* I, const float* F, const float* O,
    float* output,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= batch * numHeads) return;

    int h = gid % numHeads;
    int b = gid / numHeads;
    int hOff = h * headDim;
    int hh = headDim * headDim;
    if (headDim < 0 || headDim > XLSTM_MAX_HEADDIM || hh > XLSTM_MAX_HH) return;
    float kappa = 1.0f / sqrtf((float)headDim);

    float C[XLSTM_MAX_HH];
    float n[XLSTM_MAX_HEADDIM];
    for (int i = 0; i < hh; i++) C[i] = 0.0f;
    for (int i = 0; i < headDim; i++) n[i] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gOff = (b * seqLen + t) * numHeads + h;
        float iv = I[gOff]; if (iv > XLSTM_IGATE_CLAMP) iv = XLSTM_IGATE_CLAMP;
        float f = F[gOff], o = O[gOff];
        for (int di = 0; di < headDim; di++) {
            n[di] = f * n[di] + iv * (K[baseOff + di] * kappa);
            float vv = V[baseOff + di];
            int srow = di * headDim;
            for (int ki = 0; ki < headDim; ki++)
                C[srow + ki] = f * C[srow + ki] + iv * vv * (K[baseOff + ki] * kappa);
        }
        float nq = 0.0f;
        for (int j = 0; j < headDim; j++) nq += n[j] * Q[baseOff + j];
        float nf = fabsf(nq); if (nf < 1.0f) nf = 1.0f;
        for (int di = 0; di < headDim; di++) {
            int srow = di * headDim;
            float num = 0.0f;
            for (int ki = 0; ki < headDim; ki++) num += C[srow + ki] * Q[baseOff + ki];
            output[baseOff + di] = o * num / nf;
        }
    }
}
";
    }

    public static string[] GetKernelNames() => new[] { "xlstm_scan_forward" };
}
