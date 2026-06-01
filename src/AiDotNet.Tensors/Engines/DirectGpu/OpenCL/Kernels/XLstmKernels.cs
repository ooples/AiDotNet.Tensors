// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernel for the fused xLSTM (mLSTM) matrix-memory scan forward (issue ooples/AiDotNet#1464).
// Same design as HipXLstmKernels: one work-item per (batch, head) carries the full C + n state.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class XLstmKernels
{
    public const int MaxHeadDim = 64;

    public static string GetSource()
    {
        return @"
#define XLSTM_MAX_HEADDIM 64
#define XLSTM_MAX_HH 4096
#define XLSTM_IGATE_CLAMP 4.85e8f

__kernel void xlstm_scan_forward(
    __global const float* Q, __global const float* K, __global const float* V,
    __global const float* I, __global const float* F, __global const float* O,
    __global float* output,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    int gid = get_global_id(0);
    if (gid >= batch * numHeads) return;

    int h = gid % numHeads;
    int b = gid / numHeads;
    int hOff = h * headDim;
    int hh = headDim * headDim;
    float kappa = 1.0f / sqrt((float)headDim);

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
        float nf = fabs(nq); if (nf < 1.0f) nf = 1.0f;
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
