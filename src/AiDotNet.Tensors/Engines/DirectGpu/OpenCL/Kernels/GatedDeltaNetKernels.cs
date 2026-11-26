// Copyright (c) CrowdTrainer LLC. All rights reserved.
// OpenCL kernel for the fused Gated DeltaNet scan forward (issue ooples/AiDotNet#1464).
// Same design as HipGatedDeltaNetKernels: one work-item per (batch, head, value-row).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class GatedDeltaNetKernels
{
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#define GDN_MAX_HEADDIM 256

__kernel void gated_delta_scan_forward(
    __global const float* Q, __global const float* K, __global const float* V,
    __global const float* A, __global const float* B,
    __global float* output,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)
{
    int gid = get_global_id(0);
    int total = batch * numHeads * headDim;
    if (gid >= total) return;

    int di = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int b  = gid / (headDim * numHeads);
    int hOff = h * headDim;
    float kappa = keyScale;

    float Srow[GDN_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        float sK = 0.0f;
        for (int ki = 0; ki < headDim; ki++) sK += Srow[ki] * (K[baseOff + ki] * kappa);
        float bd = bet * (V[baseOff + di] - (decayFirst ? a : 1.0f) * sK);
        for (int ki = 0; ki < headDim; ki++)
            Srow[ki] = a * Srow[ki] + bd * (K[baseOff + ki] * kappa);
        float o = 0.0f;
        for (int ki = 0; ki < headDim; ki++) o += Srow[ki] * Q[baseOff + ki];
        output[baseOff + di] = o;
    }
}

/ Checkpointed backward (GatedDeltaNetCheckpoint): one work-group per (batch, head) whose threads stride over the
// state rows. Pass 1 saves the state entering every `chunk` steps (Ckpt); pass 2 walks the chunks in reverse,
// recomputes the chunk's states (Scratch), sums dQ over the rows, runs BPTT per row (carrying the row's adjoint in dS
// between chunks, and overwriting each Scratch row with its dK contribution once nothing reads it), then sums dK, dA
// and dB over the rows. Every sum runs over di in ascending order behind a barrier: deterministic, no atomics, every
// output element written.
__kernel void gated_delta_scan_backward_ckpt(
    __global const float* dOut, __global const float* Q, __global const float* K,
    __global const float* V, __global const float* A, __global const float* B,
    __global float* Work,
    __global float* dQ, __global float* dK, __global float* dV, __global float* dA, __global float* dB,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int chunk, int headStride,
    float keyScale, int decayFirst)
{
    int bh = get_group_id(0);
    if (bh >= batch * numHeads) return;
    int tid = get_local_id(0), nt = get_local_size(0);
    int h = bh % numHeads, b = bh / numHeads, hOff = h * headDim, hd = headDim;
    long hh = (long)hd * hd;
    int nChunks = (seqLen + chunk - 1) / chunk;
    __global float* ck = Work + (long)bh * headStride;
    __global float* sc = ck + nChunks * hh;
    __global float* dsb = sc + chunk * hh;
    __global float* part = dsb + hh;
    float kappa = keyScale;
    float R[GDN_MAX_HEADDIM];
    int tLast = (nChunks - 1) * chunk;

    for (int di = tid; di < hd; di += nt) {
        long rowOff = (long)di * hd;
        for (int ki = 0; ki < hd; ki++) { R[ki] = 0.0f; dsb[rowOff + ki] = 0.0f; }
        for (int t = 0; ; t++) {
            if (t % chunk == 0) {
                __global float* dst = ck + (t / chunk) * hh + rowOff;
                for (int ki = 0; ki < hd; ki++) dst[ki] = R[ki];
            }
            if (t == tLast) break;
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            int gIdx = (b * seqLen + t) * numHeads + h;
            float a = A[gIdx], bet = B[gIdx];
            float sK = 0.0f;
            for (int ki = 0; ki < hd; ki++) sK += R[ki] * (K[baseOff + ki] * kappa);
            float bd = bet * (V[baseOff + di] - (decayFirst ? a : 1.0f) * sK);
            for (int ki = 0; ki < hd; ki++) R[ki] = a * R[ki] + bd * (K[baseOff + ki] * kappa);
        }
    }

    for (int c = nChunks - 1; c >= 0; c--) {
        int t0 = c * chunk;
        int len = min(seqLen - t0, chunk);
        __global float* ckc = ck + c * hh;
        for (int di = tid; di < hd; di += nt) {
            long rowOff = (long)di * hd;
            for (int ki = 0; ki < hd; ki++) R[ki] = ckc[rowOff + ki];
            for (int lt = 0; lt < len; lt++) {
                int t = t0 + lt;
                int baseOff = (b * seqLen + t) * modelDim + hOff;
                int gIdx = (b * seqLen + t) * numHeads + h;
                float a = A[gIdx], bet = B[gIdx];
                float sK = 0.0f;
                for (int ki = 0; ki < hd; ki++) sK += R[ki] * (K[baseOff + ki] * kappa);
                float bd = bet * (V[baseOff + di] - (decayFirst ? a : 1.0f) * sK);
                __global float* dst = sc + lt * hh + rowOff;
                for (int ki = 0; ki < hd; ki++) {
                    float s = a * R[ki] + bd * (K[baseOff + ki] * kappa);
                    R[ki] = s;
                    dst[ki] = s;
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int e = tid; e < len * hd; e += nt) {
            int lt = e / hd, ki = e % hd;
            int base = (b * seqLen + t0 + lt) * modelDim + hOff;
            __global const float* st = sc + lt * hh;
            float acc = 0.0f;
            for (int di = 0; di < hd; di++) acc += dOut[base + di] * st[(long)di * hd + ki];
            dQ[base + ki] = acc;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int di = tid; di < hd; di += nt) {
            long rowOff = (long)di * hd;
            for (int ki = 0; ki < hd; ki++) R[ki] = dsb[rowOff + ki];
            for (int lt = len - 1; lt >= 0; lt--) {
                int t = t0 + lt;
                int baseOff = (b * seqLen + t) * modelDim + hOff;
                int gIdx = (b * seqLen + t) * numHeads + h;
                float a = A[gIdx], bet = B[gIdx];
                float pa = decayFirst ? a : 1.0f;
                __global const float* sp = (lt > 0 ? sc + (lt - 1) * hh : ckc) + rowOff;
                __global float* kOut = sc + lt * hh + rowOff;
                float sK = 0.0f;
                for (int ki = 0; ki < hd; ki++) sK += sp[ki] * (K[baseOff + ki] * kappa);
                float delta = V[baseOff + di] - pa * sK;
                float dOutVal = dOut[baseOff + di];
                for (int ki = 0; ki < hd; ki++) R[ki] += dOutVal * Q[baseOff + ki];
                float dA_acc = 0.0f, dB_acc = 0.0f, dDelta = 0.0f;
                for (int ki = 0; ki < hd; ki++) {
                    float kScaled = K[baseOff + ki] * kappa;
                    dA_acc += R[ki] * sp[ki];
                    dB_acc += R[ki] * delta * kScaled;
                    dDelta += R[ki] * bet * kScaled;
                }
                if (decayFirst) dA_acc -= dDelta * sK;
                dV[baseOff + di] = dDelta;
                part[(2 * lt) * hd + di] = dA_acc;
                part[(2 * lt + 1) * hd + di] = dB_acc;
                float dSk = -dDelta;
                for (int ki = 0; ki < hd; ki++) {
                    float dSt = R[ki];
                    float kScaled = K[baseOff + ki] * kappa;
                    kOut[ki] = (dSt * (bet * delta) + dSk * pa * sp[ki]) * kappa;
                    R[ki] = dSt * a + dSk * pa * kScaled;
                }
            }
            for (int ki = 0; ki < hd; ki++) dsb[rowOff + ki] = R[ki];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int e = tid; e < len * hd; e += nt) {
            int lt = e / hd, ki = e % hd;
            int base = (b * seqLen + t0 + lt) * modelDim + hOff;
            __global const float* pk = sc + lt * hh;
            float acc = 0.0f;
            for (int di = 0; di < hd; di++) acc += pk[(long)di * hd + ki];
            dK[base + ki] = acc;
        }
        for (int lt = tid; lt < len; lt += nt) {
            int gIdx = (b * seqLen + t0 + lt) * numHeads + h;
            float sa = 0.0f, sb = 0.0f;
            for (int di = 0; di < hd; di++) { sa += part[(2 * lt) * hd + di]; sb += part[(2 * lt + 1) * hd + di]; }
            dA[gIdx] = sa;
            dB[gIdx] = sb;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
";
    }

    public static string[] GetKernelNames() => new[]
    {
        "gated_delta_scan_forward",
        GatedDeltaNetCheckpoint.KernelName
    };
}
