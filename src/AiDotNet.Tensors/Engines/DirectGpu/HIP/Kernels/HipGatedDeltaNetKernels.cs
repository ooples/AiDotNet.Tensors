// Copyright (c) CrowdTrainer LLC. All rights reserved.
// HIP kernels for the fused Gated DeltaNet (delta-rule) scan forward + BPTT backward (issue ooples/AiDotNet#1464).
// Mirrors CpuEngine.GatedDeltaNetScan. Each value row is independent in both forward and backward, so one thread
// owns (batch b, head h, value-row di) and carries that row of the state/adjoint across time.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipGatedDeltaNetKernels
{
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>

#define GDN_MAX_HEADDIM 256

extern ""C"" __global__ __launch_bounds__(1024) void gated_delta_scan_forward(
    const float* Q, const float* K, const float* V, const float* A, const float* B,
    float* output,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, float keyScale, int decayFirst)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * numHeads * headDim;
    if (gid >= total) return;

    int di = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int b  = gid / (headDim * numHeads);
    int hOff = h * headDim;
    float kappa = keyScale;

    if (headDim < 0 || headDim > GDN_MAX_HEADDIM) return;
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

// Checkpointed backward (GatedDeltaNetCheckpoint): one work-group per (batch, head) whose threads stride over the
// state rows. Pass 1 saves the state entering every `chunk` steps (Ckpt); pass 2 walks the chunks in reverse,
// recomputes the chunk's states (Scratch), sums dQ over the rows, runs BPTT per row (carrying the row's adjoint in dS
// between chunks, and overwriting each Scratch row with its dK contribution once nothing reads it), then sums dK, dA
// and dB over the rows. Every sum runs over di in ascending order behind a barrier: deterministic, no atomics, every
// output element written.
extern ""C"" __global__ __launch_bounds__(256) void gated_delta_scan_backward_ckpt(
    const float* dOut, const float* Q, const float* K,
    const float* V, const float* A, const float* B,
    float* Work,
    float* dQ, float* dK, float* dV, float* dA, float* dB,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int chunk, int headStride,
    float keyScale, int decayFirst)
{
    int bh = (int)blockIdx.x;
    if (bh >= batch * numHeads) return;
    int tid = (int)threadIdx.x, nt = (int)blockDim.x;
    int h = bh % numHeads, b = bh / numHeads, hOff = h * headDim, hd = headDim;
    long hh = (long)hd * hd;
    int nChunks = (seqLen + chunk - 1) / chunk;
    float* ck = Work + (long)bh * headStride;
    float* sc = ck + nChunks * hh;
    float* dsb = sc + chunk * hh;
    float* part = dsb + hh;
    float kappa = keyScale;
    float R[GDN_MAX_HEADDIM];
    int tLast = (nChunks - 1) * chunk;

    for (int di = tid; di < hd; di += nt) {
        long rowOff = (long)di * hd;
        for (int ki = 0; ki < hd; ki++) { R[ki] = 0.0f; dsb[rowOff + ki] = 0.0f; }
        for (int t = 0; ; t++) {
            if (t % chunk == 0) {
                float* dst = ck + (t / chunk) * hh + rowOff;
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
        float* ckc = ck + c * hh;
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
                float* dst = sc + lt * hh + rowOff;
                for (int ki = 0; ki < hd; ki++) {
                    float s = a * R[ki] + bd * (K[baseOff + ki] * kappa);
                    R[ki] = s;
                    dst[ki] = s;
                }
            }
        }
        __syncthreads();

        for (int e = tid; e < len * hd; e += nt) {
            int lt = e / hd, ki = e % hd;
            int base = (b * seqLen + t0 + lt) * modelDim + hOff;
            const float* st = sc + lt * hh;
            float acc = 0.0f;
            for (int di = 0; di < hd; di++) acc += dOut[base + di] * st[(long)di * hd + ki];
            dQ[base + ki] = acc;
        }
        __syncthreads();

        for (int di = tid; di < hd; di += nt) {
            long rowOff = (long)di * hd;
            for (int ki = 0; ki < hd; ki++) R[ki] = dsb[rowOff + ki];
            for (int lt = len - 1; lt >= 0; lt--) {
                int t = t0 + lt;
                int baseOff = (b * seqLen + t) * modelDim + hOff;
                int gIdx = (b * seqLen + t) * numHeads + h;
                float a = A[gIdx], bet = B[gIdx];
                float pa = decayFirst ? a : 1.0f;
                const float* sp = (lt > 0 ? sc + (lt - 1) * hh : ckc) + rowOff;
                float* kOut = sc + lt * hh + rowOff;
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
        __syncthreads();

        for (int e = tid; e < len * hd; e += nt) {
            int lt = e / hd, ki = e % hd;
            int base = (b * seqLen + t0 + lt) * modelDim + hOff;
            const float* pk = sc + lt * hh;
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
        __syncthreads();
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
