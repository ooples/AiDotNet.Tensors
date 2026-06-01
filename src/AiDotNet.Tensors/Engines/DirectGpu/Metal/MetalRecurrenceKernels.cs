// Copyright (c) AiDotNet. All rights reserved.
// Metal (MSL) compute shaders for the fused recurrence / LM-head ops (#1464). Real GPU kernels —
// they mirror the HIP/CUDA/OpenCL kernels and the CpuEngine math exactly. One thread per the same
// work-item as the other backends. The cross-entropy kernels write a per-row loss vector (the host
// sums the N-element vector); everything else writes the full output. Forward only — the
// differentiable backward runs through the CpuEngine tape (same as every backend).

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

internal static class MetalRecurrenceKernels
{
    public const string Source = @"
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

#define REC_MAX_HEADDIM 256
#define XLSTM_MAX_HH 4096
#define XLSTM_IGATE_CLAMP 4.85e8f

// ── GLA scan forward ───────────────────────────────────────────────────────────────────────
kernel void gla_scan_forward(
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device const float* G [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& batch [[buffer(5)]], constant int& seqLen [[buffer(6)]],
    constant int& modelDim [[buffer(7)]], constant int& numHeads [[buffer(8)]],
    constant int& headDim [[buffer(9)]], uint gid [[thread_position_in_grid]])
{
    int total = batch * numHeads * headDim;
    if ((int)gid >= total) return;
    int di = (int)gid % headDim;
    int h = ((int)gid / headDim) % numHeads;
    int b = (int)gid / (headDim * numHeads);
    int hOff = h * headDim;
    thread float Srow[REC_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;
    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        float g = G[(b * seqLen + t) * numHeads + h];
        float vd = V[baseOff + di];
        float o = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float s = g * Srow[ki] + vd * K[baseOff + ki];
            Srow[ki] = s;
            o += s * Q[baseOff + ki];
        }
        output[baseOff + di] = o;
    }
}

// ── xLSTM scan forward ─────────────────────────────────────────────────────────────────────
kernel void xlstm_scan_forward(
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device const float* I [[buffer(3)]],
    device const float* F [[buffer(4)]], device const float* O [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant int& batch [[buffer(7)]], constant int& seqLen [[buffer(8)]],
    constant int& modelDim [[buffer(9)]], constant int& numHeads [[buffer(10)]],
    constant int& headDim [[buffer(11)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= batch * numHeads) return;
    int h = (int)gid % numHeads;
    int b = (int)gid / numHeads;
    int hOff = h * headDim;
    int hh = headDim * headDim;
    float kappa = 1.0f / sqrt((float)headDim);
    thread float C[XLSTM_MAX_HH];
    thread float n[REC_MAX_HEADDIM];
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

// ── Gated DeltaNet scan forward ────────────────────────────────────────────────────────────
kernel void gated_delta_scan_forward(
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device const float* A [[buffer(3)]],
    device const float* B [[buffer(4)]], device float* output [[buffer(5)]],
    constant int& batch [[buffer(6)]], constant int& seqLen [[buffer(7)]],
    constant int& modelDim [[buffer(8)]], constant int& numHeads [[buffer(9)]],
    constant int& headDim [[buffer(10)]], uint gid [[thread_position_in_grid]])
{
    int total = batch * numHeads * headDim;
    if ((int)gid >= total) return;
    int di = (int)gid % headDim;
    int h = ((int)gid / headDim) % numHeads;
    int b = (int)gid / (headDim * numHeads);
    int hOff = h * headDim;
    float kappa = 1.0f / sqrt((float)headDim);
    thread float Srow[REC_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;
    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        float sK = 0.0f;
        for (int ki = 0; ki < headDim; ki++) sK += Srow[ki] * (K[baseOff + ki] * kappa);
        float bd = bet * (V[baseOff + di] - sK);
        for (int ki = 0; ki < headDim; ki++)
            Srow[ki] = a * Srow[ki] + bd * (K[baseOff + ki] * kappa);
        float o = 0.0f;
        for (int ki = 0; ki < headDim; ki++) o += Srow[ki] * Q[baseOff + ki];
        output[baseOff + di] = o;
    }
}

// ── RG-LRU scan forward ────────────────────────────────────────────────────────────────────
kernel void rglru_scan_forward(
    device const float* V [[buffer(0)]], device const float* R [[buffer(1)]],
    device const float* I [[buffer(2)]], device const float* decay [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& batch [[buffer(5)]], constant int& seqLen [[buffer(6)]],
    constant int& recDim [[buffer(7)]], uint gid [[thread_position_in_grid]])
{
    int total = batch * recDim;
    if ((int)gid >= total) return;
    int c = (int)gid % recDim;
    int b = (int)gid / recDim;
    float base = 1.0f / (1.0f + exp(decay[c]));
    float h = 0.0f;
    for (int t = 0; t < seqLen; t++) {
        int off = (b * seqLen + t) * recDim + c;
        float a = R[off] * base;
        float om = 1.0f - a * a;
        float s = om > 0.0f ? sqrt(om) : 0.0f;
        h = a * h + s * (I[off] * V[off]);
        output[off] = h;
    }
}

// ── RWKV-4 WKV scan forward ────────────────────────────────────────────────────────────────
kernel void rwkv4_wkv_forward(
    device const float* R [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device const float* timeDecay [[buffer(3)]],
    device const float* timeFirst [[buffer(4)]], device float* output [[buffer(5)]],
    constant int& batch [[buffer(6)]], constant int& seqLen [[buffer(7)]],
    constant int& modelDim [[buffer(8)]], uint gid [[thread_position_in_grid]])
{
    int total = batch * modelDim;
    if ((int)gid >= total) return;
    int c = (int)gid % modelDim;
    int b = (int)gid / modelDim;
    float w = -exp(timeDecay[c]);
    float u = timeFirst[c];
    float aa = 0.0f, bb = 0.0f, pp = -1.0e38f;
    for (int t = 0; t < seqLen; t++) {
        int off = (b * seqLen + t) * modelDim + c;
        float k = K[off];
        float v = V[off];
        float ww = u + k;
        float q = fmax(pp, ww);
        float e1 = exp(pp - q);
        float e2 = exp(ww - q);
        float wkv = (e1 * aa + e2 * v) / (e1 * bb + e2);
        output[off] = (1.0f / (1.0f + exp(-R[off]))) * wkv;
        float ww2 = pp + w;
        float q2 = fmax(ww2, k);
        float e1b = exp(ww2 - q2);
        float e2b = exp(k - q2);
        aa = e1b * aa + e2b * v;
        bb = e1b * bb + e2b;
        pp = q2;
    }
}

// ── Mamba selective scan forward ───────────────────────────────────────────────────────────
kernel void mamba_selective_scan_forward(
    device const float* X [[buffer(0)]], device const float* delta [[buffer(1)]],
    device const float* aLog [[buffer(2)]], device const float* B [[buffer(3)]],
    device const float* C [[buffer(4)]], device const float* D [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant int& batch [[buffer(7)]], constant int& seqLen [[buffer(8)]],
    constant int& innerDim [[buffer(9)]], constant int& stateDim [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    int total = batch * innerDim;
    if ((int)gid >= total) return;
    int di = (int)gid % innerDim;
    int b = (int)gid / innerDim;
    int hrow = di * stateDim;
    thread float negA[REC_MAX_HEADDIM];
    thread float h[REC_MAX_HEADDIM];
    for (int n = 0; n < stateDim; n++) { negA[n] = -exp(aLog[hrow + n]); h[n] = 0.0f; }
    for (int t = 0; t < seqLen; t++) {
        int baseID = (b * seqLen + t) * innerDim;
        int baseSD = (b * seqLen + t) * stateDim;
        float dt = delta[baseID + di];
        float xv = X[baseID + di];
        float y = 0.0f;
        for (int n = 0; n < stateDim; n++) {
            float aBar = exp(dt * negA[n]);
            float hv = aBar * h[n] + dt * B[baseSD + n] * xv;
            h[n] = hv;
            y += C[baseSD + n] * hv;
        }
        output[baseID + di] = y + D[di] * xv;
    }
}

// ── Mamba-2 SSD scan forward ───────────────────────────────────────────────────────────────
kernel void mamba2_ssd_scan_forward(
    device const float* X [[buffer(0)]], device const float* delta [[buffer(1)]],
    device const float* aLog [[buffer(2)]], device const float* B [[buffer(3)]],
    device const float* C [[buffer(4)]], device const float* D [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant int& batch [[buffer(7)]], constant int& seqLen [[buffer(8)]],
    constant int& innerDim [[buffer(9)]], constant int& numHeads [[buffer(10)]],
    constant int& headDim [[buffer(11)]], constant int& sd [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    int total = batch * innerDim;
    if ((int)gid >= total) return;
    int flatD = (int)gid % innerDim;
    int b = (int)gid / innerDim;
    int hi = flatD / headDim;
    float negA = -exp(aLog[hi]);
    float dv = D[hi];
    thread float h[REC_MAX_HEADDIM];
    for (int n = 0; n < sd; n++) h[n] = 0.0f;
    for (int t = 0; t < seqLen; t++) {
        int btInner = (b * seqLen + t) * innerDim;
        int btState = (b * seqLen + t) * sd;
        int btHead = (b * seqLen + t) * numHeads;
        float dt = delta[btHead + hi];
        float aBar = exp(dt * negA);
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

// ── Fused linear + cross-entropy (per-row loss; host sums) ──────────────────────────────────
kernel void fused_linear_ce_index(
    device const float* hidden [[buffer(0)]], device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]], device const float* targetIds [[buffer(3)]],
    device float* rowLoss [[buffer(4)]],
    constant int& N [[buffer(5)]], constant int& d [[buffer(6)]], constant int& vocab [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int r = (int)gid;
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
    rowLoss[r] = (mx + log(sumExp)) - logitTarget;
}

kernel void fused_linear_ce_dense(
    device const float* hidden [[buffer(0)]], device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]], device const float* target [[buffer(3)]],
    device float* rowLoss [[buffer(4)]],
    constant int& N [[buffer(5)]], constant int& d [[buffer(6)]], constant int& vocab [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int r = (int)gid;
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
    float rl = 0.0f;
    for (int vv = 0; vv < vocab; vv++) {
        float s = bias[vv];
        for (int j = 0; j < d; j++) s += hidden[hRow + j] * weight[j * vocab + vv];
        rl += target[tRow + vv] * (s - lse);
    }
    rowLoss[r] = -rl;
}

// ── GLA BPTT backward (recompute trajectory + reverse sweep) ─────────────────────────────────
// Cross-row dQ/dK/dG accumulation uses a CAS-on-uint float atomic add (portable across GPUs that
// lack native float atomics). dV[baseOff+di] is written by a single (b,h,di) thread → plain store.
inline void gla_atomic_add_f(device atomic_uint* buf, int idx, float val) {
    uint oldv = atomic_load_explicit(&buf[idx], memory_order_relaxed);
    while (true) {
        uint newv = as_type<uint>(as_type<float>(oldv) + val);
        if (atomic_compare_exchange_weak_explicit(&buf[idx], &oldv, newv,
                memory_order_relaxed, memory_order_relaxed)) break;
    }
}

kernel void gla_scan_recompute(
    device const float* K [[buffer(0)]], device const float* V [[buffer(1)]], device const float* G [[buffer(2)]],
    device float* Straj [[buffer(3)]],
    constant int& batch [[buffer(4)]], constant int& seqLen [[buffer(5)]], constant int& modelDim [[buffer(6)]],
    constant int& numHeads [[buffer(7)]], constant int& headDim [[buffer(8)]], uint gid [[thread_position_in_grid]])
{
    int total = batch * numHeads * headDim;
    if ((int)gid >= total) return;
    int di = (int)gid % headDim; int h = ((int)gid / headDim) % numHeads; int b = (int)gid / (headDim * numHeads);
    int hOff = h * headDim; int hh = headDim * headDim;
    float Srow[REC_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;
    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        float g = G[(b * seqLen + t) * numHeads + h];
        float vd = V[baseOff + di];
        int trajBase = ((b * numHeads + h) * seqLen + t) * hh + di * headDim;
        for (int ki = 0; ki < headDim; ki++) { float s = g * Srow[ki] + vd * K[baseOff + ki]; Srow[ki] = s; Straj[trajBase + ki] = s; }
    }
}

kernel void gla_scan_backward(
    device const float* dOut [[buffer(0)]], device const float* Q [[buffer(1)]], device const float* K [[buffer(2)]],
    device const float* V [[buffer(3)]], device const float* G [[buffer(4)]], device const float* Straj [[buffer(5)]],
    device atomic_uint* dQ [[buffer(6)]], device atomic_uint* dK [[buffer(7)]], device float* dV [[buffer(8)]], device atomic_uint* dG [[buffer(9)]],
    constant int& batch [[buffer(10)]], constant int& seqLen [[buffer(11)]], constant int& modelDim [[buffer(12)]],
    constant int& numHeads [[buffer(13)]], constant int& headDim [[buffer(14)]], uint gid [[thread_position_in_grid]])
{
    int total = batch * numHeads * headDim;
    if ((int)gid >= total) return;
    int di = (int)gid % headDim; int h = ((int)gid / headDim) % numHeads; int b = (int)gid / (headDim * numHeads);
    int hOff = h * headDim; int hh = headDim * headDim;
    float dSrow[REC_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) dSrow[ki] = 0.0f;
    for (int t = seqLen - 1; t >= 0; t--) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gOff = (b * seqLen + t) * numHeads + h;
        float g = G[gOff];
        int trajBase = ((b * numHeads + h) * seqLen + t) * hh + di * headDim;
        int sprevBase = ((b * numHeads + h) * seqLen + (t - 1)) * hh + di * headDim;
        float dOutVal = dOut[baseOff + di];
        for (int ki = 0; ki < headDim; ki++) {
            float sval = Straj[trajBase + ki];
            gla_atomic_add_f(dQ, baseOff + ki, dOutVal * sval);
            dSrow[ki] += dOutVal * Q[baseOff + ki];
        }
        float vd = V[baseOff + di]; float dg = 0.0f; float dVacc = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float dStv = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            dg += dStv * sprev;
            gla_atomic_add_f(dK, baseOff + ki, dStv * vd);
            dVacc += dStv * K[baseOff + ki];
        }
        dV[baseOff + di] = dVacc;
        gla_atomic_add_f(dG, gOff, dg);
        for (int ki = 0; ki < headDim; ki++) dSrow[ki] *= g;
    }
}
";
}
