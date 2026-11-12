// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Metal (MSL) compute shaders for the fused recurrence / LM-head ops (#1464). Real GPU kernels —
// they mirror the HIP/CUDA/OpenCL kernels and the CpuEngine math exactly. One thread per the same
// work-item as the other backends. The cross-entropy kernels write a per-row loss vector (the host
// sums the N-element vector); everything else writes the full output. Forward only — the
// differentiable backward runs through the CpuEngine tape (same as every backend).

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

internal static class MetalRecurrenceKernels
{
    /// <summary>
    /// The largest head the recurrence kernels' per-thread rows hold. Must equal <c>REC_MAX_HEADDIM</c> in
    /// <see cref="Source"/>: a larger head would index past those thread arrays.
    /// </summary>
    public const int MaxHeadDim = 256;

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

kernel void complex_diagonal_ssm_scan_forward(
    device const float* X [[buffer(0)]], device const float* Ar [[buffer(1)]],
    device const float* Ai [[buffer(2)]], device const float* Br [[buffer(3)]],
    device const float* Bi [[buffer(4)]], device const float* Cr [[buffer(5)]],
    device const float* Ci [[buffer(6)]], device const float* D [[buffer(7)]],
    device float* output [[buffer(8)]], constant int& batch [[buffer(9)]],
    constant int& time [[buffer(10)]], constant int& groups [[buffer(11)]],
    constant int& width [[buffer(12)]], constant int& state [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= batch*groups) return;
    int b=(int)gid/groups, g=(int)gid%groups;
    thread float hr[256], hi[256];
    for (int n=0;n<state;n++) { hr[n]=0.0f; hi[n]=0.0f; }
    for (int t=0;t<time;t++) {
        int xBase=((b*time+t)*groups+g)*width;
        for (int n=0;n<state;n++) {
            int a=g*state+n, bm=(g*state+n)*width; float oldR=hr[n], oldI=hi[n];
            float nextR=Ar[a]*oldR-Ai[a]*oldI, nextI=Ar[a]*oldI+Ai[a]*oldR;
            for (int w=0;w<width;w++) { float xv=X[xBase+w]; nextR+=Br[bm+w]*xv; nextI+=Bi[bm+w]*xv; }
            hr[n]=nextR; hi[n]=nextI;
        }
        for (int w=0;w<width;w++) {
            int cm=(g*width+w)*state; float y=D[g*width+w]*X[xBase+w];
            for (int n=0;n<state;n++) y+=Cr[cm+n]*hr[n]-Ci[cm+n]*hi[n];
            output[xBase+w]=y;
        }
    }
}

// The Mesa head dimension is a runtime and storage contract: it bounds both the
// admission check below and the thread-local scratch arrays, so the two can never
// be allowed to drift apart.
#define MESA_MAX_HEADDIM 32
kernel void mesa_scan_forward(
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device const float* W0 [[buffer(3)]],
    device const float* regularization [[buffer(4)]], device float* output [[buffer(5)]],
    device float* workW [[buffer(6)]], device float* covariance [[buffer(7)]],
    constant int& batch [[buffer(8)]], constant int& time [[buffer(9)]],
    constant int& model [[buffer(10)]], constant int& heads [[buffer(11)]],
    constant int& headDim [[buffer(12)]], uint gid [[thread_position_in_grid]])
{
    int bh=(int)gid; if(bh>=batch*heads || headDim>MESA_MAX_HEADDIM)return;
    int b=bh/heads,h=bh%heads,matrix=headDim*headDim,base=bh*matrix,w0Base=h*matrix;
    float invLambda=1.0f/regularization[0];
    for(int i=0;i<matrix;i++){workW[base+i]=W0[w0Base+i];covariance[base+i]=0.0f;}
    for(int i=0;i<headDim;i++)covariance[base+i*headDim+i]=invLambda;
    thread float pk[MESA_MAX_HEADDIM],error[MESA_MAX_HEADDIM],row[MESA_MAX_HEADDIM];
    for(int t=0;t<time;t++){
        int vb=(b*time+t)*model+h*headDim;
        for(int i=0;i<headDim;i++){float s=0.0f;for(int j=0;j<headDim;j++)s+=covariance[base+i*headDim+j]*K[vb+j];pk[i]=s;}
        float denom=1.0f;for(int i=0;i<headDim;i++)denom+=K[vb+i]*pk[i];
        for(int i=0;i<headDim;i++)for(int j=0;j<headDim;j++)covariance[base+i*headDim+j]-=pk[i]*pk[j]/denom;
        for(int i=0;i<headDim;i++){float s=0.0f;for(int j=0;j<headDim;j++)s+=workW[base+i*headDim+j]*K[vb+j];error[i]=s-V[vb+i];}
        for(int j=0;j<headDim;j++){float s=0.0f;for(int i=0;i<headDim;i++)s+=K[vb+i]*covariance[base+i*headDim+j];row[j]=s;}
        for(int i=0;i<headDim;i++)for(int j=0;j<headDim;j++)workW[base+i*headDim+j]-=error[i]*row[j];
        for(int i=0;i<headDim;i++){float s=0.0f;for(int j=0;j<headDim;j++)s+=workW[base+i*headDim+j]*Q[vb+j];output[vb+i]=s;}
    }
}

kernel void routed_diagonal_ssm_scan_forward(
 device const float* X [[buffer(0)]],device const float* mask [[buffer(1)]],device const float* A [[buffer(2)]],
 device const float* B [[buffer(3)]],device const float* C [[buffer(4)]],device const float* D [[buffer(5)]],
 device float* output [[buffer(6)]],device float* hState [[buffer(7)]],constant int& batch [[buffer(8)]],
 constant int& time [[buffer(9)]],constant int& model [[buffer(10)]],constant int& experts [[buffer(11)]],
 constant int& state [[buffer(12)]],uint gid [[thread_position_in_grid]])
{
 int be=(int)gid;if(be>=batch*experts)return;int b=be/experts,e=be%experts,hb=be*state;
 for(int s=0;s<state;s++)hState[hb+s]=0.0f;for(int t=0;t<time;t++){int xb=(b*time+t)*model,mi=(b*time+t)*experts+e;float active=mask[mi];
  for(int s=0;s<state;s++){float next=A[e*state+s]*hState[hb+s];int bb=(e*state+s)*model;for(int d=0;d<model;d++)next+=B[bb+d]*X[xb+d];hState[hb+s]=active*next;}
  int yb=((b*time+t)*experts+e)*model;for(int d=0;d<model;d++){float y=D[e*model+d]*X[xb+d];int cb=(e*model+d)*state;for(int s=0;s<state;s++)y+=C[cb+s]*hState[hb+s];output[yb+d]=active*y;}}
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

// Checkpointed: pass 1 re-runs the forward scan saving the state entering each chunk of `chunk` steps (Ckpt,
// B*H*nChunks*d^2); pass 2 walks chunks in reverse, recomputes the chunk's states into Scratch (B*H*chunk*d^2) and
// runs BPTT over it: O(sqrt(S)) state memory instead of the O(S) trajectory (GlaScanCheckpoint).
kernel void gla_scan_backward_ckpt(
    device const float* dOut [[buffer(0)]], device const float* Q [[buffer(1)]], device const float* K [[buffer(2)]],
    device const float* V [[buffer(3)]], device const float* G [[buffer(4)]],
    device float* Ckpt [[buffer(5)]], device float* Scratch [[buffer(6)]],
    device atomic_uint* dQ [[buffer(7)]], device atomic_uint* dK [[buffer(8)]], device float* dV [[buffer(9)]], device atomic_uint* dG [[buffer(10)]],
    constant int& batch [[buffer(11)]], constant int& seqLen [[buffer(12)]], constant int& modelDim [[buffer(13)]],
    constant int& numHeads [[buffer(14)]], constant int& headDim [[buffer(15)]], constant int& chunk [[buffer(16)]],
    uint gid [[thread_position_in_grid]])
{
    int total = batch * numHeads * headDim;
    if ((int)gid >= total) return;
    int di = (int)gid % headDim; int h = ((int)gid / headDim) % numHeads; int b = (int)gid / (headDim * numHeads);
    int hOff = h * headDim; int hh = headDim * headDim;
    int bh = b * numHeads + h; int rowOff = di * headDim;
    int nChunks = (seqLen + chunk - 1) / chunk;
    float Srow[REC_MAX_HEADDIM];
    float dSrow[REC_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;
    for (int t = 0; t < seqLen; t++) {
        if (t % chunk == 0) {
            int ck = (bh * nChunks + t / chunk) * hh + rowOff;
            for (int ki = 0; ki < headDim; ki++) Ckpt[ck + ki] = Srow[ki];
        }
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        float g = G[(b * seqLen + t) * numHeads + h];
        float vd = V[baseOff + di];
        for (int ki = 0; ki < headDim; ki++) Srow[ki] = g * Srow[ki] + vd * K[baseOff + ki];
    }
    for (int ki = 0; ki < headDim; ki++) dSrow[ki] = 0.0f;
    for (int c = nChunks - 1; c >= 0; c--) {
        int t0 = c * chunk;
        int t1 = min(seqLen, t0 + chunk);
        int ck = (bh * nChunks + c) * hh + rowOff;
        for (int ki = 0; ki < headDim; ki++) Srow[ki] = Ckpt[ck + ki];
        for (int t = t0; t < t1; t++) {
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            float g = G[(b * seqLen + t) * numHeads + h];
            float vd = V[baseOff + di];
            int sc = (bh * chunk + (t - t0)) * hh + rowOff;
            for (int ki = 0; ki < headDim; ki++) { float s = g * Srow[ki] + vd * K[baseOff + ki]; Srow[ki] = s; Scratch[sc + ki] = s; }
        }
        for (int t = t1 - 1; t >= t0; t--) {
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            int gOff = (b * seqLen + t) * numHeads + h;
            float g = G[gOff];
            int sc = (bh * chunk + (t - t0)) * hh + rowOff;
            int sp = (t > t0) ? (bh * chunk + (t - 1 - t0)) * hh + rowOff : -1;
            float dOutVal = dOut[baseOff + di];
            for (int ki = 0; ki < headDim; ki++) {
                gla_atomic_add_f(dQ, baseOff + ki, dOutVal * Scratch[sc + ki]);
                dSrow[ki] += dOutVal * Q[baseOff + ki];
            }
            float vd = V[baseOff + di]; float dg = 0.0f; float dVacc = 0.0f;
            for (int ki = 0; ki < headDim; ki++) {
                float dStv = dSrow[ki];
                float sprev = (sp >= 0) ? Scratch[sp + ki] : Ckpt[ck + ki];
                dg += dStv * sprev;
                gla_atomic_add_f(dK, baseOff + ki, dStv * vd);
                dVacc += dStv * K[baseOff + ki];
            }
            dV[baseOff + di] = dVacc;
            gla_atomic_add_f(dG, gOff, dg);
            for (int ki = 0; ki < headDim; ki++) dSrow[ki] *= g;
        }
    }
}

// ── Channels-last causal depthwise conv (ICausalDepthwiseConvBackend) ─────────────────────────────
// x, y, dy, dx are [rows, seqLen, C]; w, dw are [K, C]; tap j reads j steps back. Every output element is written and
// nothing is atomic, so all three are deterministic.
kernel void causal_dwconv1d_forward(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device float* y [[buffer(2)]],
    constant int& rows [[buffer(3)]], constant int& seqLen [[buffer(4)]], constant int& channels [[buffer(5)]],
    constant int& taps [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    int total = rows * seqLen * channels;
    if ((int)gid >= total) return;
    int idx = (int)gid;
    int c = idx % channels; int t = (idx / channels) % seqLen;
    int maxJ = min(taps - 1, t);
    float acc = 0.0f;
    for (int j = 0; j <= maxJ; j++) acc += w[j * channels + c] * x[idx - j * channels];
    y[idx] = acc;
}

kernel void causal_dwconv1d_backward_input(
    device const float* dy [[buffer(0)]], device const float* w [[buffer(1)]], device float* dx [[buffer(2)]],
    constant int& rows [[buffer(3)]], constant int& seqLen [[buffer(4)]], constant int& channels [[buffer(5)]],
    constant int& taps [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    int total = rows * seqLen * channels;
    if ((int)gid >= total) return;
    int idx = (int)gid;
    int c = idx % channels; int t = (idx / channels) % seqLen;
    int maxJ = min(taps - 1, seqLen - 1 - t);
    float acc = 0.0f;
    for (int j = 0; j <= maxJ; j++) acc += w[j * channels + c] * dy[idx + j * channels];
    dx[idx] = acc;
}

// One thread per (tap j, channel c), summing every row and step in ascending order.
kernel void causal_dwconv1d_backward_kernel(
    device const float* dy [[buffer(0)]], device const float* x [[buffer(1)]], device float* dw [[buffer(2)]],
    constant int& rows [[buffer(3)]], constant int& seqLen [[buffer(4)]], constant int& channels [[buffer(5)]],
    constant int& taps [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    int jc = (int)gid;
    if (jc >= taps * channels) return;
    int j = jc / channels; int c = jc % channels;
    float acc = 0.0f;
    for (int r = 0; r < rows; r++) {
        int rowBase = r * seqLen * channels;
        for (int t = j; t < seqLen; t++) acc += dy[rowBase + t * channels + c] * x[rowBase + (t - j) * channels + c];
    }
    dw[jc] = acc;
}

kernel void gated_delta_scan_recompute(
    device const float* K [[buffer(0)]], device const float* V [[buffer(1)]],
    device const float* A [[buffer(2)]], device const float* B [[buffer(3)]],
    device float* Straj [[buffer(4)]],
    constant int& batch [[buffer(5)]], constant int& seqLen [[buffer(6)]], constant int& modelDim [[buffer(7)]],
    constant int& numHeads [[buffer(8)]], constant int& headDim [[buffer(9)]], uint gid [[thread_position_in_grid]])
{
    int total = batch * numHeads * headDim;
    if ((int)gid >= total) return;
    int di = (int)gid % headDim; int h = ((int)gid / headDim) % numHeads; int b = (int)gid / (headDim * numHeads);
    int hOff = h * headDim; int hh = headDim * headDim;
    float kappa = 1.0f / sqrt((float)headDim);
    float Srow[REC_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;
    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        float sK = 0.0f;
        for (int ki = 0; ki < headDim; ki++) sK += Srow[ki] * (K[baseOff + ki] * kappa);
        float bd = bet * (V[baseOff + di] - sK);
        int trajBase = ((b * numHeads + h) * seqLen + t) * hh + di * headDim;
        for (int ki = 0; ki < headDim; ki++) { float s = a * Srow[ki] + bd * (K[baseOff + ki] * kappa); Srow[ki] = s; Straj[trajBase + ki] = s; }
    }
}

kernel void gated_delta_scan_backward(
    device const float* dOut [[buffer(0)]], device const float* Q [[buffer(1)]], device const float* K [[buffer(2)]],
    device const float* V [[buffer(3)]], device const float* A [[buffer(4)]], device const float* B [[buffer(5)]],
    device const float* Straj [[buffer(6)]],
    device atomic_uint* dQ [[buffer(7)]], device atomic_uint* dK [[buffer(8)]], device float* dV [[buffer(9)]],
    device atomic_uint* dA [[buffer(10)]], device atomic_uint* dB [[buffer(11)]],
    constant int& batch [[buffer(12)]], constant int& seqLen [[buffer(13)]], constant int& modelDim [[buffer(14)]],
    constant int& numHeads [[buffer(15)]], constant int& headDim [[buffer(16)]], uint gid [[thread_position_in_grid]])
{
    int total = batch * numHeads * headDim;
    if ((int)gid >= total) return;
    int di = (int)gid % headDim; int h = ((int)gid / headDim) % numHeads; int b = (int)gid / (headDim * numHeads);
    int hOff = h * headDim; int hh = headDim * headDim;
    float kappa = 1.0f / sqrt((float)headDim);
    float dSrow[REC_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) dSrow[ki] = 0.0f;
    for (int t = seqLen - 1; t >= 0; t--) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        int trajBase = ((b * numHeads + h) * seqLen + t) * hh + di * headDim;
        int sprevBase = (t > 0) ? (((b * numHeads + h) * seqLen + (t - 1)) * hh + di * headDim) : 0;
        float sK = 0.0f;
        for (int ki = 0; ki < headDim; ki++) { float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f; sK += sprev * (K[baseOff + ki] * kappa); }
        float delta = V[baseOff + di] - sK;
        float dOutVal = dOut[baseOff + di];
        for (int ki = 0; ki < headDim; ki++) {
            float sval = Straj[trajBase + ki];
            gla_atomic_add_f(dQ, baseOff + ki, dOutVal * sval);
            dSrow[ki] += dOutVal * Q[baseOff + ki];
        }
        float dA_acc = 0.0f, dB_acc = 0.0f, dDelta = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            float kScaled = K[baseOff + ki] * kappa;
            dA_acc += dSt * sprev;
            dB_acc += dSt * delta * kScaled;
            dDelta += dSt * bet * kScaled;
        }
        dV[baseOff + di] = dDelta;
        gla_atomic_add_f(dA, gIdx, dA_acc);
        gla_atomic_add_f(dB, gIdx, dB_acc);
        float dSk = -dDelta;
        for (int ki = 0; ki < headDim; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            float kScaled = K[baseOff + ki] * kappa;
            float dKS = dSt * (bet * delta) + dSk * sprev;
            gla_atomic_add_f(dK, baseOff + ki, dKS * kappa);
            dSrow[ki] = dSt * a + dSk * kScaled;
        }
    }
}

// gated_delta_scan_backward without atomics, for deterministic mode: dQ comes from gated_delta_scan_dq, each row's dK
// contribution overwrites that row of Straj (read only by this thread, and last read at step t + 1), and dA/dB go to
// per-row partials P = [dA | dB]. gated_delta_scan_reduce then sums the rows in a fixed order.
kernel void gated_delta_scan_backward_det(
    device const float* dOut [[buffer(0)]], device const float* Q [[buffer(1)]], device const float* K [[buffer(2)]],
    device const float* V [[buffer(3)]], device const float* A [[buffer(4)]], device const float* B [[buffer(5)]],
    device float* Straj [[buffer(6)]], device float* P [[buffer(7)]], device float* dV [[buffer(8)]],
    constant int& batch [[buffer(9)]], constant int& seqLen [[buffer(10)]], constant int& modelDim [[buffer(11)]],
    constant int& numHeads [[buffer(12)]], constant int& headDim [[buffer(13)]], uint gid [[thread_position_in_grid]])
{
    int total = batch * numHeads * headDim;
    if ((int)gid >= total) return;
    int di = (int)gid % headDim; int h = ((int)gid / headDim) % numHeads; int b = (int)gid / (headDim * numHeads);
    int hOff = h * headDim; int hh = headDim * headDim;
    float kappa = 1.0f / sqrt((float)headDim);
    float dSrow[REC_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) dSrow[ki] = 0.0f;
    for (int t = seqLen - 1; t >= 0; t--) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        int gIdx = (b * seqLen + t) * numHeads + h;
        float a = A[gIdx], bet = B[gIdx];
        int trajBase = ((b * numHeads + h) * seqLen + t) * hh + di * headDim;
        int sprevBase = (t > 0) ? (((b * numHeads + h) * seqLen + (t - 1)) * hh + di * headDim) : 0;
        float sK = 0.0f;
        for (int ki = 0; ki < headDim; ki++) { float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f; sK += sprev * (K[baseOff + ki] * kappa); }
        float delta = V[baseOff + di] - sK;
        float dOutVal = dOut[baseOff + di];
        for (int ki = 0; ki < headDim; ki++) {
            dSrow[ki] += dOutVal * Q[baseOff + ki];
        }
        float dA_acc = 0.0f, dB_acc = 0.0f, dDelta = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            float kScaled = K[baseOff + ki] * kappa;
            dA_acc += dSt * sprev;
            dB_acc += dSt * delta * kScaled;
            dDelta += dSt * bet * kScaled;
        }
        dV[baseOff + di] = dDelta;
        P[gIdx * headDim + di] = dA_acc;
        P[batch * seqLen * modelDim + gIdx * headDim + di] = dB_acc;
        float dSk = -dDelta;
        for (int ki = 0; ki < headDim; ki++) {
            float dSt = dSrow[ki];
            float sprev = (t > 0) ? Straj[sprevBase + ki] : 0.0f;
            float kScaled = K[baseOff + ki] * kappa;
            float dKS = dSt * (bet * delta) + dSk * sprev;
            Straj[trajBase + ki] = dKS * kappa;
            dSrow[ki] = dSt * a + dSk * kScaled;
        }
    }
}

// Deterministic dQ: dQ[b,t,h,ki] = sum over di (ascending) of dOut[b,t,h,di] * S_t[di,ki].
kernel void gated_delta_scan_dq(
    device const float* dOut [[buffer(0)]], device const float* Straj [[buffer(1)]], device float* dQ [[buffer(2)]],
    constant int& batch [[buffer(3)]], constant int& seqLen [[buffer(4)]], constant int& modelDim [[buffer(5)]],
    constant int& numHeads [[buffer(6)]], constant int& headDim [[buffer(7)]], uint gid [[thread_position_in_grid]])
{
    int g = (int)gid;
    if (g >= batch * seqLen * modelDim) return;
    int ki = g % headDim; int h = (g / headDim) % numHeads; int bt = g / modelDim; int t = bt % seqLen; int b = bt / seqLen;
    int trajT = ((b * numHeads + h) * seqLen + t) * headDim * headDim;
    int base = bt * modelDim + h * headDim;
    float acc = 0.0f;
    for (int di = 0; di < headDim; di++) acc += dOut[base + di] * Straj[trajT + di * headDim + ki];
    dQ[g] = acc;
}

// Deterministic reduction of gated_delta_scan_backward_det's partials: dK from the rows of PK (the overwritten
// trajectory) and dA/dB from P = [dA | dB], each summed over di in ascending order.
kernel void gated_delta_scan_reduce(
    device const float* PK [[buffer(0)]], device const float* P [[buffer(1)]], device float* dK [[buffer(2)]],
    device float* dA [[buffer(3)]], device float* dB [[buffer(4)]],
    constant int& batch [[buffer(5)]], constant int& seqLen [[buffer(6)]], constant int& modelDim [[buffer(7)]],
    constant int& numHeads [[buffer(8)]], constant int& headDim [[buffer(9)]], uint gid [[thread_position_in_grid]])
{
    int g = (int)gid;
    int n = batch * seqLen * modelDim;
    if (g >= n) return;
    int ki = g % headDim; int h = (g / headDim) % numHeads; int bt = g / modelDim; int t = bt % seqLen; int b = bt / seqLen;
    int trajT = ((b * numHeads + h) * seqLen + t) * headDim * headDim;
    float acc = 0.0f;
    for (int di = 0; di < headDim; di++) acc += PK[trajT + di * headDim + ki];
    dK[g] = acc;
    if (ki == 0) {
        int gIdx = bt * numHeads + h;
        float sa = 0.0f, sb = 0.0f;
        for (int di = 0; di < headDim; di++) { sa += P[gIdx * headDim + di]; sb += P[n + gIdx * headDim + di]; }
        dA[gIdx] = sa;
        dB[gIdx] = sb;
    }
}

// ── Sparse routing & codebook primitives (ISparseRoutingBackend) ──────────────────────────────────
kernel void gather_row_values_forward(
    device const float* valBuf [[buffer(0)]], device const int* idxBuf [[buffer(1)]], device float* outBuf [[buffer(2)]],
    constant int& rows [[buffer(3)]], constant int& cols [[buffer(4)]], constant int& k [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = rows * k;
    if ((int)gid >= total) return;
    int r = (int)gid / k;
    int col = idxBuf[gid];
    outBuf[gid] = (col >= 0 && col < cols) ? valBuf[r * cols + col] : 0.0f;
}

kernel void gather_row_values_backward(
    device const float* gradBuf [[buffer(0)]], device const int* idxBuf [[buffer(1)]], device float* gradValBuf [[buffer(2)]],
    constant int& rows [[buffer(3)]], constant int& cols [[buffer(4)]], constant int& k [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = rows * cols;
    if ((int)gid >= total) return;
    int r = (int)gid / cols;
    int c = (int)gid % cols;
    int kBase = r * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        if (idxBuf[kBase + j] == c) acc += gradBuf[kBase + j];
    }
    gradValBuf[gid] = acc;
}

kernel void topk_weighted_row_mix_forward(
    device const float* tblBuf [[buffer(0)]], device const int* idxBuf [[buffer(1)]],
    device const float* wBuf [[buffer(2)]], device float* outBuf [[buffer(3)]],
    constant int& rows [[buffer(4)]], constant int& numEntries [[buffer(5)]],
    constant int& k [[buffer(6)]], constant int& channels [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int total = rows * channels;
    if ((int)gid >= total) return;
    int r = (int)gid / channels;
    int c = (int)gid % channels;
    int kBase = r * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        int entry = idxBuf[kBase + j];
        if (entry >= 0 && entry < numEntries)
            acc += wBuf[kBase + j] * tblBuf[entry * channels + c];
    }
    outBuf[gid] = acc;
}

kernel void topk_weighted_row_mix_backward_weights(
    device const float* gradBuf [[buffer(0)]], device const float* tblBuf [[buffer(1)]],
    device const int* idxBuf [[buffer(2)]], device float* gradWBuf [[buffer(3)]],
    constant int& rows [[buffer(4)]], constant int& numEntries [[buffer(5)]],
    constant int& k [[buffer(6)]], constant int& channels [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int total = rows * k;
    if ((int)gid >= total) return;
    int r = (int)gid / k;
    int entry = idxBuf[gid];
    float acc = 0.0f;
    if (entry >= 0 && entry < numEntries) {
        int goBase = r * channels;
        int tbBase = entry * channels;
        for (int c = 0; c < channels; c++)
            acc += gradBuf[goBase + c] * tblBuf[tbBase + c];
    }
    gradWBuf[gid] = acc;
}

kernel void sparse_zero_int32(
    device int* buf [[buffer(0)]],
    constant int& offset [[buffer(1)]], constant int& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid < count) buf[(long)offset + (long)gid] = 0;
}

kernel void topk_csr_tile_counts(
    device const int* idxBuf [[buffer(0)]], device int* csrBuf [[buffer(1)]],
    constant int& totalSlots [[buffer(2)]], constant int& numEntries [[buffer(3)]],
    constant int& numTiles [[buffer(4)]], constant int& offTile [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int t = (int)gid;
    if (t >= numTiles || totalSlots <= 0) return;
    int tileSpan = ((totalSlots - 1) / numTiles) + 1;
    if (t > totalSlots / tileSpan) return;
    int start = t * tileSpan;
    int rem = totalSlots - start;
    int end = start + (rem < tileSpan ? rem : tileSpan);
    long rowBase = (long)offTile + (long)t * numEntries;
    for (int s = start; s < end; s++) {
        int e = idxBuf[s];
        if (e >= 0 && e < numEntries)
            csrBuf[rowBase + e] += 1;
    }
}

kernel void topk_csr_entry_prefix(
    device int* csrBuf [[buffer(0)]],
    constant int& numEntries [[buffer(1)]], constant int& numTiles [[buffer(2)]],
    constant int& offTile [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int g = (int)gid;
    if (g >= numEntries) return;
    int acc = 0;
    for (int t = 0; t < numTiles; t++) {
        long idx = (long)offTile + (long)t * numEntries + g;
        int c = csrBuf[idx];
        csrBuf[idx] = acc;
        acc += c;
    }
    csrBuf[g] = acc;
}

kernel void topk_csr_chunk_sums(
    device int* csrBuf [[buffer(0)]],
    constant int& numEntries [[buffer(1)]], constant int& offChunk [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int cIdx = (int)gid;
    if (cIdx >= 256) return;
    int acc = 0;
    if (numEntries > 0) {
        int chunkSpan = ((numEntries - 1) / 256) + 1;
        if (cIdx <= numEntries / chunkSpan) {
            int start = cIdx * chunkSpan;
            int rem = numEntries - start;
            int end = start + (rem < chunkSpan ? rem : chunkSpan);
            for (int g = start; g < end; g++) {
                int c = csrBuf[g];
                csrBuf[g] = acc;
                acc += c;
            }
        }
    }
    csrBuf[offChunk + cIdx] = acc;
}

kernel void topk_csr_chunk_finalize(
    device int* csrBuf [[buffer(0)]],
    constant int& numEntries [[buffer(1)]], constant int& offChunk [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int g = (int)gid;
    if (g > numEntries) return;
    if (g == numEntries) {
        int total = 0;
        for (int m = 0; m < 256; m++) total += csrBuf[offChunk + m];
        csrBuf[numEntries] = total;
        return;
    }
    int chunkSpan = ((numEntries - 1) / 256) + 1;
    int chunkIdx = g / chunkSpan;
    int baseOff = 0;
    for (int m = 0; m < chunkIdx; m++) baseOff += csrBuf[offChunk + m];
    csrBuf[g] += baseOff;
}

kernel void topk_csr_scatter_slots(
    device const int* idxBuf [[buffer(0)]], device int* csrBuf [[buffer(1)]],
    constant int& totalSlots [[buffer(2)]], constant int& numEntries [[buffer(3)]],
    constant int& numTiles [[buffer(4)]], constant int& offSorted [[buffer(5)]],
    constant int& offTile [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int t = (int)gid;
    if (t >= numTiles || totalSlots <= 0) return;
    int tileSpan = ((totalSlots - 1) / numTiles) + 1;
    if (t > totalSlots / tileSpan) return;
    int start = t * tileSpan;
    int rem = totalSlots - start;
    int end = start + (rem < tileSpan ? rem : tileSpan);
    long rowBase = (long)offTile + (long)t * numEntries;
    for (int s = start; s < end; s++) {
        int e = idxBuf[s];
        if (e >= 0 && e < numEntries) {
            int localOff = csrBuf[rowBase + e];
            csrBuf[rowBase + e] = localOff + 1;
            int pos = csrBuf[e] + localOff;
            csrBuf[offSorted + pos] = s;
        }
    }
}

kernel void topk_weighted_row_mix_backward_table(
    device const float* gradBuf [[buffer(0)]], device const float* wBuf [[buffer(1)]],
    device const int* csrBuf [[buffer(2)]], device float* gradTblBuf [[buffer(3)]],
    constant int& numEntries [[buffer(4)]], constant int& k [[buffer(5)]],
    constant int& channels [[buffer(6)]], constant int& offSorted [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    long total = (long)numEntries * channels;
    if ((long)gid >= total) return;
    int g = (int)((long)gid / channels);
    int c = (int)((long)gid % channels);
    int start = csrBuf[g];
    int end = csrBuf[g + 1];
    float acc = 0.0f;
    for (int p = start; p < end; p++) {
        int slot = csrBuf[offSorted + p];
        int r = slot / k;
        acc += wBuf[slot] * gradBuf[(long)r * channels + c];
    }
    gradTblBuf[gid] = acc;
}

kernel void count_indices_accumulate_int32(
    device const int* idxBuf [[buffer(0)]], device atomic_int* cntBuf [[buffer(1)]],
    constant int& length [[buffer(2)]], constant int& numBins [[buffer(3)]],
    constant int& totalThreads [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint bdim [[threads_per_threadgroup]])
{
    threadgroup atomic_int s_counts[256];
    uint uLen = (uint)length;
    uint uStride = (uint)totalThreads;
    if (numBins <= 256) {
        for (int b = (int)tid; b < numBins; b += (int)bdim)
            atomic_store_explicit(&s_counts[b], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)gid < totalThreads) {
            for (uint i = gid; i < uLen; i += uStride) {
                int bin = idxBuf[i];
                if (bin >= 0 && bin < numBins)
                    atomic_fetch_add_explicit(&s_counts[bin], 1, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int b = (int)tid; b < numBins; b += (int)bdim) {
            int c = atomic_load_explicit(&s_counts[b], memory_order_relaxed);
            if (c != 0) atomic_fetch_add_explicit(&cntBuf[b], c, memory_order_relaxed);
        }
    } else {
        if ((int)gid < totalThreads) {
            for (uint i = gid; i < uLen; i += uStride) {
                int bin = idxBuf[i];
                if (bin >= 0 && bin < numBins)
                    atomic_fetch_add_explicit(&cntBuf[bin], 1, memory_order_relaxed);
            }
        }
    }
}

kernel void count_indices_finalize_float(
    device int* cntBuf [[buffer(0)]], constant int& numBins [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid < numBins) {
        int c = cntBuf[gid];
        cntBuf[gid] = as_type<int>((float)c);
    }
}

kernel void minmax_init_int32(
    device int* outBuf [[buffer(0)]], constant int& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= count) return;
    if (gid == 0) outBuf[0] = 2147483647;
    else if (gid == 1) outBuf[1] = (-2147483647 - 1);
}

kernel void minmax_reduce_int32(
    device const int* dataBuf [[buffer(0)]], device atomic_int* outBuf [[buffer(1)]],
    constant int& length [[buffer(2)]], constant int& totalThreads [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup atomic_int s_min;
    threadgroup atomic_int s_max;
    if (tid == 0) {
        atomic_store_explicit(&s_min, 2147483647, memory_order_relaxed);
        atomic_store_explicit(&s_max, (-2147483647 - 1), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)gid < totalThreads) {
        uint uLen = (uint)length;
        uint uStride = (uint)totalThreads;
        int lo = 2147483647;
        int hi = (-2147483647 - 1);
        int anyVal = 0;
        for (uint i = gid; i < uLen; i += uStride) {
            int v = dataBuf[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
            anyVal = 1;
        }
        if (anyVal != 0) {
            atomic_fetch_min_explicit(&s_min, lo, memory_order_relaxed);
            atomic_fetch_max_explicit(&s_max, hi, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int bMin = atomic_load_explicit(&s_min, memory_order_relaxed);
        int bMax = atomic_load_explicit(&s_max, memory_order_relaxed);
        if (bMin <= bMax) {
            atomic_fetch_min_explicit(&outBuf[0], bMin, memory_order_relaxed);
            atomic_fetch_max_explicit(&outBuf[1], bMax, memory_order_relaxed);
        }
    }
}
";
}
