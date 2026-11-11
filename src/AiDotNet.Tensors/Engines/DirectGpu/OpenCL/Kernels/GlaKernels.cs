// Copyright (c) CrowdTrainer LLC. All rights reserved.
// OpenCL kernels for the fused Gated Linear Attention (GLA) sequence scan (issue ooples/AiDotNet#1464).
// Mirrors CpuEngine.GlaScan.cs / HipGlaKernels exactly:
//   S_t[di,ki] = g_t * S_{t-1}[di,ki] + V_t[di] * K_t[ki]
//   O_t[di]    = sum_ki S_t[di,ki] * Q_t[ki]
// One work-item owns (batch b, head h, value-row di), carrying that state row across time.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for the fused GLA scan (forward + BPTT backward).
/// </summary>
internal static class GlaKernels
{
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#define GLA_MAX_HEADDIM 256

// CAS-based float atomic add (OpenCL 1.2 has no native float atomics).
inline void atomic_add_float(__global volatile float* addr, float val) {
    union { unsigned int u; float f; } oldval, newval;
    do {
        oldval.f = *addr;
        newval.f = oldval.f + val;
    } while (atomic_cmpxchg((__global volatile unsigned int*)addr, oldval.u, newval.u) != oldval.u);
}

__kernel void gla_scan_forward(
    __global const float* Q, __global const float* K, __global const float* V, __global const float* G,
    __global float* output,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    int gid = get_global_id(0);
    int total = batch * numHeads * headDim;
    if (gid >= total) return;

    int di = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int b  = gid / (headDim * numHeads);
    int hOff = h * headDim;

    float Srow[GLA_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        float g  = G[(b * seqLen + t) * numHeads + h];
        float vd = V[baseOff + di];
        float o  = 0.0f;
        for (int ki = 0; ki < headDim; ki++) {
            float s = g * Srow[ki] + vd * K[baseOff + ki];
            Srow[ki] = s;
            o += s * Q[baseOff + ki];
        }
        output[baseOff + di] = o;
    }
}

// Checkpointed backward (#gla-ckpt): pass 1 re-runs the forward scan saving the state entering each chunk of
// `chunk` steps (Ckpt, B*H*nChunks*d^2 floats); pass 2 walks chunks in reverse, recomputes the chunk's states into
// Scratch (B*H*chunk*d^2 floats) and runs BPTT over it. chunk = ceil(sqrt(S)) makes both O(sqrt(S)) instead of the
// O(S) state trajectory, at the cost of one extra forward scan.
__kernel void gla_scan_backward_ckpt(
    __global const float* dOut, __global const float* Q, __global const float* K,
    __global const float* V, __global const float* G,
    __global float* Ckpt, __global float* Scratch,
    __global volatile float* dQ, __global volatile float* dK,
    __global volatile float* dV, __global volatile float* dG,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int chunk)
{
    int gid = get_global_id(0);
    int total = batch * numHeads * headDim;
    if (gid >= total) return;

    int di = gid % headDim;
    int h  = (gid / headDim) % numHeads;
    int b  = gid / (headDim * numHeads);
    int hOff = h * headDim;
    long hh = (long)headDim * headDim;
    long bh = (long)b * numHeads + h;
    long rowOff = (long)di * headDim;
    int nChunks = (seqLen + chunk - 1) / chunk;

    float Srow[GLA_MAX_HEADDIM];
    float dSrow[GLA_MAX_HEADDIM];
    for (int ki = 0; ki < headDim; ki++) Srow[ki] = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        if (t % chunk == 0) {
            long ck = (bh * nChunks + t / chunk) * hh + rowOff;
            for (int ki = 0; ki < headDim; ki++) Ckpt[ck + ki] = Srow[ki];
        }
        int baseOff = (b * seqLen + t) * modelDim + hOff;
        float g  = G[(b * seqLen + t) * numHeads + h];
        float vd = V[baseOff + di];
        for (int ki = 0; ki < headDim; ki++) Srow[ki] = g * Srow[ki] + vd * K[baseOff + ki];
    }

    for (int ki = 0; ki < headDim; ki++) dSrow[ki] = 0.0f;
    for (int c = nChunks - 1; c >= 0; c--) {
        int t0 = c * chunk;
        int t1 = min(seqLen, t0 + chunk);
        long ck = (bh * nChunks + c) * hh + rowOff;
        for (int ki = 0; ki < headDim; ki++) Srow[ki] = Ckpt[ck + ki];
        for (int t = t0; t < t1; t++) {
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            float g  = G[(b * seqLen + t) * numHeads + h];
            float vd = V[baseOff + di];
            long sc = (bh * chunk + (t - t0)) * hh + rowOff;
            for (int ki = 0; ki < headDim; ki++) {
                float s = g * Srow[ki] + vd * K[baseOff + ki];
                Srow[ki] = s;
                Scratch[sc + ki] = s;
            }
        }
        for (int t = t1 - 1; t >= t0; t--) {
            int baseOff = (b * seqLen + t) * modelDim + hOff;
            int gOff = (b * seqLen + t) * numHeads + h;
            float g = G[gOff];
            long sc = (bh * chunk + (t - t0)) * hh + rowOff;
            long sp = (t > t0) ? (bh * chunk + (t - 1 - t0)) * hh + rowOff : -1;

            float dOutVal = dOut[baseOff + di];
            for (int ki = 0; ki < headDim; ki++) {
                atomic_add_float(&dQ[baseOff + ki], dOutVal * Scratch[sc + ki]);
                dSrow[ki] += dOutVal * Q[baseOff + ki];
            }

            float vd = V[baseOff + di];
            float dg = 0.0f;
            float dVacc = 0.0f;
            for (int ki = 0; ki < headDim; ki++) {
                float dStv = dSrow[ki];
                float sprev = (sp >= 0) ? Scratch[sp + ki] : Ckpt[ck + ki];
                dg += dStv * sprev;
                atomic_add_float(&dK[baseOff + ki], dStv * vd);
                dVacc += dStv * K[baseOff + ki];
            }
            atomic_add_float(&dV[baseOff + di], dVacc);
            atomic_add_float(&dG[gOff], dg);

            for (int ki = 0; ki < headDim; ki++) dSrow[ki] *= g;
        }
    }
}

// Channels-last causal depthwise conv (ICausalDepthwiseConvBackend): x, y, dy, dx are [rows, seqLen, C]; w, dw are
// [K, C]; tap j reads j steps back. Every output element is written; no atomics, so all three are deterministic.
__kernel void causal_dwconv1d_forward(
    __global const float* x, __global const float* w, __global float* y,
    int rows, int seqLen, int channels, int taps)
{
    long idx = get_global_id(0);
    long total = (long)rows * seqLen * channels;
    if (idx >= total) return;
    int c = (int)(idx % channels);
    int t = (int)((idx / channels) % seqLen);
    int maxJ = min(taps - 1, t);
    float acc = 0.0f;
    for (int j = 0; j <= maxJ; j++) acc += w[j * channels + c] * x[idx - (long)j * channels];
    y[idx] = acc;
}

__kernel void causal_dwconv1d_backward_input(
    __global const float* dy, __global const float* w, __global float* dx,
    int rows, int seqLen, int channels, int taps)
{
    long idx = get_global_id(0);
    long total = (long)rows * seqLen * channels;
    if (idx >= total) return;
    int c = (int)(idx % channels);
    int t = (int)((idx / channels) % seqLen);
    int maxJ = min(taps - 1, seqLen - 1 - t);
    float acc = 0.0f;
    for (int j = 0; j <= maxJ; j++) acc += w[j * channels + c] * dy[idx + (long)j * channels];
    dx[idx] = acc;
}

// One work item per (tap j, channel c), summing every row and step in ascending order.
__kernel void causal_dwconv1d_backward_kernel(
    __global const float* dy, __global const float* x, __global float* dw,
    int rows, int seqLen, int channels, int taps)
{
    int jc = get_global_id(0);
    if (jc >= taps * channels) return;
    int j = jc / channels, c = jc % channels;
    float acc = 0.0f;
    for (int r = 0; r < rows; r++) {
        long rowBase = (long)r * seqLen * channels;
        for (int t = j; t < seqLen; t++)
            acc += dy[rowBase + (long)t * channels + c] * x[rowBase + (long)(t - j) * channels + c];
    }
    dw[jc] = acc;
}

// Sparse routing & codebook primitives (ISparseRoutingBackend): zero atomics, deterministic ascending reductions.
__kernel void gather_row_values_forward(
    __global const float* values, __global const int* indices, __global float* output,
    int rows, int cols, int k)
{
    long gid = get_global_id(0);
    long total = (long)rows * k;
    if (gid >= total) return;
    int r = (int)(gid / k);
    int col = indices[gid];
    output[gid] = (col >= 0 && col < cols) ? values[(long)r * cols + col] : 0.0f;
}

__kernel void gather_row_values_backward(
    __global const float* gradOut, __global const int* indices, __global float* gradValues,
    int rows, int cols, int k)
{
    long gid = get_global_id(0);
    long total = (long)rows * cols;
    if (gid >= total) return;
    int r = (int)(gid / cols);
    int c = (int)(gid % cols);
    long kBase = (long)r * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        if (indices[kBase + j] == c) acc += gradOut[kBase + j];
    }
    gradValues[gid] = acc;
}

__kernel void topk_weighted_row_mix_forward(
    __global const float* table, __global const int* indices, __global const float* weights, __global float* output,
    int rows, int numEntries, int k, int channels)
{
    long gid = get_global_id(0);
    long total = (long)rows * channels;
    if (gid >= total) return;
    int r = (int)(gid / channels);
    int c = (int)(gid % channels);
    long kBase = (long)r * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        int entry = indices[kBase + j];
        if (entry >= 0 && entry < numEntries)
            acc += weights[kBase + j] * table[(long)entry * channels + c];
    }
    output[gid] = acc;
}

__kernel void topk_weighted_row_mix_backward_weights(
    __global const float* gradOut, __global const float* table, __global const int* indices, __global float* gradWeights,
    int rows, int numEntries, int k, int channels)
{
    long gid = get_global_id(0);
    long total = (long)rows * k;
    if (gid >= total) return;
    int r = (int)(gid / k);
    int entry = indices[gid];
    float acc = 0.0f;
    if (entry >= 0 && entry < numEntries) {
        long goBase = (long)r * channels;
        long tbBase = (long)entry * channels;
        for (int c = 0; c < channels; c++)
            acc += gradOut[goBase + c] * table[tbBase + c];
    }
    gradWeights[gid] = acc;
}

__kernel void topk_weighted_row_mix_backward_table(
    __global const float* gradOut, __global const int* indices, __global const float* weights, __global float* gradTable,
    int rows, int numEntries, int k, int channels)
{
    long gid = get_global_id(0);
    long total = (long)numEntries * channels;
    if (gid >= total) return;
    int g = (int)(gid / channels);
    int c = (int)(gid % channels);
    float acc = 0.0f;
    for (int r = 0; r < rows; r++) {
        long kBase = (long)r * k;
        for (int j = 0; j < k; j++) {
            if (indices[kBase + j] == g)
                acc += weights[kBase + j] * gradOut[(long)r * channels + c];
        }
    }
    gradTable[gid] = acc;
}

__kernel void count_indices_int32(
    __global const int* indices, __global float* counts, int length, int numBins)
{
    int gid = get_global_id(0);
    if (gid >= numBins) return;
    int cnt = 0;
    for (int i = 0; i < length; i++) {
        if (indices[i] == gid) cnt++;
    }
    counts[gid] = (float)cnt;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "gla_scan_forward",
            "gla_scan_backward_ckpt",
            "causal_dwconv1d_forward",
            "causal_dwconv1d_backward_input",
            "causal_dwconv1d_backward_kernel",
            "gather_row_values_forward",
            "gather_row_values_backward",
            "topk_weighted_row_mix_forward",
            "topk_weighted_row_mix_backward_weights",
            "topk_weighted_row_mix_backward_table",
            "count_indices_int32"
        };
    }
}
