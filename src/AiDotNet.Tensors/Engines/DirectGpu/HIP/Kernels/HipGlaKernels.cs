// Copyright (c) CrowdTrainer LLC. All rights reserved.
// HIP kernels for the fused Gated Linear Attention (GLA) sequence scan (issue ooples/AiDotNet#1464).
// Mirrors the CPU reference in CpuEngine.GlaScan.cs exactly:
//   per head, scalar-per-head gate g_t, matrix state S[di,ki]:
//     S_t[di,ki] = g_t * S_{t-1}[di,ki] + V_t[di] * K_t[ki]
//     O_t[di]    = sum_ki S_t[di,ki] * Q_t[ki]   (query contracts the key dim)
// One thread owns (batch b, head h, value-row di) and carries that row of the state
// (headDim floats) through the sequential time loop.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for the fused GLA scan (forward + BPTT backward) on AMD GPUs.
/// </summary>
internal static class HipGlaKernels
{
    // Max headDim supported by the register-resident state row. Matches the CPU
    // path's expectation that headDim = modelDim / numHeads is modest.
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>

#define GLA_MAX_HEADDIM 256

// Forward: output[b,t,h,di] = sum_ki S_t[di,ki] * Q[b,t,h,ki].
// Thread = (b, h, di). Carries state row S[di,ki] (ki in [0,headDim)) across time.
extern ""C"" __global__ __launch_bounds__(1024) void gla_scan_forward(
    const float* Q, const float* K, const float* V, const float* G,
    float* output,
    int batch, int seqLen, int modelDim, int numHeads, int headDim)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
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

// Checkpointed BPTT backward. The previous recompute + backward pair kept the whole state
// trajectory, batch*heads*seqLen*headDim^2 floats (2.1 GB at batch 16, seqLen 256, 8 heads of 128).
// This kernel stores the state only at chunk boundaries (Ckpt, batch*heads*nChunks*headDim^2) and
// recomputes one chunk at a time into Scratch (batch*heads*chunk*headDim^2), so memory is
// O(seqLen/chunk + chunk) instead of O(seqLen); chunk = ceil(sqrt(seqLen)) gives O(sqrt(seqLen)).
// Extra work: one forward pass, the same as the recompute kernel it replaces. Each thread reads and
// writes only its own state row, so Ckpt/Scratch need no synchronization.
extern ""C"" __global__ __launch_bounds__(1024) void gla_scan_backward_ckpt(
    const float* dOut, const float* Q, const float* K, const float* V, const float* G,
    float* Ckpt, float* Scratch,
    float* dQ, float* dK, float* dV, float* dG,
    int batch, int seqLen, int modelDim, int numHeads, int headDim, int chunk)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
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

    // Pass 1: forward scan, saving S_{t0-1} (the state entering each chunk) at every chunk start.
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

    // Pass 2: chunks in reverse; recompute the chunk's states, then run BPTT over it.
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
            // S_{t-1}: the previous scratch row inside the chunk, else the chunk's checkpoint.
            const float* sprevRow = (t > t0) ? (Scratch + (bh * chunk + (t - 1 - t0)) * hh + rowOff)
                                             : (Ckpt + ck);

            float dOutVal = dOut[baseOff + di];
            for (int ki = 0; ki < headDim; ki++) {
                atomicAdd(&dQ[baseOff + ki], dOutVal * Scratch[sc + ki]);
                dSrow[ki] += dOutVal * Q[baseOff + ki];
            }

            float vd = V[baseOff + di];
            float dg = 0.0f;
            float dVacc = 0.0f;
            for (int ki = 0; ki < headDim; ki++) {
                float dStv = dSrow[ki];
                dg += dStv * sprevRow[ki];
                atomicAdd(&dK[baseOff + ki], dStv * vd);
                dVacc += dStv * K[baseOff + ki];
            }
            atomicAdd(&dV[baseOff + di], dVacc);
            atomicAdd(&dG[gOff], dg);

            for (int ki = 0; ki < headDim; ki++) dSrow[ki] *= g;
        }
    }
}

// Channels-last causal depthwise conv (ICausalDepthwiseConvBackend): x, y, dy, dx are [rows, seqLen, C]; w, dw are
// [K, C]; tap j reads j steps back. Every output element is written; no atomics, so all three are deterministic.
extern ""C"" __global__ void causal_dwconv1d_forward(
    const float* x, const float* w, float* y, int rows, int seqLen, int channels, int taps)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * seqLen * channels;
    if (idx >= total) return;
    int c = (int)(idx % channels);
    int t = (int)((idx / channels) % seqLen);
    int maxJ = min(taps - 1, t);
    float acc = 0.0f;
    for (int j = 0; j <= maxJ; j++) acc += w[j * channels + c] * x[idx - (long)j * channels];
    y[idx] = acc;
}

extern ""C"" __global__ void causal_dwconv1d_backward_input(
    const float* dy, const float* w, float* dx, int rows, int seqLen, int channels, int taps)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * seqLen * channels;
    if (idx >= total) return;
    int c = (int)(idx % channels);
    int t = (int)((idx / channels) % seqLen);
    int maxJ = min(taps - 1, seqLen - 1 - t);
    float acc = 0.0f;
    for (int j = 0; j <= maxJ; j++) acc += w[j * channels + c] * dy[idx + (long)j * channels];
    dx[idx] = acc;
}

// Block (32 channel lanes x 8 groups) per (32-channel tile, tap): group g sums steps g, g + 8, ... of the flattened
// [rows * seqLen] grid (lanes read consecutive channels, so loads coalesce), then lane 0..31 of group 0 adds the 8
// partials in a fixed order. Launch: grid (ceil(C / 32), K), block (32, 8).
extern ""C"" __global__ void causal_dwconv1d_backward_kernel(
    const float* dy, const float* x, float* dw, int rows, int seqLen, int channels, int taps)
{
    __shared__ float partial[8][32];
    int lane = threadIdx.x, group = threadIdx.y;
    int c = blockIdx.x * 32 + lane;
    int j = blockIdx.y;
    float acc = 0.0f;
    if (c < channels) {
        long steps = (long)rows * seqLen;
        for (long rt = group; rt < steps; rt += 8) {
            int t = (int)(rt % seqLen);
            if (t >= j) acc += dy[rt * channels + c] * x[(rt - j) * channels + c];
        }
    }
    partial[group][lane] = acc;
    __syncthreads();
    if (group == 0 && c < channels) {
        float s = 0.0f;
        for (int g = 0; g < 8; g++) s += partial[g][lane];
        dw[j * channels + c] = s;
    }
}

// Sparse routing & codebook primitives (ISparseRoutingBackend): zero atomics, deterministic ascending reductions.
extern ""C"" __global__ void gather_row_values_forward(
    const float* values, const int* indices, float* output, int rows, int cols, int k)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * k;
    if (gid >= total) return;
    int r = (int)(gid / k);
    int col = indices[gid];
    output[gid] = (col >= 0 && col < cols) ? values[(long)r * cols + col] : 0.0f;
}

extern ""C"" __global__ void gather_row_values_backward(
    const float* gradOut, const int* indices, float* gradValues, int rows, int cols, int k)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
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

extern ""C"" __global__ void topk_weighted_row_mix_forward(
    const float* table, const int* indices, const float* weights, float* output,
    int rows, int numEntries, int k, int channels)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
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

extern ""C"" __global__ void topk_weighted_row_mix_backward_weights(
    const float* gradOut, const float* table, const int* indices, float* gradWeights,
    int rows, int numEntries, int k, int channels)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
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

extern ""C"" __global__ void sparse_zero_int32(int* buf, int offset, int count)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) buf[(long)offset + gid] = 0;
}

extern ""C"" __global__ void topk_csr_tile_counts(
    const int* indices, int* csr, int totalSlots, int numEntries, int numTiles, int offTile)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numTiles || totalSlots <= 0) return;
    int tileSpan = ((totalSlots - 1) / numTiles) + 1;
    if (t > totalSlots / tileSpan) return;
    int start = t * tileSpan;
    int rem = totalSlots - start;
    int end = start + (rem < tileSpan ? rem : tileSpan);
    long rowBase = (long)offTile + (long)t * numEntries;
    for (int s = start; s < end; s++) {
        int e = indices[s];
        if (e >= 0 && e < numEntries)
            csr[rowBase + e] += 1;
    }
}

extern ""C"" __global__ void topk_csr_entry_prefix(
    int* csr, int numEntries, int numTiles, int offTile)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= numEntries) return;
    int acc = 0;
    for (int t = 0; t < numTiles; t++) {
        long idx = (long)offTile + (long)t * numEntries + g;
        int c = csr[idx];
        csr[idx] = acc;
        acc += c;
    }
    csr[g] = acc;
}

extern ""C"" __global__ void topk_csr_chunk_sums(
    int* csr, int numEntries, int offChunk)
{
    int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cIdx >= 256) return;
    int acc = 0;
    if (numEntries > 0) {
        int chunkSpan = ((numEntries - 1) / 256) + 1;
        if (cIdx <= numEntries / chunkSpan) {
            int start = cIdx * chunkSpan;
            int rem = numEntries - start;
            int end = start + (rem < chunkSpan ? rem : chunkSpan);
            for (int g = start; g < end; g++) {
                int c = csr[g];
                csr[g] = acc;
                acc += c;
            }
        }
    }
    csr[offChunk + cIdx] = acc;
}

extern ""C"" __global__ void topk_csr_chunk_finalize(
    int* csr, int numEntries, int offChunk)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g > numEntries) return;
    if (g == numEntries) {
        int total = 0;
        for (int m = 0; m < 256; m++) total += csr[offChunk + m];
        csr[numEntries] = total;
        return;
    }
    int chunkSpan = ((numEntries - 1) / 256) + 1;
    int chunkIdx = g / chunkSpan;
    int baseOff = 0;
    for (int m = 0; m < chunkIdx; m++) baseOff += csr[offChunk + m];
    csr[g] += baseOff;
}

extern ""C"" __global__ void topk_csr_scatter_slots(
    const int* indices, int* csr, int totalSlots, int numEntries, int numTiles, int offSorted, int offTile)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numTiles || totalSlots <= 0) return;
    int tileSpan = ((totalSlots - 1) / numTiles) + 1;
    if (t > totalSlots / tileSpan) return;
    int start = t * tileSpan;
    int rem = totalSlots - start;
    int end = start + (rem < tileSpan ? rem : tileSpan);
    long rowBase = (long)offTile + (long)t * numEntries;
    for (int s = start; s < end; s++) {
        int e = indices[s];
        if (e >= 0 && e < numEntries) {
            int localOff = csr[rowBase + e];
            csr[rowBase + e] = localOff + 1;
            int pos = csr[e] + localOff;
            csr[offSorted + pos] = s;
        }
    }
}

extern ""C"" __global__ void topk_weighted_row_mix_backward_table(
    const float* gradOut, const float* weights, const int* csr, float* gradTable,
    int numEntries, int k, int channels, int offSorted)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)numEntries * channels;
    if (gid >= total) return;
    int g = (int)(gid / channels);
    int c = (int)(gid % channels);
    int start = csr[g];
    int end = csr[g + 1];
    float acc = 0.0f;
    for (int p = start; p < end; p++) {
        int slot = csr[offSorted + p];
        int r = slot / k;
        acc += weights[slot] * gradOut[(long)r * channels + c];
    }
    gradTable[gid] = acc;
}

extern ""C"" __global__ void count_indices_accumulate_int32(
    const int* indices, int* countsInt, int length, int numBins, int totalThreads)
{
    __shared__ int s_counts[256];
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    long gid = (long)blockIdx.x * bdim + tid;
    unsigned int uLen = (unsigned int)length;
    unsigned int uStride = (unsigned int)totalThreads;
    if (numBins <= 256) {
        for (int b = tid; b < numBins; b += bdim) s_counts[b] = 0;
        __syncthreads();
        if (gid < totalThreads) {
            for (unsigned int i = (unsigned int)gid; i < uLen; i += uStride) {
                int bin = indices[i];
                if (bin >= 0 && bin < numBins) atomicAdd(&s_counts[bin], 1);
            }
        }
        __syncthreads();
        for (int b = tid; b < numBins; b += bdim) {
            int c = s_counts[b];
            if (c != 0) atomicAdd(&countsInt[b], c);
        }
    } else {
        if (gid < totalThreads) {
            for (unsigned int i = (unsigned int)gid; i < uLen; i += uStride) {
                int bin = indices[i];
                if (bin >= 0 && bin < numBins) atomicAdd(&countsInt[bin], 1);
            }
        }
    }
}

extern ""C"" __global__ void count_indices_finalize_float(int* countsBuf, int numBins)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < numBins) {
        int c = countsBuf[gid];
        ((float*)countsBuf)[gid] = (float)c;
    }
}

extern ""C"" __global__ void minmax_init_int32(int* outMinMax)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0) outMinMax[0] = 2147483647;
    else if (gid == 1) outMinMax[1] = (-2147483647 - 1);
}

extern ""C"" __global__ void minmax_reduce_int32(
    const int* data, int* outMinMax, int length, int totalThreads)
{
    __shared__ int s_min;
    __shared__ int s_max;
    int tid = threadIdx.x;
    long gid = (long)blockIdx.x * blockDim.x + tid;
    if (tid == 0) {
        s_min = 2147483647;
        s_max = (-2147483647 - 1);
    }
    __syncthreads();
    if (gid < totalThreads) {
        unsigned int uLen = (unsigned int)length;
        unsigned int uStride = (unsigned int)totalThreads;
        int lo = 2147483647;
        int hi = (-2147483647 - 1);
        int any = 0;
        for (unsigned int i = (unsigned int)gid; i < uLen; i += uStride) {
            int v = data[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
            any = 1;
        }
        if (any != 0) {
            atomicMin(&s_min, lo);
            atomicMax(&s_max, hi);
        }
    }
    __syncthreads();
    if (tid == 0 && s_min <= s_max) {
        atomicMin(&outMinMax[0], s_min);
        atomicMax(&outMinMax[1], s_max);
    }
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
            "sparse_zero_int32",
            "topk_csr_tile_counts",
            "topk_csr_entry_prefix",
            "topk_csr_chunk_sums",
            "topk_csr_chunk_finalize",
            "topk_csr_scatter_slots",
            "topk_weighted_row_mix_backward_table",
            "count_indices_accumulate_int32",
            "count_indices_finalize_float",
            "minmax_init_int32",
            "minmax_reduce_int32"
        };
    }
}
