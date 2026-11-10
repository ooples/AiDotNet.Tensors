// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CUDA kernels for the fused Gated Linear Attention (GLA) sequence scan (issue ooples/AiDotNet#1464).
// Mirrors CpuEngine.GlaScan.cs exactly (see HipGlaKernels for the shared design notes):
//   S_t[di,ki] = g_t * S_{t-1}[di,ki] + V_t[di] * K_t[ki]
//   O_t[di]    = sum_ki S_t[di,ki] * Q_t[ki]
// One thread owns (batch b, head h, value-row di) and carries that state row across time.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for the fused GLA scan (forward + BPTT backward) on NVIDIA GPUs.
/// </summary>
internal static class CudaGlaKernels
{
    public const int MaxHeadDim = 256;

    public static string GetSource()
    {
        return @"
#include <math.h>

#define GLA_MAX_HEADDIM 256

extern ""C"" __global__ void gla_scan_forward(
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
extern ""C"" __global__ void gla_scan_backward_ckpt(
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
            "causal_dwconv1d_backward_kernel"
        };
    }
}
