// Copyright (c) AiDotNet. All rights reserved.
// Dedicated OpenCL kernels for Issue #160 spectral/audio perf operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    internal static class SpectralPerfKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "atan2_elementwise",
            "analytic_signal_mask",
            "normalize_rows_fused",
            "bispectrum_gather",
            "trispectrum_gather",
            "cavity_bounce_inplace",
            "wideband_log_bin_pool",
            "pac_phase_bin_mi",
            "mel_filterbank_apply",
            "mfcc_log1p",
        };

        public static string GetSource()
        {
            return @"
__kernel void atan2_elementwise(__global const float* imag,
                                __global const float* real,
                                __global float* output,
                                int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    output[idx] = atan2(imag[idx], real[idx]);
}

__kernel void analytic_signal_mask(__global const float* specReal,
                                   __global const float* specImag,
                                   __global float* outReal,
                                   __global float* outImag,
                                   int batch, int fftSize, int binLow, int binHigh)
{
    int idx = get_global_id(0);
    int total = batch * fftSize;
    if (idx >= total) return;
    int k = idx % fftSize;
    int halfN = fftSize >> 1;
    float gain;
    if (k == 0 || k == halfN) {
        gain = (k < binLow || k >= binHigh) ? 0.0f : 1.0f;
    } else if (k < halfN) {
        gain = (k < binLow || k >= binHigh) ? 0.0f : 2.0f;
    } else {
        gain = 0.0f;
    }
    outReal[idx] = specReal[idx] * gain;
    outImag[idx] = specImag[idx] * gain;
}

__kernel void normalize_rows_fused(__global const float* input,
                                   __global float* output,
                                   __local float* sdata,
                                   int rows, int cols)
{
    int row = get_group_id(0);
    if (row >= rows) return;
    int tid = get_local_id(0);
    int blockDim = get_local_size(0);
    int rowOff = row * cols;

    float local_acc = 0.0f;
    for (int c = tid; c < cols; c += blockDim) {
        float v = input[rowOff + c];
        local_acc += v * v;
    }
    sdata[tid] = local_acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = blockDim >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float invNorm = 0.0f;
    if (sdata[0] > 0.0f) invNorm = rsqrt(sdata[0]);

    for (int c = tid; c < cols; c += blockDim) {
        output[rowOff + c] = input[rowOff + c] * invNorm;
    }
}

__kernel void bispectrum_gather(__global const float* specReal,
                                __global const float* specImag,
                                __global float* outReal,
                                __global float* outImag,
                                int maxF1, int maxF2)
{
    int idx = get_global_id(0);
    int total = maxF1 * maxF2;
    if (idx >= total) return;
    int f1 = idx / maxF2;
    int f2 = idx % maxF2;
    int sumIdx = f1 + f2;
    float ar = specReal[f1], ai = specImag[f1];
    float br = specReal[f2], bi = specImag[f2];
    float cr = specReal[sumIdx], ci = -specImag[sumIdx];
    float abr = ar * br - ai * bi;
    float abi = ar * bi + ai * br;
    outReal[idx] = abr * cr - abi * ci;
    outImag[idx] = abr * ci + abi * cr;
}

__kernel void trispectrum_gather(__global const float* specReal,
                                 __global const float* specImag,
                                 __global float* outReal,
                                 __global float* outImag,
                                 int maxF1, int maxF2, int maxF3)
{
    int idx = get_global_id(0);
    int total = maxF1 * maxF2 * maxF3;
    if (idx >= total) return;
    int f1 = idx / (maxF2 * maxF3);
    int rem = idx - f1 * maxF2 * maxF3;
    int f2 = rem / maxF3;
    int f3 = rem - f2 * maxF3;
    int sumIdx = f1 + f2 + f3;
    float ar = specReal[f1], ai = specImag[f1];
    float br = specReal[f2], bi = specImag[f2];
    float cr = specReal[f3], ci = specImag[f3];
    float dr = specReal[sumIdx], di = -specImag[sumIdx];
    float t1r = ar * br - ai * bi;
    float t1i = ar * bi + ai * br;
    float t2r = t1r * cr - t1i * ci;
    float t2i = t1r * ci + t1i * cr;
    outReal[idx] = t2r * dr - t2i * di;
    outImag[idx] = t2r * di + t2i * dr;
}

__kernel void cavity_bounce_inplace(__global float* workReal,
                                    __global float* workImag,
                                    int total, float invN)
{
    int idx = get_global_id(0);
    if (idx >= total) return;
    float r = workReal[idx] * invN;
    r = fmin(fmax(r, -20.0f), 20.0f);
    workReal[idx] = tanh(r);
    workImag[idx] = 0.0f;
}

__kernel void wideband_log_bin_pool(__global const float* magBuf,
                                    __global float* output,
                                    int totalSegBatch, int fftSize, int numBins, int usable)
{
    int outIdx = get_global_id(0);
    int total = totalSegBatch * numBins;
    if (outIdx >= total) return;
    int seg = outIdx / numBins;
    int k = outIdx % numBins;
    float r0 = (float)k / (float)numBins;
    float r1 = (float)(k + 1) / (float)numBins;
    int binStart = 1 + (int)(r0 * r0 * (float)(usable - 1));
    int binEnd = 1 + (int)(r1 * r1 * (float)(usable - 1));
    if (binEnd <= binStart) binEnd = binStart + 1;
    if (binEnd > usable) binEnd = usable;
    int magOff = seg * fftSize;
    float sum = 0.0f; int cnt = 0;
    for (int i = binStart; i < binEnd; i++) {
        sum += magBuf[magOff + i];
        cnt++;
    }
    float avg = (cnt > 0) ? (sum / (float)cnt) : 0.0f;
    output[outIdx] = log1p(avg);
}

// PAC: one work-group per batch, single-thread histogram + reduction.
// Avoids float atomics (not portable in OpenCL 1.x).
__kernel void pac_phase_bin_mi(__global const float* thetaPhase,
                               __global const float* gammaAmp,
                               __global float* output,
                               int batch, int numSamples, int numGammaBands, int gammaIdx)
{
    int b = get_global_id(0);
    if (b >= batch) return;
    const int NUM_PHASE_BINS = 18;
    float binSum[18];
    float binCount[18];
    for (int k = 0; k < NUM_PHASE_BINS; k++) { binSum[k] = 0.0f; binCount[k] = 0.0f; }
    int sampleOff = b * numSamples;
    int gammaOff = (gammaIdx * batch + b) * numSamples;
    for (int i = 0; i < numSamples; i++) {
        float phase = thetaPhase[sampleOff + i];
        float amp = gammaAmp[gammaOff + i];
        float fbin = (phase + 3.14159265358979f) / (2.0f * 3.14159265358979f) * (float)NUM_PHASE_BINS;
        int bin = (int)fbin;
        if (bin < 0) bin = 0;
        if (bin >= NUM_PHASE_BINS) bin = NUM_PHASE_BINS - 1;
        binSum[bin] += amp;
        binCount[bin] += 1.0f;
    }
    float totalAmp = 0.0f;
    for (int k = 0; k < NUM_PHASE_BINS; k++) {
        float avg = (binCount[k] > 0.0f) ? (binSum[k] / binCount[k]) : 0.0f;
        totalAmp += avg;
    }
    float mi = 0.0f;
    if (totalAmp > 0.0f) {
        float entropy = 0.0f;
        for (int k = 0; k < NUM_PHASE_BINS; k++) {
            float avg = (binCount[k] > 0.0f) ? (binSum[k] / binCount[k]) : 0.0f;
            float p = avg / totalAmp;
            if (p > 1e-12f) entropy -= p * log(p);
        }
        mi = (log((float)NUM_PHASE_BINS) - entropy) / log((float)NUM_PHASE_BINS);
    }
    output[b * numGammaBands + gammaIdx] = mi;
}

__kernel void mel_filterbank_apply(__global const float* powerSpec,
                                   __global const float* melFilters,
                                   __global float* melEnergy,
                                   int totalSegBatch, int specBins, int melBins)
{
    int outIdx = get_global_id(0);
    int total = totalSegBatch * melBins;
    if (outIdx >= total) return;
    int seg = outIdx / melBins;
    int m = outIdx % melBins;
    int powerOff = seg * specBins;
    int filtOff = m * specBins;
    float sum = 0.0f;
    for (int i = 0; i < specBins; i++)
        sum += powerSpec[powerOff + i] * melFilters[filtOff + i];
    melEnergy[outIdx] = sum;
}

__kernel void mfcc_log1p(__global const float* input,
                         __global float* output,
                         int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    output[idx] = log1p(input[idx]);
}
";
        }
    }
}
