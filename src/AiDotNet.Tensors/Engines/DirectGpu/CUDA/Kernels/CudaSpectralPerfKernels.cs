// Copyright (c) AiDotNet. All rights reserved.
// Dedicated CUDA kernels for Issue #160 spectral/audio perf operations.
// All kernels are fully GPU-resident — no host-side loops.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    internal static class CudaSpectralPerfKernels
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
#include <math.h>

// =================================================================
// Atan2 element-wise: output[i] = atan2(imag[i], real[i])
// =================================================================
extern ""C"" __global__ __launch_bounds__(256)
void atan2_elementwise(const float* __restrict__ imag,
                       const float* __restrict__ real,
                       float* __restrict__ output,
                       int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = atan2f(imag[idx], real[idx]);
}

// =================================================================
// Analytic signal Hilbert mask: apply gain (0, 1, or 2) per frequency bin.
// Bins are organized as [batch, fftSize] flat. For each bin k:
//   k == 0 or k == fftSize/2: gain = (k in [binLow, binHigh)) ? 1 : 0
//   k < fftSize/2:            gain = (k in [binLow, binHigh)) ? 2 : 0
//   k > fftSize/2:            gain = 0
// Multiplies (specReal, specImag) by gain in-place into (outReal, outImag).
// =================================================================
extern ""C"" __global__ __launch_bounds__(256)
void analytic_signal_mask(const float* __restrict__ specReal,
                          const float* __restrict__ specImag,
                          float* __restrict__ outReal,
                          float* __restrict__ outImag,
                          int batch, int fftSize, int binLow, int binHigh)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

// =================================================================
// Per-row L2 normalize. One block per row; threads cooperate on
// sum-of-squares reduction, then divide all elements by sqrt(sumSq).
// =================================================================
extern ""C"" __global__ __launch_bounds__(256)
void normalize_rows_fused(const float* __restrict__ input,
                          float* __restrict__ output,
                          int rows, int cols)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int rowOff = row * cols;

    // Phase 1: sum of squares
    float local = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = input[rowOff + c];
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float invNorm = 0.0f;
    if (sdata[0] > 0.0f) invNorm = rsqrtf(sdata[0]);

    // Phase 2: write normalized output
    for (int c = tid; c < cols; c += blockDim.x) {
        output[rowOff + c] = input[rowOff + c] * invNorm;
    }
}

// =================================================================
// Bispectrum gather: B(f1, f2) = X(f1) * X(f2) * conj(X(f1+f2))
// One thread per output element. Output shape [maxF1, maxF2] complex.
// =================================================================
extern ""C"" __global__ __launch_bounds__(256)
void bispectrum_gather(const float* __restrict__ specReal,
                       const float* __restrict__ specImag,
                       float* __restrict__ outReal,
                       float* __restrict__ outImag,
                       int maxF1, int maxF2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = maxF1 * maxF2;
    if (idx >= total) return;
    int f1 = idx / maxF2;
    int f2 = idx % maxF2;
    int sumIdx = f1 + f2;

    float ar = specReal[f1], ai = specImag[f1];
    float br = specReal[f2], bi = specImag[f2];
    float cr = specReal[sumIdx], ci = -specImag[sumIdx]; // conjugate

    // (ar+i*ai) * (br+i*bi) = (ar*br - ai*bi) + i*(ar*bi + ai*br)
    float abr = ar * br - ai * bi;
    float abi = ar * bi + ai * br;
    // (abr+i*abi) * (cr+i*ci)
    outReal[idx] = abr * cr - abi * ci;
    outImag[idx] = abr * ci + abi * cr;
}

// =================================================================
// Trispectrum gather: T(f1,f2,f3) = X(f1)*X(f2)*X(f3)*conj(X(f1+f2+f3))
// =================================================================
extern ""C"" __global__ __launch_bounds__(256)
void trispectrum_gather(const float* __restrict__ specReal,
                        const float* __restrict__ specImag,
                        float* __restrict__ outReal,
                        float* __restrict__ outImag,
                        int maxF1, int maxF2, int maxF3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

// =================================================================
// Cavity bounce in-place: applies the per-bounce nonlinearity to a
// time-domain signal that was just IFFT'd. Computes tanh(real / N) for
// real part, zeros imag part. N is fftSize for IFFT normalization.
// Designed to fuse the post-IFFT scale + tanh + imag-zero work.
// =================================================================
extern ""C"" __global__ __launch_bounds__(256)
void cavity_bounce_inplace(float* __restrict__ workReal,
                           float* __restrict__ workImag,
                           int total, float invN)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float r = workReal[idx] * invN;
    // Clamp to avoid NaN on extreme inputs
    r = fminf(fmaxf(r, -20.0f), 20.0f);
    workReal[idx] = tanhf(r);
    workImag[idx] = 0.0f;
}

// =================================================================
// Wideband log-bin pool: per (batch, segment), pool magnitudes into
// numBins logarithmically-spaced bins, take log(1+avg).
// magBuf shape: [totalSegBatch, fftSize]; output: [totalSegBatch, numBins].
// =================================================================
extern ""C"" __global__ __launch_bounds__(256)
void wideband_log_bin_pool(const float* __restrict__ magBuf,
                           float* __restrict__ output,
                           int totalSegBatch, int fftSize, int numBins, int usable)
{
    int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = totalSegBatch * numBins;
    if (outIdx >= total) return;
    int seg = outIdx / numBins;
    int k = outIdx % numBins;

    // Logarithmic bin layout: binStart = 1 + (k/numBins)^2 * (usable-1)
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
    output[outIdx] = log1pf(avg);
}

// =================================================================
// PAC phase-binned modulation index. For each (batch, gammaBand):
// uses 18 phase bins; computes KL-divergence of amplitude distribution
// from uniform, normalized by log(numBins).
// thetaPhase: [batch, numSamples], gammaAmp: [batch, numSamples]
// output: [batch, numGammaBands], one MI value per (batch, band).
// One block per (batch, gammaBand) pair; threads cooperate on histogram + reduction.
// =================================================================
extern ""C"" __global__ __launch_bounds__(256)
void pac_phase_bin_mi(const float* __restrict__ thetaPhase,
                      const float* __restrict__ gammaAmp,
                      float* __restrict__ output,
                      int batch, int numSamples, int numGammaBands, int gammaIdx)
{
    extern __shared__ float sdata[];
    const int NUM_PHASE_BINS = 18;
    int b = blockIdx.x;
    if (b >= batch) return;
    int tid = threadIdx.x;

    // sdata layout: [NUM_PHASE_BINS sums | NUM_PHASE_BINS counts]
    if (tid < NUM_PHASE_BINS) {
        sdata[tid] = 0.0f;
        sdata[NUM_PHASE_BINS + tid] = 0.0f;
    }
    __syncthreads();

    // Histogram: each thread accumulates partial sums into shared memory via atomicAdd
    int sampleOff = b * numSamples;
    int gammaOff = (gammaIdx * batch + b) * numSamples;
    for (int i = tid; i < numSamples; i += blockDim.x) {
        float phase = thetaPhase[sampleOff + i];
        float amp = gammaAmp[gammaOff + i];
        float fbin = (phase + 3.14159265358979f) / (2.0f * 3.14159265358979f) * (float)NUM_PHASE_BINS;
        int bin = (int)fbin;
        if (bin < 0) bin = 0;
        if (bin >= NUM_PHASE_BINS) bin = NUM_PHASE_BINS - 1;
        atomicAdd(&sdata[bin], amp);
        atomicAdd(&sdata[NUM_PHASE_BINS + bin], 1.0f);
    }
    __syncthreads();

    // Compute MI on thread 0
    if (tid == 0) {
        float totalAmp = 0.0f;
        for (int k = 0; k < NUM_PHASE_BINS; k++) {
            float c = sdata[NUM_PHASE_BINS + k];
            float avg = (c > 0.0f) ? (sdata[k] / c) : 0.0f;
            totalAmp += avg;
        }
        float mi = 0.0f;
        if (totalAmp > 0.0f) {
            float entropy = 0.0f;
            for (int k = 0; k < NUM_PHASE_BINS; k++) {
                float c = sdata[NUM_PHASE_BINS + k];
                float avg = (c > 0.0f) ? (sdata[k] / c) : 0.0f;
                float p = avg / totalAmp;
                if (p > 1e-12f) entropy -= p * logf(p);
            }
            mi = (logf((float)NUM_PHASE_BINS) - entropy) / logf((float)NUM_PHASE_BINS);
        }
        output[b * numGammaBands + gammaIdx] = mi;
    }
}

// =================================================================
// Mel filterbank apply: for each (segment, melBin), sum power[i] * melFilter[melBin*specBins + i].
// powerSpec: [totalSegBatch, specBins], melFilters: [melBins, specBins], output: [totalSegBatch, melBins].
// =================================================================
extern ""C"" __global__ __launch_bounds__(256)
void mel_filterbank_apply(const float* __restrict__ powerSpec,
                          const float* __restrict__ melFilters,
                          float* __restrict__ melEnergy,
                          int totalSegBatch, int specBins, int melBins)
{
    int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
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

// =================================================================
// MFCC log1p: log(1 + e) compression
// =================================================================
extern ""C"" __global__ __launch_bounds__(256)
void mfcc_log1p(const float* __restrict__ input,
                float* __restrict__ output,
                int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = log1pf(input[idx]);
}
";
        }
    }
}
