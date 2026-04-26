// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for native Complex<T> tensor operations.
// HIP is source-compatible with CUDA device code.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipComplexKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "split_complex_multiply", "split_complex_conjugate", "split_complex_magnitude",
        "split_complex_magnitude_squared", "split_complex_phase", "split_complex_from_polar",
        "split_complex_scale", "split_complex_add", "split_complex_cross_spectral",
        "split_complex_topk", "softmax_rows",
        // Issue #248 — HRR binding primitives
        "hrr_unit_phase_codebook",
        "hrr_phase_coherence_decode",
        "hrr_bind_accumulate"
    };

    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

#define PI 3.14159265358979323846f

extern ""C"" __global__ void split_complex_multiply(
    const float* aReal, const float* aImag,
    const float* bReal, const float* bImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float ar = aReal[idx], ai = aImag[idx];
    float br = bReal[idx], bi = bImag[idx];
    outReal[idx] = ar * br - ai * bi;
    outImag[idx] = ar * bi + ai * br;
}

extern ""C"" __global__ void split_complex_conjugate(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}

extern ""C"" __global__ void split_complex_magnitude(
    const float* inReal, const float* inImag,
    float* outMag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMag[idx] = sqrtf(re * re + im * im);
}

extern ""C"" __global__ void split_complex_magnitude_squared(
    const float* inReal, const float* inImag,
    float* outMagSq, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}

extern ""C"" __global__ void split_complex_phase(
    const float* inReal, const float* inImag,
    float* outPhase, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outPhase[idx] = atan2f(inImag[idx], inReal[idx]);
}

extern ""C"" __global__ void split_complex_from_polar(
    const float* mag, const float* phase,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float m = mag[idx], p = phase[idx];
    outReal[idx] = m * cosf(p);
    outImag[idx] = m * sinf(p);
}

extern ""C"" __global__ void split_complex_scale(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = inReal[idx] * scalar;
    outImag[idx] = inImag[idx] * scalar;
}

extern ""C"" __global__ void split_complex_add(
    const float* aReal, const float* aImag,
    const float* bReal, const float* bImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}

extern ""C"" __global__ void split_complex_cross_spectral(
    const float* xReal, const float* xImag,
    const float* yReal, const float* yImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xr = xReal[idx], xi = xImag[idx];
    float yr = yReal[idx], yi = yImag[idx];
    outReal[idx] = xr * yr + xi * yi;
    outImag[idx] = xi * yr - xr * yi;
}

extern ""C"" __global__ void split_complex_topk(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag,
    float thresholdMagSq, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    float magSq = re * re + im * im;
    outReal[idx] = (magSq >= thresholdMagSq) ? re : 0.0f;
    outImag[idx] = (magSq >= thresholdMagSq) ? im : 0.0f;
}

extern ""C"" __global__ void softmax_rows(
    const float* input, float* output, int rows, int cols)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;
    const float* rowIn = input + row * cols;
    float* rowOut = output + row * cols;
    float maxVal = -1e30f;
    for (int c = tid; c < cols; c += blockDim.x) maxVal = fmaxf(maxVal, rowIn[c]);
    sdata[tid] = maxVal; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]); __syncthreads(); }
    maxVal = sdata[0];
    float sumExp = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) { float e = expf(rowIn[c]-maxVal); rowOut[c] = e; sumExp += e; }
    sdata[tid] = sumExp; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads(); }
    sumExp = sdata[0];
    for (int c = tid; c < cols; c += blockDim.x) rowOut[c] /= sumExp;
}

// ─── HRR binding primitives (issue #248) ────────────────────────────
// Matches CUDA/OpenCL/Metal/Vulkan/WebGPU — see CudaComplexKernels.cs
// for the hash rationale (32-bit Murmur3 fmix chosen for WebGPU
// compatibility; per-backend GPU determinism preserved, CPU
// xorshift64* path intentionally divergent for single-thread speed).
__device__ __forceinline__ unsigned int hrr_hash(unsigned int seed_u, unsigned int cell_u)
{
    unsigned int z = seed_u * 0x9E3779B9u + cell_u * 0x85EBCA6Bu;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z =  z ^ (z >> 16);
    return z;
}

__device__ __forceinline__ float hrr_phase_from_cell(int seed, long long cellIdx)
{
    unsigned int z = hrr_hash((unsigned int)seed, (unsigned int)cellIdx);
    unsigned int top24 = z >> 8;
    return (float)top24 * (1.0f / 16777216.0f) * 6.28318530717958647692f;
}

extern ""C"" __global__ void hrr_unit_phase_codebook(
    float* outReal, float* outImag,
    int seed, int V, int D, int kPsk, int k)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)V * D;
    if (idx >= total) return;
    float phase = hrr_phase_from_cell(seed, idx);
    if (kPsk) {
        float step = 6.28318530717958647692f / (float)k;
        phase = floorf(phase / step + 0.5f) * step;
    }
    float c, s;
    sincosf(phase, &s, &c);
    outReal[idx] = c;
    outImag[idx] = s;
}

extern ""C"" __global__ void hrr_phase_coherence_decode(
    const float* codesReal, const float* codesImag,
    const float* queryReal, const float* queryImag,
    float* outScores, int V, int D)
{
    extern __shared__ float sdata[];
    int v = blockIdx.x;
    int tid = threadIdx.x;
    if (v >= V) return;
    const float* cR = codesReal + (long long)v * D;
    const float* cI = codesImag + (long long)v * D;
    float acc = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        acc += cR[d] * queryReal[d] + cI[d] * queryImag[d];
    }
    sdata[tid] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) outScores[v] = sdata[0];
}

// nKeys / nVals match the CUDA kernel: codebook row counts passed
// by the host so out-of-range ids are rejected without OOB reads.
// See CudaComplexKernels.hrr_bind_accumulate for the full rationale.
extern ""C"" __global__ void hrr_bind_accumulate(
    const float* keyCodeReal, const float* keyCodeImag,
    const float* valPermCodeReal, const float* valPermCodeImag,
    const int* keyIds, const int* valIds,
    float* memoryReal, float* memoryImag,
    int N, int D, int nKeys, int nVals)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float accR = memoryReal[d];
    float accI = memoryImag[d];
    for (int n = 0; n < N; n++) {
        int kId = keyIds[n];
        int vId = valIds[n];
        // Unsigned-comparison trick rejects both negative and
        // too-large indices in one branch.
        if ((unsigned)kId >= (unsigned)nKeys || (unsigned)vId >= (unsigned)nVals) continue;
        long long kOff = (long long)kId * D;
        long long vOff = (long long)vId * D;
        float ar = keyCodeReal[kOff + d];
        float ai = keyCodeImag[kOff + d];
        float br = valPermCodeReal[vOff + d];
        float bi = valPermCodeImag[vOff + d];
        accR += ar * br - ai * bi;
        accI += ar * bi + ai * br;
    }
    memoryReal[d] = accR;
    memoryImag[d] = accI;
}
";
    }
}
