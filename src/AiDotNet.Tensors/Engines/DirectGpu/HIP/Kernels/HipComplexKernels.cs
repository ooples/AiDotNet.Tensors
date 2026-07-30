// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for native Complex<T> tensor operations.
// HIP is source-compatible with CUDA device code.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipComplexKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "interleave_complex", "deinterleave_complex",
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

// Pack split real/imag into interleaved [re0,im0,re1,im1,...]. Replaces the dispatch layer's
// two-Copy-per-frequency-bin bridge (~1.06M device copies per batched RFFT).
extern ""C"" __global__ void interleave_complex(
    const float* real, const float* imag, float* interleaved, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    interleaved[2 * idx] = real[idx];
    interleaved[2 * idx + 1] = imag[idx];
}

extern ""C"" __global__ void deinterleave_complex(
    const float* interleaved, float* real, float* imag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    real[idx] = interleaved[2 * idx];
    imag[idx] = interleaved[2 * idx + 1];
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
    int k, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    float magSq = re * re + im * im;
    bool magIsNan = (__float_as_uint(magSq) & 0x7fffffffu) > 0x7f800000u;
    int rank = 0;
    for (int other = 0; other < n; ++other) {
        float otherRe = inReal[other], otherIm = inImag[other];
        float otherMagSq = otherRe * otherRe + otherIm * otherIm;
        bool otherIsNan = (__float_as_uint(otherMagSq) & 0x7fffffffu) > 0x7f800000u;
        bool otherBefore = magIsNan
            ? (!otherIsNan || (otherIsNan && other < idx))
            : (!otherIsNan && (otherMagSq > magSq || (otherMagSq == magSq && other < idx)));
        rank += otherBefore ? 1 : 0;
    }
    bool keep = rank < k;
    outReal[idx] = keep ? re : 0.0f;
    outImag[idx] = keep ? im : 0.0f;
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
    // Non-power-of-2-safe reduction (blockDim.x=min(256,cols) can be e.g. 3/6): ceil-halve the active
    // count + bound-check, else the top odd element is dropped (corrupts max/sum -> unnormalized softmax).
    for (int n = blockDim.x; n > 1; ) { int half = (n+1)>>1; if (tid < half && tid+half < n) sdata[tid] = fmaxf(sdata[tid], sdata[tid+half]); __syncthreads(); n = half; }
    maxVal = sdata[0];
    float sumExp = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) { float e = expf(rowIn[c]-maxVal); rowOut[c] = e; sumExp += e; }
    sdata[tid] = sumExp; __syncthreads();
    for (int n = blockDim.x; n > 1; ) { int half = (n+1)>>1; if (tid < half && tid+half < n) sdata[tid] += sdata[tid+half]; __syncthreads(); n = half; }
    sumExp = sdata[0];
    for (int c = tid; c < cols; c += blockDim.x) rowOut[c] /= sumExp;
}

// ─── HRR binding primitives (issue #248) ────────────────────────────
// Matches CUDA/OpenCL/Metal/Vulkan/WebGPU — see CudaComplexKernels.cs
// for the hash rationale (32-bit Murmur3 fmix chosen for WebGPU
// compatibility and shared with the CPU reference).
__device__ __forceinline__ unsigned int hrr_hash(unsigned int seed_u, unsigned int cell_u)
{
    unsigned int z = seed_u * 0x9E3779B9u + cell_u * 0x85EBCA6Bu;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z =  z ^ (z >> 16);
    return z;
}

__device__ __forceinline__ unsigned int hrr_mul_hi(unsigned int a, unsigned int b)
{
    unsigned int a0 = a & 0xFFFFu, a1 = a >> 16;
    unsigned int b0 = b & 0xFFFFu, b1 = b >> 16;
    unsigned int p0 = a0 * b0, p1 = a1 * b0, p2 = a0 * b1;
    unsigned int carry = (p0 >> 16) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
    return a1 * b1 + (p1 >> 16) + (p2 >> 16) + (carry >> 16);
}

__device__ __forceinline__ unsigned int hrr_quantize_turn(unsigned int turn, unsigned int k)
{
    unsigned int lattice = hrr_mul_hi(turn, k);
    if (turn * k >= 0x80000000u) lattice++;
    if (lattice == k) lattice = 0;
    unsigned int quotient = 0, remainder = lattice;
    for (int i = 0; i < 32; i++) {
        remainder <<= 1;
        quotient <<= 1;
        if (remainder >= k) { remainder -= k; quotient |= 1u; }
    }
    return quotient;
}

__device__ __constant__ int hrr_cordic_angles[30] = {
    536870912, 316933406, 167458907, 85004756, 42667331,
    21354465, 10679838, 5340245, 2670163, 1335087,
    667544, 333772, 166886, 83443, 41722,
    20861, 10430, 5215, 2608, 1304,
    652, 326, 163, 81, 41, 20, 10, 5, 3, 1
};

__device__ __forceinline__ void hrr_sincos(unsigned int turn, float* outCos, float* outSin)
{
    unsigned int quadrant = turn >> 30;
    unsigned int offset = turn & 0x3FFFFFFFu;
    int z = (int)((quadrant == 1u || quadrant == 3u) ? 0x40000000u - offset : offset);
    int x = 652032874, y = 0;
    for (int i = 0; i < 30; i++) {
        int oldX = x;
        if (z >= 0) {
            x -= y >> i; y += oldX >> i; z -= hrr_cordic_angles[i];
        } else {
            x += y >> i; y -= oldX >> i; z += hrr_cordic_angles[i];
        }
    }
    if (quadrant == 1u || quadrant == 2u) x = -x;
    if (quadrant >= 2u) y = -y;
    *outCos = (float)x * (1.0f / 1073741824.0f);
    *outSin = (float)y * (1.0f / 1073741824.0f);
}

extern ""C"" __global__ void hrr_unit_phase_codebook(
    float* outReal, float* outImag,
    int seed, int V, int D, int kPsk, int k)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)V * D;
    if (idx >= total) return;
    unsigned int turn = hrr_hash((unsigned int)seed, (unsigned int)idx) & 0xFFFFFF00u;
    if (kPsk) turn = hrr_quantize_turn(turn, (unsigned int)k);
    float c, s;
    hrr_sincos(turn, &c, &s);
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
