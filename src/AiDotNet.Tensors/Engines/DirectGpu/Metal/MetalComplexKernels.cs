// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language (MSL) compute kernels for native Complex<T> tensor operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

internal static class MetalComplexKernels
{
    public static string GetSource() => @"
#include <metal_stdlib>
using namespace metal;

kernel void split_complex_multiply(
    device const float* aReal [[buffer(0)]],
    device const float* aImag [[buffer(1)]],
    device const float* bReal [[buffer(2)]],
    device const float* bImag [[buffer(3)]],
    device float* outReal [[buffer(4)]],
    device float* outImag [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float ar = aReal[idx], ai = aImag[idx];
    float br = bReal[idx], bi = bImag[idx];
    outReal[idx] = ar * br - ai * bi;
    outImag[idx] = ar * bi + ai * br;
}

kernel void split_complex_conjugate(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outReal [[buffer(2)]],
    device float* outImag [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}

kernel void split_complex_magnitude(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outMag [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMag[idx] = sqrt(re * re + im * im);
}

kernel void split_complex_magnitude_squared(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outMagSq [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}

kernel void split_complex_phase(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outPhase [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    outPhase[idx] = atan2(inImag[idx], inReal[idx]);
}

kernel void split_complex_from_polar(
    device const float* mag [[buffer(0)]],
    device const float* phase [[buffer(1)]],
    device float* outReal [[buffer(2)]],
    device float* outImag [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float m = mag[idx], p = phase[idx];
    outReal[idx] = m * cos(p);
    outImag[idx] = m * sin(p);
}

kernel void split_complex_scale(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outReal [[buffer(2)]],
    device float* outImag [[buffer(3)]],
    constant float& scalar [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    outReal[idx] = inReal[idx] * scalar;
    outImag[idx] = inImag[idx] * scalar;
}

kernel void split_complex_add(
    device const float* aReal [[buffer(0)]],
    device const float* aImag [[buffer(1)]],
    device const float* bReal [[buffer(2)]],
    device const float* bImag [[buffer(3)]],
    device float* outReal [[buffer(4)]],
    device float* outImag [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}

kernel void split_complex_cross_spectral(
    device const float* xReal [[buffer(0)]],
    device const float* xImag [[buffer(1)]],
    device const float* yReal [[buffer(2)]],
    device const float* yImag [[buffer(3)]],
    device float* outReal [[buffer(4)]],
    device float* outImag [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float xr = xReal[idx], xi = xImag[idx];
    float yr = yReal[idx], yi = yImag[idx];
    outReal[idx] = xr * yr + xi * yi;
    outImag[idx] = xi * yr - xr * yi;
}

kernel void split_complex_topk(
    device const float* inReal [[buffer(0)]],
    device const float* inImag [[buffer(1)]],
    device float* outReal [[buffer(2)]],
    device float* outImag [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    float magSq = re * re + im * im;
    bool magIsNan = (as_type<uint>(magSq) & 0x7fffffffu) > 0x7f800000u;
    uint rank = 0;
    for (uint other = 0; other < n; ++other) {
        float otherRe = inReal[other], otherIm = inImag[other];
        float otherMagSq = otherRe * otherRe + otherIm * otherIm;
        bool otherIsNan = (as_type<uint>(otherMagSq) & 0x7fffffffu) > 0x7f800000u;
        bool otherBefore = magIsNan
            ? (!otherIsNan || (otherIsNan && other < idx))
            : (!otherIsNan && (otherMagSq > magSq || (otherMagSq == magSq && other < idx)));
        rank += otherBefore ? 1u : 0u;
    }
    bool keep = rank < k;
    outReal[idx] = keep ? re : 0.0f;
    outImag[idx] = keep ? im : 0.0f;
}

kernel void softmax_rows(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    uint offset = row * cols;
    float maxVal = -1e30f;
    for (uint c = 0; c < cols; c++) maxVal = max(maxVal, input[offset + c]);
    float sumExp = 0.0f;
    for (uint c = 0; c < cols; c++) { float e = exp(input[offset + c] - maxVal); output[offset + c] = e; sumExp += e; }
    for (uint c = 0; c < cols; c++) output[offset + c] /= sumExp;
}

// ─── HRR binding primitives (issue #248) ────────────────────────────
// Matches CUDA/HIP/OpenCL/Vulkan/WebGPU — see CudaComplexKernels.cs
// for the hash rationale (32-bit Murmur3 fmix chosen for WebGPU
// compatibility and shared with the CPU reference).
inline uint hrr_hash(uint seed_u, uint cell_u)
{
    uint z = seed_u * 0x9E3779B9u + cell_u * 0x85EBCA6Bu;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z =  z ^ (z >> 16);
    return z;
}

inline uint hrr_mul_hi(uint a, uint b)
{
    uint a0 = a & 0xFFFFu, a1 = a >> 16;
    uint b0 = b & 0xFFFFu, b1 = b >> 16;
    uint p0 = a0 * b0, p1 = a1 * b0, p2 = a0 * b1;
    uint carry = (p0 >> 16) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
    return a1 * b1 + (p1 >> 16) + (p2 >> 16) + (carry >> 16);
}

inline uint hrr_quantize_turn(uint turn, uint k)
{
    uint lattice = hrr_mul_hi(turn, k);
    if (turn * k >= 0x80000000u) lattice++;
    if (lattice == k) lattice = 0;
    uint quotient = 0, remainder = lattice;
    for (int i = 0; i < 32; i++) {
        remainder <<= 1;
        quotient <<= 1;
        if (remainder >= k) { remainder -= k; quotient |= 1u; }
    }
    return quotient;
}

constant int hrr_cordic_angles[30] = {
    536870912, 316933406, 167458907, 85004756, 42667331,
    21354465, 10679838, 5340245, 2670163, 1335087,
    667544, 333772, 166886, 83443, 41722,
    20861, 10430, 5215, 2608, 1304,
    652, 326, 163, 81, 41, 20, 10, 5, 3, 1
};

inline float2 hrr_sincos(uint turn)
{
    uint quadrant = turn >> 30;
    uint offset = turn & 0x3FFFFFFFu;
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
    return float2((float)x, (float)y) * (1.0f / 1073741824.0f);
}

kernel void hrr_unit_phase_codebook(
    device float* outReal [[buffer(0)]],
    device float* outImag [[buffer(1)]],
    constant int& seed [[buffer(2)]],
    constant int& V [[buffer(3)]],
    constant int& D [[buffer(4)]],
    constant int& kPsk [[buffer(5)]],
    constant int& k [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    ulong total = (ulong)V * (ulong)D;
    if ((ulong)idx >= total) return;
    uint turn = hrr_hash((uint)seed, idx) & 0xFFFFFF00u;
    if (kPsk != 0) turn = hrr_quantize_turn(turn, (uint)k);
    float2 value = hrr_sincos(turn);
    outReal[idx] = value.x;
    outImag[idx] = value.y;
}

kernel void hrr_phase_coherence_decode(
    device const float* codesReal [[buffer(0)]],
    device const float* codesImag [[buffer(1)]],
    device const float* queryReal [[buffer(2)]],
    device const float* queryImag [[buffer(3)]],
    device float* outScores [[buffer(4)]],
    constant int& V [[buffer(5)]],
    constant int& D [[buffer(6)]],
    uint v [[thread_position_in_grid]])
{
    if ((int)v >= V) return;
    device const float* cR = codesReal + (long)v * D;
    device const float* cI = codesImag + (long)v * D;
    float acc = 0.0f;
    for (int d = 0; d < D; d++) {
        acc += cR[d] * queryReal[d] + cI[d] * queryImag[d];
    }
    outScores[v] = acc;
}

// nKeys / nVals match the CUDA/HIP kernels: codebook row counts
// passed by the host so out-of-range ids are rejected without OOB
// reads. See CudaComplexKernels.hrr_bind_accumulate for the full
// rationale.
kernel void hrr_bind_accumulate(
    device const float* keyCodeReal [[buffer(0)]],
    device const float* keyCodeImag [[buffer(1)]],
    device const float* valPermCodeReal [[buffer(2)]],
    device const float* valPermCodeImag [[buffer(3)]],
    device const int* keyIds [[buffer(4)]],
    device const int* valIds [[buffer(5)]],
    device float* memoryReal [[buffer(6)]],
    device float* memoryImag [[buffer(7)]],
    constant int& N [[buffer(8)]],
    constant int& D [[buffer(9)]],
    constant int& nKeys [[buffer(10)]],
    constant int& nVals [[buffer(11)]],
    uint d [[thread_position_in_grid]])
{
    if ((int)d >= D) return;
    float accR = memoryReal[d];
    float accI = memoryImag[d];
    for (int n = 0; n < N; n++) {
        int kId = keyIds[n];
        int vId = valIds[n];
        // Unsigned-comparison trick rejects both negative and
        // too-large indices in one branch.
        if ((uint)kId >= (uint)nKeys || (uint)vId >= (uint)nVals) continue;
        long kOff = (long)kId * D;
        long vOff = (long)vId * D;
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
