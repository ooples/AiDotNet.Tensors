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
    constant float& thresholdMagSq [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    float magSq = re * re + im * im;
    outReal[idx] = (magSq >= thresholdMagSq) ? re : 0.0f;
    outImag[idx] = (magSq >= thresholdMagSq) ? im : 0.0f;
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
inline float hrr_phase_from_cell(int seed, ulong cellIdx)
{
    ulong z = (ulong)seed * 0x9E3779B97F4A7C15UL
            + cellIdx * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z =  z ^ (z >> 31);
    uint top24 = (uint)(z >> 40);
    return (float)top24 * (1.0f / 16777216.0f) * 6.28318530717958647692f;
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
    float phase = hrr_phase_from_cell(seed, (ulong)idx);
    if (kPsk != 0) {
        float step = 6.28318530717958647692f / (float)k;
        phase = floor(phase / step + 0.5f) * step;
    }
    outReal[idx] = cos(phase);
    outImag[idx] = sin(phase);
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
    uint d [[thread_position_in_grid]])
{
    if ((int)d >= D) return;
    float accR = memoryReal[d];
    float accI = memoryImag[d];
    for (int n = 0; n < N; n++) {
        long kOff = (long)keyIds[n] * D;
        long vOff = (long)valIds[n] * D;
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
