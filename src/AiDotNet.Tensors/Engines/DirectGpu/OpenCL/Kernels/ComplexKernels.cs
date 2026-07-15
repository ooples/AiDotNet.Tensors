// OpenCL kernels for native Complex<T> tensor operations.
// Uses split real/imaginary buffers for coalesced memory access.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

public static class ComplexKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "split_complex_multiply", "split_complex_conjugate",
        "split_complex_magnitude", "split_complex_magnitude_squared",
        "split_complex_phase", "split_complex_from_polar",
        "split_complex_scale", "split_complex_add",
        "split_complex_cross_spectral",
        "split_complex_topk", "softmax_rows",
        // Issue #248 — HRR binding primitives
        "hrr_unit_phase_codebook",
        "hrr_phase_coherence_decode",
        "hrr_bind_accumulate"
    };

    public static string GetSource() => @"
__kernel void split_complex_multiply(
    __global const float* aReal, __global const float* aImag,
    __global const float* bReal, __global const float* bImag,
    __global float* outReal, __global float* outImag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float ar = aReal[idx], ai = aImag[idx];
    float br = bReal[idx], bi = bImag[idx];
    outReal[idx] = ar * br - ai * bi;
    outImag[idx] = ar * bi + ai * br;
}

__kernel void split_complex_conjugate(
    __global const float* inReal, __global const float* inImag,
    __global float* outReal, __global float* outImag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}

__kernel void split_complex_magnitude(
    __global const float* inReal, __global const float* inImag,
    __global float* outMag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMag[idx] = sqrt(re * re + im * im);
}

__kernel void split_complex_magnitude_squared(
    __global const float* inReal, __global const float* inImag,
    __global float* outMagSq, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}

__kernel void split_complex_phase(
    __global const float* inReal, __global const float* inImag,
    __global float* outPhase, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    outPhase[idx] = atan2(inImag[idx], inReal[idx]);
}

__kernel void split_complex_from_polar(
    __global const float* mag, __global const float* phase,
    __global float* outReal, __global float* outImag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float m = mag[idx], p = phase[idx];
    outReal[idx] = m * cos(p);
    outImag[idx] = m * sin(p);
}

__kernel void split_complex_scale(
    __global const float* inReal, __global const float* inImag,
    __global float* outReal, __global float* outImag,
    const float scalar, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    outReal[idx] = inReal[idx] * scalar;
    outImag[idx] = inImag[idx] * scalar;
}

__kernel void split_complex_add(
    __global const float* aReal, __global const float* aImag,
    __global const float* bReal, __global const float* bImag,
    __global float* outReal, __global float* outImag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}

__kernel void split_complex_cross_spectral(
    __global const float* xReal, __global const float* xImag,
    __global const float* yReal, __global const float* yImag,
    __global float* outReal, __global float* outImag, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float xr = xReal[idx], xi = xImag[idx];
    float yr = yReal[idx], yi = yImag[idx];
    outReal[idx] = xr * yr + xi * yi;
    outImag[idx] = xi * yr - xr * yi;
}

__kernel void split_complex_topk(
    __global const float* inReal, __global const float* inImag,
    __global float* outReal, __global float* outImag,
    const int k, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    float magSq = re * re + im * im;
    bool magIsNan = (as_uint(magSq) & 0x7fffffffu) > 0x7f800000u;
    int rank = 0;
    for (int other = 0; other < n; ++other) {
        float otherRe = inReal[other], otherIm = inImag[other];
        float otherMagSq = otherRe * otherRe + otherIm * otherIm;
        bool otherIsNan = (as_uint(otherMagSq) & 0x7fffffffu) > 0x7f800000u;
        bool otherBefore = magIsNan
            ? (!otherIsNan || (otherIsNan && other < idx))
            : (!otherIsNan && (otherMagSq > magSq || (otherMagSq == magSq && other < idx)));
        rank += otherBefore ? 1 : 0;
    }
    bool keep = rank < k;
    outReal[idx] = keep ? re : 0.0f;
    outImag[idx] = keep ? im : 0.0f;
}

__kernel void softmax_rows(
    __global const float* input, __global float* output,
    const int rows, const int cols)
{
    int row = get_global_id(0);
    if (row >= rows) return;
    int offset = row * cols;
    float maxVal = -1e30f;
    for (int c = 0; c < cols; c++) maxVal = fmax(maxVal, input[offset + c]);
    float sumExp = 0.0f;
    for (int c = 0; c < cols; c++) { float e = exp(input[offset + c] - maxVal); output[offset + c] = e; sumExp += e; }
    for (int c = 0; c < cols; c++) output[offset + c] /= sumExp;
}

// ─── HRR binding primitives (issue #248) ────────────────────────────
// Matches CUDA/HIP/Metal/Vulkan/WebGPU — see CudaComplexKernels.cs for
// the hash rationale (32-bit Murmur3 fmix chosen for WebGPU
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

__constant int hrr_cordic_angles[30] = {
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
    return (float2)((float)x, (float)y) * (1.0f / 1073741824.0f);
}

__kernel void hrr_unit_phase_codebook(
    __global float* outReal, __global float* outImag,
    const int seed, const int V, const int D,
    const int kPsk, const int k)
{
    long idx = get_global_id(0);
    long total = (long)V * D;
    if (idx >= total) return;
    uint turn = hrr_hash((uint)seed, (uint)idx) & 0xFFFFFF00u;
    if (kPsk != 0) turn = hrr_quantize_turn(turn, (uint)k);
    float2 value = hrr_sincos(turn);
    outReal[idx] = value.x;
    outImag[idx] = value.y;
}

// One work-item per V row. Each work-item iterates all D entries of
// its row and produces one score. Simpler than a tree reduction (no
// local memory needed), and V is usually small enough (≤ 1024) that
// the parallelism across V covers any GPU's wavefront size — same
// tradeoff as softmax_rows above.
__kernel void hrr_phase_coherence_decode(
    __global const float* codesReal, __global const float* codesImag,
    __global const float* queryReal, __global const float* queryImag,
    __global float* outScores,
    const int V, const int D)
{
    int v = get_global_id(0);
    if (v >= V) return;
    __global const float* cR = codesReal + (long)v * D;
    __global const float* cI = codesImag + (long)v * D;
    float acc = 0.0f;
    for (int d = 0; d < D; d++) {
        acc += cR[d] * queryReal[d] + cI[d] * queryImag[d];
    }
    outScores[v] = acc;
}

__kernel void hrr_bind_accumulate(
    __global const float* keyCodeReal, __global const float* keyCodeImag,
    __global const float* valPermCodeReal, __global const float* valPermCodeImag,
    __global const int* keyIds, __global const int* valIds,
    __global float* memoryReal, __global float* memoryImag,
    const int N, const int D)
{
    int d = get_global_id(0);
    if (d >= D) return;
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
