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
    const float thresholdMagSq, const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    float magSq = re * re + im * im;
    outReal[idx] = (magSq >= thresholdMagSq) ? re : 0.0f;
    outImag[idx] = (magSq >= thresholdMagSq) ? im : 0.0f;
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
// compatibility; per-backend GPU determinism preserved, CPU
// xorshift64* path intentionally divergent for single-thread speed).
inline uint hrr_hash(uint seed_u, uint cell_u)
{
    uint z = seed_u * 0x9E3779B9u + cell_u * 0x85EBCA6Bu;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z =  z ^ (z >> 16);
    return z;
}

inline float hrr_phase_from_cell(int seed, long cellIdx)
{
    uint z = hrr_hash((uint)seed, (uint)cellIdx);
    uint top24 = z >> 8;
    return (float)top24 * (1.0f / 16777216.0f) * 6.28318530717958647692f;
}

__kernel void hrr_unit_phase_codebook(
    __global float* outReal, __global float* outImag,
    const int seed, const int V, const int D,
    const int kPsk, const int k)
{
    long idx = get_global_id(0);
    long total = (long)V * D;
    if (idx >= total) return;
    float phase = hrr_phase_from_cell(seed, idx);
    if (kPsk != 0) {
        float step = 6.28318530717958647692f / (float)k;
        phase = floor(phase / step + 0.5f) * step;
    }
    outReal[idx] = cos(phase);
    outImag[idx] = sin(phase);
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
