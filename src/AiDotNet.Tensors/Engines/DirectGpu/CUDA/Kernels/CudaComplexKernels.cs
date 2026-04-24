// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for native Complex<T> split-buffer tensor operations.
// Uses separate real/imaginary GPU buffers for coalesced memory access.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    internal static class CudaComplexKernels
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

        public static string GetSource()
        {
            return @"
extern ""C"" __global__ __launch_bounds__(256) void split_complex_multiply(
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

extern ""C"" __global__ __launch_bounds__(256) void split_complex_conjugate(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_magnitude(
    const float* inReal, const float* inImag,
    float* outMag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMag[idx] = sqrtf(re * re + im * im);
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_magnitude_squared(
    const float* inReal, const float* inImag,
    float* outMagSq, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_phase(
    const float* inReal, const float* inImag,
    float* outPhase, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outPhase[idx] = atan2f(inImag[idx], inReal[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_from_polar(
    const float* mag, const float* phase,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float m = mag[idx], p = phase[idx];
    outReal[idx] = m * cosf(p);
    outImag[idx] = m * sinf(p);
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_scale(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = inReal[idx] * scalar;
    outImag[idx] = inImag[idx] * scalar;
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_add(
    const float* aReal, const float* aImag,
    const float* bReal, const float* bImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void split_complex_cross_spectral(
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

// Top-K by magnitude: zero elements not in top-K.
// Step 1: compute magnitude squared into scratch buffer (caller does this).
// Step 2: this kernel zeros elements below the K-th largest magnitude.
// Caller provides the threshold magnitude squared value.
extern ""C"" __global__ __launch_bounds__(256) void split_complex_topk(
    const float* inReal, const float* inImag,
    float* outReal, float* outImag,
    float thresholdMagSq, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    float magSq = re * re + im * im;
    if (magSq >= thresholdMagSq) {
        outReal[idx] = re;
        outImag[idx] = im;
    } else {
        outReal[idx] = 0.0f;
        outImag[idx] = 0.0f;
    }
}

// Per-row softmax: each block handles one row.
// Shared memory used for reduction (max and sum).
extern ""C"" __global__ void softmax_rows(
    const float* input, float* output, int rows, int cols)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* rowIn = input + row * cols;
    float* rowOut = output + row * cols;

    // Step 1: find max (reduction)
    float maxVal = -1e30f;
    for (int c = tid; c < cols; c += blockDim.x)
        maxVal = fmaxf(maxVal, rowIn[c]);
    sdata[tid] = maxVal;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    maxVal = sdata[0];

    // Step 2: exp and sum
    float sumExp = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float e = expf(rowIn[c] - maxVal);
        rowOut[c] = e;
        sumExp += e;
    }
    sdata[tid] = sumExp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sumExp = sdata[0];

    // Step 3: normalize
    for (int c = tid; c < cols; c += blockDim.x)
        rowOut[c] /= sumExp;
}

// ─── HRR binding primitives (issue #248) ────────────────────────────
//
// Deterministic per-cell phase via splitmix64 hash of (seed, cell_idx).
// Each thread generates its own phase independently — no shared state,
// no cross-thread sync. Same seed → same codebook across runs; same
// seed + same (V, D) → identical output. Does NOT match the CPU
// xorshift64* sequence bit-for-bit (GPU phases are a different
// uniform sample), which is fine — users don't need cross-device
// bit-identity on random init.
__device__ __forceinline__ float hrr_phase_from_cell(int seed, long long cellIdx)
{
    unsigned long long z = (unsigned long long)seed * 0x9E3779B97F4A7C15ULL
                         + (unsigned long long)cellIdx * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z =  z ^ (z >> 31);
    // Upper 24 bits → float in [0, 1). 24 bits is exact-representable
    // in float32 so the distribution has no rounding artefacts at the
    // top of the range.
    unsigned int top24 = (unsigned int)(z >> 40);
    return (float)top24 * (1.0f / 16777216.0f) * 6.28318530717958647692f;
}

extern ""C"" __global__ __launch_bounds__(256) void hrr_unit_phase_codebook(
    float* outReal, float* outImag,
    int seed, int V, int D, int kPsk, int k)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)V * D;
    if (idx >= total) return;

    float phase = hrr_phase_from_cell(seed, idx);
    if (kPsk) {
        // Snap to nearest multiple of 2π/k.
        float step = 6.28318530717958647692f / (float)k;
        phase = floorf(phase / step + 0.5f) * step;
    }
    float c, s;
    sincosf(phase, &s, &c);
    outReal[idx] = c;
    outImag[idx] = s;
}

// Per-row phase-coherence reduction: one block per V, block-level
// tree sum over D. scores[v] = Σ_d (queryR·codesR[v,d] + queryI·codesI[v,d]).
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
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) outScores[v] = sdata[0];
}

// Fused gather + complex multiply + accumulate across N training pairs.
// One thread per D lane loops all N pairs — each thread writes only its
// own d so no atomics are needed. Coalesced reads within a warp
// because d varies consecutively and key/val offsets are broadcast
// (same kId for all threads in a warp at a given n iteration).
extern ""C"" __global__ __launch_bounds__(256) void hrr_bind_accumulate(
    const float* keyCodeReal, const float* keyCodeImag,
    const float* valPermCodeReal, const float* valPermCodeImag,
    const int* keyIds, const int* valIds,
    float* memoryReal, float* memoryImag,
    int N, int D)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    float accR = memoryReal[d];
    float accI = memoryImag[d];
    for (int n = 0; n < N; n++) {
        long long kOff = (long long)keyIds[n] * D;
        long long vOff = (long long)valIds[n] * D;
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
}
