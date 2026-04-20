// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for the parity-210 op surface. Source-compatible with
// CUDA/Parity210 (HIP's hiprtc provides device intrinsics without needing
// <math.h>) — the two files stay structurally identical on purpose so we
// can point at the same reference implementations during review.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels
{
    /// <summary>
    /// Mirror of <c>CudaParity210Kernels</c> for HIP/ROCm. Every kernel name
    /// matches the CUDA version bit-for-bit so the two dispatch tables can
    /// share test harnesses.
    /// </summary>
    internal static class HipParity210Kernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "parity210_roll_1d","parity210_flip_axis","parity210_triu","parity210_tril",
            "parity210_diag_embed",
            "parity210_cumsum_axis","parity210_cumprod_axis","parity210_cummax_axis",
            "parity210_cummin_axis","parity210_logcumsumexp_axis",
            "parity210_cumsum_block_hillis_steele",
            "parity210_take_linear","parity210_take_along_dim","parity210_index_add",
            "parity210_index_copy","parity210_index_fill","parity210_masked_scatter",
            "parity210_hypot","parity210_copysign","parity210_fmod","parity210_remainder",
            "parity210_float_power","parity210_log_add_exp","parity210_log_add_exp2",
            "parity210_xlogy","parity210_xlog1py",
            "parity210_erfc","parity210_erfinv","parity210_lgamma_approx","parity210_digamma",
            "parity210_i0","parity210_i1","parity210_i0e","parity210_i1e",
            "parity210_is_finite","parity210_is_nan","parity210_is_inf","parity210_nan_to_num",
            "parity210_cosine_similarity_last","parity210_cdist_l2",
            "parity210_clamp_min_max",
        };

        public static string GetSource() => @"
// HIP-RTC device defines — no #include needed.
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==========================================================================
// MOVEMENT
// ==========================================================================

extern ""C"" __global__ __launch_bounds__(256) void parity210_roll_1d(
    const float* input, float* output,
    int outerSize, int axisSize, int innerSize, int shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * axisSize * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int tmp = idx / innerSize;
    int a = tmp % axisSize;
    int outer = tmp / axisSize;

    int srcAxis = a - shift;
    srcAxis %= axisSize;
    if (srcAxis < 0) srcAxis += axisSize;

    int srcIdx = (outer * axisSize + srcAxis) * innerSize + inner;
    output[idx] = input[srcIdx];
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_flip_axis(
    const float* input, float* output,
    int outerSize, int axisSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * axisSize * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int tmp = idx / innerSize;
    int a = tmp % axisSize;
    int outer = tmp / axisSize;
    int srcAxis = axisSize - 1 - a;
    output[idx] = input[(outer * axisSize + srcAxis) * innerSize + inner];
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_triu(
    const float* input, float* output,
    int batchSize, int rows, int cols, int diagonal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * rows * cols;
    if (idx >= total) return;

    int col = idx % cols;
    int tmp = idx / cols;
    int row = tmp % rows;
    output[idx] = ((col - row) >= diagonal) ? input[idx] : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_tril(
    const float* input, float* output,
    int batchSize, int rows, int cols, int diagonal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * rows * cols;
    if (idx >= total) return;

    int col = idx % cols;
    int tmp = idx / cols;
    int row = tmp % rows;
    output[idx] = ((col - row) <= diagonal) ? input[idx] : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_diag_embed(
    const float* input, float* output,
    int batchSize, int diagLen, int matSize, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * matSize * matSize;
    if (idx >= total) return;

    int col = idx % matSize;
    int tmp = idx / matSize;
    int row = tmp % matSize;
    int b = tmp / matSize;

    int diagRow = (offset >= 0) ? row : row + offset;
    int diagCol = (offset >= 0) ? col - offset : col;
    if (diagRow == diagCol && diagRow >= 0 && diagRow < diagLen)
        output[idx] = input[b * diagLen + diagRow];
    else
        output[idx] = 0.0f;
}

// ==========================================================================
// CUMULATIVE
// ==========================================================================

// Block-level Hillis-Steele scan for axes ≤ 1024. See the CUDA counterpart
// for algorithmic detail; HIP's shared-memory + sync semantics match CUDA
// line-for-line.
extern ""C"" __global__ void parity210_cumsum_block_hillis_steele(
    const float* input, float* output,
    int outerSize, int axisSize, int innerSize)
{
    HIP_DYNAMIC_SHARED(float, smem);
    int line = blockIdx.x;
    int inner = line % innerSize;
    int outer = line / innerSize;
    if (outer >= outerSize) return;
    int base_ = outer * axisSize * innerSize + inner;

    int tid = threadIdx.x;
    float* s0 = smem;
    float* s1 = smem + blockDim.x;

    s0[tid] = (tid < axisSize) ? input[base_ + tid * innerSize] : 0.0f;
    __syncthreads();

    int limit = axisSize;
    for (int offset = 1; offset < limit; offset *= 2) {
        if (tid < limit) {
            s1[tid] = (tid >= offset) ? s0[tid] + s0[tid - offset] : s0[tid];
        }
        __syncthreads();
        float* tmp = s0; s0 = s1; s1 = tmp;
    }

    if (tid < axisSize) {
        output[base_ + tid * innerSize] = s0[tid];
    }
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_cumsum_axis(
    const float* input, float* output,
    int outerSize, int axisSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int outer = idx / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = 0.0f;
    for (int a = 0; a < axisSize; ++a) {
        acc += input[base_ + a * innerSize];
        output[base_ + a * innerSize] = acc;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_cumprod_axis(
    const float* input, float* output,
    int outerSize, int axisSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int outer = idx / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = 1.0f;
    for (int a = 0; a < axisSize; ++a) {
        acc *= input[base_ + a * innerSize];
        output[base_ + a * innerSize] = acc;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_cummax_axis(
    const float* input, float* output,
    int outerSize, int axisSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int outer = idx / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = -INFINITY;
    for (int a = 0; a < axisSize; ++a) {
        float v = input[base_ + a * innerSize];
        if (v > acc) acc = v;
        output[base_ + a * innerSize] = acc;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_cummin_axis(
    const float* input, float* output,
    int outerSize, int axisSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int outer = idx / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = INFINITY;
    for (int a = 0; a < axisSize; ++a) {
        float v = input[base_ + a * innerSize];
        if (v < acc) acc = v;
        output[base_ + a * innerSize] = acc;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_logcumsumexp_axis(
    const float* input, float* output,
    int outerSize, int axisSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int outer = idx / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    if (axisSize <= 0) return;
    // Bootstrap from the first element instead of -INFINITY. The original
    // seeding would compute exp(-INF - -INF) = exp(NaN) on the first
    // iteration when input[0] is -INFINITY, propagating NaN down the scan.
    float m = input[base_];
    float s = 1.0f;
    output[base_] = m;
    for (int a = 1; a < axisSize; ++a) {
        float x = input[base_ + a * innerSize];
        if (x > m) { s = s * expf(m - x) + 1.0f; m = x; }
        else { s += expf(x - m); }
        output[base_ + a * innerSize] = m + logf(s);
    }
}

// ==========================================================================
// INDEXING
// ==========================================================================

extern ""C"" __global__ __launch_bounds__(256) void parity210_take_linear(
    const float* input, const int* indices, float* output,
    int outSize, int inputLinearLen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outSize) return;
    int pos = indices[idx];
    output[idx] = (pos >= 0 && pos < inputLinearLen) ? input[pos] : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_take_along_dim(
    const float* input, const int* indices, float* output,
    int outerSize, int idxAxis, int innerSize, int srcAxis)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * idxAxis * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int tmp = idx / innerSize;
    int i = tmp % idxAxis;
    int outer = tmp / idxAxis;

    int target = indices[idx];
    int srcIdx = (outer * srcAxis + target) * innerSize + inner;
    output[idx] = (target >= 0 && target < srcAxis) ? input[srcIdx] : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_index_add(
    float* output, const int* indices, const float* source,
    int outerSize, int dstAxis, int innerSize, int idxLen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * idxLen * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int tmp = idx / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;

    int target = indices[i];
    if (target < 0 || target >= dstAxis) return;
    int dstPos = (outer * dstAxis + target) * innerSize + inner;
    int srcPos = (outer * idxLen + i) * innerSize + inner;
    atomicAdd(&output[dstPos], source[srcPos]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_index_copy(
    float* output, const int* indices, const float* source,
    int outerSize, int dstAxis, int innerSize, int idxLen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * idxLen * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int tmp = idx / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;

    int target = indices[i];
    if (target < 0 || target >= dstAxis) return;
    output[(outer * dstAxis + target) * innerSize + inner] =
        source[(outer * idxLen + i) * innerSize + inner];
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_index_fill(
    float* output, const int* indices, float fillValue,
    int outerSize, int dstAxis, int innerSize, int idxLen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * idxLen * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int tmp = idx / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;

    int target = indices[i];
    if (target < 0 || target >= dstAxis) return;
    output[(outer * dstAxis + target) * innerSize + inner] = fillValue;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_masked_scatter(
    float* output, const char* mask, const int* prefixSum,
    const float* source, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (mask[idx]) output[idx] = source[prefixSum[idx]];
}

// ==========================================================================
// ELEMENT-WISE BINARY SPECIAL
// ==========================================================================

extern ""C"" __global__ __launch_bounds__(256) void parity210_hypot(
    const float* a, const float* b, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = hypotf(a[idx], b[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_copysign(
    const float* a, const float* b, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = copysignf(a[idx], b[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_fmod(
    const float* a, const float* b, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float bv = b[idx];
    out[idx] = (bv == 0.0f) ? 0.0f : fmodf(a[idx], bv);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_remainder(
    const float* a, const float* b, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float av = a[idx], bv = b[idx];
    if (bv == 0.0f) { out[idx] = 0.0f; return; }
    float q = floorf(av / bv);
    out[idx] = av - q * bv;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_float_power(
    const float* a, const float* b, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = powf(a[idx], b[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_log_add_exp(
    const float* a, const float* b, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float av = a[idx], bv = b[idx];
    float m = fmaxf(av, bv);
    float s = fminf(av, bv);
    out[idx] = m + log1pf(expf(s - m));
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_log_add_exp2(
    const float* a, const float* b, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float av = a[idx], bv = b[idx];
    float m = fmaxf(av, bv);
    float s = fminf(av, bv);
    out[idx] = m + log2f(1.0f + exp2f(s - m));
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_xlogy(
    const float* x, const float* y, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float xv = x[idx];
    out[idx] = (xv == 0.0f) ? 0.0f : xv * logf(y[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_xlog1py(
    const float* x, const float* y, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float xv = x[idx];
    out[idx] = (xv == 0.0f) ? 0.0f : xv * log1pf(y[idx]);
}

// ==========================================================================
// ELEMENT-WISE UNARY SPECIAL
// ==========================================================================

extern ""C"" __global__ __launch_bounds__(256) void parity210_erfc(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = erfcf(input[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_erfinv(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float y = input[idx];
    if (y >= 1.0f) { output[idx] = INFINITY; return; }
    if (y <= -1.0f) { output[idx] = -INFINITY; return; }
    float ln = logf(1.0f - y * y);
    float a = 0.147f;
    float t = 2.0f / ((float)M_PI * a) + ln * 0.5f;
    float xs = copysignf(sqrtf(sqrtf(t * t - ln / a) - t), y);
    for (int k = 0; k < 2; ++k) {
        float e = erff(xs);
        float df = 2.0f / sqrtf((float)M_PI) * expf(-xs * xs);
        xs -= (e - y) / df;
    }
    output[idx] = xs;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_lgamma_approx(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = lgammaf(input[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_digamma(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float result = 0.0f;
    while (x < 6.0f) { result -= 1.0f / x; x += 1.0f; }
    float inv = 1.0f / x;
    float inv2 = inv * inv;
    result += logf(x) - 0.5f * inv
            - inv2 * ((1.0f/12.0f)
              - inv2 * ((1.0f/120.0f)
                - inv2 * (1.0f/252.0f)));
    output[idx] = result;
}

__device__ inline float parity210_dev_i0(float x) {
    float ax = fabsf(x);
    if (ax < 3.75f) {
        float y = (x / 3.75f); y = y * y;
        return 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f
                + y * (0.2659732f + y * (0.0360768f + y * 0.0045813f)))));
    } else {
        float y = 3.75f / ax;
        float ans = 0.39894228f + y * (0.01328592f + y * (0.00225319f
                + y * (-0.00157565f + y * (0.00916281f + y * (-0.02057706f
                + y * (0.02635537f + y * (-0.01647633f + y * 0.00392377f)))))));
        return (expf(ax) / sqrtf(ax)) * ans;
    }
}

__device__ inline float parity210_dev_i1(float x) {
    float ax = fabsf(x);
    float ans;
    if (ax < 3.75f) {
        float y = (x / 3.75f); y = y * y;
        ans = ax * (0.5f + y * (0.87890594f + y * (0.51498869f + y * (0.15084934f
                + y * (0.02658733f + y * (0.00301532f + y * 0.00032411f))))));
    } else {
        float y = 3.75f / ax;
        ans = 0.39894228f + y * (-0.03988024f + y * (-0.00362018f
                + y * (0.00163801f + y * (-0.01031555f + y * (0.02282967f
                + y * (-0.02895312f + y * (0.01787654f + y * -0.00420059f)))))));
        ans *= (expf(ax) / sqrtf(ax));
    }
    return (x < 0.0f) ? -ans : ans;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_i0(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = parity210_dev_i0(input[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_i1(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = parity210_dev_i1(input[idx]);
}

// Overflow-safe form — see the CUDA counterpart for full commentary.
extern ""C"" __global__ __launch_bounds__(256) void parity210_i0e(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float ax = fabsf(x);
    float ans;
    if (ax < 3.75f) {
        float y = (x / 3.75f); y = y * y;
        ans = 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f
                + y * (0.2659732f + y * (0.0360768f + y * 0.0045813f)))));
        ans = expf(-ax) * ans;
    } else {
        float y = 3.75f / ax;
        ans = 0.39894228f + y * (0.01328592f + y * (0.00225319f
                + y * (-0.00157565f + y * (0.00916281f + y * (-0.02057706f
                + y * (0.02635537f + y * (-0.01647633f + y * 0.00392377f)))))));
        ans = ans / sqrtf(ax);
    }
    output[idx] = ans;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_i1e(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float ax = fabsf(x);
    float ans;
    if (ax < 3.75f) {
        float y = (x / 3.75f); y = y * y;
        ans = ax * (0.5f + y * (0.87890594f + y * (0.51498869f + y * (0.15084934f
                + y * (0.02658733f + y * (0.00301532f + y * 0.00032411f))))));
        ans = expf(-ax) * ans;
    } else {
        float y = 3.75f / ax;
        ans = 0.39894228f + y * (-0.03988024f + y * (-0.00362018f
                + y * (0.00163801f + y * (-0.01031555f + y * (0.02282967f
                + y * (-0.02895312f + y * (0.01787654f + y * -0.00420059f)))))));
        ans = ans / sqrtf(ax);
    }
    output[idx] = (x < 0.0f) ? -ans : ans;
}

// ==========================================================================
// PREDICATES + NUMERIC HYGIENE
// ==========================================================================

extern ""C"" __global__ __launch_bounds__(256) void parity210_is_finite(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = isfinite(input[idx]) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_is_nan(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = isnan(input[idx]) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_is_inf(
    const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = isinf(input[idx]) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_nan_to_num(
    const float* input, float* output,
    int size, float nanVal, float posInfVal, float negInfVal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    if (isnan(x)) output[idx] = nanVal;
    else if (isinf(x)) output[idx] = (x > 0.0f) ? posInfVal : negInfVal;
    else output[idx] = x;
}

// ==========================================================================
// PAIRWISE
// ==========================================================================

extern ""C"" __global__ __launch_bounds__(256) void parity210_cosine_similarity_last(
    const float* a, const float* b, float* out, int n, int d, float eps)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    int base_ = row * d;
    for (int k = 0; k < d; ++k) {
        float av = a[base_ + k], bv = b[base_ + k];
        dot += av * bv;
        na  += av * av;
        nb  += bv * bv;
    }
    float denom = fmaxf(sqrtf(na * nb), eps);
    out[row] = dot / denom;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_cdist_l2(
    const float* x1, const float* x2, float* out, int n, int m, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    if (idx >= total) return;
    int j = idx % m;
    int i = idx / m;
    float acc = 0.0f;
    for (int k = 0; k < d; ++k) {
        float v = x1[i * d + k] - x2[j * d + k];
        acc += v * v;
    }
    out[idx] = sqrtf(acc);
}

// ==========================================================================
// CLAMP
// ==========================================================================

extern ""C"" __global__ __launch_bounds__(256) void parity210_clamp_min_max(
    const float* input, const float* lo, const float* hi,
    float* output, int size, int hasLo, int hasHi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    if (hasLo) { float l = lo[idx]; if (x < l) x = l; }
    if (hasHi) { float h = hi[idx]; if (x > h) x = h; }
    output[idx] = x;
}
";
    }
}
