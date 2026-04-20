// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language (MSL) kernels for the parity-210 op surface.
// Mirrors CudaParity210Kernels / HipParity210Kernels in function coverage.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    /// <summary>
    /// Metal Shading Language implementations for the parity-210 kernels.
    /// Each kernel is launched with a 1-D dispatch of <c>(totalOutputElements,
    /// 1, 1)</c> (block size 256) and mirrors the semantics of the CUDA/HIP
    /// versions bit-for-bit so we can diff the three side-by-side.
    /// </summary>
    internal static class MetalParity210Kernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "parity210_roll_1d","parity210_flip_axis","parity210_triu","parity210_tril",
            "parity210_diag_embed",
            "parity210_cumsum_axis","parity210_cumprod_axis","parity210_cummax_axis",
            "parity210_cummin_axis","parity210_logcumsumexp_axis",
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

        public const string Source = @"
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

constant float PARITY210_PI = 3.14159265358979323846f;

// Local helper: positive modulo for int.
inline int p210_pmod(int a, int m) {
    int r = a % m;
    return (r < 0) ? r + m : r;
}

// ==========================================================================
// MOVEMENT
// ==========================================================================

kernel void parity210_roll_1d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& outerSize [[buffer(2)]],
    constant int& axisSize [[buffer(3)]],
    constant int& innerSize [[buffer(4)]],
    constant int& shift [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * axisSize * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int tmp = (int)gid / innerSize;
    int a = tmp % axisSize;
    int outer = tmp / axisSize;
    int srcAxis = p210_pmod(a - shift, axisSize);
    output[gid] = input[(outer * axisSize + srcAxis) * innerSize + inner];
}

kernel void parity210_flip_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& outerSize [[buffer(2)]],
    constant int& axisSize [[buffer(3)]],
    constant int& innerSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * axisSize * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int tmp = (int)gid / innerSize;
    int a = tmp % axisSize;
    int outer = tmp / axisSize;
    int srcAxis = axisSize - 1 - a;
    output[gid] = input[(outer * axisSize + srcAxis) * innerSize + inner];
}

kernel void parity210_triu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batchSize [[buffer(2)]],
    constant int& rows [[buffer(3)]],
    constant int& cols [[buffer(4)]],
    constant int& diagonal [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = batchSize * rows * cols;
    if ((int)gid >= total) return;
    int col = (int)gid % cols;
    int tmp = (int)gid / cols;
    int row = tmp % rows;
    output[gid] = ((col - row) >= diagonal) ? input[gid] : 0.0f;
}

kernel void parity210_tril(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batchSize [[buffer(2)]],
    constant int& rows [[buffer(3)]],
    constant int& cols [[buffer(4)]],
    constant int& diagonal [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = batchSize * rows * cols;
    if ((int)gid >= total) return;
    int col = (int)gid % cols;
    int tmp = (int)gid / cols;
    int row = tmp % rows;
    output[gid] = ((col - row) <= diagonal) ? input[gid] : 0.0f;
}

kernel void parity210_diag_embed(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batchSize [[buffer(2)]],
    constant int& diagLen [[buffer(3)]],
    constant int& matSize [[buffer(4)]],
    constant int& offset [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = batchSize * matSize * matSize;
    if ((int)gid >= total) return;
    int col = (int)gid % matSize;
    int tmp = (int)gid / matSize;
    int row = tmp % matSize;
    int b = tmp / matSize;
    int diagRow = (offset >= 0) ? row : row + offset;
    int diagCol = (offset >= 0) ? col - offset : col;
    if (diagRow == diagCol && diagRow >= 0 && diagRow < diagLen)
        output[gid] = input[b * diagLen + diagRow];
    else
        output[gid] = 0.0f;
}

// ==========================================================================
// CUMULATIVE
// ==========================================================================

kernel void parity210_cumsum_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& outerSize [[buffer(2)]],
    constant int& axisSize [[buffer(3)]],
    constant int& innerSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int outer = (int)gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = 0.0f;
    for (int a = 0; a < axisSize; ++a) {
        acc += input[base_ + a * innerSize];
        output[base_ + a * innerSize] = acc;
    }
}

kernel void parity210_cumprod_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& outerSize [[buffer(2)]],
    constant int& axisSize [[buffer(3)]],
    constant int& innerSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int outer = (int)gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = 1.0f;
    for (int a = 0; a < axisSize; ++a) {
        acc *= input[base_ + a * innerSize];
        output[base_ + a * innerSize] = acc;
    }
}

kernel void parity210_cummax_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& outerSize [[buffer(2)]],
    constant int& axisSize [[buffer(3)]],
    constant int& innerSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int outer = (int)gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = -INFINITY;
    for (int a = 0; a < axisSize; ++a) {
        float v = input[base_ + a * innerSize];
        if (v > acc) acc = v;
        output[base_ + a * innerSize] = acc;
    }
}

kernel void parity210_cummin_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& outerSize [[buffer(2)]],
    constant int& axisSize [[buffer(3)]],
    constant int& innerSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int outer = (int)gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = INFINITY;
    for (int a = 0; a < axisSize; ++a) {
        float v = input[base_ + a * innerSize];
        if (v < acc) acc = v;
        output[base_ + a * innerSize] = acc;
    }
}

kernel void parity210_logcumsumexp_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& outerSize [[buffer(2)]],
    constant int& axisSize [[buffer(3)]],
    constant int& innerSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int outer = (int)gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    if (axisSize <= 0) return;
    // Bootstrap from input[0] so initial -INFINITY doesn't produce NaN.
    float m = input[base_];
    float s = 1.0f;
    output[base_] = m;
    for (int a = 1; a < axisSize; ++a) {
        float x = input[base_ + a * innerSize];
        if (x > m) { s = s * exp(m - x) + 1.0f; m = x; }
        else { s += exp(x - m); }
        output[base_ + a * innerSize] = m + log(s);
    }
}

// ==========================================================================
// INDEXING
// ==========================================================================

kernel void parity210_take_linear(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& outSize [[buffer(3)]],
    constant int& inputLinearLen [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= outSize) return;
    int pos = indices[gid];
    output[gid] = (pos >= 0 && pos < inputLinearLen) ? input[pos] : 0.0f;
}

kernel void parity210_take_along_dim(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& outerSize [[buffer(3)]],
    constant int& idxAxis [[buffer(4)]],
    constant int& innerSize [[buffer(5)]],
    constant int& srcAxis [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * idxAxis * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int tmp = (int)gid / innerSize;
    int i = tmp % idxAxis;
    int outer = tmp / idxAxis;
    int target = indices[gid];
    int srcIdx = (outer * srcAxis + target) * innerSize + inner;
    output[gid] = (target >= 0 && target < srcAxis) ? input[srcIdx] : 0.0f;
}

kernel void parity210_index_add(
    device atomic_float* output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* source [[buffer(2)]],
    constant int& outerSize [[buffer(3)]],
    constant int& dstAxis [[buffer(4)]],
    constant int& innerSize [[buffer(5)]],
    constant int& idxLen [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * idxLen * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int tmp = (int)gid / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;
    int target = indices[i];
    if (target < 0 || target >= dstAxis) return;
    int dstPos = (outer * dstAxis + target) * innerSize + inner;
    int srcPos = (outer * idxLen + i) * innerSize + inner;
    atomic_fetch_add_explicit(&output[dstPos], source[srcPos], memory_order_relaxed);
}

kernel void parity210_index_copy(
    device float* output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* source [[buffer(2)]],
    constant int& outerSize [[buffer(3)]],
    constant int& dstAxis [[buffer(4)]],
    constant int& innerSize [[buffer(5)]],
    constant int& idxLen [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * idxLen * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int tmp = (int)gid / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;
    int target = indices[i];
    if (target < 0 || target >= dstAxis) return;
    output[(outer * dstAxis + target) * innerSize + inner] =
        source[(outer * idxLen + i) * innerSize + inner];
}

kernel void parity210_index_fill(
    device float* output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    constant float& fillValue [[buffer(2)]],
    constant int& outerSize [[buffer(3)]],
    constant int& dstAxis [[buffer(4)]],
    constant int& innerSize [[buffer(5)]],
    constant int& idxLen [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * idxLen * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int tmp = (int)gid / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;
    int target = indices[i];
    if (target < 0 || target >= dstAxis) return;
    output[(outer * dstAxis + target) * innerSize + inner] = fillValue;
}

kernel void parity210_masked_scatter(
    device float* output [[buffer(0)]],
    device const char* mask [[buffer(1)]],
    device const int* prefixSum [[buffer(2)]],
    device const float* source [[buffer(3)]],
    constant int& total [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= total) return;
    if (mask[gid]) output[gid] = source[prefixSum[gid]];
}

// ==========================================================================
// ELEMENT-WISE BINARY
// ==========================================================================

kernel void parity210_hypot(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    out[gid] = precise::hypot(a[gid], b[gid]);
}

kernel void parity210_copysign(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    out[gid] = copysign(a[gid], b[gid]);
}

kernel void parity210_fmod(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float bv = b[gid];
    out[gid] = (bv == 0.0f) ? 0.0f : fmod(a[gid], bv);
}

kernel void parity210_remainder(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float av = a[gid], bv = b[gid];
    if (bv == 0.0f) { out[gid] = 0.0f; return; }
    float q = floor(av / bv);
    out[gid] = av - q * bv;
}

kernel void parity210_float_power(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    out[gid] = pow(a[gid], b[gid]);
}

kernel void parity210_log_add_exp(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float av = a[gid], bv = b[gid];
    float m = max(av, bv);
    float s = min(av, bv);
    out[gid] = m + log1p(exp(s - m));
}

kernel void parity210_log_add_exp2(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float av = a[gid], bv = b[gid];
    float m = max(av, bv);
    float s = min(av, bv);
    out[gid] = m + log2(1.0f + exp2(s - m));
}

kernel void parity210_xlogy(
    device const float* x [[buffer(0)]], device const float* y [[buffer(1)]],
    device float* out [[buffer(2)]], constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float xv = x[gid];
    out[gid] = (xv == 0.0f) ? 0.0f : xv * log(y[gid]);
}

kernel void parity210_xlog1py(
    device const float* x [[buffer(0)]], device const float* y [[buffer(1)]],
    device float* out [[buffer(2)]], constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float xv = x[gid];
    out[gid] = (xv == 0.0f) ? 0.0f : xv * log1p(y[gid]);
}

// ==========================================================================
// ELEMENT-WISE UNARY SPECIAL
// ==========================================================================

kernel void parity210_erfc(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    output[gid] = erfc(input[gid]);
}

kernel void parity210_erfinv(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float y = input[gid];
    if (y >= 1.0f) { output[gid] = INFINITY; return; }
    if (y <= -1.0f) { output[gid] = -INFINITY; return; }
    float ln = log(1.0f - y * y);
    float a = 0.147f;
    float t = 2.0f / (PARITY210_PI * a) + ln * 0.5f;
    float xs = copysign(sqrt(sqrt(t * t - ln / a) - t), y);
    for (int k = 0; k < 2; ++k) {
        float e = erf(xs);
        float df = 2.0f / sqrt(PARITY210_PI) * exp(-xs * xs);
        xs -= (e - y) / df;
    }
    output[gid] = xs;
}

kernel void parity210_lgamma_approx(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    output[gid] = lgamma(input[gid]);
}

kernel void parity210_digamma(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float x = input[gid];
    float result = 0.0f;
    while (x < 6.0f) { result -= 1.0f / x; x += 1.0f; }
    float inv = 1.0f / x;
    float inv2 = inv * inv;
    result += log(x) - 0.5f * inv
            - inv2 * ((1.0f/12.0f)
              - inv2 * ((1.0f/120.0f)
                - inv2 * (1.0f/252.0f)));
    output[gid] = result;
}

inline float p210_i0(float x) {
    float ax = fabs(x);
    if (ax < 3.75f) {
        float y = (x / 3.75f); y = y * y;
        return 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f
                + y * (0.2659732f + y * (0.0360768f + y * 0.0045813f)))));
    } else {
        float y = 3.75f / ax;
        float ans = 0.39894228f + y * (0.01328592f + y * (0.00225319f
                + y * (-0.00157565f + y * (0.00916281f + y * (-0.02057706f
                + y * (0.02635537f + y * (-0.01647633f + y * 0.00392377f)))))));
        return (exp(ax) / sqrt(ax)) * ans;
    }
}

inline float p210_i1(float x) {
    float ax = fabs(x);
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
        ans *= (exp(ax) / sqrt(ax));
    }
    return (x < 0.0f) ? -ans : ans;
}

kernel void parity210_i0(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    output[gid] = p210_i0(input[gid]);
}

kernel void parity210_i1(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    output[gid] = p210_i1(input[gid]);
}

// i0e / i1e compute their own scaled polynomials directly so the
// intermediate `exp(ax) * exp(-ax)` never overflows for large |x|.
// For |x| < 3.75 the series is already bounded; for |x| >= 3.75 the
// exp(ax) factor the helpers would produce cancels with exp(-|x|), so
// we fold the cancellation in and evaluate (1/sqrt(|x|)) * ans directly.
kernel void parity210_i0e(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float x = input[gid];
    float ax = fabs(x);
    float ans;
    if (ax < 3.75f) {
        float y = (x / 3.75f); y = y * y;
        ans = 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f
                + y * (0.2659732f + y * (0.0360768f + y * 0.0045813f)))));
        ans = exp(-ax) * ans;
    } else {
        float y = 3.75f / ax;
        ans = 0.39894228f + y * (0.01328592f + y * (0.00225319f
                + y * (-0.00157565f + y * (0.00916281f + y * (-0.02057706f
                + y * (0.02635537f + y * (-0.01647633f + y * 0.00392377f)))))));
        // Large-|x| form already absorbs the exp(-ax) cancellation.
        ans = ans / sqrt(ax);
    }
    output[gid] = ans;
}

kernel void parity210_i1e(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float x = input[gid];
    float ax = fabs(x);
    float ans;
    if (ax < 3.75f) {
        float y = (x / 3.75f); y = y * y;
        ans = ax * (0.5f + y * (0.87890594f + y * (0.51498869f + y * (0.15084934f
                + y * (0.02658733f + y * (0.00301532f + y * 0.00032411f))))));
        ans = exp(-ax) * ans;
    } else {
        float y = 3.75f / ax;
        ans = 0.39894228f + y * (-0.03988024f + y * (-0.00362018f
                + y * (0.00163801f + y * (-0.01031555f + y * (0.02282967f
                + y * (-0.02895312f + y * (0.01787654f + y * -0.00420059f)))))));
        ans = ans / sqrt(ax);
    }
    output[gid] = (x < 0.0f) ? -ans : ans;
}

// ==========================================================================
// PREDICATES + NUMERIC HYGIENE
// ==========================================================================

kernel void parity210_is_finite(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float x = input[gid];
    output[gid] = (isfinite(x)) ? 1.0f : 0.0f;
}

kernel void parity210_is_nan(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    output[gid] = isnan(input[gid]) ? 1.0f : 0.0f;
}

kernel void parity210_is_inf(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    output[gid] = isinf(input[gid]) ? 1.0f : 0.0f;
}

kernel void parity210_nan_to_num(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    constant float& nanVal [[buffer(3)]],
    constant float& posInfVal [[buffer(4)]],
    constant float& negInfVal [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float x = input[gid];
    if (isnan(x)) output[gid] = nanVal;
    else if (isinf(x)) output[gid] = (x > 0.0f) ? posInfVal : negInfVal;
    else output[gid] = x;
}

// ==========================================================================
// PAIRWISE
// ==========================================================================

kernel void parity210_cosine_similarity_last(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& n [[buffer(3)]], constant int& d [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= n) return;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    int base_ = (int)gid * d;
    for (int k = 0; k < d; ++k) {
        float av = a[base_ + k], bv = b[base_ + k];
        dot += av * bv; na += av * av; nb += bv * bv;
    }
    float denom = max(sqrt(na * nb), eps);
    out[gid] = dot / denom;
}

kernel void parity210_cdist_l2(
    device const float* x1 [[buffer(0)]], device const float* x2 [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& n [[buffer(3)]], constant int& m [[buffer(4)]],
    constant int& d [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int total = n * m;
    if ((int)gid >= total) return;
    int j = (int)gid % m;
    int i = (int)gid / m;
    float acc = 0.0f;
    for (int k = 0; k < d; ++k) {
        float v = x1[i * d + k] - x2[j * d + k];
        acc += v * v;
    }
    out[gid] = sqrt(acc);
}

// ==========================================================================
// CLAMP
// ==========================================================================

kernel void parity210_clamp_min_max(
    device const float* input [[buffer(0)]],
    device const float* lo [[buffer(1)]],
    device const float* hi [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& size [[buffer(4)]],
    constant int& hasLo [[buffer(5)]],
    constant int& hasHi [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float x = input[gid];
    if (hasLo != 0) { float l = lo[gid]; if (x < l) x = l; }
    if (hasHi != 0) { float h = hi[gid]; if (x > h) x = h; }
    output[gid] = x;
}
";
    }
}
