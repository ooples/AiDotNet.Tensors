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
            "parity210_reflect_pad_1d","parity210_stft_mag_phase","parity210_phase_vocoder","parity210_build_spectrum","parity210_istft_from_spectrum","parity210_istft_normalize","parity210_logical_op","parity210_logical_not","parity210_shifted_diff","parity210_masks_to_boxes","parity210_pairwise_iou","parity210_histogramdd","parity210_gridsample_backward_input","parity210_gridsample_backward_grid","parity210_take_along_dim_f","parity210_cross3","parity210_ldexp","parity210_kron2d","parity210_search_sorted","parity210_next_after","parity210_index_write","parity210_cdist","parity210_pdist","parity210_histc","parity210_bitonic_step","parity210_copy_rows","parity210_iota_pad","parity210_hsoftmax_paths","parity210_isin","parity210_copy_block_2d","parity210_scatter_reduce","parity210_unfold","parity210_classify_float","parity210_zeta","parity210_polygamma","parity210_rwkv7_forward",
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
    if (axisSize <= 0) return;
    // Bootstrap from input[0] so a leading NaN propagates (NaN > -INF is
    // false, which would otherwise silently shadow it with the sentinel).
    float acc = input[base_];
    output[base_] = acc;
    for (int a = 1; a < axisSize; ++a) {
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
    if (axisSize <= 0) return;
    // Bootstrap from input[0] so a leading NaN propagates (NaN < +INF is
    // false, which would otherwise silently shadow it with the sentinel).
    float acc = input[base_];
    output[base_] = acc;
    for (int a = 1; a < axisSize; ++a) {
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
        // Equal infinities would produce exp(NaN); count directly so
        // [-inf,-inf] and [+inf,+inf] stay inf throughout the scan.
        if (x == m && isinf(x)) { s += 1.0f; }
        else if (x > m) { s = s * exp(m - x) + 1.0f; m = x; }
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
    // PyTorch-style negative index normalization.
    if (pos < 0) pos += inputLinearLen;
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
    if (target < 0) target += srcAxis;
    if (target < 0 || target >= srcAxis) { output[gid] = 0.0f; return; }
    int srcIdx = (outer * srcAxis + target) * innerSize + inner;
    output[gid] = input[srcIdx];
}

// NON-DETERMINISTIC (issue #382); see parity210_index_add_deterministic below.
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
    if (target < 0) target += dstAxis;
    if (target < 0 || target >= dstAxis) return;
    int dstPos = (outer * dstAxis + target) * innerSize + inner;
    int srcPos = (outer * idxLen + i) * innerSize + inner;
    atomic_fetch_add_explicit(&output[dstPos], source[srcPos], memory_order_relaxed);
}

// parity210_index_add — bit-deterministic variant (issue #382).
// One thread per (outer, dstTarget, inner) output cell scans idxLen in fixed order.
kernel void parity210_index_add_deterministic(
    device float* output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* source [[buffer(2)]],
    constant int& outerSize [[buffer(3)]],
    constant int& dstAxis [[buffer(4)]],
    constant int& innerSize [[buffer(5)]],
    constant int& idxLen [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int total = outerSize * dstAxis * innerSize;
    if ((int)gid >= total) return;
    int inner = (int)gid % innerSize;
    int tmp = (int)gid / innerSize;
    int dstTarget = tmp % dstAxis;
    int outer = tmp / dstAxis;

    float sum = 0.0f;
    for (int i = 0; i < idxLen; i++) {
        int t = indices[i];
        if (t < 0) t += dstAxis;
        if (t == dstTarget) {
            int srcPos = (outer * idxLen + i) * innerSize + inner;
            sum += source[srcPos];
        }
    }
    int dstPos = (outer * dstAxis + dstTarget) * innerSize + inner;
    output[dstPos] += sum;
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
    if (target < 0) target += dstAxis;
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
    if (target < 0) target += dstAxis;
    if (target < 0 || target >= dstAxis) return;
    output[(outer * dstAxis + target) * innerSize + inner] = fillValue;
}

kernel void parity210_masked_scatter(
    device float* output [[buffer(0)]],
    device const char* mask [[buffer(1)]],
    device const int* prefixSum [[buffer(2)]],
    device const float* source [[buffer(3)]],
    constant int& total [[buffer(4)]],
    constant int& sourceLen [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= total) return;
    if (mask[gid]) {
        int srcIdx = prefixSum[gid];
        // Guard against inconsistent prefix metadata or a short source.
        if (srcIdx >= 0 && srcIdx < sourceLen) output[gid] = source[srcIdx];
    }
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
    // torch.fmod(x, 0) = NaN for fp; Metal's fmod follows IEEE.
    out[gid] = fmod(a[gid], b[gid]);
}

kernel void parity210_remainder(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= size) return;
    float av = a[gid], bv = b[gid];
    if (bv == 0.0f) { out[gid] = NAN; return; }
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
    // Short-circuit equal infinities so inf-inf = NaN doesn't contaminate.
    if (av == bv && isinf(av)) { out[gid] = av; return; }
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
    if (av == bv && isinf(av)) { out[gid] = av; return; }
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
    // Domain: [-1, 1]. Exact +/-1 -> +/-infinity; anything strictly outside is NaN.
    if (y == 1.0f) { output[gid] = INFINITY; return; }
    if (y == -1.0f) { output[gid] = -INFINITY; return; }
    if (!(y > -1.0f && y < 1.0f)) { output[gid] = NAN; return; }
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
    // Reflection for x <= 0 (avoids log of non-positive in asymptotic tail).
    // psi(x) = psi(1-x) - pi * cot(pi*x); poles at non-positive integers.
    if (x <= 0.0f) {
        if (x == floor(x)) { output[gid] = NAN; return; }
        float sp = sin(PARITY210_PI * x);
        result = -PARITY210_PI * cos(PARITY210_PI * x) / sp;
        x = 1.0f - x;
    }
    // Bounded for-loop — x += 1 on -INFINITY stays at -INFINITY.
    for (int step = 0; step < 64 && x < 6.0f; ++step) { result -= 1.0f / x; x += 1.0f; }
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
    // Asymptotic branch is exp(ax)/sqrt(ax); at ax=+inf that's inf/inf=NaN.
    if (isinf(ax)) return INFINITY;
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
    // I1 is odd: I1(+inf) = +inf, I1(-inf) = -inf.  Avoid exp/sqrt NaN.
    if (isinf(ax)) return copysign((float)INFINITY, x);
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

// ===== Missing-kernels audit (MSL, translated from verified CUDA-C) =====
float p210_zeta_scalar(float x, float q) {
    if (x == 1.0f) return INFINITY;
    if (q <= 0.0f && q == floor(q)) return INFINITY;
    float sum = 0.0f;
    for (int k = 0; k < 12; k++) sum += pow(q + (float)k, -x);
    float Nq = 12.0f + q; float lnNq = log(Nq);
    float cont = exp((1.0f - x) * lnNq) / (x - 1.0f);
    float halfTerm = 0.5f * exp(-x * lnNq);
    const float b2k[8] = { 1.0f/6.0f, -1.0f/30.0f, 1.0f/42.0f, -1.0f/30.0f, 5.0f/66.0f, -691.0f/2730.0f, 7.0f/6.0f, -3617.0f/510.0f };
    const float fact2k[8] = { 2.0f, 24.0f, 720.0f, 40320.0f, 3628800.0f, 479001600.0f, 87178291200.0f, 20922789888000.0f };
    float corr = 0.0f; float xPow = exp(-x * lnNq) / Nq; float invNq2 = 1.0f / (Nq * Nq); float rising = x;
    for (int j = 1; j <= 8; j++) {
        if (j > 1) { rising *= (x + 2.0f*(float)(j-1) - 2.0f) * (x + 2.0f*(float)(j-1) - 1.0f); xPow *= invNq2; }
        corr += (b2k[j-1] / fact2k[j-1]) * rising * xPow;
    }
    return sum + cont + halfTerm + corr;
}
float p210_polygamma_scalar(int n, float x) {
    if (x <= 0.0f && x == floor(x)) return INFINITY;
    float recurrence = 0.0f; float xd = x; int signN = (n & 1) == 1 ? 1 : -1;
    while (xd < 10.0f) { recurrence += (float)signN * exp(lgamma((float)(n+1)) - (float)(n+1) * log(xd)); xd += 1.0f; }
    float lnX = log(xd);
    float asympt = exp(lgamma((float)n) - (float)n * lnX) + 0.5f * exp(lgamma((float)(n+1)) - (float)(n+1) * lnX);
    const float b2k[8] = { 1.0f/6.0f, -1.0f/30.0f, 1.0f/42.0f, -1.0f/30.0f, 5.0f/66.0f, -691.0f/2730.0f, 7.0f/6.0f, -3617.0f/510.0f };
    float invX2 = 1.0f / (xd * xd); float xPow = exp(-(float)n * lnX);
    for (int k = 1; k <= 8; k++) { xPow *= invX2; asympt += b2k[k-1] * exp(lgamma((float)(2*k+n)) - lgamma((float)(2*k+1))) * xPow; }
    return recurrence + (float)signN * asympt;
}

static inline void p210_atomicReduceF(device atomic_uint* a, float val, int mode) {
    uint old = atomic_load_explicit(a, memory_order_relaxed);
    bool ok = false;
    while (!ok) {
        float pf = as_type<float>(old);
        float nv = (mode==0)?(pf+val):(mode==1)?(pf*val):(mode==2)?fmax(pf,val):fmin(pf,val);
        uint des = as_type<uint>(nv);
        ok = atomic_compare_exchange_weak_explicit(a, &old, des, memory_order_relaxed, memory_order_relaxed);
    }
}

kernel void parity210_reflect_pad_1d(
    device const float* inp [[buffer(0)]],
    device float* outp [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& L [[buffer(3)]],
    constant int& Lp [[buffer(4)]],
    constant int& pad [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= batch*Lp) return;
    int j = idx % Lp; int b = idx / Lp; int src;
    if (j < pad) src = pad - j; else if (j < pad + L) src = j - pad; else src = L - 2 - (j - pad - L);
    outp[idx] = inp[b*L + src];
}

kernel void parity210_stft_mag_phase(
    device const float* padded [[buffer(0)]],
    device const float* window [[buffer(1)]],
    device float* mag [[buffer(2)]],
    device float* phase [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& Lp [[buffer(5)]],
    constant int& nFft [[buffer(6)]],
    constant int& hop [[buffer(7)]],
    constant int& numFrames [[buffer(8)]],
    constant int& numFreqs [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; int total = batch*numFreqs*numFrames; if (idx >= total) return;
    int frame = idx % numFrames; int tmp = idx / numFrames; int k = tmp % numFreqs; int b = tmp / numFreqs;
    int start = frame * hop; int inOff = b * Lp; float re = 0.0f, im = 0.0f;
    float bk = -2.0f * M_PI_F * (float)k / (float)nFft;
    for (int i = 0; i < nFft; i++) { float x = padded[inOff + start + i] * window[i]; float a = bk * (float)i; re += x*cos(a); im += x*sin(a); }
    int outOff = b*numFreqs*numFrames + k*numFrames + frame;
    mag[outOff] = sqrt(re*re + im*im); phase[outOff] = atan2(im, re);
}

kernel void parity210_phase_vocoder(
    device const float* mag [[buffer(0)]],
    device const float* phase [[buffer(1)]],
    device float* newMag [[buffer(2)]],
    device float* newPhase [[buffer(3)]],
    constant int& leading [[buffer(4)]],
    constant int& nFramesV [[buffer(5)]],
    constant int& nFreqV [[buffer(6)]],
    constant int& outFrames [[buffer(7)]],
    constant float& rate [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= leading*nFreqV) return;
    int f = idx % nFreqV; int b = idx / nFreqV; int stride = nFramesV*nFreqV; int outStride = outFrames*nFreqV;
    float accPhase = 0.0f;
    for (int t = 0; t < outFrames; t++) {
        float srcT = (float)t * rate; int t0 = (int)floor(srcT); int t1 = min(t0+1, nFramesV-1); float frac = srcT - (float)t0;
        float m0 = mag[b*stride + t0*nFreqV + f]; float m1 = mag[b*stride + t1*nFreqV + f];
        newMag[b*outStride + t*nFreqV + f] = (1.0f-frac)*m0 + frac*m1;
        float dp = 0.0f;
        if (t0+1 < nFramesV) { dp = phase[b*stride + (t0+1)*nFreqV + f] - phase[b*stride + t0*nFreqV + f]; dp -= 2.0f*M_PI_F * round(dp/(2.0f*M_PI_F)); }
        accPhase += dp; newPhase[b*outStride + t*nFreqV + f] = accPhase;
    }
}

kernel void parity210_build_spectrum(
    device const float* mag [[buffer(0)]],
    device const float* phase [[buffer(1)]],
    device float* specRe [[buffer(2)]],
    device float* specIm [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& numFreqs [[buffer(5)]],
    constant int& numFrames [[buffer(6)]],
    constant int& nFft [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= batch*numFrames) return;
    int frame = idx % numFrames; int b = idx / numFrames; int magOff = b*numFreqs*numFrames; int specOff = idx * nFft;
    for (int k = 0; k < nFft; k++) { specRe[specOff+k] = 0.0f; specIm[specOff+k] = 0.0f; }
    for (int k = 0; k < numFreqs && k < nFft; k++) { float m = mag[magOff + k*numFrames + frame]; float p = phase[magOff + k*numFrames + frame]; specRe[specOff+k] = m*cos(p); specIm[specOff+k] = m*sin(p); }
    for (int k = 1; k < numFreqs - 1; k++) { int dst = nFft - k; if (dst >= 0 && dst < nFft && k < nFft) { specRe[specOff+dst] = specRe[specOff+k]; specIm[specOff+dst] = -specIm[specOff+k]; } }
}

kernel void parity210_istft_from_spectrum(
    device const float* specRe [[buffer(0)]],
    device const float* specIm [[buffer(1)]],
    device const float* window [[buffer(2)]],
    device atomic_float* result [[buffer(3)]],
    device atomic_float* windowSum [[buffer(4)]],
    constant int& batch [[buffer(5)]],
    constant int& numFrames [[buffer(6)]],
    constant int& nFft [[buffer(7)]],
    constant int& hop [[buffer(8)]],
    constant int& outputLength [[buffer(9)]],
    constant int& center [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; int total = batch*numFrames*nFft; if (idx >= total) return;
    int i = idx % nFft; int tmp = idx / nFft; int frame = tmp % numFrames; int b = tmp / numFrames; int specOff = (b*numFrames + frame) * nFft;
    float acc = 0.0f;
    for (int k = 0; k < nFft; k++) { float a = 2.0f*M_PI_F*(float)k*(float)i/(float)nFft; acc += specRe[specOff+k]*cos(a) - specIm[specOff+k]*sin(a); }
    int writeStart = center ? max(0, frame*hop - nFft/2) : frame*hop; int outIdx = writeStart + i;
    if (outIdx >= 0 && outIdx < outputLength) { float w = window[i]; atomic_fetch_add_explicit(&result[b*outputLength + outIdx], acc*(1.0f/(float)nFft)*w, memory_order_relaxed); atomic_fetch_add_explicit(&windowSum[b*outputLength + outIdx], w*w, memory_order_relaxed); }
}

kernel void parity210_istft_normalize(
    device float* result [[buffer(0)]],
    device const float* windowSum [[buffer(1)]],
    constant int& total [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= total) return;
    float ws = windowSum[idx]; if (ws > 1e-8f) result[idx] = result[idx] / ws;
}

kernel void parity210_logical_op(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& mode [[buffer(3)]],
    constant int& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    int i = (int)gid; if (i >= n) return;
    int ba = (a[i] != 0.0f); int bb = (b[i] != 0.0f);
    int r = (mode == 0) ? (ba && bb) : (mode == 1) ? (ba || bb) : (ba != bb);
    out[i] = r ? 1.0f : 0.0f;
}

kernel void parity210_logical_not(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    int i = (int)gid; if (i >= n) return;
    out[i] = (a[i] != 0.0f) ? 0.0f : 1.0f;
}

kernel void parity210_shifted_diff(
    device const float* x [[buffer(0)]],
    device float* mask [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    int i = (int)gid; if (i >= n) return;
    mask[i] = (i == 0 || x[i] != x[i-1]) ? 1.0f : 0.0f;
}

kernel void parity210_masks_to_boxes(
    device const float* masks [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& H [[buffer(3)]],
    constant int& W [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    int n = (int)gid; if (n >= N) return;
    int xMin = W, yMin = H, xMax = -1, yMax = -1; int planeOff = n * H * W;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (masks[planeOff + y*W + x] != 0.0f) { if (x<xMin)xMin=x; if (x>xMax)xMax=x; if (y<yMin)yMin=y; if (y>yMax)yMax=y; }
    int o = n * 4;
    if (xMax < 0) { out[o]=0.0f; out[o+1]=0.0f; out[o+2]=0.0f; out[o+3]=0.0f; }
    else { out[o]=(float)xMin; out[o+1]=(float)yMin; out[o+2]=(float)xMax; out[o+3]=(float)yMax; }
}

kernel void parity210_pairwise_iou(
    device const float* boxes [[buffer(0)]],
    device float* iou [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= N*N) return; int j = idx % N; int i = idx / N;
    float ix1=boxes[i*4],iy1=boxes[i*4+1],ix2=boxes[i*4+2],iy2=boxes[i*4+3];
    float jx1=boxes[j*4],jy1=boxes[j*4+1],jx2=boxes[j*4+2],jy2=boxes[j*4+3];
    float ai=fmax(0.0f,ix2-ix1)*fmax(0.0f,iy2-iy1); float aj=fmax(0.0f,jx2-jx1)*fmax(0.0f,jy2-jy1);
    float iw=fmax(0.0f,fmin(ix2,jx2)-fmax(ix1,jx1)); float ih=fmax(0.0f,fmin(iy2,jy2)-fmax(iy1,jy1));
    float inter=iw*ih; float uni=ai+aj-inter; iou[idx]=(uni>0.0f)?inter/uni:0.0f;
}

kernel void parity210_histogramdd(
    device const float* samples [[buffer(0)]],
    device atomic_float* hist [[buffer(1)]],
    device const int* bins [[buffer(2)]],
    device const float* mins [[buffer(3)]],
    device const float* maxs [[buffer(4)]],
    constant int& n [[buffer(5)]],
    constant int& d [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    int i = (int)gid; if (i >= n) return; int linIdx = 0; int valid = 1;
    for (int k = 0; k < d; k++) { float v = samples[i*d + k]; float mn = mins[k]; float mx = maxs[k]; if (!(v >= mn && v <= mx)) { valid = 0; break; } float width = (mx - mn) / (float)bins[k]; int kIdx = (int)floor((v - mn) / width); if (kIdx >= bins[k]) kIdx = bins[k] - 1; if (kIdx < 0) kIdx = 0; linIdx = linIdx * bins[k] + kIdx; }
    if (valid) atomic_fetch_add_explicit(&hist[linIdx], 1.0f, memory_order_relaxed);
}

kernel void parity210_gridsample_backward_input(
    device const float* gradOut [[buffer(0)]],
    device const float* grid [[buffer(1)]],
    device atomic_float* gradIn [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& C [[buffer(6)]],
    constant int& outH [[buffer(7)]],
    constant int& outW [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; int total = batch*outH*outW*C; if (idx >= total) return;
    int c = idx % C; int tmp = idx / C; int ow = tmp % outW; tmp /= outW; int oh = tmp % outH; int b = tmp / outH;
    int gridBase = ((b*outH + oh)*outW + ow)*2; float gx = grid[gridBase]; float gy = grid[gridBase+1];
    float srcH = (gy + 1.0f) * 0.5f * (float)(H - 1); float srcW = (gx + 1.0f) * 0.5f * (float)(W - 1);
    if (srcH <= -1.0f || srcH >= (float)H || srcW <= -1.0f || srcW >= (float)W) return;
    int h0 = (int)floor(srcH); int h1 = h0 + 1; int w0 = (int)floor(srcW); int w1 = w0 + 1;
    float lh = srcH - (float)h0; float lw = srcW - (float)w0; float g = gradOut[idx];
    if (h0>=0 && h0<H && w0>=0 && w0<W) atomic_fetch_add_explicit(&gradIn[((b*H+h0)*W+w0)*C+c], g*(1.0f-lh)*(1.0f-lw), memory_order_relaxed);
    if (h0>=0 && h0<H && w1>=0 && w1<W) atomic_fetch_add_explicit(&gradIn[((b*H+h0)*W+w1)*C+c], g*(1.0f-lh)*lw, memory_order_relaxed);
    if (h1>=0 && h1<H && w0>=0 && w0<W) atomic_fetch_add_explicit(&gradIn[((b*H+h1)*W+w0)*C+c], g*lh*(1.0f-lw), memory_order_relaxed);
    if (h1>=0 && h1<H && w1>=0 && w1<W) atomic_fetch_add_explicit(&gradIn[((b*H+h1)*W+w1)*C+c], g*lh*lw, memory_order_relaxed);
}

kernel void parity210_gridsample_backward_grid(
    device const float* gradOut [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* grid [[buffer(2)]],
    device float* gradGrid [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& H [[buffer(5)]],
    constant int& W [[buffer(6)]],
    constant int& C [[buffer(7)]],
    constant int& outH [[buffer(8)]],
    constant int& outW [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= batch*outH*outW) return;
    int ow = idx % outW; int tmp = idx / outW; int oh = tmp % outH; int b = tmp / outH;
    int gridBase = ((b*outH+oh)*outW+ow)*2; float gx = grid[gridBase]; float gy = grid[gridBase+1];
    float srcH = (gy+1.0f)*0.5f*(float)(H-1); float srcW = (gx+1.0f)*0.5f*(float)(W-1);
    float gradGx = 0.0f; float gradGy = 0.0f;
    if (!(srcH <= -1.0f || srcH >= (float)H || srcW <= -1.0f || srcW >= (float)W)) {
        int h0 = (int)floor(srcH); int h1 = h0+1; int w0 = (int)floor(srcW); int w1 = w0+1;
        float lh = srcH-(float)h0; float lw = srcW-(float)w0;
        int in00 = (h0>=0&&h0<H&&w0>=0&&w0<W); int in01 = (h0>=0&&h0<H&&w1>=0&&w1<W);
        int in10 = (h1>=0&&h1<H&&w0>=0&&w0<W); int in11 = (h1>=0&&h1<H&&w1>=0&&w1<W);
        for (int c = 0; c < C; c++) {
            float v00 = in00 ? input[((b*H+h0)*W+w0)*C+c] : 0.0f; float v01 = in01 ? input[((b*H+h0)*W+w1)*C+c] : 0.0f;
            float v10 = in10 ? input[((b*H+h1)*W+w0)*C+c] : 0.0f; float v11 = in11 ? input[((b*H+h1)*W+w1)*C+c] : 0.0f;
            float dH = (1.0f-lw)*(v10-v00) + lw*(v11-v01); float dW = (1.0f-lh)*(v01-v00) + lh*(v11-v10);
            float go = gradOut[((b*outH+oh)*outW+ow)*C+c];
            gradGx += go * dW * (float)(W-1)*0.5f; gradGy += go * dH * (float)(H-1)*0.5f;
        }
    }
    gradGrid[gridBase] = gradGx; gradGrid[gridBase+1] = gradGy;
}

kernel void parity210_take_along_dim_f(
    device const float* input [[buffer(0)]],
    device const float* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& outerSize [[buffer(3)]],
    constant int& axisOut [[buffer(4)]],
    constant int& innerSize [[buffer(5)]],
    constant int& axisIn [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; int total = outerSize*axisOut*innerSize; if (idx >= total) return;
    int inner = idx % innerSize; int outer = (idx / innerSize) / axisOut; int srcJ = (int)indices[idx];
    if (srcJ < 0 || srcJ >= axisIn) { output[idx] = 0.0f; return; }
    output[idx] = input[(outer * axisIn + srcJ) * innerSize + inner];
}

kernel void parity210_cross3(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& outerSize [[buffer(3)]],
    constant int& innerSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= outerSize * innerSize) return;
    int inner = idx % innerSize; int outer = idx / innerSize; int p = outer * 3 * innerSize + inner;
    float a0=a[p],a1=a[p+innerSize],a2=a[p+2*innerSize]; float b0=b[p],b1=b[p+innerSize],b2=b[p+2*innerSize];
    output[p]=a1*b2-a2*b1; output[p+innerSize]=a2*b0-a0*b2; output[p+2*innerSize]=a0*b1-a1*b0;
}

kernel void parity210_ldexp(
    device const float* input [[buffer(0)]],
    device const int* exponents [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= size) return; output[idx] = ldexp(input[idx], exponents[idx]);
}

kernel void parity210_kron2d(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& am [[buffer(3)]],
    constant int& an [[buffer(4)]],
    constant int& bp [[buffer(5)]],
    constant int& bq [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; int outCols = an*bq; int total=(am*bp)*outCols; if (idx>=total) return;
    int oc=idx%outCols; int orow=idx/outCols; int i=orow/bp; int k=orow%bp; int j=oc/bq; int l=oc%bq;
    output[idx] = a[i*an+j] * b[k*bq+l];
}

kernel void parity210_search_sorted(
    device const float* seq [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& seqLen [[buffer(3)]],
    constant int& numValues [[buffer(4)]],
    constant int& right [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= numValues) return; float v = values[idx]; int lo=0, hi=seqLen;
    while (lo < hi) { int mid=(lo+hi)>>1; int cond=(right!=0)?(seq[mid]<=v):(seq[mid]<v); if (cond) lo=mid+1; else hi=mid; }
    output[idx] = (float)lo;
}

kernel void parity210_next_after(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= size) return; float av=a[idx], bv=b[idx];
    uint ua=as_type<uint>(av), ub=as_type<uint>(bv);
    int aNan=(((ua>>23)&0xFFu)==0xFFu)&&((ua&0x7FFFFFu)!=0u); int bNan=(((ub>>23)&0xFFu)==0xFFu)&&((ub&0x7FFFFFu)!=0u);
    if (aNan||bNan) { output[idx]=as_type<float>(0x7FC00000u); return; }
    if (av==bv) { output[idx]=bv; return; }
    if (av==0.0f) { output[idx]=as_type<float>(bv>0.0f?0x00000001u:0x80000001u); return; }
    uint r=ua;
    if (bv>av) r=(av>0.0f)?(r+1u):(r-1u); else r=(av>0.0f)?(r-1u):(r+1u);
    output[idx]=as_type<float>(r);
}

kernel void parity210_index_write(
    device float* output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* source [[buffer(2)]],
    constant float& fillValue [[buffer(3)]],
    constant int& mode [[buffer(4)]],
    constant int& outerSize [[buffer(5)]],
    constant int& idxAxis [[buffer(6)]],
    constant int& innerSize [[buffer(7)]],
    constant int& dstAxis [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; int total=outerSize*idxAxis*innerSize; if (idx>=total) return;
    int inner=idx%innerSize; int j=(idx/innerSize)%idxAxis; int outer=(idx/innerSize)/idxAxis; int dstJ=indices[j];
    if (dstJ<0||dstJ>=dstAxis) return; float v=(mode==0)?source[idx]:fillValue;
    output[(outer*dstAxis+dstJ)*innerSize+inner]=v;
}

kernel void parity210_cdist(
    device const float* x1 [[buffer(0)]],
    device const float* x2 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& m [[buffer(3)]],
    constant int& n [[buffer(4)]],
    constant int& d [[buffer(5)]],
    constant float& p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx>=m*n) return; int j=idx%n; int i=idx/n; float sum=0.0f;
    for (int k=0;k<d;k++){ float diff=fabs(x1[i*d+k]-x2[j*d+k]); if (p==1.0f) sum+=diff; else if (p==2.0f) sum+=diff*diff; else sum+=pow(diff,p); }
    output[idx]=(p==1.0f)?sum:(p==2.0f)?sqrt(sum):pow(sum,1.0f/p);
}

kernel void parity210_pdist(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant int& d [[buffer(3)]],
    constant float& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    int flat = (int)gid; if (flat>=n*n) return; int j=flat%n; int i=flat/n; if (i>=j) return; float sum=0.0f;
    for (int k=0;k<d;k++){ float diff=fabs(input[i*d+k]-input[j*d+k]); if (p==1.0f) sum+=diff; else if (p==2.0f) sum+=diff*diff; else sum+=pow(diff,p); }
    float dist=(p==1.0f)?sum:(p==2.0f)?sqrt(sum):pow(sum,1.0f/p); int outIdx=i*n-(i*(i+1))/2+(j-i-1); output[outIdx]=dist;
}

kernel void parity210_histc(
    device const float* input [[buffer(0)]],
    device atomic_float* hist [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant int& bins [[buffer(3)]],
    constant float& mn [[buffer(4)]],
    constant float& mx [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx>=n) return; float x=input[idx]; if (!(x>=mn&&x<=mx)) return;
    float bw=(mx-mn)/(float)bins; int b=(int)((x-mn)/bw); if (b>=bins) b=bins-1; if (b<0) b=0; atomic_fetch_add_explicit(&hist[b], 1.0f, memory_order_relaxed);
}

kernel void parity210_bitonic_step(
    device float* values [[buffer(0)]],
    device float* indices [[buffer(1)]],
    constant int& rowLen [[buffer(2)]],
    constant int& k [[buffer(3)]],
    constant int& j [[buffer(4)]],
    constant int& numRows [[buffer(5)]],
    constant int& descending [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    int gi = (int)gid; if (gi>=numRows*rowLen) return; int i=gi%rowLen; int ixj=i^j; if (ixj<=i) return;
    int base=(gi/rowLen)*rowLen; float a=values[base+i], b=values[base+ixj];
    uint ua=as_type<uint>(a), ub=as_type<uint>(b);
    float ka=(((ua>>23)&0xFFu)==0xFFu&&(ua&0x7FFFFFu)!=0u)?INFINITY:a; float kb=(((ub>>23)&0xFFu)==0xFFu&&(ub&0x7FFFFFu)!=0u)?INFINITY:b;
    int up=((i&k)==0); if (descending!=0) up=!up; int doSwap=up?(ka>kb):(ka<kb);
    if (doSwap) { values[base+i]=b; values[base+ixj]=a; float t=indices[base+i]; indices[base+i]=indices[base+ixj]; indices[base+ixj]=t; }
}

kernel void parity210_copy_rows(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant int& srcRowLen [[buffer(2)]],
    constant int& dstRowLen [[buffer(3)]],
    constant int& numRows [[buffer(4)]],
    constant int& copyLen [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    int gi = (int)gid; if (gi>=numRows*copyLen) return; int i=gi%copyLen; int r=gi/copyLen; dst[r*dstRowLen+i]=src[r*srcRowLen+i];
}

kernel void parity210_iota_pad(
    device float* idx [[buffer(0)]],
    constant int& L [[buffer(1)]],
    constant int& P [[buffer(2)]],
    constant int& numRows [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int gi = (int)gid; if (gi>=numRows*P) return; int i=gi%P; idx[gi]=(i<L)?(float)i:-1.0f;
}

kernel void parity210_hsoftmax_paths(
    device const float* acts [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& treeDepth [[buffer(3)]],
    constant int& numClasses [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx>=rows*numClasses) return; int c=idx%numClasses; int r=idx/numClasses; int gbase=r*treeDepth; float prob=1.0f; int node=1;
    for (int level=0; level<treeDepth; level++) { int goRight=(c&(1<<(treeDepth-level-1)))!=0; float g=1.0f/(1.0f+exp(-acts[gbase+level])); prob*=goRight?g:(1.0f-g); node=node*2+(goRight?1:0); if (node>=numClasses) break; }
    out[idx]=prob;
}

kernel void parity210_isin(
    device const float* elements [[buffer(0)]],
    device const float* sortedTest [[buffer(1)]],
    device float* mask [[buffer(2)]],
    constant int& numElements [[buffer(3)]],
    constant int& testLen [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx>=numElements) return; float v=elements[idx]; int lo=0, hi=testLen;
    while (lo<hi) { int mid=(lo+hi)>>1; if (sortedTest[mid]<v) lo=mid+1; else hi=mid; } mask[idx]=(lo<testLen&&sortedTest[lo]==v)?1.0f:0.0f;
}

kernel void parity210_copy_block_2d(
    device const float* block [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& blockRows [[buffer(2)]],
    constant int& blockCols [[buffer(3)]],
    constant int& totalCols [[buffer(4)]],
    constant int& rowOff [[buffer(5)]],
    constant int& colOff [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx>=blockRows*blockCols) return; int j=idx%blockCols; int i=idx/blockCols; output[(rowOff+i)*totalCols+(colOff+j)]=block[i*blockCols+j];
}

kernel void parity210_scatter_reduce(
    device atomic_uint* output [[buffer(0)]],
    device const float* source [[buffer(1)]],
    device const int* index [[buffer(2)]],
    constant int& outerSize [[buffer(3)]],
    constant int& srcDim [[buffer(4)]],
    constant int& dstDim [[buffer(5)]],
    constant int& innerSize [[buffer(6)]],
    constant int& mode [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    int idx=(int)gid; int total=outerSize*srcDim*innerSize; if (idx>=total) return;
    int inner=idx%innerSize; int tmp=idx/innerSize; int outer=tmp/srcDim; int t=index[idx];
    if (t<0||t>=dstDim) return; p210_atomicReduceF(&output[(outer*dstDim+t)*innerSize+inner], source[idx], mode);
}

kernel void parity210_unfold(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant int& outerSize [[buffer(2)]],
    constant int& dimSize [[buffer(3)]],
    constant int& innerSize [[buffer(4)]],
    constant int& nWindows [[buffer(5)]],
    constant int& size [[buffer(6)]],
    constant int& step [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; int total=outerSize*nWindows*innerSize*size; if (idx>=total) return; int s=idx%size; int tmp=idx/size; int inner=tmp%innerSize; tmp/=innerSize; int w=tmp%nWindows; int outer=tmp/nWindows;
    dst[idx]=src[(outer*dimSize+(w*step+s))*innerSize+inner];
}

kernel void parity210_classify_float(
    device const float* a [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& mode [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx>=size) return; uint bits=as_type<uint>(a[idx]); uint expo=(bits>>23)&0xFFu; uint mant=bits&0x7FFFFFu;
    int isNan=(expo==0xFFu)&&(mant!=0u); int isInf=(expo==0xFFu)&&(mant==0u); float r;
    if (mode==0) r=isNan?1.0f:0.0f; else if (mode==1) r=isInf?1.0f:0.0f; else r=(!isNan&&!isInf)?1.0f:0.0f; output[idx]=r;
}

kernel void parity210_zeta(
    device const float* x [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int i = (int)gid; if (i>=size) return; out[i]=p210_zeta_scalar(x[i], q[i]);
}

kernel void parity210_polygamma(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int i = (int)gid; if (i>=size) return; out[i]=p210_polygamma_scalar(n, x[i]);
}

kernel void parity210_rwkv7_forward(
    device const float* R [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* A [[buffer(3)]],
    device const float* B [[buffer(4)]],
    device float* outp [[buffer(5)]],
    device float* Sbuf [[buffer(6)]],
    constant int& batch [[buffer(7)]],
    constant int& seqLen [[buffer(8)]],
    constant int& modelDim [[buffer(9)]],
    constant int& numHeads [[buffer(10)]],
    constant int& headDim [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
    int bh = (int)gid; if (bh>=batch*numHeads) return; int b=bh/numHeads; int h=bh%numHeads; int hOff=h*headDim; int hh=headDim*headDim; float* S=Sbuf+bh*hh;
    for (int i=0;i<hh;i++) S[i]=0.0f;
    for (int t=0;t<seqLen;t++) {
        int baseOff=(b*seqLen+t)*modelDim+hOff;
        for (int di=0;di<headDim;di++) { float ga=1.0f/(1.0f+exp(-A[baseOff+di])); float gbk=(1.0f/(1.0f+exp(-B[baseOff+di])))*K[baseOff+di]; int srow=di*headDim; for (int vi=0;vi<headDim;vi++) S[srow+vi]=ga*S[srow+vi]+gbk*V[baseOff+vi]; }
        for (int di=0;di<headDim;di++) { int srow=di*headDim; float sk=0.0f; for (int vi=0;vi<headDim;vi++) sk+=S[srow+vi]*K[baseOff+vi]; outp[baseOff+di]=(1.0f/(1.0f+exp(-R[baseOff+di])))*sk; }
    }
}
";
    }
}
