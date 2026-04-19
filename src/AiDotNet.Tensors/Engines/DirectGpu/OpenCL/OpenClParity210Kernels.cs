#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// OpenCL C kernels for the parity-210 op surface. Mirrors
/// CudaParity210Kernels / HipParity210Kernels / MetalParity210Kernels /
/// VulkanParity210Kernels function-for-function.
/// </summary>
public static class OpenClParity210Kernels
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

    public static string GetSource() => @"
// OpenCL 1.2 parity-210 kernels.  float, log, exp, hypot, fmod, floor,
// pow, erf, erfc, lgamma are all standard OpenCL C built-ins — we use
// native_* only where accuracy is acceptable.

// ============================================================================
// MOVEMENT
// ============================================================================

__kernel void parity210_roll_1d(
    __global const float* input, __global float* output,
    const int outerSize, const int axisSize, const int innerSize, const int shift)
{
    int gid = get_global_id(0);
    int total = outerSize * axisSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int a = tmp % axisSize;
    int outer = tmp / axisSize;
    int src = ((a - shift) % axisSize + axisSize) % axisSize;
    output[gid] = input[(outer * axisSize + src) * innerSize + inner];
}

__kernel void parity210_flip_axis(
    __global const float* input, __global float* output,
    const int outerSize, const int axisSize, const int innerSize)
{
    int gid = get_global_id(0);
    int total = outerSize * axisSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int a = tmp % axisSize;
    int outer = tmp / axisSize;
    int src = axisSize - 1 - a;
    output[gid] = input[(outer * axisSize + src) * innerSize + inner];
}

__kernel void parity210_triu(
    __global const float* input, __global float* output,
    const int batchSize, const int rows, const int cols, const int diagonal)
{
    int gid = get_global_id(0);
    int total = batchSize * rows * cols;
    if (gid >= total) return;
    int col = gid % cols;
    int tmp = gid / cols;
    int row = tmp % rows;
    output[gid] = ((col - row) >= diagonal) ? input[gid] : 0.0f;
}

__kernel void parity210_tril(
    __global const float* input, __global float* output,
    const int batchSize, const int rows, const int cols, const int diagonal)
{
    int gid = get_global_id(0);
    int total = batchSize * rows * cols;
    if (gid >= total) return;
    int col = gid % cols;
    int tmp = gid / cols;
    int row = tmp % rows;
    output[gid] = ((col - row) <= diagonal) ? input[gid] : 0.0f;
}

__kernel void parity210_diag_embed(
    __global const float* input, __global float* output,
    const int batchSize, const int diagLen, const int matSize, const int offset)
{
    int gid = get_global_id(0);
    int total = batchSize * matSize * matSize;
    if (gid >= total) return;
    int col = gid % matSize;
    int tmp = gid / matSize;
    int row = tmp % matSize;
    int b = tmp / matSize;
    int diagRow = (offset >= 0) ? row : row + offset;
    int diagCol = (offset >= 0) ? col - offset : col;
    if (diagRow == diagCol && diagRow >= 0 && diagRow < diagLen)
        output[gid] = input[b * diagLen + diagRow];
    else
        output[gid] = 0.0f;
}

// ============================================================================
// CUMULATIVE
// ============================================================================

__kernel void parity210_cumsum_axis(
    __global const float* input, __global float* output,
    const int outerSize, const int axisSize, const int innerSize)
{
    int gid = get_global_id(0);
    int total = outerSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int outer = gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = 0.0f;
    for (int k = 0; k < axisSize; ++k) {
        acc += input[base_ + k * innerSize];
        output[base_ + k * innerSize] = acc;
    }
}

__kernel void parity210_cumprod_axis(
    __global const float* input, __global float* output,
    const int outerSize, const int axisSize, const int innerSize)
{
    int gid = get_global_id(0);
    int total = outerSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int outer = gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = 1.0f;
    for (int k = 0; k < axisSize; ++k) {
        acc *= input[base_ + k * innerSize];
        output[base_ + k * innerSize] = acc;
    }
}

__kernel void parity210_cummax_axis(
    __global const float* input, __global float* output,
    const int outerSize, const int axisSize, const int innerSize)
{
    int gid = get_global_id(0);
    int total = outerSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int outer = gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = -INFINITY;
    for (int k = 0; k < axisSize; ++k) {
        float v = input[base_ + k * innerSize];
        if (v > acc) acc = v;
        output[base_ + k * innerSize] = acc;
    }
}

__kernel void parity210_cummin_axis(
    __global const float* input, __global float* output,
    const int outerSize, const int axisSize, const int innerSize)
{
    int gid = get_global_id(0);
    int total = outerSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int outer = gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = INFINITY;
    for (int k = 0; k < axisSize; ++k) {
        float v = input[base_ + k * innerSize];
        if (v < acc) acc = v;
        output[base_ + k * innerSize] = acc;
    }
}

__kernel void parity210_logcumsumexp_axis(
    __global const float* input, __global float* output,
    const int outerSize, const int axisSize, const int innerSize)
{
    int gid = get_global_id(0);
    int total = outerSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int outer = gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float m = -INFINITY;
    float s = 0.0f;
    for (int k = 0; k < axisSize; ++k) {
        float x = input[base_ + k * innerSize];
        if (x > m) { s = s * exp(m - x) + 1.0f; m = x; }
        else { s += exp(x - m); }
        output[base_ + k * innerSize] = m + log(s);
    }
}

// ============================================================================
// INDEXING
// ============================================================================

__kernel void parity210_take_linear(
    __global const float* input, __global const int* indices,
    __global float* output, const int outSize, const int inputLinearLen)
{
    int gid = get_global_id(0);
    if (gid >= outSize) return;
    int pos = indices[gid];
    output[gid] = (pos >= 0 && pos < inputLinearLen) ? input[pos] : 0.0f;
}

__kernel void parity210_take_along_dim(
    __global const float* input, __global const int* indices,
    __global float* output,
    const int outerSize, const int idxAxis, const int innerSize, const int srcAxis)
{
    int gid = get_global_id(0);
    int total = outerSize * idxAxis * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int i = tmp % idxAxis;
    int outer = tmp / idxAxis;
    int target = indices[gid];
    int srcIdx = (outer * srcAxis + target) * innerSize + inner;
    output[gid] = (target >= 0 && target < srcAxis) ? input[srcIdx] : 0.0f;
}

// OpenCL 2.0 atomic_fetch_add on float via cl_khr_int32_base_atomics +
// compare-and-swap loop (works on OpenCL 1.2 too). We emulate with
// atomic_cmpxchg on the int reinterpretation.
inline void p210_atomic_add(__global volatile float* addr, float val) {
    union { unsigned int i; float f; } old_u, new_u;
    do {
        old_u.f = *addr;
        new_u.f = old_u.f + val;
    } while (atomic_cmpxchg((__global volatile unsigned int*)addr, old_u.i, new_u.i) != old_u.i);
}

__kernel void parity210_index_add(
    __global volatile float* output, __global const int* indices,
    __global const float* source,
    const int outerSize, const int dstAxis, const int innerSize, const int idxLen)
{
    int gid = get_global_id(0);
    int total = outerSize * idxLen * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;
    int target = indices[i];
    if (target < 0 || target >= dstAxis) return;
    int dstPos = (outer * dstAxis + target) * innerSize + inner;
    int srcPos = (outer * idxLen + i) * innerSize + inner;
    p210_atomic_add(&output[dstPos], source[srcPos]);
}

__kernel void parity210_index_copy(
    __global float* output, __global const int* indices,
    __global const float* source,
    const int outerSize, const int dstAxis, const int innerSize, const int idxLen)
{
    int gid = get_global_id(0);
    int total = outerSize * idxLen * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;
    int target = indices[i];
    if (target < 0 || target >= dstAxis) return;
    output[(outer * dstAxis + target) * innerSize + inner] =
        source[(outer * idxLen + i) * innerSize + inner];
}

__kernel void parity210_index_fill(
    __global float* output, __global const int* indices,
    const float fillValue,
    const int outerSize, const int dstAxis, const int innerSize, const int idxLen)
{
    int gid = get_global_id(0);
    int total = outerSize * idxLen * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;
    int target = indices[i];
    if (target < 0 || target >= dstAxis) return;
    output[(outer * dstAxis + target) * innerSize + inner] = fillValue;
}

__kernel void parity210_masked_scatter(
    __global float* output, __global const char* mask,
    __global const int* prefixSum, __global const float* source,
    const int total)
{
    int gid = get_global_id(0);
    if (gid >= total) return;
    if (mask[gid]) output[gid] = source[prefixSum[gid]];
}

// ============================================================================
// ELEMENT-WISE BINARY
// ============================================================================

__kernel void parity210_hypot(
    __global const float* a, __global const float* b,
    __global float* out, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    out[gid] = hypot(a[gid], b[gid]);
}

__kernel void parity210_copysign(
    __global const float* a, __global const float* b,
    __global float* out, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    out[gid] = copysign(a[gid], b[gid]);
}

__kernel void parity210_fmod(
    __global const float* a, __global const float* b,
    __global float* out, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float bv = b[gid];
    out[gid] = (bv == 0.0f) ? 0.0f : fmod(a[gid], bv);
}

__kernel void parity210_remainder(
    __global const float* a, __global const float* b,
    __global float* out, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float av = a[gid], bv = b[gid];
    if (bv == 0.0f) { out[gid] = 0.0f; return; }
    float q = floor(av / bv);
    out[gid] = av - q * bv;
}

__kernel void parity210_float_power(
    __global const float* a, __global const float* b,
    __global float* out, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    out[gid] = pow(a[gid], b[gid]);
}

__kernel void parity210_log_add_exp(
    __global const float* a, __global const float* b,
    __global float* out, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float av = a[gid], bv = b[gid];
    float m = fmax(av, bv);
    float s = fmin(av, bv);
    out[gid] = m + log(1.0f + exp(s - m));
}

__kernel void parity210_log_add_exp2(
    __global const float* a, __global const float* b,
    __global float* out, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float av = a[gid], bv = b[gid];
    float m = fmax(av, bv);
    float s = fmin(av, bv);
    float inv_ln2 = 1.4426950408889634f;
    out[gid] = m + log(1.0f + exp((s - m) * 0.6931471805599453f)) * inv_ln2;
}

__kernel void parity210_xlogy(
    __global const float* x, __global const float* y,
    __global float* out, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float xv = x[gid];
    out[gid] = (xv == 0.0f) ? 0.0f : xv * log(y[gid]);
}

__kernel void parity210_xlog1py(
    __global const float* x, __global const float* y,
    __global float* out, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float xv = x[gid];
    out[gid] = (xv == 0.0f) ? 0.0f : xv * log1p(y[gid]);
}

// ============================================================================
// ELEMENT-WISE UNARY SPECIAL
// ============================================================================

__kernel void parity210_erfc(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    output[gid] = erfc(input[gid]);
}

__kernel void parity210_erfinv(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float y = input[gid];
    if (y >= 1.0f) { output[gid] = INFINITY; return; }
    if (y <= -1.0f) { output[gid] = -INFINITY; return; }
    float ln = log(1.0f - y * y);
    float a = 0.147f;
    const float pi = 3.14159265358979f;
    float t = 2.0f / (pi * a) + ln * 0.5f;
    float xs = copysign(sqrt(sqrt(t * t - ln / a) - t), y);
    for (int k = 0; k < 2; ++k) {
        float e = erf(xs);
        float df = 2.0f / sqrt(pi) * exp(-xs * xs);
        xs -= (e - y) / df;
    }
    output[gid] = xs;
}

__kernel void parity210_lgamma_approx(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    output[gid] = lgamma(input[gid]);
}

__kernel void parity210_digamma(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
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

__kernel void parity210_i0(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    output[gid] = p210_i0(input[gid]);
}

__kernel void parity210_i1(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    output[gid] = p210_i1(input[gid]);
}

__kernel void parity210_i0e(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float x = input[gid];
    output[gid] = exp(-fabs(x)) * p210_i0(x);
}

__kernel void parity210_i1e(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float x = input[gid];
    output[gid] = exp(-fabs(x)) * p210_i1(x);
}

// ============================================================================
// PREDICATES + NUMERIC HYGIENE
// ============================================================================

__kernel void parity210_is_finite(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    output[gid] = isfinite(input[gid]) ? 1.0f : 0.0f;
}

__kernel void parity210_is_nan(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    output[gid] = isnan(input[gid]) ? 1.0f : 0.0f;
}

__kernel void parity210_is_inf(
    __global const float* input, __global float* output, const int size)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    output[gid] = isinf(input[gid]) ? 1.0f : 0.0f;
}

__kernel void parity210_nan_to_num(
    __global const float* input, __global float* output,
    const int size, const float nanVal, const float posInfVal, const float negInfVal)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float x = input[gid];
    if (isnan(x)) output[gid] = nanVal;
    else if (isinf(x)) output[gid] = (x > 0.0f) ? posInfVal : negInfVal;
    else output[gid] = x;
}

// ============================================================================
// PAIRWISE
// ============================================================================

__kernel void parity210_cosine_similarity_last(
    __global const float* a, __global const float* b,
    __global float* out, const int n, const int d, const float eps)
{
    int row = get_global_id(0);
    if (row >= n) return;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    int base_ = row * d;
    for (int k = 0; k < d; ++k) {
        float av = a[base_ + k], bv = b[base_ + k];
        dot += av * bv; na += av * av; nb += bv * bv;
    }
    float denom = fmax(sqrt(na * nb), eps);
    out[row] = dot / denom;
}

__kernel void parity210_cdist_l2(
    __global const float* x1, __global const float* x2,
    __global float* out, const int n, const int m, const int d)
{
    int gid = get_global_id(0);
    int total = n * m;
    if (gid >= total) return;
    int j = gid % m;
    int i = gid / m;
    float acc = 0.0f;
    for (int k = 0; k < d; ++k) {
        float v = x1[i * d + k] - x2[j * d + k];
        acc += v * v;
    }
    out[gid] = sqrt(acc);
}

// ============================================================================
// CLAMP
// ============================================================================

__kernel void parity210_clamp_min_max(
    __global const float* input,
    __global const float* lo, __global const float* hi,
    __global float* output,
    const int size, const int hasLo, const int hasHi)
{
    int gid = get_global_id(0);
    if (gid >= size) return;
    float x = input[gid];
    if (hasLo) { float l = lo[gid]; if (x < l) x = l; }
    if (hasHi) { float h = hi[gid]; if (x > h) x = h; }
    output[gid] = x;
}
";
}
#endif
