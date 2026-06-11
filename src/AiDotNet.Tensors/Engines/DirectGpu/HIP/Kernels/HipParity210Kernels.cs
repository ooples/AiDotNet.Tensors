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
            "parity210_take_linear","parity210_take_along_dim","parity210_index_add","parity210_index_add_deterministic",
            "parity210_index_copy","parity210_index_fill","parity210_masked_scatter",
            "parity210_hypot","parity210_copysign","parity210_fmod","parity210_remainder",
            "parity210_float_power","parity210_log_add_exp","parity210_log_add_exp2",
            "parity210_xlogy","parity210_xlog1py",
            "parity210_erfc","parity210_erfinv","parity210_lgamma_approx","parity210_digamma",
            "parity210_i0","parity210_i1","parity210_i0e","parity210_i1e",
            "parity210_is_finite","parity210_is_nan","parity210_is_inf","parity210_nan_to_num",
            "parity210_cosine_similarity_last","parity210_cdist_l2",
            "parity210_clamp_min_max",
            // Missing-kernels audit (audio/FFT/logical/detection/geometry/movement/indexing/special)
            "parity210_reflect_pad_1d","parity210_stft_mag_phase","parity210_phase_vocoder",
            "parity210_build_spectrum","parity210_istft_from_spectrum","parity210_istft_normalize",
            "parity210_logical_op","parity210_logical_not","parity210_shifted_diff","parity210_masks_to_boxes",
            "parity210_pairwise_iou","parity210_histogramdd","parity210_gridsample_backward_input","parity210_gridsample_backward_grid",
            "parity210_take_along_dim_f","parity210_cross3","parity210_ldexp","parity210_kron2d",
            "parity210_search_sorted","parity210_next_after","parity210_index_write","parity210_cdist",
            "parity210_pdist","parity210_histc","parity210_bitonic_step","parity210_copy_rows",
            "parity210_iota_pad","parity210_hsoftmax_paths","parity210_isin","parity210_copy_block_2d",
            "parity210_scatter_reduce","parity210_unfold","parity210_classify_float","parity210_zeta",
            "parity210_polygamma","parity210_rwkv7_forward",
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
        // Equal infinities would produce expf(NaN); count directly so
        // [-inf,-inf] and [+inf,+inf] stay inf throughout the scan.
        if (x == m && isinf(x)) { s += 1.0f; }
        else if (x > m) { s = s * expf(m - x) + 1.0f; m = x; }
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

// NON-DETERMINISTIC (issue #382); see parity210_index_add_deterministic.
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

extern ""C"" __global__ __launch_bounds__(256) void parity210_index_add_deterministic(
    float* output, const int* indices, const float* source,
    int outerSize, int dstAxis, int innerSize, int idxLen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * dstAxis * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int tmp = idx / innerSize;
    int dstTarget = tmp % dstAxis;
    int outer = tmp / dstAxis;

    float sum = 0.0f;
    for (int i = 0; i < idxLen; i++) {
        if (indices[i] == dstTarget) {
            int srcPos = (outer * idxLen + i) * innerSize + inner;
            sum += source[srcPos];
        }
    }
    int dstPos = (outer * dstAxis + dstTarget) * innerSize + inner;
    output[dstPos] += sum;
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
    const float* source, int total, int sourceLen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (mask[idx]) {
        int srcIdx = prefixSum[idx];
        // Guard against inconsistent prefix metadata or a short source.
        if (srcIdx >= 0 && srcIdx < sourceLen) output[idx] = source[srcIdx];
    }
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
    // torch.fmod(x, 0) returns NaN for fp; fmodf already does so under IEEE 754.
    out[idx] = fmodf(a[idx], b[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_remainder(
    const float* a, const float* b, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float av = a[idx], bv = b[idx];
    if (bv == 0.0f) { out[idx] = nanf(""""); return; }
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

// Short-circuit equal infinities to avoid inf-inf = NaN on `s - m`.
extern ""C"" __global__ __launch_bounds__(256) void parity210_log_add_exp(
    const float* a, const float* b, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float av = a[idx], bv = b[idx];
    if (av == bv && isinf(av)) { out[idx] = av; return; }
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
    if (av == bv && isinf(av)) { out[idx] = av; return; }
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
    // Domain: [-1, 1]. Exact +/-1 -> +/-infinity; anything strictly outside is NaN.
    if (y == 1.0f) { output[idx] = INFINITY; return; }
    if (y == -1.0f) { output[idx] = -INFINITY; return; }
    if (!(y > -1.0f && y < 1.0f)) { output[idx] = nanf(""""); return; }
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
    const float pi = 3.14159265358979323846f;
    // Reflection for x <= 0 (avoids log of non-positive in asymptotic tail).
    // psi(x) = psi(1-x) - pi * cot(pi*x); poles at non-positive integers.
    if (x <= 0.0f) {
        if (x == floorf(x)) { output[idx] = nanf(""""); return; }
        float sp = sinf(pi * x);
        result = -pi * cosf(pi * x) / sp;
        x = 1.0f - x;
    }
    // Bounded for-loop — x += 1.0f never advances -INFINITY, so an
    // unbounded while would hang the kernel.
    for (int step = 0; step < 64 && x < 6.0f; ++step) { result -= 1.0f / x; x += 1.0f; }
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
    // Asymptotic branch is expf(ax)/sqrtf(ax); at ax=+inf that's inf/inf=NaN.
    // I0 is even and diverges to +inf, so bypass the formula explicitly.
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
        return (expf(ax) / sqrtf(ax)) * ans;
    }
}

__device__ inline float parity210_dev_i1(float x) {
    float ax = fabsf(x);
    // I1 is odd: I1(+inf) = +inf, I1(-inf) = -inf.  Avoid expf/sqrtf NaN.
    if (isinf(ax)) return copysignf(INFINITY, x);
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

// ==========================================================================
// AUDIO / FFT (missing-kernels audit) — translated from the verified OpenCL kernels.
// ==========================================================================
extern ""C"" __global__ void parity210_reflect_pad_1d(
    const float* __restrict__ inp, float* __restrict__ outp, int batch, int L, int Lp, int pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch*Lp) return;
    int j = idx % Lp; int b = idx / Lp; int src;
    if (j < pad) src = pad - j; else if (j < pad + L) src = j - pad; else src = L - 2 - (j - pad - L);
    outp[idx] = inp[b*L + src];
}
extern ""C"" __global__ void parity210_stft_mag_phase(
    const float* __restrict__ padded, const float* __restrict__ window,
    float* __restrict__ mag, float* __restrict__ phase, int batch, int Lp, int nFft, int hop, int numFrames, int numFreqs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; int total = batch*numFreqs*numFrames; if (idx >= total) return;
    int frame = idx % numFrames; int tmp = idx / numFrames; int k = tmp % numFreqs; int b = tmp / numFreqs;
    int start = frame * hop; int inOff = b * Lp; float re = 0.0f, im = 0.0f;
    float bk = -2.0f * (float)M_PI * (float)k / (float)nFft;
    for (int i = 0; i < nFft; i++) { float x = padded[inOff + start + i] * window[i]; float a = bk * (float)i; re += x*cosf(a); im += x*sinf(a); }
    int outOff = b*numFreqs*numFrames + k*numFrames + frame;
    mag[outOff] = sqrtf(re*re + im*im); phase[outOff] = atan2f(im, re);
}
extern ""C"" __global__ void parity210_phase_vocoder(
    const float* __restrict__ mag, const float* __restrict__ phase,
    float* __restrict__ newMag, float* __restrict__ newPhase, int leading, int nFramesV, int nFreqV, int outFrames, float rate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= leading*nFreqV) return;
    int f = idx % nFreqV; int b = idx / nFreqV; int stride = nFramesV*nFreqV; int outStride = outFrames*nFreqV;
    float accPhase = 0.0f;
    for (int t = 0; t < outFrames; t++) {
        float srcT = (float)t * rate; int t0 = (int)floorf(srcT); int t1 = min(t0+1, nFramesV-1); float frac = srcT - (float)t0;
        float m0 = mag[b*stride + t0*nFreqV + f]; float m1 = mag[b*stride + t1*nFreqV + f];
        newMag[b*outStride + t*nFreqV + f] = (1.0f-frac)*m0 + frac*m1;
        float dp = 0.0f;
        if (t0+1 < nFramesV) { dp = phase[b*stride + (t0+1)*nFreqV + f] - phase[b*stride + t0*nFreqV + f]; dp -= 2.0f*(float)M_PI * roundf(dp/(2.0f*(float)M_PI)); }
        accPhase += dp; newPhase[b*outStride + t*nFreqV + f] = accPhase;
    }
}
extern ""C"" __global__ void parity210_build_spectrum(
    const float* __restrict__ mag, const float* __restrict__ phase,
    float* __restrict__ specRe, float* __restrict__ specIm, int batch, int numFreqs, int numFrames, int nFft)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch*numFrames) return;
    int frame = idx % numFrames; int b = idx / numFrames; int magOff = b*numFreqs*numFrames; int specOff = idx * nFft;
    for (int k = 0; k < nFft; k++) { specRe[specOff+k] = 0.0f; specIm[specOff+k] = 0.0f; }
    for (int k = 0; k < numFreqs && k < nFft; k++) { float m = mag[magOff + k*numFrames + frame]; float p = phase[magOff + k*numFrames + frame]; specRe[specOff+k] = m*cosf(p); specIm[specOff+k] = m*sinf(p); }
    for (int k = 1; k < numFreqs - 1; k++) { int dst = nFft - k; if (dst >= 0 && dst < nFft && k < nFft) { specRe[specOff+dst] = specRe[specOff+k]; specIm[specOff+dst] = -specIm[specOff+k]; } }
}
extern ""C"" __global__ void parity210_istft_from_spectrum(
    const float* __restrict__ specRe, const float* __restrict__ specIm, const float* __restrict__ window,
    float* __restrict__ result, float* __restrict__ windowSum, int batch, int numFrames, int nFft, int hop, int outputLength, int center)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; int total = batch*numFrames*nFft; if (idx >= total) return;
    int i = idx % nFft; int tmp = idx / nFft; int frame = tmp % numFrames; int b = tmp / numFrames; int specOff = (b*numFrames + frame) * nFft;
    float acc = 0.0f;
    for (int k = 0; k < nFft; k++) { float a = 2.0f*(float)M_PI*(float)k*(float)i/(float)nFft; acc += specRe[specOff+k]*cosf(a) - specIm[specOff+k]*sinf(a); }
    int writeStart = center ? max(0, frame*hop - nFft/2) : frame*hop; int outIdx = writeStart + i;
    if (outIdx >= 0 && outIdx < outputLength) { float w = window[i]; atomicAdd(&result[b*outputLength + outIdx], acc*(1.0f/(float)nFft)*w); atomicAdd(&windowSum[b*outputLength + outIdx], w*w); }
}
extern ""C"" __global__ void parity210_istft_normalize(float* __restrict__ result, const float* __restrict__ windowSum, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total) return;
    float ws = windowSum[idx]; if (ws > 1e-8f) result[idx] = result[idx] / ws;
}

// ==========================================================================
// LOGICAL / DETECTION / GEOMETRY / MISC (audit) — from verified OpenCL kernels.
// ==========================================================================
extern ""C"" __global__ void parity210_logical_op(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int mode, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return;
    int ba = (a[i] != 0.0f); int bb = (b[i] != 0.0f);
    int r = (mode == 0) ? (ba && bb) : (mode == 1) ? (ba || bb) : (ba != bb);
    out[i] = r ? 1.0f : 0.0f;
}
extern ""C"" __global__ void parity210_logical_not(const float* __restrict__ a, float* __restrict__ out, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return;
    out[i] = (a[i] != 0.0f) ? 0.0f : 1.0f;
}
extern ""C"" __global__ void parity210_shifted_diff(const float* __restrict__ x, float* __restrict__ mask, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return;
    mask[i] = (i == 0 || x[i] != x[i-1]) ? 1.0f : 0.0f;
}
extern ""C"" __global__ void parity210_masks_to_boxes(const float* __restrict__ masks, float* __restrict__ out, int N, int H, int W)
{
    int n = blockIdx.x*blockDim.x+threadIdx.x; if (n >= N) return;
    int xMin = W, yMin = H, xMax = -1, yMax = -1; int planeOff = n * H * W;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (masks[planeOff + y*W + x] != 0.0f) { if (x<xMin)xMin=x; if (x>xMax)xMax=x; if (y<yMin)yMin=y; if (y>yMax)yMax=y; }
    int o = n * 4;
    if (xMax < 0) { out[o]=0.0f; out[o+1]=0.0f; out[o+2]=0.0f; out[o+3]=0.0f; }
    else { out[o]=(float)xMin; out[o+1]=(float)yMin; out[o+2]=(float)xMax; out[o+3]=(float)yMax; }
}
extern ""C"" __global__ void parity210_pairwise_iou(const float* __restrict__ boxes, float* __restrict__ iou, int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx >= N*N) return; int j = idx % N; int i = idx / N;
    float ix1=boxes[i*4],iy1=boxes[i*4+1],ix2=boxes[i*4+2],iy2=boxes[i*4+3];
    float jx1=boxes[j*4],jy1=boxes[j*4+1],jx2=boxes[j*4+2],jy2=boxes[j*4+3];
    float ai=fmaxf(0.0f,ix2-ix1)*fmaxf(0.0f,iy2-iy1); float aj=fmaxf(0.0f,jx2-jx1)*fmaxf(0.0f,jy2-jy1);
    float iw=fmaxf(0.0f,fminf(ix2,jx2)-fmaxf(ix1,jx1)); float ih=fmaxf(0.0f,fminf(iy2,jy2)-fmaxf(iy1,jy1));
    float inter=iw*ih; float uni=ai+aj-inter; iou[idx]=(uni>0.0f)?inter/uni:0.0f;
}
extern ""C"" __global__ void parity210_histogramdd(const float* __restrict__ samples, float* __restrict__ hist, const int* __restrict__ bins, const float* __restrict__ mins, const float* __restrict__ maxs, int n, int d)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return; int linIdx = 0; int valid = 1;
    for (int k = 0; k < d; k++) { float v = samples[i*d + k]; float mn = mins[k]; float mx = maxs[k]; if (v < mn || v > mx) { valid = 0; break; } float width = (mx - mn) / (float)bins[k]; int kIdx = (int)floorf((v - mn) / width); if (kIdx >= bins[k]) kIdx = bins[k] - 1; if (kIdx < 0) kIdx = 0; linIdx = linIdx * bins[k] + kIdx; }
    if (valid) atomicAdd(&hist[linIdx], 1.0f);
}
extern ""C"" __global__ void parity210_gridsample_backward_input(const float* __restrict__ gradOut, const float* __restrict__ grid, float* __restrict__ gradIn, int batch, int H, int W, int C, int outH, int outW)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x; int total = batch*outH*outW*C; if (idx >= total) return;
    int c = idx % C; int tmp = idx / C; int ow = tmp % outW; tmp /= outW; int oh = tmp % outH; int b = tmp / outH;
    int gridBase = ((b*outH + oh)*outW + ow)*2; float gx = grid[gridBase]; float gy = grid[gridBase+1];
    float srcH = (gy + 1.0f) * 0.5f * (float)(H - 1); float srcW = (gx + 1.0f) * 0.5f * (float)(W - 1);
    if (srcH <= -1.0f || srcH >= (float)H || srcW <= -1.0f || srcW >= (float)W) return;
    int h0 = (int)floorf(srcH); int h1 = h0 + 1; int w0 = (int)floorf(srcW); int w1 = w0 + 1;
    float lh = srcH - (float)h0; float lw = srcW - (float)w0; float g = gradOut[idx];
    if (h0>=0 && h0<H && w0>=0 && w0<W) atomicAdd(&gradIn[((b*H+h0)*W+w0)*C+c], g*(1.0f-lh)*(1.0f-lw));
    if (h0>=0 && h0<H && w1>=0 && w1<W) atomicAdd(&gradIn[((b*H+h0)*W+w1)*C+c], g*(1.0f-lh)*lw);
    if (h1>=0 && h1<H && w0>=0 && w0<W) atomicAdd(&gradIn[((b*H+h1)*W+w0)*C+c], g*lh*(1.0f-lw));
    if (h1>=0 && h1<H && w1>=0 && w1<W) atomicAdd(&gradIn[((b*H+h1)*W+w1)*C+c], g*lh*lw);
}
extern ""C"" __global__ void parity210_gridsample_backward_grid(const float* __restrict__ gradOut, const float* __restrict__ input, const float* __restrict__ grid, float* __restrict__ gradGrid, int batch, int H, int W, int C, int outH, int outW)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx >= batch*outH*outW) return;
    int ow = idx % outW; int tmp = idx / outW; int oh = tmp % outH; int b = tmp / outH;
    int gridBase = ((b*outH+oh)*outW+ow)*2; float gx = grid[gridBase]; float gy = grid[gridBase+1];
    float srcH = (gy+1.0f)*0.5f*(float)(H-1); float srcW = (gx+1.0f)*0.5f*(float)(W-1);
    float gradGx = 0.0f; float gradGy = 0.0f;
    if (!(srcH <= -1.0f || srcH >= (float)H || srcW <= -1.0f || srcW >= (float)W)) {
        int h0 = (int)floorf(srcH); int h1 = h0+1; int w0 = (int)floorf(srcW); int w1 = w0+1;
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

// ==========================================================================
// REMAINING AUDIT OPS (movement/indexing/special/sort/etc.) — from verified OpenCL.
// ==========================================================================
__device__ void p210_atomicReduceF(float* addr, float val, int mode) {
    int* ia = (int*)addr; int old = *ia; int assumed;
    do { assumed = old; float pf = __int_as_float(assumed);
         float nv = (mode==0)?(pf+val):(mode==1)?(pf*val):(mode==2)?fmaxf(pf,val):fminf(pf,val);
         old = atomicCAS(ia, assumed, __float_as_int(nv));
    } while (assumed != old);
}
__device__ float p210_zeta_scalar(float x, float q) {
    if (x == 1.0f) return INFINITY;
    if (q <= 0.0f && q == floorf(q)) return INFINITY;
    float sum = 0.0f;
    for (int k = 0; k < 12; k++) sum += powf(q + (float)k, -x);
    float Nq = 12.0f + q; float lnNq = logf(Nq);
    float cont = expf((1.0f - x) * lnNq) / (x - 1.0f);
    float halfTerm = 0.5f * expf(-x * lnNq);
    const float b2k[8] = { 1.0f/6.0f, -1.0f/30.0f, 1.0f/42.0f, -1.0f/30.0f, 5.0f/66.0f, -691.0f/2730.0f, 7.0f/6.0f, -3617.0f/510.0f };
    const float fact2k[8] = { 2.0f, 24.0f, 720.0f, 40320.0f, 3628800.0f, 479001600.0f, 87178291200.0f, 20922789888000.0f };
    float corr = 0.0f; float xPow = expf(-x * lnNq) / Nq; float invNq2 = 1.0f / (Nq * Nq); float rising = x;
    for (int j = 1; j <= 8; j++) {
        if (j > 1) { rising *= (x + 2.0f*(float)(j-1) - 2.0f) * (x + 2.0f*(float)(j-1) - 1.0f); xPow *= invNq2; }
        corr += (b2k[j-1] / fact2k[j-1]) * rising * xPow;
    }
    return sum + cont + halfTerm + corr;
}
__device__ float p210_polygamma_scalar(int n, float x) {
    if (x <= 0.0f && x == floorf(x)) return INFINITY;
    float recurrence = 0.0f; float xd = x; int signN = (n & 1) == 1 ? 1 : -1;
    while (xd < 10.0f) { recurrence += (float)signN * expf(lgammaf((float)(n+1)) - (float)(n+1) * logf(xd)); xd += 1.0f; }
    float lnX = logf(xd);
    float asympt = expf(lgammaf((float)n) - (float)n * lnX) + 0.5f * expf(lgammaf((float)(n+1)) - (float)(n+1) * lnX);
    const float b2k[8] = { 1.0f/6.0f, -1.0f/30.0f, 1.0f/42.0f, -1.0f/30.0f, 5.0f/66.0f, -691.0f/2730.0f, 7.0f/6.0f, -3617.0f/510.0f };
    float invX2 = 1.0f / (xd * xd); float xPow = expf(-(float)n * lnX);
    for (int k = 1; k <= 8; k++) { xPow *= invX2; asympt += b2k[k-1] * expf(lgammaf((float)(2*k+n)) - lgammaf((float)(2*k+1))) * xPow; }
    return recurrence + (float)signN * asympt;
}
extern ""C"" __global__ void parity210_take_along_dim_f(const float* __restrict__ input, const float* __restrict__ indices, float* __restrict__ output, int outerSize, int axisOut, int innerSize, int axisIn) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; int total = outerSize*axisOut*innerSize; if (idx >= total) return;
    int inner = idx % innerSize; int outer = (idx / innerSize) / axisOut; int srcJ = (int)indices[idx];
    if (srcJ < 0 || srcJ >= axisIn) { output[idx] = 0.0f; return; }
    output[idx] = input[(outer * axisIn + srcJ) * innerSize + inner];
}
extern ""C"" __global__ void parity210_cross3(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ output, int outerSize, int innerSize) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx >= outerSize * innerSize) return;
    int inner = idx % innerSize; int outer = idx / innerSize; int p = outer * 3 * innerSize + inner;
    float a0=a[p],a1=a[p+innerSize],a2=a[p+2*innerSize]; float b0=b[p],b1=b[p+innerSize],b2=b[p+2*innerSize];
    output[p]=a1*b2-a2*b1; output[p+innerSize]=a2*b0-a0*b2; output[p+2*innerSize]=a0*b1-a1*b0;
}
extern ""C"" __global__ void parity210_ldexp(const float* __restrict__ input, const int* __restrict__ exponents, float* __restrict__ output, int size) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx >= size) return; output[idx] = ldexpf(input[idx], exponents[idx]);
}
extern ""C"" __global__ void parity210_kron2d(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ output, int am, int an, int bp, int bq) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; int outCols = an*bq; int total=(am*bp)*outCols; if (idx>=total) return;
    int oc=idx%outCols; int orow=idx/outCols; int i=orow/bp; int k=orow%bp; int j=oc/bq; int l=oc%bq;
    output[idx] = a[i*an+j] * b[k*bq+l];
}
extern ""C"" __global__ void parity210_search_sorted(const float* __restrict__ seq, const float* __restrict__ values, float* __restrict__ output, int seqLen, int numValues, int right) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx >= numValues) return; float v = values[idx]; int lo=0, hi=seqLen;
    while (lo < hi) { int mid=(lo+hi)>>1; int cond=(right!=0)?(seq[mid]<=v):(seq[mid]<v); if (cond) lo=mid+1; else hi=mid; }
    output[idx] = (float)lo;
}
extern ""C"" __global__ void parity210_next_after(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ output, int size) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx >= size) return; float av=a[idx], bv=b[idx];
    unsigned int ua=__float_as_uint(av), ub=__float_as_uint(bv);
    int aNan=(((ua>>23)&0xFFu)==0xFFu)&&((ua&0x7FFFFFu)!=0u); int bNan=(((ub>>23)&0xFFu)==0xFFu)&&((ub&0x7FFFFFu)!=0u);
    if (aNan||bNan) { output[idx]=__uint_as_float(0x7FC00000u); return; }
    if (av==bv) { output[idx]=bv; return; }
    if (av==0.0f) { output[idx]=__uint_as_float(bv>0.0f?0x00000001u:0x80000001u); return; }
    unsigned int r=ua;
    if (bv>av) r=(av>0.0f)?(r+1u):(r-1u); else r=(av>0.0f)?(r-1u):(r+1u);
    output[idx]=__uint_as_float(r);
}
extern ""C"" __global__ void parity210_index_write(float* __restrict__ output, const int* __restrict__ indices, const float* __restrict__ source, float fillValue, int mode, int outerSize, int idxAxis, int innerSize, int dstAxis) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; int total=outerSize*idxAxis*innerSize; if (idx>=total) return;
    int inner=idx%innerSize; int j=(idx/innerSize)%idxAxis; int outer=(idx/innerSize)/idxAxis; int dstJ=indices[j];
    if (dstJ<0||dstJ>=dstAxis) return; float v=(mode==0)?source[idx]:fillValue;
    output[(outer*dstAxis+dstJ)*innerSize+inner]=v;
}
extern ""C"" __global__ void parity210_cdist(const float* __restrict__ x1, const float* __restrict__ x2, float* __restrict__ output, int m, int n, int d, float p) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx>=m*n) return; int j=idx%n; int i=idx/n; float sum=0.0f;
    for (int k=0;k<d;k++){ float diff=fabsf(x1[i*d+k]-x2[j*d+k]); if (p==1.0f) sum+=diff; else if (p==2.0f) sum+=diff*diff; else sum+=powf(diff,p); }
    output[idx]=(p==1.0f)?sum:(p==2.0f)?sqrtf(sum):powf(sum,1.0f/p);
}
extern ""C"" __global__ void parity210_pdist(const float* __restrict__ input, float* __restrict__ output, int n, int d, float p) {
    int flat = blockIdx.x*blockDim.x+threadIdx.x; if (flat>=n*n) return; int j=flat%n; int i=flat/n; if (i>=j) return; float sum=0.0f;
    for (int k=0;k<d;k++){ float diff=fabsf(input[i*d+k]-input[j*d+k]); if (p==1.0f) sum+=diff; else if (p==2.0f) sum+=diff*diff; else sum+=powf(diff,p); }
    float dist=(p==1.0f)?sum:(p==2.0f)?sqrtf(sum):powf(sum,1.0f/p); int outIdx=i*n-(i*(i+1))/2+(j-i-1); output[outIdx]=dist;
}
extern ""C"" __global__ void parity210_histc(const float* __restrict__ input, float* __restrict__ hist, int n, int bins, float mn, float mx) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx>=n) return; float x=input[idx]; if (x<mn||x>mx) return;
    float bw=(mx-mn)/(float)bins; int b=(int)((x-mn)/bw); if (b>=bins) b=bins-1; if (b<0) b=0; atomicAdd(&hist[b], 1.0f);
}
extern ""C"" __global__ void parity210_bitonic_step(float* __restrict__ values, float* __restrict__ indices, int rowLen, int k, int j, int numRows, int descending) {
    int gid = blockIdx.x*blockDim.x+threadIdx.x; if (gid>=numRows*rowLen) return; int i=gid%rowLen; int ixj=i^j; if (ixj<=i) return;
    int base=(gid/rowLen)*rowLen; float a=values[base+i], b=values[base+ixj];
    unsigned int ua=__float_as_uint(a), ub=__float_as_uint(b);
    float ka=(((ua>>23)&0xFFu)==0xFFu&&(ua&0x7FFFFFu)!=0u)?INFINITY:a; float kb=(((ub>>23)&0xFFu)==0xFFu&&(ub&0x7FFFFFu)!=0u)?INFINITY:b;
    int up=((i&k)==0); if (descending!=0) up=!up; int doSwap=up?(ka>kb):(ka<kb);
    if (doSwap) { values[base+i]=b; values[base+ixj]=a; float t=indices[base+i]; indices[base+i]=indices[base+ixj]; indices[base+ixj]=t; }
}
extern ""C"" __global__ void parity210_copy_rows(const float* __restrict__ src, float* __restrict__ dst, int srcRowLen, int dstRowLen, int numRows, int copyLen) {
    int gid = blockIdx.x*blockDim.x+threadIdx.x; if (gid>=numRows*copyLen) return; int i=gid%copyLen; int r=gid/copyLen; dst[r*dstRowLen+i]=src[r*srcRowLen+i];
}
extern ""C"" __global__ void parity210_iota_pad(float* __restrict__ idx, int L, int P, int numRows) {
    int gid = blockIdx.x*blockDim.x+threadIdx.x; if (gid>=numRows*P) return; int i=gid%P; idx[gid]=(i<L)?(float)i:-1.0f;
}
extern ""C"" __global__ void parity210_hsoftmax_paths(const float* __restrict__ acts, float* __restrict__ out, int rows, int treeDepth, int numClasses) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx>=rows*numClasses) return; int c=idx%numClasses; int r=idx/numClasses; int gbase=r*treeDepth; float prob=1.0f; int node=1;
    for (int level=0; level<treeDepth; level++) { int goRight=(c&(1<<(treeDepth-level-1)))!=0; float g=1.0f/(1.0f+expf(-acts[gbase+level])); prob*=goRight?g:(1.0f-g); node=node*2+(goRight?1:0); if (node>=numClasses) break; }
    out[idx]=prob;
}
extern ""C"" __global__ void parity210_isin(const float* __restrict__ elements, const float* __restrict__ sortedTest, float* __restrict__ mask, int numElements, int testLen) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx>=numElements) return; float v=elements[idx]; int lo=0, hi=testLen;
    while (lo<hi) { int mid=(lo+hi)>>1; if (sortedTest[mid]<v) lo=mid+1; else hi=mid; } mask[idx]=(lo<testLen&&sortedTest[lo]==v)?1.0f:0.0f;
}
extern ""C"" __global__ void parity210_copy_block_2d(const float* __restrict__ block, float* __restrict__ output, int blockRows, int blockCols, int totalCols, int rowOff, int colOff) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx>=blockRows*blockCols) return; int j=idx%blockCols; int i=idx/blockCols; output[(rowOff+i)*totalCols+(colOff+j)]=block[i*blockCols+j];
}
extern ""C"" __global__ void parity210_scatter_reduce(float* __restrict__ output, const float* __restrict__ source, const int* __restrict__ index, int outerSize, int srcDim, int dstDim, int innerSize, int mode) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; int total=outerSize*srcDim*innerSize; if (idx>=total) return; int inner=idx%innerSize; int tmp=idx/innerSize; int outer=tmp/srcDim; int t=index[idx];
    if (t<0||t>=dstDim) return; p210_atomicReduceF(&output[(outer*dstDim+t)*innerSize+inner], source[idx], mode);
}
extern ""C"" __global__ void parity210_unfold(const float* __restrict__ src, float* __restrict__ dst, int outerSize, int dimSize, int innerSize, int nWindows, int size, int step) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; int total=outerSize*nWindows*innerSize*size; if (idx>=total) return; int s=idx%size; int tmp=idx/size; int inner=tmp%innerSize; tmp/=innerSize; int w=tmp%nWindows; int outer=tmp/nWindows;
    dst[idx]=src[(outer*dimSize+(w*step+s))*innerSize+inner];
}
extern ""C"" __global__ void parity210_classify_float(const float* __restrict__ a, float* __restrict__ output, int mode, int size) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; if (idx>=size) return; unsigned int bits=__float_as_uint(a[idx]); unsigned int expo=(bits>>23)&0xFFu; unsigned int mant=bits&0x7FFFFFu;
    int isNan=(expo==0xFFu)&&(mant!=0u); int isInf=(expo==0xFFu)&&(mant==0u); float r;
    if (mode==0) r=isNan?1.0f:0.0f; else if (mode==1) r=isInf?1.0f:0.0f; else r=(!isNan&&!isInf)?1.0f:0.0f; output[idx]=r;
}
extern ""C"" __global__ void parity210_zeta(const float* __restrict__ x, const float* __restrict__ q, float* __restrict__ out, int size) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=size) return; out[i]=p210_zeta_scalar(x[i], q[i]);
}
extern ""C"" __global__ void parity210_polygamma(const float* __restrict__ x, float* __restrict__ out, int n, int size) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=size) return; out[i]=p210_polygamma_scalar(n, x[i]);
}
extern ""C"" __global__ void parity210_rwkv7_forward(const float* __restrict__ R, const float* __restrict__ K, const float* __restrict__ V, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ outp, float* __restrict__ Sbuf, int batch, int seqLen, int modelDim, int numHeads, int headDim) {
    int bh = blockIdx.x*blockDim.x+threadIdx.x; if (bh>=batch*numHeads) return; int b=bh/numHeads; int h=bh%numHeads; int hOff=h*headDim; int hh=headDim*headDim; float* S=Sbuf+bh*hh;
    for (int i=0;i<hh;i++) S[i]=0.0f;
    for (int t=0;t<seqLen;t++) {
        int baseOff=(b*seqLen+t)*modelDim+hOff;
        for (int di=0;di<headDim;di++) { float ga=1.0f/(1.0f+expf(-A[baseOff+di])); float gbk=(1.0f/(1.0f+expf(-B[baseOff+di])))*K[baseOff+di]; int srow=di*headDim; for (int vi=0;vi<headDim;vi++) S[srow+vi]=ga*S[srow+vi]+gbk*V[baseOff+vi]; }
        for (int di=0;di<headDim;di++) { int srow=di*headDim; float sk=0.0f; for (int vi=0;vi<headDim;vi++) sk+=S[srow+vi]*K[baseOff+vi]; outp[baseOff+di]=(1.0f/(1.0f+expf(-R[baseOff+di])))*sk; }
    }
}
";
    }
}
