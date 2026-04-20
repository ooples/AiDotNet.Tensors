// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for the parity-210 op surface. Covers the hot-path subset of
// Issue #210's 90-op beat-PyTorch gate: movement (roll/flip/triu/tril/diag_embed/
// rot90), cumulative (cumsum/cumprod/cummax/cummin/logcumsumexp), indexing
// (take/take_along_dim/index_add/index_copy/index_fill/masked_scatter/
// masked_select), element-wise binary special (hypot/copysign/fmod/remainder/
// float_power/logaddexp/logaddexp2/xlogy/xlog1py), and element-wise unary
// special (erfc/erfinv/lgamma/digamma/i0/i1/i0e/i1e/is_finite/is_nan/is_inf/
// nan_to_num).
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// Source bundle for the parity-210 CUDA kernels. Each function is a
    /// single-pass kernel launched over the output element count. fp32 is the
    /// primary compute dtype — the tensor engine is responsible for casting to
    /// fp32 on entry and back to T on exit (<see cref="AiDotNet.Tensors.Engines.DirectGpu.DirectGpuTensorEngine"/>
    /// uses the same lift/lower strategy as its activation overrides).
    /// </summary>
    public static class CudaParity210Kernels
    {
        public static string[] GetKernelNames() => new[]
        {
            // Movement
            "parity210_roll_1d","parity210_flip_axis","parity210_triu","parity210_tril",
            "parity210_diag_embed",
            // Cumulative
            "parity210_cumsum_axis","parity210_cumprod_axis","parity210_cummax_axis",
            "parity210_cummin_axis","parity210_logcumsumexp_axis",
            "parity210_cumsum_block_hillis_steele",
            // Indexing
            "parity210_take_linear","parity210_take_along_dim","parity210_index_add",
            "parity210_index_copy","parity210_index_fill","parity210_masked_scatter",
            // Element-wise binary special
            "parity210_hypot","parity210_copysign","parity210_fmod","parity210_remainder",
            "parity210_float_power","parity210_log_add_exp","parity210_log_add_exp2",
            "parity210_xlogy","parity210_xlog1py",
            // Element-wise unary special
            "parity210_erfc","parity210_erfinv","parity210_lgamma_approx","parity210_digamma",
            "parity210_i0","parity210_i1","parity210_i0e","parity210_i1e",
            // Predicate / numeric hygiene
            "parity210_is_finite","parity210_is_nan","parity210_is_inf","parity210_nan_to_num",
            // Pairwise
            "parity210_cosine_similarity_last","parity210_cdist_l2",
            // Triangular + bitwise helpers
            "parity210_clamp_min_max",
        };

        public static string GetSource() => @"
#include <math.h>

// Small utility: 1/Pi, sqrt(2/Pi), etc.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==========================================================================
// MOVEMENT
// ==========================================================================

// Rolls a single axis by `shift` positions. Flattened indexing assumes the
// tensor has been contiguously reshaped to [outer, axisSize, inner] before
// launch (the C# caller does this by computing outer/inner strides from the
// real axis index).
extern ""C"" __global__ __launch_bounds__(256) void parity210_roll_1d(
    const float* __restrict__ input, float* __restrict__ output,
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
    // Signed-safe positive modulo so shift can be negative without UB.
    srcAxis %= axisSize;
    if (srcAxis < 0) srcAxis += axisSize;

    int srcIdx = (outer * axisSize + srcAxis) * innerSize + inner;
    output[idx] = input[srcIdx];
}

// Flips a single axis. Multi-axis flip is composed in C# as a sequence of
// launches (cheap because each is O(N) and flips are commutative).
extern ""C"" __global__ __launch_bounds__(256) void parity210_flip_axis(
    const float* __restrict__ input, float* __restrict__ output,
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
    int srcIdx = (outer * axisSize + srcAxis) * innerSize + inner;
    output[idx] = input[srcIdx];
}

// Upper-triangular mask: keep cells where (col - row) >= diagonal.
// Input is a batch of rows×cols matrices. batchSize = prod(shape[:-2]).
extern ""C"" __global__ __launch_bounds__(256) void parity210_triu(
    const float* __restrict__ input, float* __restrict__ output,
    int batchSize, int rows, int cols, int diagonal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * rows * cols;
    if (idx >= total) return;

    int col = idx % cols;
    int tmp = idx / cols;
    int row = tmp % rows;
    int keep = ((col - row) >= diagonal) ? 1 : 0;
    output[idx] = keep ? input[idx] : 0.0f;
}

// Lower-triangular mask: keep cells where (col - row) <= diagonal.
extern ""C"" __global__ __launch_bounds__(256) void parity210_tril(
    const float* __restrict__ input, float* __restrict__ output,
    int batchSize, int rows, int cols, int diagonal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * rows * cols;
    if (idx >= total) return;

    int col = idx % cols;
    int tmp = idx / cols;
    int row = tmp % rows;
    int keep = ((col - row) <= diagonal) ? 1 : 0;
    output[idx] = keep ? input[idx] : 0.0f;
}

// DiagEmbed: place input[..., i] on the diagonal of a square (matSize×matSize)
// with the given offset. Output is zero-initialised by the kernel itself — we
// write both diagonal and off-diagonal positions.
extern ""C"" __global__ __launch_bounds__(256) void parity210_diag_embed(
    const float* __restrict__ input, float* __restrict__ output,
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
// CUMULATIVE (sequential along axis — one thread per (outer, inner) line)
// ==========================================================================

// Block-level Hillis-Steele prefix sum for axes up to 1024 elements.
// One block per (outer * inner) line; each thread handles one axis position.
// Shared memory holds 2 * blockDim.x floats for ping-pong scan.
//
// O(log n) depth, O(n log n) work — simpler than Blelloch and faster in
// practice up to ~1024 elements (Blelloch's extra sync overhead dominates
// for these sizes; beyond 1024 the tree-based approach wins and we fall
// through to a multi-block carry scheme via the per-line serial kernel).
extern ""C"" __global__ void parity210_cumsum_block_hillis_steele(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int axisSize, int innerSize)
{
    extern __shared__ float smem[];
    int line = blockIdx.x;             // one block per outer*inner line
    int inner = line % innerSize;
    int outer = line / innerSize;
    if (outer >= outerSize) return;
    int base_ = outer * axisSize * innerSize + inner;

    int tid = threadIdx.x;
    float* s0 = smem;
    float* s1 = smem + blockDim.x;

    // Load into s0 (thread tid owns position tid along axis).
    s0[tid] = (tid < axisSize) ? input[base_ + tid * innerSize] : 0.0f;
    __syncthreads();

    // Hillis-Steele scan: each step reads from the prior buffer and writes
    // to the other, then swaps.
    int limit = axisSize;
    for (int offset = 1; offset < limit; offset *= 2) {
        if (tid < limit) {
            s1[tid] = (tid >= offset) ? s0[tid] + s0[tid - offset] : s0[tid];
        }
        __syncthreads();
        // Swap s0 <-> s1 for next iteration.
        float* tmp = s0; s0 = s1; s1 = tmp;
    }

    if (tid < axisSize) {
        output[base_ + tid * innerSize] = s0[tid];
    }
}

// Each thread owns one 1-D line of length axisSize and does a serial scan.
// Fallback path for axisSize > 1024 (above the single-block Hillis-Steele
// cutoff); also used for cumprod / cummax / cummin / logcumsumexp which
// don't yet have a block-scan specialization.
extern ""C"" __global__ __launch_bounds__(256) void parity210_cumsum_axis(
    const float* __restrict__ input, float* __restrict__ output,
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
    const float* __restrict__ input, float* __restrict__ output,
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
    const float* __restrict__ input, float* __restrict__ output,
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
    const float* __restrict__ input, float* __restrict__ output,
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

// LogCumSumExp: numerically-stable scan. max-track + running exp-shift.
extern ""C"" __global__ __launch_bounds__(256) void parity210_logcumsumexp_axis(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int axisSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx >= total) return;

    int inner = idx % innerSize;
    int outer = idx / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    if (axisSize <= 0) return;
    // Bootstrap from input[0] so an initial -INFINITY doesn't force
    // exp(-INF - -INF) = exp(NaN). The scanning invariant
    //   output[i] = m_i + log(s_i)
    // is satisfied at i=0 with m = x0, s = 1 (i.e. output[0] = x0).
    float m = input[base_];
    float s = 1.0f;
    output[base_] = m;
    for (int a = 1; a < axisSize; ++a) {
        float x = input[base_ + a * innerSize];
        if (x > m) {
            // Rebase to new max: s' = s * exp(m - x) + 1
            s = s * expf(m - x) + 1.0f;
            m = x;
        } else {
            s += expf(x - m);
        }
        output[base_ + a * innerSize] = m + logf(s);
    }
}

// ==========================================================================
// INDEXING
// ==========================================================================

// Flat-index take: output[i] = input[indices[i]]. Shape of output matches
// indices; caller validates bounds in the wrapping CPU code.
extern ""C"" __global__ __launch_bounds__(256) void parity210_take_linear(
    const float* __restrict__ input, const int* __restrict__ indices,
    float* __restrict__ output, int outSize, int inputLinearLen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outSize) return;
    int pos = indices[idx];
    output[idx] = (pos >= 0 && pos < inputLinearLen) ? input[pos] : 0.0f;
}

// Take along a specific dim. outer/inner are strides w.r.t. the dim axis.
// axisSize is the input's size along the dim axis; idxAxis is the indices'
// size along the same dim.
extern ""C"" __global__ __launch_bounds__(256) void parity210_take_along_dim(
    const float* __restrict__ input, const int* __restrict__ indices,
    float* __restrict__ output,
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

// IndexAdd requires atomicAdd to be safe under repeated indices.
extern ""C"" __global__ __launch_bounds__(256) void parity210_index_add(
    float* __restrict__ output, const int* __restrict__ indices,
    const float* __restrict__ source,
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

// IndexCopy replaces target rows with source rows (no accumulation).
extern ""C"" __global__ __launch_bounds__(256) void parity210_index_copy(
    float* __restrict__ output, const int* __restrict__ indices,
    const float* __restrict__ source,
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
    output[dstPos] = source[srcPos];
}

// IndexFill sets output[.., idx, ..] = fillValue for each idx in indices.
extern ""C"" __global__ __launch_bounds__(256) void parity210_index_fill(
    float* __restrict__ output, const int* __restrict__ indices,
    float fillValue,
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
    output[dstPos] = fillValue;
}

// MaskedScatter writes source elements into output positions where mask is 1.
// srcCursor is advanced linearly through source per mask-true element; we
// pre-compute the prefix sum of the mask on host to assign cursor indices.
// prefixSum[i] = number of 1-bits in mask[0..i) — so the cursor for position i
// (when mask[i] == 1) is prefixSum[i].
extern ""C"" __global__ __launch_bounds__(256) void parity210_masked_scatter(
    float* __restrict__ output, const char* __restrict__ mask,
    const int* __restrict__ prefixSum, const float* __restrict__ source,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (mask[idx]) {
        output[idx] = source[prefixSum[idx]];
    }
}

// ==========================================================================
// ELEMENT-WISE BINARY SPECIAL
// ==========================================================================

extern ""C"" __global__ __launch_bounds__(256) void parity210_hypot(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = hypotf(a[idx], b[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_copysign(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = copysignf(a[idx], b[idx]);
}

// fmod: truncation-based. Matches C fmodf, which PyTorch torch.fmod mirrors.
extern ""C"" __global__ __launch_bounds__(256) void parity210_fmod(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // torch.fmod(x, 0) returns NaN for floating types. fmodf(x, 0) is
    // already NaN under IEEE 754 so no branch is needed.
    out[idx] = fmodf(a[idx], b[idx]);
}

// torch.remainder: floor-based mod (result has the same sign as divisor).
// Returns NaN when the divisor is zero to match PyTorch.
extern ""C"" __global__ __launch_bounds__(256) void parity210_remainder(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float av = a[idx];
    float bv = b[idx];
    if (bv == 0.0f) { out[idx] = nanf(""""); return; }
    float q = floorf(av / bv);
    out[idx] = av - q * bv;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_float_power(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = powf(a[idx], b[idx]);
}

// log(exp(a)+exp(b)) — stable.  When both inputs are equal infinities,
// `s - m` evaluates to inf - inf = NaN, so short-circuit to the shared
// infinity value to match PyTorch.
extern ""C"" __global__ __launch_bounds__(256) void parity210_log_add_exp(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int size)
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
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float av = a[idx], bv = b[idx];
    if (av == bv && isinf(av)) { out[idx] = av; return; }
    float m = fmaxf(av, bv);
    float s = fminf(av, bv);
    // log2(2^m + 2^s) = m + log2(1 + 2^(s-m))
    out[idx] = m + log2f(1.0f + exp2f(s - m));
}

// xlogy: x * log(y), with 0*log(0) := 0 (matches torch.xlogy).
extern ""C"" __global__ __launch_bounds__(256) void parity210_xlogy(
    const float* __restrict__ x, const float* __restrict__ y,
    float* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float xv = x[idx];
    out[idx] = (xv == 0.0f) ? 0.0f : xv * logf(y[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_xlog1py(
    const float* __restrict__ x, const float* __restrict__ y,
    float* __restrict__ out, int size)
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
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = erfcf(input[idx]);
}

// erfinv via Winitzki's approximation + 2 Newton refinements. PyTorch's
// kernel uses the same shape of algorithm; 2 Newton iterations is enough
// for single-precision agreement.
extern ""C"" __global__ __launch_bounds__(256) void parity210_erfinv(
    const float* __restrict__ input, float* __restrict__ output, int size)
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

// lgamma: CUDA ships lgammaf built-in — we just wrap it.
extern ""C"" __global__ __launch_bounds__(256) void parity210_lgamma_approx(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = lgammaf(input[idx]);
}

// Digamma via asymptotic expansion with recurrence shift-up.
extern ""C"" __global__ __launch_bounds__(256) void parity210_digamma(
    const float* __restrict__ input, float* __restrict__ output, int size)
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
    // psi(x) = psi(x+1) - 1/x, shift until x >= 6.  Bounded for-loop —
    // x += 1.0f never advances -INFINITY, so an unbounded while would hang.
    for (int step = 0; step < 64 && x < 6.0f; ++step) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    // Asymptotic: psi(x) ~ ln(x) - 1/(2x) - sum B_{2k} / (2k * x^{2k})
    float inv = 1.0f / x;
    float inv2 = inv * inv;
    result += logf(x) - 0.5f * inv
            - inv2 * ((1.0f/12.0f)
              - inv2 * ((1.0f/120.0f)
                - inv2 * (1.0f/252.0f)));
    output[idx] = result;
}

// Modified Bessel I0 via series for |x|<3.75 and asymptotic for |x|>=3.75.
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
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = parity210_dev_i0(input[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_i1(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = parity210_dev_i1(input[idx]);
}

// I0e / I1e: scaled by exp(-|x|).  The naive form `expf(-fabsf(x)) *
// parity210_dev_i0(x)` overflows before the cancellation can happen
// because the large-|x| branch of parity210_dev_i0 multiplies by
// exp(|x|) / sqrt(|x|). Inline the polynomial here and fold in
// exp(-|x|) analytically: for |x| < 3.75 the series is small enough
// to multiply by exp(-|x|); for |x| >= 3.75 the exp factors cancel
// to 1/sqrt(|x|), so we drop them entirely.
extern ""C"" __global__ __launch_bounds__(256) void parity210_i0e(
    const float* __restrict__ input, float* __restrict__ output, int size)
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
    const float* __restrict__ input, float* __restrict__ output, int size)
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
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = isfinite(input[idx]) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_is_nan(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = isnan(input[idx]) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_is_inf(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = isinf(input[idx]) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity210_nan_to_num(
    const float* __restrict__ input, float* __restrict__ output,
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

// Cosine similarity along the last axis: batch of [N, D] — compute dot / ||a||·||b||
extern ""C"" __global__ __launch_bounds__(256) void parity210_cosine_similarity_last(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int n, int d, float eps)
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

// Pairwise L2 distance: out[i, j] = ||x1[i] - x2[j]||. One thread per (i, j).
extern ""C"" __global__ __launch_bounds__(256) void parity210_cdist_l2(
    const float* __restrict__ x1, const float* __restrict__ x2,
    float* __restrict__ out, int n, int m, int d)
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
// CLAMP (tensor-bounds variant)
// ==========================================================================

extern ""C"" __global__ __launch_bounds__(256) void parity210_clamp_min_max(
    const float* __restrict__ input,
    const float* __restrict__ lo, const float* __restrict__ hi,
    float* __restrict__ output, int size, int hasLo, int hasHi)
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
