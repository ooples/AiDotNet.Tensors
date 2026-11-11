// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Device kernels of the model-level ops in <c>CpuEngine.TorchModelOps</c> (the eight pointwise activations, fake
/// quantization, top-k softmax routing, the sinusoidal time embedding and the linear-chain CRF log-partition), each
/// with its backward. One source serves every backend: it is written in the portable CUDA dialect
/// <see cref="WebGpu.CudaDialectToWgsl"/> documents, so CUDA and HIP compile it as is, Metal and WebGPU run it as part
/// of <see cref="TorchParityF32Kernels"/>, and OpenCL and Vulkan translate it (<see cref="TorchModelOpsTranslation"/>).
/// </summary>
/// <remarks>
/// <para>
/// The kernels compute in float and are written to stay accurate where fast or relaxed math is in play: NaN and
/// infinity are tested on their bits, tanh is evaluated without the exp overflow of (e^x - e^-x) / (e^x + e^-x) or the
/// cancellation near 0, sin and cos come from a reduction whose large product is formed exactly from bit-masked halves
/// (no fma, no rounding of the argument before the reduction) followed by minimax polynomials on [-pi/4, pi/4],
/// log-sum-exp subtracts the maximum, and the bent identity / ISRU / sigmoid-slope use the cancellation- and
/// overflow-free forms of their CPU formulas. Index math is integer throughout.
/// </para>
/// <para>
/// The CRF recursions run one work group per sequence that walks the time steps itself, separated by a group barrier
/// (<c>__syncthreads()</c>, with <c>threadIdx.x</c> / <c>blockIdx.x</c> / <c>blockDim.x</c>, which every dialect
/// translation maps); the forward is one launch, the backward three. Top-k softmax over a long row selects per chunk
/// and then among the chunk candidates. Every other op is one to three launches.
/// </para>
/// </remarks>
internal static class TorchModelOpsKernels
{
    /// <summary>Kernel names, for the backends that register kernels by name.</summary>
    public static string[] KernelNames { get; } =
    {
        "mo_lisht", "mo_bent_identity", "mo_gaussian", "mo_scaled_tanh", "mo_thresholded_relu", "mo_isru", "mo_sqrbf",
        "mo_surrogate_spike", "mo_activation_backward",
        "mo_minmax_partial", "mo_minmax_final", "mo_fake_quantize",
        "mo_topk_select", "mo_topk_normalize", "mo_topk_chunk", "mo_topk_rank", "mo_topk_zero", "mo_topk_scatter",
        "mo_time_embedding", "mo_time_embedding_backward",
        "mo_crf_forward", "mo_crf_backward",
        "mo_crf_boundary_grad", "mo_crf_transition_grad",
    };

    /// <summary>The grid-index helper for CUDA and HIP, which compile the dialect directly.</summary>
    public const string CudaPrelude = "__device__ __forceinline__ int bx_gid() { return (int)(blockIdx.x * blockDim.x + threadIdx.x); }\n";

    /// <summary>The kernels in the portable dialect.</summary>
    public const string Source = @"
// ---- model-op helpers ----
__device__ __forceinline__ bool mo_isnan(float u) { return (__float_as_uint(u) & 0x7FFFFFFFu) > 0x7F800000u; }
__device__ __forceinline__ bool mo_isfinite(float u) { return (__float_as_uint(u) & 0x7FFFFFFFu) < 0x7F800000u; }
__device__ __forceinline__ bool mo_is_neg_inf(float u) { return __float_as_uint(u) == 0xFF800000u; }
__device__ __forceinline__ float mo_nan() { return __uint_as_float(0x7FC00000u); }
__device__ __forceinline__ float mo_inf() { return __uint_as_float(0x7F800000u); }

// tanh as Cephes' tanhf: an odd polynomial below 0.625 (no cancellation near 0), 1 - 2 / (e^(2|u|) + 1) up to 9 (no
// overflow), and +-1 past 9, where the float tanh is 1.
__device__ __forceinline__ float mo_tanh(float u)
{
    if (mo_isnan(u)) { return u; }
    float a = fabsf(u);
    if (a < 0.625f)
    {
        float z = u * u;
        float q = (((-5.70498872745e-3f * z + 2.06390887954e-2f) * z - 5.37397155531e-2f) * z + 1.33314422036e-1f) * z - 3.33332819422e-1f;
        return u + u * z * q;
    }
    float r = 1.0f;
    if (a < 9.0f) { r = 1.0f - 2.0f / (expf(2.0f * a) + 1.0f); }
    if (u < 0.0f) { return -r; }
    return r;
}

// sigma(u) (1 - sigma(u)) = e / (1 + e)^2 with e = exp(-|u|): never exp of a positive argument, never 1 - sigma.
__device__ __forceinline__ float mo_sigmoid_slope(float u)
{
    if (mo_isnan(u)) { return u; }
    float e = expf(-fabsf(u));
    float d = 1.0f + e;
    return e / (d * d);
}

// u with its low 12 significand bits cleared: the high half of an exact split u = hi + (u - hi).
__device__ __forceinline__ float mo_high_half(float u) { return __uint_as_float(__float_as_uint(u) & 0xFFFFF000u); }

// (sin, cos) of t * f, where f = f1 + f2 + f3 arrives as three floats of 12, 12 and 24 significant bits (the host's
// split of the exact frequency). t = th + tl splits on its bits, so th * f1, th * f2 and tl * f1 are exact and the
// reduction by pi / 2 (pi / 2 in three parts, the first of 8 bits so j * c1 is exact) starts from an exact product.
// The small terms carry only their own rounding. The reduced r lies in [-pi/4, pi/4] where Cephes' sinf / cosf
// polynomials are accurate to an ulp; the quadrant picks and signs them. Accurate for |t * f| up to about 1e4.
__device__ __forceinline__ float2 mo_sincos_product(float tv, float f1, float f2, float f3)
{
    float th = mo_high_half(tv);
    float tl = tv - th;
    float big = th * f1;
    float small = (th * f2 + tl * f1) + (tl * f2 + tv * f3);
    float j = rintf((big + small) * 0.636619772367581343f);
    float r = big - j * 1.5703125f;
    r = r + small;
    r = r - j * 4.837512969970703125e-4f;
    r = r - j * 7.54978995489188216e-8f;
    float z = r * r;
    float s = r + r * z * ((-1.9515295891e-4f * z + 8.3321608736e-3f) * z - 1.6666654611e-1f);
    float c = 1.0f - 0.5f * z + z * z * ((2.443315711809948e-5f * z - 1.388731625493765e-3f) * z + 4.166664568298827e-2f);
    if (!mo_isfinite(j)) { return make_float2(mo_nan(), mo_nan()); }
    int q = ((int)j) & 3;
    if (q == 0) { return make_float2(s, c); }
    if (q == 1) { return make_float2(c, -s); }
    if (q == 2) { return make_float2(-s, -c); }
    return make_float2(-c, s);
}

// The CPU's top-k order: larger first (NaN below every number, NaN equal to NaN), ties to the lower index.
__device__ __forceinline__ bool mo_before(float a, int ia, float b, int ib)
{
    bool an = mo_isnan(a);
    bool bn = mo_isnan(b);
    if (!an && bn) { return true; }
    if (a > b) { return true; }
    bool same = a == b || (an && bn);
    return same && ia < ib;
}

// ---- pointwise activations (the parity unary contract: x, y, n, p0, p1) ----

// LiSHT: x tanh(x)
extern ""C"" __global__ __launch_bounds__(256) void mo_lisht(const float* x, float* y, int n, float p0, float p1)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float v = x[i];
    y[i] = v * mo_tanh(v);
}

// bent identity: (sqrt(x^2 + 1) - 1) / 2 + x, with sqrt(x^2 + 1) - 1 = x^2 / (sqrt(x^2 + 1) + 1) (no cancellation) and
// |x| - 1 past 1e18 (no overflow of x^2).
extern ""C"" __global__ __launch_bounds__(256) void mo_bent_identity(const float* x, float* y, int n, float p0, float p1)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float v = x[i];
    float a = fabsf(v);
    float h = 0.0f;
    if (a > 1e18f) { h = a - 1.0f; }
    else { h = v * v / (sqrtf(v * v + 1.0f) + 1.0f); }
    y[i] = 0.5f * h + v;
}

// gaussian: exp(-x^2)
extern ""C"" __global__ __launch_bounds__(256) void mo_gaussian(const float* x, float* y, int n, float p0, float p1)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float v = x[i];
    y[i] = expf(-v * v);
}

// scaled tanh: alpha tanh(beta x), alpha = p0, beta = p1
extern ""C"" __global__ __launch_bounds__(256) void mo_scaled_tanh(const float* x, float* y, int n, float p0, float p1)
{
    int i = bx_gid();
    if (i >= n) { return; }
    y[i] = p0 * mo_tanh(p1 * x[i]);
}

// thresholded ReLU: x > theta ? x : 0 (NaN gives 0), theta = p0
extern ""C"" __global__ __launch_bounds__(256) void mo_thresholded_relu(const float* x, float* y, int n, float p0, float p1)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float v = x[i];
    float r = 0.0f;
    if (v > p0) { r = v; }
    y[i] = r;
}

// ISRU: x / sqrt(1 + alpha x^2), alpha = p0 >= 0. Where alpha x^2 overflows float for a finite x the value is its
// limit sign(x) / sqrt(alpha), as the CPU's wider arithmetic gives.
extern ""C"" __global__ __launch_bounds__(256) void mo_isru(const float* x, float* y, int n, float p0, float p1)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float v = x[i];
    float d = 1.0f + p0 * v * v;
    float r = v / sqrtf(d);
    if (d > 3.0e38f && mo_isfinite(v))
    {
        float s = 1.0f;
        if (v < 0.0f) { s = -1.0f; }
        r = s / sqrtf(p0);
    }
    y[i] = r;
}

// SQRBF: exp(-beta x^2), beta = p0
extern ""C"" __global__ __launch_bounds__(256) void mo_sqrbf(const float* x, float* y, int n, float p0, float p1)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float v = x[i];
    y[i] = expf(-p0 * v * v);
}

// surrogate spike forward: the Heaviside step x > threshold (p0, the host's threshold rounded down to a float so
// the comparison agrees with the CPU's exact one)
extern ""C"" __global__ __launch_bounds__(256) void mo_surrogate_spike(const float* x, float* y, int n, float p0, float p1)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float r = 0.0f;
    if (x[i] > p0) { r = 1.0f; }
    y[i] = r;
}

// dx = dy * f'(x) for activation op: 0 LiSHT, 1 bent identity, 2 gaussian, 3 scaled tanh (p0 alpha, p1 beta),
// 4 thresholded ReLU (p0 theta), 5 ISRU (p0 alpha), 6 SQRBF (p0 beta), 7 surrogate spike (p0 threshold, p1 beta:
// beta sigma'(beta (x - threshold))).
extern ""C"" __global__ __launch_bounds__(256) void mo_activation_backward(const float* dy, const float* x, float* dx, int n, int op, float p0, float p1)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float v = x[i];
    float d = 0.0f;
    if (op == 0)
    {
        float t = mo_tanh(v);
        d = t + v * (1.0f - t * t);
    }
    else if (op == 1)
    {
        if (fabsf(v) > 1e18f && mo_isfinite(v))
        {
            d = 1.5f;
            if (v < 0.0f) { d = 0.5f; }
        }
        else { d = v / (2.0f * sqrtf(v * v + 1.0f)) + 1.0f; }
    }
    else if (op == 2) { d = -2.0f * v * expf(-v * v); }
    else if (op == 3)
    {
        float t = mo_tanh(p1 * v);
        d = p0 * p1 * (1.0f - t * t);
    }
    else if (op == 4)
    {
        if (v > p0) { d = 1.0f; }
    }
    else if (op == 5)
    {
        float den = 1.0f + p0 * v * v;
        d = (1.0f / den) / sqrtf(den);
    }
    else if (op == 6) { d = -2.0f * p0 * v * expf(-p0 * v * v); }
    else { d = p1 * mo_sigmoid_slope(p1 * (v - p0)); }
    dx[i] = dy[i] * d;
}

// ---- fake quantization ----

// Slot t's minimum and maximum over x[t], x[t + slots], ...; NaN is skipped, as the CPU's < and > skip it.
extern ""C"" __global__ __launch_bounds__(256) void mo_minmax_partial(const float* x, float* partial, int n, int slots)
{
    int t = bx_gid();
    if (t >= slots) { return; }
    float lo = mo_inf();
    float hi = -mo_inf();
    for (int i = t; i < n; i = i + slots)
    {
        float v = x[i];
        if (v < lo) { lo = v; }
        if (v > hi) { hi = v; }
    }
    partial[2 * t] = lo;
    partial[2 * t + 1] = hi;
}

extern ""C"" __global__ __launch_bounds__(256) void mo_minmax_final(const float* partial, float* minmax, int slots)
{
    int t = bx_gid();
    if (t >= 1) { return; }
    float lo = mo_inf();
    float hi = -mo_inf();
    for (int s = 0; s < slots; s++)
    {
        float a = partial[2 * s];
        float b = partial[2 * s + 1];
        if (a < lo) { lo = a; }
        if (b > hi) { hi = b; }
    }
    minmax[0] = lo;
    minmax[1] = hi;
}

// y = (clamp(round(x / step) + zp, qMin, qMax) - zp) * step and mask = 1 where round(x / step) + zp was in range.
// derive != 0 forms step (and, without a given zero point, an asymmetric zp) from minmax as the CPU does; round is
// half to even, as Math.Round.
extern ""C"" __global__ __launch_bounds__(256) void mo_fake_quantize(const float* x, const float* minmax, float* y, float* mask, int n, int derive, int symmetric, int hasZeroPoint, float qMin, float qMax, float fixedStep, float zeroPoint)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float stp = fixedStep;
    float zp = zeroPoint;
    if (derive != 0)
    {
        float lo = minmax[0];
        float hi = minmax[1];
        if (symmetric != 0) { stp = fmaxf(fabsf(lo), fabsf(hi)) / qMax; }
        else { stp = (hi - lo) / (qMax - qMin); }
        if (stp == 0.0f || !mo_isfinite(stp)) { stp = 1.0f; }
        if (hasZeroPoint == 0)
        {
            zp = 0.0f;
            if (symmetric == 0)
            {
                zp = rintf(qMin - lo / stp);
                if (zp < qMin) { zp = qMin; }
                if (zp > qMax) { zp = qMax; }
            }
        }
    }
    float q = rintf(x[i] / stp) + zp;
    float keep = 0.0f;
    if (q >= qMin && q <= qMax) { keep = 1.0f; }
    float c = q;
    if (q < qMin) { c = qMin; }
    else if (q > qMax) { c = qMax; }
    mask[i] = keep;
    y[i] = (c - zp) * stp;
}

// ---- top-k softmax over the last axis ----

// e = exp(x - top) for the k entries of each row that come first in the CPU's order, 0 elsewhere.
extern ""C"" __global__ __launch_bounds__(256) void mo_topk_select(const float* x, float* e, int rows, int width, int k)
{
    int g = bx_gid();
    if (g >= rows * width) { return; }
    int r = g / width;
    int i = g - r * width;
    int base = r * width;
    float xi = x[g];
    float top = x[base];
    int topIndex = 0;
    int rank = 0;
    for (int j = 0; j < width; j++)
    {
        float xj = x[base + j];
        if (mo_before(xj, j, xi, i)) { rank = rank + 1; }
        if (mo_before(xj, j, top, topIndex)) { top = xj; topIndex = j; }
    }
    float r0 = 0.0f;
    if (rank < k) { r0 = expf(xi - top); }
    e[g] = r0;
}

// y = e / (row sum of e); entries outside the top k stay exactly 0.
extern ""C"" __global__ __launch_bounds__(256) void mo_topk_normalize(const float* e, float* y, int rows, int width)
{
    int g = bx_gid();
    if (g >= rows * width) { return; }
    int base = (g / width) * width;
    float v = e[g];
    if (v == 0.0f) { y[g] = 0.0f; return; }
    float s = 0.0f;
    for (int j = 0; j < width; j++) { s = s + e[base + j]; }
    y[g] = v / s;
}

// Long rows (a vocabulary): each (row, chunk) slot keeps its chunk's kk best entries, best first, by insertion (one
// comparison per element once the list is full and the element loses to its last entry). The global top k of a row
// are among these candidates: every entry ranking ahead of a top-k entry is itself top-k, so it is in its chunk's list.
// Unfilled slots get (NaN, an index past the row), which rank after every real entry.
extern ""C"" __global__ __launch_bounds__(256) void mo_topk_chunk(const float* x, float* candVal, int* candIdx, int rows, int width, int kk, int chunk, int chunks)
{
    int g = bx_gid();
    if (g >= rows * chunks) { return; }
    int r = g / chunks;
    int c = g - r * chunks;
    int base = r * width;
    int dst = g * kk;
    int begin = c * chunk;
    int end = begin + chunk;
    if (end > width) { end = width; }
    int count = 0;
    for (int j = begin; j < end; j++)
    {
        float v = x[base + j];
        int slot = count;
        if (count == kk)
        {
            if (!mo_before(v, j, candVal[dst + kk - 1], candIdx[dst + kk - 1])) { continue; }
            slot = kk - 1;
        }
        int pos = slot;
        for (int s = slot; s > 0; s--)
        {
            if (!mo_before(v, j, candVal[dst + s - 1], candIdx[dst + s - 1])) { break; }
            candVal[dst + s] = candVal[dst + s - 1];
            candIdx[dst + s] = candIdx[dst + s - 1];
            pos = s - 1;
        }
        candVal[dst + pos] = v;
        candIdx[dst + pos] = j;
        if (count < kk) { count = count + 1; }
    }
    for (int s = count; s < kk; s++)
    {
        candVal[dst + s] = mo_nan();
        candIdx[dst + s] = width + s;
    }
}

// A candidate's rank among its row's candidates: per chunk list (sorted best first), a binary search for the entries
// ahead of it. w = exp(v - top) for the k best, -1 for the rest.
extern ""C"" __global__ __launch_bounds__(256) void mo_topk_rank(const float* candVal, const int* candIdx, float* w, int rows, int kk, int chunks, int k)
{
    int g = bx_gid();
    int m = chunks * kk;
    if (g >= rows * m) { return; }
    int base = (g / m) * m;
    float v = candVal[g];
    int vi = candIdx[g];
    int rank = 0;
    float top = candVal[base];
    int topIndex = candIdx[base];
    for (int c = 0; c < chunks; c++)
    {
        int list = base + c * kk;
        if (mo_before(candVal[list], candIdx[list], top, topIndex)) { top = candVal[list]; topIndex = candIdx[list]; }
        int lo = 0;
        int hi = kk;
        for (int it = 0; it < 32; it++)
        {
            if (lo >= hi) { break; }
            int mid = (lo + hi) / 2;
            if (mo_before(candVal[list + mid], candIdx[list + mid], v, vi)) { lo = mid + 1; }
            else { hi = mid; }
        }
        rank = rank + lo;
    }
    float r0 = -1.0f;
    if (rank < k) { r0 = expf(v - top); }
    w[g] = r0;
}

extern ""C"" __global__ __launch_bounds__(256) void mo_topk_zero(float* y, int n)
{
    int g = bx_gid();
    if (g >= n) { return; }
    y[g] = 0.0f;
}

// y[row, index] = w / (row sum of the selected w) for the k selected candidates (w -1 marks the rest; NaN is kept).
extern ""C"" __global__ __launch_bounds__(256) void mo_topk_scatter(const int* candIdx, const float* w, float* y, int rows, int m, int width)
{
    int g = bx_gid();
    if (g >= rows * m) { return; }
    float v = w[g];
    if (v < 0.0f) { return; }
    int r = g / m;
    int base = r * m;
    float s = 0.0f;
    for (int j = 0; j < m; j++)
    {
        float wj = w[base + j];
        if (!(wj < 0.0f)) { s = s + wj; }
    }
    y[r * width + candIdx[g]] = v / s;
}

// ---- sinusoidal time embedding ----

// y[b, c] = sin(t_b f_c) for c < half, cos(t_b f_(c - half)) for half <= c < 2 half, 0 past (an odd dim); freq holds
// each frequency as three floats (see mo_sincos_product).
extern ""C"" __global__ __launch_bounds__(256) void mo_time_embedding(const float* t, const float* freq, float* y, int batch, int halfDim, int dim)
{
    int g = bx_gid();
    if (g >= batch * dim) { return; }
    int b = g / dim;
    int c = g - b * dim;
    if (c >= 2 * halfDim) { y[g] = 0.0f; return; }
    int f = c;
    if (c >= halfDim) { f = c - halfDim; }
    float2 sc = mo_sincos_product(t[b], freq[3 * f], freq[3 * f + 1], freq[3 * f + 2]);
    if (c >= halfDim) { y[g] = sc.y; }
    else { y[g] = sc.x; }
}

// dt_b = sum_i f_i (dy[b, i] cos(t_b f_i) - dy[b, i + half] sin(t_b f_i))
extern ""C"" __global__ __launch_bounds__(256) void mo_time_embedding_backward(const float* t, const float* freq, const float* dy, float* dt, int batch, int halfDim, int dim)
{
    int b = bx_gid();
    if (b >= batch) { return; }
    float tb = t[b];
    float s = 0.0f;
    for (int f = 0; f < halfDim; f++)
    {
        float2 sc = mo_sincos_product(tb, freq[3 * f], freq[3 * f + 1], freq[3 * f + 2]);
        float fv = freq[3 * f] + (freq[3 * f + 1] + freq[3 * f + 2]);
        s = s + fv * (dy[b * dim + f] * sc.y - dy[b * dim + f + halfDim] * sc.x);
    }
    dt[b] = s;
}

// ---- linear-chain CRF log-partition (log space, [batch, seqLen, tags]; transitions [from, to]) ----
// One work group per sequence (launched with batch * 256 threads) walks the time steps itself, its lanes covering the
// tags, with a group barrier (device-memory scope) between steps: the forward recursion is one launch. The tag loops
// step by the group size from 0, so every lane reaches every barrier (uniform control flow).
//
// Precision: log alpha grows with t (and log beta with seqLen - t), so plain float recursions lose an ulp of a value
// near 1e3 per step, and the marginals exp(alpha + beta - logZ) cancel those large values. Each step is therefore kept
// relative to an offset: the stored row is a_t = alpha_t - A_t with A_t = A_(t-1) + d_t, d_t the previous stored row's
// maximum (an exact float), and A_t is summed exactly as an unevaluated hi + lo pair. The marginals' exponent
// A_t + B_t - logZ is formed in that pair arithmetic, so it is a small number with float's full relative accuracy.
//
// Buffer layout: alpha holds the stored rows [batch, seqLen, tags], then per sequence (seqLen + 1) (hi, lo) pairs: A_t,
// then logZ. beta holds the stored rows, then per sequence seqLen pairs (G_t, H_t) = (A_t + B_t - logZ,
// A_(t-1) + B_t - logZ). A log-sum-exp subtracts the maximum; a -inf or NaN maximum is the result itself, as the CPU's.

// exp and log to about an ulp on every backend: GLSL and WGSL bound their builtins only loosely (exp to 3 + 2|x| ulp, log
// to an absolute 2^-21 near 1), an error a CRF recursion would add once per time step. exp: x = k ln2 + r with ln2 in
// two parts (the first of 8 bits, so k ln2_hi is exact), Cephes' expf polynomial on |r| <= ln2 / 2, and 2^k applied as
// two exact power-of-two factors. log: x = m 2^e with m in [sqrt(1/2), sqrt(2)), log m = 2 atanh((m - 1) / (m + 1)) by its
// odd series. Arguments outside the normal range use the builtin.
__device__ __forceinline__ float mo_exp(float x)
{
    if (!(x > -87.0f) || !(x < 88.0f)) { return expf(x); }
    float k = rintf(x * 1.44269504088896341f);
    float r = (x - k * 0.693359375f) - k * -2.12194440e-4f;
    float q = ((((1.9875691500e-4f * r + 1.3981999507e-3f) * r + 8.3334519073e-3f) * r + 4.1665795894e-2f) * r + 1.6666665459e-1f) * r + 5.0000001201e-1f;
    float er = 1.0f + r + r * r * q;
    int ki = (int)k;
    int k1 = ki / 2;
    int k2 = ki - k1;
    float s1 = __uint_as_float((unsigned int)(k1 + 127) << 23);
    float s2 = __uint_as_float((unsigned int)(k2 + 127) << 23);
    return er * s1 * s2;
}

__device__ __forceinline__ float mo_log(float x)
{
    unsigned int bits = __float_as_uint(x);
    if (bits < 0x00800000u || bits >= 0x7F800000u) { return logf(x); }
    int e = (int)(bits >> 23) - 127;
    float m = __uint_as_float((bits & 0x007FFFFFu) | 0x3F800000u);
    if (m > 1.41421356237f)
    {
        m = 0.5f * m;
        e = e + 1;
    }
    float f = m - 1.0f;
    float s = f / (2.0f + f);
    float z = s * s;
    float series = 2.0f * s + 2.0f * s * z * ((((z * 0.111111111f + 0.142857143f) * z + 0.2f) * z) + 0.333333333f);
    float ef = (float)e;
    return ef * 0.693145752f + (series + ef * 1.42860677e-6f);
}

// hi + lo + y as a renormalized pair (Knuth's two-sum: exact without fma); a non-finite sum carries through.
__device__ __forceinline__ float2 mo_pair_add(float hi, float lo, float y)
{
    float s = hi + y;
    if (!mo_isfinite(s)) { return make_float2(s, 0.0f); }
    float bb = s - hi;
    float e = (hi - (s - bb)) + (y - bb);
    float t = lo + e;
    float h = s + t;
    float l = t - (h - s);
    return make_float2(h, l);
}

// (aHi + aLo) + (bHi + bLo) - (zHi + zLo), rounded once to float.
__device__ __forceinline__ float mo_pair_exponent(float aHi, float aLo, float bHi, float bLo, float zHi, float zLo)
{
    float2 x = mo_pair_add(aHi, aLo, bHi);
    x = mo_pair_add(x.x, x.y, bLo);
    x = mo_pair_add(x.x, x.y, -zHi);
    x = mo_pair_add(x.x, x.y, -zLo);
    float r = x.x + x.y;
    return r;
}

extern ""C"" __global__ __launch_bounds__(256) void mo_crf_forward(const float* emit, const float* trans, const float* start, const float* fin, float* alpha, float* logZ, int batch, int seqLen, int tags)
{
    int b = blockIdx.x;
    if (b >= batch) { return; }
    int lane = threadIdx.x;
    int lanes = blockDim.x;
    int seq = b * seqLen * tags;
    int offs = batch * seqLen * tags + b * (seqLen + 1) * 2;
    for (int base = 0; base < tags; base = base + lanes)
    {
        int j = base + lane;
        if (j < tags) { alpha[seq + j] = start[j] + emit[seq + j]; }
    }
    float accHi = 0.0f;
    float accLo = 0.0f;
    if (lane == 0)
    {
        alpha[offs] = 0.0f;
        alpha[offs + 1] = 0.0f;
    }
    __syncthreads();
    for (int t = 1; t < seqLen; t++)
    {
        int prev = seq + (t - 1) * tags;
        float d = -mo_inf();
        for (int i = 0; i < tags; i++)
        {
            float a = alpha[prev + i];
            if (a > d) { d = a; }
        }
        if (!mo_isfinite(d)) { d = 0.0f; }
        for (int base = 0; base < tags; base = base + lanes)
        {
            int j = base + lane;
            if (j < tags)
            {
                float m = -mo_inf();
                for (int i = 0; i < tags; i++)
                {
                    float v = (alpha[prev + i] - d) + trans[i * tags + j];
                    if (v > m) { m = v; }
                }
                float lse = m;
                if (!mo_is_neg_inf(m) && !mo_isnan(m))
                {
                    float s = 0.0f;
                    for (int i = 0; i < tags; i++) { s = s + mo_exp((alpha[prev + i] - d) + trans[i * tags + j] - m); }
                    lse = m + mo_log(s);
                }
                alpha[prev + tags + j] = emit[prev + tags + j] + lse;
            }
        }
        if (lane == 0)
        {
            float2 acc = mo_pair_add(accHi, accLo, d);
            accHi = acc.x;
            accLo = acc.y;
            alpha[offs + 2 * t] = accHi;
            alpha[offs + 2 * t + 1] = accLo;
        }
        __syncthreads();
    }
    if (lane == 0)
    {
        int last = seq + (seqLen - 1) * tags;
        float m = -mo_inf();
        for (int j = 0; j < tags; j++)
        {
            float v = alpha[last + j] + fin[j];
            if (v > m) { m = v; }
        }
        float lse = m;
        if (!mo_is_neg_inf(m) && !mo_isnan(m))
        {
            float s = 0.0f;
            for (int j = 0; j < tags; j++) { s = s + mo_exp(alpha[last + j] + fin[j] - m); }
            lse = m + mo_log(s);
        }
        float2 z = mo_pair_add(accHi, accLo, lse);
        alpha[offs + 2 * seqLen] = z.x;
        alpha[offs + 2 * seqLen + 1] = z.y;
        logZ[b] = z.x + z.y;
    }
}

// The backward recursion (stored rows b_t = beta_t - B_t, beta at the last step being end), the per-step exponents
// G_t / H_t, and the emission gradient d logZ / d emit[b, t, j] = exp(alpha + beta - logZ) * dz[b], in the sequence's
// work group.
extern ""C"" __global__ __launch_bounds__(256) void mo_crf_backward(const float* emit, const float* trans, const float* fin, const float* alpha, const float* dz, float* beta, float* demit, int batch, int seqLen, int tags)
{
    int b = blockIdx.x;
    if (b >= batch) { return; }
    int lane = threadIdx.x;
    int lanes = blockDim.x;
    int seq = b * seqLen * tags;
    int volume = batch * seqLen * tags;
    int aoffs = volume + b * (seqLen + 1) * 2;
    int goffs = volume + b * seqLen * 2;
    float zHi = alpha[aoffs + 2 * seqLen];
    float zLo = alpha[aoffs + 2 * seqLen + 1];
    for (int base = 0; base < tags; base = base + lanes)
    {
        int j = base + lane;
        if (j < tags) { beta[seq + (seqLen - 1) * tags + j] = fin[j]; }
    }
    float accHi = 0.0f;
    float accLo = 0.0f;
    if (lane == 0)
    {
        int t = seqLen - 1;
        beta[goffs + 2 * t] = mo_pair_exponent(alpha[aoffs + 2 * t], alpha[aoffs + 2 * t + 1], 0.0f, 0.0f, zHi, zLo);
        if (t >= 1) { beta[goffs + 2 * t + 1] = mo_pair_exponent(alpha[aoffs + 2 * t - 2], alpha[aoffs + 2 * t - 1], 0.0f, 0.0f, zHi, zLo); }
    }
    __syncthreads();
    for (int t = seqLen - 2; t >= 0; t--)
    {
        int nxt = seq + (t + 1) * tags;
        float e = -mo_inf();
        for (int j = 0; j < tags; j++)
        {
            float v = beta[nxt + j];
            if (v > e) { e = v; }
        }
        if (!mo_isfinite(e)) { e = 0.0f; }
        for (int base = 0; base < tags; base = base + lanes)
        {
            int i = base + lane;
            if (i < tags)
            {
                float m = -mo_inf();
                for (int j = 0; j < tags; j++)
                {
                    float v = trans[i * tags + j] + emit[nxt + j] + (beta[nxt + j] - e);
                    if (v > m) { m = v; }
                }
                float lse = m;
                if (!mo_is_neg_inf(m) && !mo_isnan(m))
                {
                    float s = 0.0f;
                    for (int j = 0; j < tags; j++) { s = s + mo_exp(trans[i * tags + j] + emit[nxt + j] + (beta[nxt + j] - e) - m); }
                    lse = m + mo_log(s);
                }
                beta[nxt - tags + i] = lse;
            }
        }
        if (lane == 0)
        {
            float2 acc = mo_pair_add(accHi, accLo, e);
            accHi = acc.x;
            accLo = acc.y;
            beta[goffs + 2 * t] = mo_pair_exponent(alpha[aoffs + 2 * t], alpha[aoffs + 2 * t + 1], accHi, accLo, zHi, zLo);
            if (t >= 1) { beta[goffs + 2 * t + 1] = mo_pair_exponent(alpha[aoffs + 2 * t - 2], alpha[aoffs + 2 * t - 1], accHi, accLo, zHi, zLo); }
        }
        __syncthreads();
    }
    float scale = dz[b];
    int span = seqLen * tags;
    for (int base = 0; base < span; base = base + lanes)
    {
        int at = base + lane;
        if (at < span)
        {
            int t = at / tags;
            demit[seq + at] = mo_exp(alpha[seq + at] + beta[seq + at] + beta[goffs + 2 * t]) * scale;
        }
    }
}

// d start[j] = sum_b p(y_0 = j), d end[j] = sum_b p(y_last = j): the emission gradients at the first and last step.
extern ""C"" __global__ __launch_bounds__(256) void mo_crf_boundary_grad(const float* demit, float* dstart, float* dend, int batch, int seqLen, int tags)
{
    int j = bx_gid();
    if (j >= tags) { return; }
    float s0 = 0.0f;
    float s1 = 0.0f;
    for (int b = 0; b < batch; b++)
    {
        s0 = s0 + demit[b * seqLen * tags + j];
        s1 = s1 + demit[b * seqLen * tags + (seqLen - 1) * tags + j];
    }
    dstart[j] = s0;
    dend[j] = s1;
}

// d trans[i, j] = sum_b sum_(t >= 1) p(y_(t-1) = i, y_t = j) dz[b] = exp(a_(t-1, i) + trans + emit_t,j + b_t,j + H_t) dz[b].
extern ""C"" __global__ __launch_bounds__(256) void mo_crf_transition_grad(const float* emit, const float* trans, const float* alpha, const float* beta, const float* dz, float* dtrans, int batch, int seqLen, int tags)
{
    int g = bx_gid();
    if (g >= tags * tags) { return; }
    int i = g / tags;
    int j = g - i * tags;
    int volume = batch * seqLen * tags;
    float tr = trans[g];
    float s = 0.0f;
    for (int b = 0; b < batch; b++)
    {
        float scale = dz[b];
        int goffs = volume + b * seqLen * 2;
        for (int t = 1; t < seqLen; t++)
        {
            int at = b * seqLen * tags + t * tags;
            s = s + mo_exp(alpha[at - tags + i] + tr + emit[at + j] + beta[at + j] + beta[goffs + 2 * t + 1]) * scale;
        }
    }
    dtrans[g] = s;
}

";
}
