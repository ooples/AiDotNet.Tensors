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
/// The CRF recursions launch once per time step (no work-group memory or barriers, which the portable dialect does
/// not have); every other op is one or two launches.
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
        "mo_topk_select", "mo_topk_normalize",
        "mo_time_embedding", "mo_time_embedding_backward",
        "mo_crf_alpha_init", "mo_crf_alpha_step", "mo_crf_log_partition", "mo_crf_beta_init", "mo_crf_beta_step",
        "mo_crf_emission_grad", "mo_crf_boundary_grad", "mo_crf_transition_grad",
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

// ---- linear-chain CRF log-partition (alpha / beta in log space, [batch, seqLen, tags]; transitions [from, to]) ----

extern ""C"" __global__ __launch_bounds__(256) void mo_crf_alpha_init(const float* emit, const float* start, float* alpha, int batch, int seqLen, int tags)
{
    int g = bx_gid();
    if (g >= batch * tags) { return; }
    int b = g / tags;
    int j = g - b * tags;
    int at = b * seqLen * tags + j;
    alpha[at] = start[j] + emit[at];
}

// alpha[b, t, j] = emit[b, t, j] + logsumexp_i(alpha[b, t - 1, i] + trans[i, j]), the maximum subtracted; a -inf or
// NaN maximum is the result itself, as the CPU's LogSumExp.
extern ""C"" __global__ __launch_bounds__(256) void mo_crf_alpha_step(const float* emit, const float* trans, float* alpha, int batch, int seqLen, int tags, int stepIndex)
{
    int g = bx_gid();
    if (g >= batch * tags) { return; }
    int b = g / tags;
    int j = g - b * tags;
    int prev = b * seqLen * tags + (stepIndex - 1) * tags;
    float m = -mo_inf();
    for (int i = 0; i < tags; i++)
    {
        float v = alpha[prev + i] + trans[i * tags + j];
        if (v > m) { m = v; }
    }
    float lse = m;
    if (!mo_is_neg_inf(m) && !mo_isnan(m))
    {
        float s = 0.0f;
        for (int i = 0; i < tags; i++) { s = s + expf(alpha[prev + i] + trans[i * tags + j] - m); }
        lse = m + logf(s);
    }
    int at = prev + tags + j;
    alpha[at] = emit[at] + lse;
}

extern ""C"" __global__ __launch_bounds__(256) void mo_crf_log_partition(const float* alpha, const float* fin, float* logZ, int batch, int seqLen, int tags)
{
    int b = bx_gid();
    if (b >= batch) { return; }
    int last = b * seqLen * tags + (seqLen - 1) * tags;
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
        for (int j = 0; j < tags; j++) { s = s + expf(alpha[last + j] + fin[j] - m); }
        lse = m + logf(s);
    }
    logZ[b] = lse;
}

extern ""C"" __global__ __launch_bounds__(256) void mo_crf_beta_init(const float* fin, float* beta, int batch, int seqLen, int tags)
{
    int g = bx_gid();
    if (g >= batch * tags) { return; }
    int b = g / tags;
    int j = g - b * tags;
    beta[b * seqLen * tags + (seqLen - 1) * tags + j] = fin[j];
}

// beta[b, t, i] = logsumexp_j(trans[i, j] + emit[b, t + 1, j] + beta[b, t + 1, j])
extern ""C"" __global__ __launch_bounds__(256) void mo_crf_beta_step(const float* emit, const float* trans, float* beta, int batch, int seqLen, int tags, int stepIndex)
{
    int g = bx_gid();
    if (g >= batch * tags) { return; }
    int b = g / tags;
    int i = g - b * tags;
    int nxt = b * seqLen * tags + (stepIndex + 1) * tags;
    float m = -mo_inf();
    for (int j = 0; j < tags; j++)
    {
        float v = trans[i * tags + j] + emit[nxt + j] + beta[nxt + j];
        if (v > m) { m = v; }
    }
    float lse = m;
    if (!mo_is_neg_inf(m) && !mo_isnan(m))
    {
        float s = 0.0f;
        for (int j = 0; j < tags; j++) { s = s + expf(trans[i * tags + j] + emit[nxt + j] + beta[nxt + j] - m); }
        lse = m + logf(s);
    }
    beta[nxt - tags + i] = lse;
}

// d logZ / d emit[b, t, j] = p(y_t = j) = exp(alpha + beta - logZ), scaled by the row's upstream gradient.
extern ""C"" __global__ __launch_bounds__(256) void mo_crf_emission_grad(const float* alpha, const float* beta, const float* logZ, const float* dz, float* demit, int batch, int seqLen, int tags)
{
    int g = bx_gid();
    if (g >= batch * seqLen * tags) { return; }
    int b = g / (seqLen * tags);
    demit[g] = expf(alpha[g] + beta[g] - logZ[b]) * dz[b];
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

// d trans[i, j] = sum_b sum_(t >= 1) p(y_(t-1) = i, y_t = j), scaled by each row's upstream gradient.
extern ""C"" __global__ __launch_bounds__(256) void mo_crf_transition_grad(const float* emit, const float* trans, const float* alpha, const float* beta, const float* logZ, const float* dz, float* dtrans, int batch, int seqLen, int tags)
{
    int g = bx_gid();
    if (g >= tags * tags) { return; }
    int i = g / tags;
    int j = g - i * tags;
    float tr = trans[g];
    float s = 0.0f;
    for (int b = 0; b < batch; b++)
    {
        float lz = logZ[b];
        float scale = dz[b];
        for (int t = 1; t < seqLen; t++)
        {
            int at = b * seqLen * tags + t * tags;
            s = s + expf(alpha[at - tags + i] + tr + emit[at + j] + beta[at + j] - lz) * scale;
        }
    }
    dtrans[g] = s;
}
";
}
