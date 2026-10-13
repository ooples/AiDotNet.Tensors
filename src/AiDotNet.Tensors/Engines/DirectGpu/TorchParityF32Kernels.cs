// Copyright (c) AiDotNet. All rights reserved.
using System.Globalization;
using System.Linq;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The PyTorch-parity kernels (#1125, #1128) in FP32 for backends without 64-bit floats (Metal, WebGPU), written in the
/// portable CUDA dialect <see cref="WebGpu.CudaDialectToWgsl"/> documents, so one source serves WGSL (that translator)
/// and MSL (MetalBatchExecutionKernels' translator). The algorithms are the CPU references' (Helpers/SpecialFunctions,
/// as the OpenCL port runs them in double), evaluated in float with float-sized convergence thresholds and an exp
/// underflow bound of 104; NaN and infinity are tested on their bits so fast math cannot fold them away. Verified on
/// Dawn against the same CPU references the device parity suites hold every backend to.
/// </summary>
internal static class TorchParityF32Kernels
{
    public static string GetSource() => Tables() + Helpers + SpecialFunctions + Elementwise + Sampling + Indexing;

    private static string Tables()
    {
        var sb = new StringBuilder();
        foreach (int n in new[] { 96, 32 })
        {
            var (x, w) = AiDotNet.Tensors.Helpers.SpecialFunctions.GaussLegendreRule(n);
            sb.Append("__device__ const float TP_GL").Append(n).Append("_X[").Append(n).Append("] = { ").Append(Join(x)).Append(" };\n");
            sb.Append("__device__ const float TP_GL").Append(n).Append("_W[").Append(n).Append("] = { ").Append(Join(w)).Append(" };\n");
        }
        return sb.ToString();
    }

    private static string Join(double[] values)
        => string.Join(", ", values.Select(v => ((float)v).ToString("R", CultureInfo.InvariantCulture) + "f"));

    private const string Helpers = @"
__device__ __forceinline__ bool tp_isnan(float v) { return (__float_as_uint(v) & 0x7FFFFFFFu) > 0x7F800000u; }
__device__ __forceinline__ bool tp_isinf(float v) { return (__float_as_uint(v) & 0x7FFFFFFFu) == 0x7F800000u; }
__device__ __forceinline__ bool tp_signbit_f(float v) { return (__float_as_uint(v) & 0x80000000u) != 0u; }
__device__ __forceinline__ float tp_nan() { return __uint_as_float(0x7FC00000u); }
__device__ __forceinline__ float tp_inf() { return __uint_as_float(0x7F800000u); }
__device__ __forceinline__ float tp_ninf() { return __uint_as_float(0xFF800000u); }
__device__ __forceinline__ float tp_pick(bool c, float a, float b) { if (c) { return a; } return b; }

// min(max(v, lo), hi) with NaN propagating from v or either bound (Math.Max / Math.Min semantics).
__device__ __forceinline__ float tp_clamp_nan(float v, float lo, float hi)
{
    if (tp_isnan(v) || tp_isnan(lo) || tp_isnan(hi)) { return tp_nan(); }
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}

__device__ __forceinline__ float tp_log1p(float y)
{
    float u = 1.0f + y;
    if (u == 1.0f) { return y; }
    return logf(u) * y / (u - 1.0f);
}

// sin(pi a) for a in [0, 0.5], from the Maclaurin series: sin on [0, pi/4], cos of the complement above it (0.5 - a is
// exact there). Not the device's sinf: a shading language may give sin an ABSOLUTE error bound and no relative one,
// and one D3D12 driver returns sin(3.14e-7) = 1.87e-7, which is 40% low (#1220). sinc, and lgamma's reflection through
// log(sin(pi x)), need the relative accuracy near zero. The truncation error here is below 1e-10.
__device__ __forceinline__ float tp_sinpi_reduced(float a)
{
    if (a <= 0.25f)
    {
        float ts = 3.14159265358979f * a;
        float us = ts * ts;
        return ts * (1.0f + us * (-0.16666666666666666f + us * (8.333333333333333e-3f + us * (-1.984126984126984e-4f + us * (2.7557319223985893e-6f + us * (-2.505210838544172e-8f))))));
    }
    float tc = 3.14159265358979f * (0.5f - a);
    float uc = tc * tc;
    return 1.0f + uc * (-0.5f + uc * (4.1666666666666664e-2f + uc * (-1.3888888888888889e-3f + uc * (2.48015873015873e-5f + uc * (-2.755731922398589e-7f + uc * 2.08767569878681e-9f)))));
}

// sin(pi x) with exact argument reduction (SpecialFunctions.SinPi).
__device__ __forceinline__ float tp_sinpi(float x)
{
    if (tp_isnan(x) || tp_isinf(x)) { return tp_nan(); }
    float r = x - 2.0f * rintf(x * 0.5f);
    float a = fabsf(r);
    if (a == 0.0f || a == 1.0f) { return 0.0f; }
    if (a > 0.5f) { a = 1.0f - a; }
    float s = tp_sinpi_reduced(a);
    if (r < 0.0f) { return -s; }
    return s;
}

// The Lanczos series of Numerical Recipes' gammln (g = 5, six terms): its terms cancel far less than the g = 7 set, so
// it stays accurate in float. Gamma(a) = (a + 5.5)^(a + 0.5) e^-(a + 5.5) sqrt(2 pi) S(a) / a.
__device__ __forceinline__ float tp_lanczos_series(float a)
{
    float s = 1.000000000190015f;
    s = s + 76.18009172947146f / (a + 1.0f);
    s = s - 86.50532032941677f / (a + 2.0f);
    s = s + 24.01409824083091f / (a + 3.0f);
    s = s - 1.231739572450155f / (a + 4.0f);
    s = s + 0.1208650973866179e-2f / (a + 5.0f);
    s = s - 0.5395239384953e-5f / (a + 6.0f);
    return s;
}

__device__ __forceinline__ float tp_lgamma_positive(float x)
{
    float g = x + 5.5f;
    return (x + 0.5f) * logf(g) - g + logf(2.5066282746310005f * tp_lanczos_series(x) / x);
}

// ln Gamma(x): the series for x >= 0.5, the reflection formula below.
__device__ __forceinline__ float tp_lgamma(float x)
{
    if (tp_isnan(x)) { return x; }
    if (tp_isinf(x)) { return tp_inf(); }
    if (x >= 0.5f) { return tp_lgamma_positive(x); }
    float s = tp_sinpi(x);
    if (s == 0.0f) { return tp_inf(); }
    return 1.1447298858494002f - logf(fabsf(s)) - tp_lgamma_positive(1.0f - x);
}

// x^a e^-x / Gamma(a), the prefix of the regularized incomplete gamma, without the cancellation of
// a log x - x - ln Gamma(a): for a >= 0.5 it is (x / g)^a e^(g - x) a / (sqrt(2 pi g) S(a)) with g = a + 5.5, whose
// exponent a log1p((x - g) / g) - (x - g) stays small where x is near a.
__device__ __forceinline__ float tp_gamma_prefix(float a, float x)
{
    if (a < 0.5f) { return expf(-x + a * logf(x) - tp_lgamma(a)); }
    float g = a + 5.5f;
    float delta = x - g;
    float l = 0.0f;
    if (x < 0.5f * g) { l = a * (logf(x) - logf(g)) - delta; }
    else { l = a * tp_log1p(delta / g) - delta; }
    return expf(l) * a / (sqrtf(6.283185307179586f * g) * tp_lanczos_series(a));
}

// log of a positive float, subnormals included: GPUs flush subnormal operands to zero, so one is rebuilt from its bits
// (mantissa * 2^-149).
__device__ __forceinline__ float tp_log(float x)
{
    unsigned int bits = __float_as_uint(x);
    if ((bits & 0x7F800000u) == 0u && (bits & 0x007FFFFFu) != 0u && (bits & 0x80000000u) == 0u)
    {
        return logf((float)(bits & 0x007FFFFFu)) - 103.27892990343185f;
    }
    return logf(x);
}

// True for +0 and -0 only, never a subnormal a flush-to-zero comparison would call zero.
__device__ __forceinline__ bool tp_is_zero(float x) { return (__float_as_uint(x) & 0x7FFFFFFFu) == 0u; }

// The rounding error of p = a * b, exactly (Dekker's TwoProduct over Veltkamp splits). Not fma: WGSL only requires
// fma to round like a * b + c, and Dawn's D3D12 path returns 0 for fma(a, b, -a * b). Magnitudes past 1e30 would
// overflow the split, and their products are past float's useful range anyway, so their error is taken as 0.
__device__ __forceinline__ float tp_split_hi(float a)
{
    float c = 4097.0f * a;
    return c - (c - a);
}

__device__ __forceinline__ float tp_two_prod_err(float a, float b, float p)
{
    if (fabsf(a) > 1e30f || fabsf(b) > 1e30f) { return 0.0f; }
    float ah = tp_split_hi(a);
    float al = a - ah;
    float bh = tp_split_hi(b);
    float bl = b - bh;
    return ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}
// ---- double-float arithmetic: a value hi + lo held as float2 (x = hi, y = lo), exact sums and products via fma ----
__device__ __forceinline__ float2 tp_df(float a) { return make_float2(a, 0.0f); }

__device__ __forceinline__ float2 tp_df_add(float2 a, float2 b)
{
    float s = a.x + b.x;
    // Past overflow the error term is inf - inf: carry the plain float result, as the CPU's double overflows to it.
    if (tp_isinf(s) || tp_isnan(s)) { return make_float2(s, 0.0f); }
    float bb = s - a.x;
    float e = (a.x - (s - bb)) + (b.x - bb) + a.y + b.y;
    float hi = s + e;
    return make_float2(hi, e - (hi - s));
}

__device__ __forceinline__ float2 tp_df_neg(float2 a) { return make_float2(-a.x, -a.y); }

__device__ __forceinline__ float2 tp_df_mul(float2 a, float2 b)
{
    float p = a.x * b.x;
    if (tp_isinf(p) || tp_isnan(p)) { return make_float2(p, 0.0f); }
    float e = tp_two_prod_err(a.x, b.x, p) + (a.x * b.y + a.y * b.x);
    float hi = p + e;
    return make_float2(hi, e - (hi - p));
}

__device__ __forceinline__ float2 tp_df_div(float2 a, float2 b)
{
    float q = a.x / b.x;
    if (tp_isinf(q) || tp_isnan(q) || tp_isinf(a.x)) { return make_float2(q, 0.0f); }
    float2 r = tp_df_add(a, tp_df_neg(tp_df_mul(make_float2(q, 0.0f), b)));
    float q2 = r.x / b.x;
    float hi = q + q2;
    return make_float2(hi, q2 - (hi - q));
}";

    private const string SpecialFunctions = @"
// erf(x) = 2/sqrt(pi) sum (-1)^n x^(2n+1) / (n! (2n+1)), |x| < 0.5.
__device__ __forceinline__ float tp_erf_series(float x)
{
    float x2 = x * x;
    float term = x;
    float sum = x;
    for (int k = 1; k < 60; k++)
    {
        term = term * (-x2 / (float)k);
        float add = term / (float)(2 * k + 1);
        sum = sum + add;
        if (fabsf(add) < 1e-9f * fabsf(sum)) { break; }
    }
    return 1.1283791670955126f * sum;
}

// An int as an exact double-float: each half fits float's 24-bit significand, so (float)v cannot merge neighbours past 2^24.
__device__ __forceinline__ float2 tp_df_int(int v)
{
    return tp_df_add(tp_df((float)(v >> 12) * 4096.0f), tp_df((float)(v & 4095)));
}

// sqrt(pi) exp(x^2) erfc(x) by the modified-Lentz continued fraction, x >= 2.5.
__device__ __forceinline__ float tp_erfc_continued_fraction(float x)
{
    float f = x;
    float c = x;
    float d = 0.0f;
    for (int k = 1; k < 400; k++)
    {
        float a = (float)k / 2.0f;
        d = x + a * d;
        if (fabsf(d) < 1e-30f) { d = 1e-30f; }
        c = x + a / c;
        if (fabsf(c) < 1e-30f) { c = 1e-30f; }
        d = 1.0f / d;
        float delta = c * d;
        f = f * delta;
        if (fabsf(delta - 1.0f) < 1e-8f) { break; }
    }
    return 1.0f / f;
}

__device__ __forceinline__ float tp_erfc_large(float x) { return expf(-x * x) * tp_erfc_continued_fraction(x) / 1.7724538509055160f; }

// erfc on [0.5, 2.5): erfc(2.5) + (2 / sqrt(pi)) * integral_x^2.5 e^{-t^2} dt (32-node Gauss-Legendre).
__device__ __forceinline__ float tp_erfc_mid(float x)
{
    float mid = 0.5f * (x + 2.5f);
    float halfWidth = 0.5f * (2.5f - x);
    float sum = 0.0f;
    for (int k = 0; k < 32; k++)
    {
        float t = mid + halfWidth * TP_GL32_X[k];
        sum = sum + TP_GL32_W[k] * expf(-t * t);
    }
    return tp_erfc_large(2.5f) + 1.1283791670955126f * sum * halfWidth;
}

__device__ __forceinline__ float tp_erfc(float x)
{
    if (tp_isnan(x)) { return x; }
    if (tp_isinf(x)) { return tp_pick(x > 0.0f, 0.0f, 2.0f); }
    if (x >= 2.5f) { return tp_erfc_large(x); }
    if (x >= 0.5f) { return tp_erfc_mid(x); }
    if (x <= -0.5f)
    {
        float ax = -x;
        if (ax >= 2.5f) { return 2.0f - tp_erfc_large(ax); }
        return 2.0f - tp_erfc_mid(ax);
    }
    return 1.0f - tp_erf_series(x);
}

__device__ __forceinline__ float tp_erf_f(float x)
{
    if (tp_isnan(x)) { return x; }
    if (tp_isinf(x)) { return tp_pick(x > 0.0f, 1.0f, -1.0f); }
    if (fabsf(x) < 0.5f) { return tp_erf_series(x); }
    float tail = tp_erfc(fabsf(x));
    if (x > 0.0f) { return 1.0f - tail; }
    return tail - 1.0f;
}

__device__ __forceinline__ float tp_erfcx_f(float x)
{
    if (tp_isnan(x)) { return x; }
    if (tp_isinf(x) && x > 0.0f) { return 0.0f; }
    if (x >= 2.5f) { return tp_erfc_continued_fraction(x) / 1.7724538509055160f; }
    if (x >= 0.5f) { return expf(x * x) * tp_erfc_mid(x); }
    if (x > -0.5f) { return expf(x * x) * (1.0f - tp_erf_series(x)); }
    if (x < -9.4f) { return tp_inf(); }
    float ax = -x;
    float positive = 0.0f;
    if (ax >= 2.5f) { positive = tp_erfc_continued_fraction(ax) / 1.7724538509055160f; }
    else { positive = expf(ax * ax) * tp_erfc_mid(ax); }
    return 2.0f * expf(x * x) - positive;
}

__device__ __forceinline__ float tp_ndtr_f(float x) { return 0.5f * tp_erfc(-x / 1.4142135623730951f); }

__device__ __forceinline__ float tp_log_ndtr_f(float x)
{
    if (tp_isnan(x)) { return x; }
    if (tp_isinf(x) && x < 0.0f) { return tp_ninf(); }
    if (x > 0.0f) { return tp_log1p(-tp_ndtr_f(-x)); }
    if (x > -1.0f) { return logf(tp_ndtr_f(x)); }
    return logf(0.5f * tp_erfcx_f(-x / 1.4142135623730951f)) - 0.5f * x * x;
}

// Acklam's rational approximation refined by two Halley steps against ndtr.
__device__ __forceinline__ float tp_ndtri_f(float p)
{
    if (tp_isnan(p) || p < 0.0f || p > 1.0f) { return tp_nan(); }
    if (tp_is_zero(p)) { return tp_ninf(); }
    if (p == 1.0f) { return tp_inf(); }
    float x = 0.0f;
    if (p < 0.02425f)
    {
        float q = sqrtf(-2.0f * tp_log(p));
        x = (((((-7.784894002430293e-03f * q - 3.223964580411365e-01f) * q - 2.400758277161838f) * q - 2.549732539343734f) * q
            + 4.374664141464968f) * q + 2.938163982698783f)
            / ((((7.784695709041462e-03f * q + 3.224671290700398e-01f) * q + 2.445134137142996f) * q + 3.754408661907416f) * q + 1.0f);
    }
    else if (p > 0.97575f)
    {
        float q = sqrtf(-2.0f * logf(1.0f - p));
        x = -(((((-7.784894002430293e-03f * q - 3.223964580411365e-01f) * q - 2.400758277161838f) * q - 2.549732539343734f) * q
            + 4.374664141464968f) * q + 2.938163982698783f)
            / ((((7.784695709041462e-03f * q + 3.224671290700398e-01f) * q + 2.445134137142996f) * q + 3.754408661907416f) * q + 1.0f);
    }
    else
    {
        float q = p - 0.5f;
        float r = q * q;
        x = (((((-3.969683028665376e+01f * r + 2.209460984245205e+02f) * r - 2.759285104469687e+02f) * r + 1.383577518672690e+02f) * r
            - 3.066479806614716e+01f) * r + 2.506628277459239f) * q
            / (((((-5.447609879822406e+01f * r + 1.615858368580409e+02f) * r - 1.556989798598866e+02f) * r + 6.680131188771972e+01f) * r
            - 1.328068155288572e+01f) * r + 1.0f);
    }
    for (int k = 0; k < 2; k++)
    {
        float e = tp_ndtr_f(x) - p;
        float u = e * 2.5066282746310002f * expf(x * x / 2.0f);
        x = x - u / (1.0f + x * u / 2.0f);
    }
    return x;
}

// ---- Bessel functions (SpecialFunctions.BesselJ / BesselY / BesselIScaled / BesselKScaled) ----

__device__ __forceinline__ float tp_bessel_series(int order, float x, bool alternating)
{
    float halfX = x / 2.0f;
    float term = 1.0f;
    for (int k = 1; k <= order; k++) { term = term * halfX / (float)k; }
    float sum = term;
    float q = halfX * halfX;
    if (alternating) { q = -q; }
    for (int k = 1; k < 40; k++)
    {
        term = term * q / ((float)k * (float)(k + order));
        sum = sum + term;
        if (fabsf(term) <= 1e-9f * fabsf(sum)) { break; }
    }
    return sum;
}

// The large-x series of e^x K (alternating false) or sqrt(2 pi x) e^-x I (alternating true), summed to its smallest term.
__device__ __forceinline__ float tp_modified_bessel_asymptotic(float mu, float x, bool alternating)
{
    float term = 1.0f;
    float sum = 1.0f;
    float last = 3.0e38f;
    for (int k = 1; k < 100; k++)
    {
        float odd = 2.0f * (float)k - 1.0f;
        term = term * (mu - odd * odd) / (8.0f * (float)k * x);
        if (alternating) { term = -term; }
        if (fabsf(term) > last) { break; }
        last = fabsf(term);
        sum = sum + term;
        if (fabsf(term) < 1e-9f * fabsf(sum)) { break; }
    }
    return sum;
}

// Hankel's asymptotic expansion: J (y false) or Y (y true) for x > 25.
__device__ __forceinline__ float tp_hankel_asymptotic(int order, float x, bool y)
{
    float mu = 4.0f * (float)(order * order);
    float p = 1.0f;
    float q = 0.0f;
    float term = 1.0f;
    float lastP = 3.0e38f;
    for (int k = 1; k < 40; k++)
    {
        float odd = (float)(2 * k - 1);
        term = term * (mu - odd * odd) / ((float)k * 8.0f * x);
        if (fabsf(term) > lastP) { break; }
        lastP = fabsf(term);
        float sgn = 1.0f;
        if ((k / 2) % 2 != 0) { sgn = -1.0f; }
        if (k % 2 == 1) { q = q + sgn * term; }
        else { p = p + sgn * term; }
        if (fabsf(term) < 1e-9f) { break; }
    }
    float phi = ((float)order / 2.0f + 0.25f) * 3.14159265358979f;
    float s = sqrtf(2.0f / (3.14159265358979f * x));
    float cx = cosf(x);
    float sx = sinf(x);
    float cp = cosf(phi);
    float sp = sinf(phi);
    float cosChi = cx * cp + sx * sp;
    float sinChi = sx * cp - cx * sp;
    if (y) { return s * (p * sinChi + q * cosChi); }
    return s * (p * cosChi - q * sinChi);
}

__device__ __forceinline__ float tp_bessel_j(int order, float x)
{
    if (tp_isnan(x)) { return x; }
    float ax = fabsf(x);
    if (tp_isinf(ax)) { return 0.0f; }
    if (ax < 2.0f) { return tp_bessel_series(order, x, true); }
    if (ax > 25.0f)
    {
        float h = tp_hankel_asymptotic(order, ax, false);
        if (order % 2 == 1 && x < 0.0f) { return -h; }
        return h;
    }
    int points = 2 * (int)ceilf(ax) + 48;
    float sum = 0.0f;
    for (int k = 0; k < points; k++)
    {
        float tau = 6.283185307179586f * (float)k / (float)points;
        sum = sum + cosf((float)order * tau - x * sinf(tau));
    }
    return sum / (float)points;
}

__device__ __forceinline__ float tp_bessel_y(int order, float x)
{
    if (tp_isnan(x) || x < 0.0f) { return tp_nan(); }
    if (x == 0.0f) { return tp_ninf(); }
    if (tp_isinf(x)) { return 0.0f; }
    if (x > 25.0f) { return tp_hankel_asymptotic(order, x, true); }
    float first = 0.0f;
    for (int k = 0; k < 96; k++)
    {
        float tau = 1.5707963267948966f + 1.5707963267948966f * TP_GL96_X[k];
        first = first + TP_GL96_W[k] * sinf(x * sinf(tau) - (float)order * tau);
    }
    first = first * 1.5707963267948966f;
    float sgn = 1.0f;
    if (order % 2 != 0) { sgn = -1.0f; }
    float second = 0.0f;
    for (int panel = 0; panel < 2000; panel++)
    {
        float pf = (float)panel;
        if (x * 0.5f * (expf(pf) - expf(-pf)) - (float)order * pf > 104.0f) { break; }
        float part = 0.0f;
        for (int k = 0; k < 32; k++)
        {
            float t = pf + 0.5f + 0.5f * TP_GL32_X[k];
            float sinhT = 0.5f * (expf(t) - expf(-t));
            part = part + TP_GL32_W[k] * (expf((float)order * t) + sgn * expf(-(float)order * t)) * expf(-x * sinhT);
        }
        second = second + 0.5f * part;
    }
    return (first - second) / 3.14159265358979f;
}

__device__ __forceinline__ float tp_bessel_i_scaled(int order, float x)
{
    if (tp_isnan(x)) { return x; }
    float ax = fabsf(x);
    float sgn = 1.0f;
    if (order % 2 == 1 && x < 0.0f) { sgn = -1.0f; }
    if (ax < 2.0f) { return sgn * expf(-ax) * tp_bessel_series(order, ax, false); }
    if (ax > 50.0f)
    {
        return sgn * tp_modified_bessel_asymptotic(4.0f * (float)(order * order), ax, true) / sqrtf(6.283185307179586f * ax);
    }
    int points = 40 + (int)ceilf(sqrtf(80.0f * ax));
    float sum = 0.0f;
    for (int k = 0; k < points; k++)
    {
        float tau = 6.283185307179586f * (float)k / (float)points;
        sum = sum + expf(ax * (cosf(tau) - 1.0f)) * cosf((float)order * tau);
    }
    return sgn * sum / (float)points;
}

__device__ __forceinline__ float tp_bessel_i(int order, float x)
{
    if (tp_isinf(x))
    {
        if (order % 2 == 1 && x < 0.0f) { return tp_ninf(); }
        return tp_inf();
    }
    return tp_bessel_i_scaled(order, x) * expf(fabsf(x));
}

// e^x K_nu(x): the asymptotic series above x = 50, else the h = 0.02 trapezoid of e^{-x (cosh t - 1)} cosh(nu t).
__device__ __forceinline__ float tp_bessel_k_scaled(float nu, float x)
{
    if (tp_isnan(x) || x < 0.0f) { return tp_nan(); }
    if (x == 0.0f) { return tp_inf(); }
    if (x > 50.0f) { return sqrtf(3.14159265358979f / (2.0f * x)) * tp_modified_bessel_asymptotic(4.0f * nu * nu, x, false); }
    float sum = 0.5f;
    for (int k = 1; k < 100000; k++)
    {
        float t = (float)k * 0.02f;
        float coshM1 = 0.5f * (expf(t) + expf(-t)) - 1.0f;
        if (x * coshM1 - fabsf(nu) * t > 104.0f) { break; }
        sum = sum + expf(-x * coshM1) * 0.5f * (expf(nu * t) + expf(-nu * t));
    }
    return sum * 0.02f;
}

__device__ __forceinline__ float tp_bessel_k(float nu, float x)
{
    if (tp_isinf(x) && x > 0.0f) { return 0.0f; }
    return tp_bessel_k_scaled(nu, x) * expf(-x);
}

// J_nu(x) for non-integer nu (Schlafli's integral) over the CPU's Gauss-Legendre nodes.
__device__ __forceinline__ float tp_bessel_j_fractional(float nu, float x)
{
    float first = 0.0f;
    for (int k = 0; k < 96; k++)
    {
        float tau = 1.5707963267948966f + 1.5707963267948966f * TP_GL96_X[k];
        first = first + TP_GL96_W[k] * cosf(nu * tau - x * sinf(tau));
    }
    first = first * 1.5707963267948966f;
    float second = 0.0f;
    for (int panel = 0; panel < 2000; panel++)
    {
        float pf = (float)panel;
        if (x * 0.5f * (expf(pf) - expf(-pf)) + nu * pf > 104.0f) { break; }
        float part = 0.0f;
        for (int k = 0; k < 32; k++)
        {
            float t = pf + 0.5f + 0.5f * TP_GL32_X[k];
            part = part + TP_GL32_W[k] * expf(-x * 0.5f * (expf(t) - expf(-t)) - nu * t);
        }
        second = second + 0.5f * part;
    }
    return (first - sinf(nu * 3.14159265358979f) * second) / 3.14159265358979f;
}

__device__ __forceinline__ float tp_airy_ai_f(float x)
{
    if (tp_isnan(x)) { return x; }
    if (tp_isinf(x)) { return 0.0f; }
    if (x >= -5.0f && x <= 1.0f)
    {
        float x3 = x * x * x;
        float f = 1.0f;
        float g = x;
        float tf = 1.0f;
        float tg = x;
        for (int k = 1; k < 100; k++)
        {
            float k3 = 3.0f * (float)k;
            tf = tf * x3 / ((k3 - 1.0f) * k3);
            tg = tg * x3 / (k3 * (k3 + 1.0f));
            f = f + tf;
            g = g + tg;
            if (fabsf(tf) + fabsf(tg) < 1e-9f * (fabsf(f) + fabsf(g))) { break; }
        }
        return 0.355028053887817239f * f - 0.258819403792806798f * g;
    }
    float z = fabsf(x);
    float zeta = 2.0f / 3.0f * z * sqrtf(z);
    if (x > 0.0f) { return sqrtf(x / 3.0f) / 3.14159265358979f * tp_bessel_k(1.0f / 3.0f, zeta); }
    if (z <= 15.0f)
    {
        return sqrtf(z) / 3.0f * (tp_bessel_j_fractional(1.0f / 3.0f, zeta) + tp_bessel_j_fractional(-1.0f / 3.0f, zeta));
    }
    float u = 1.0f;
    float even = 1.0f;
    float odd = 0.0f;
    float power = 1.0f;
    float last = 3.0e38f;
    for (int k = 1; k < 40; k++)
    {
        float kf = (float)k;
        u = u * (6.0f * kf - 5.0f) * (6.0f * kf - 3.0f) * (6.0f * kf - 1.0f) / ((2.0f * kf - 1.0f) * 216.0f * kf);
        power = power / zeta;
        float term = u * power;
        if (term > last) { break; }
        last = term;
        float s = term;
        if ((k / 2) % 2 != 0) { s = -term; }
        if (k % 2 == 0) { even = even + s; }
        else { odd = odd + s; }
        if (term < 1e-9f) { break; }
    }
    // The phase zeta + pi / 4 must be reduced mod 2 pi to well under float's spacing of zeta (z = 800 gives zeta ~ 15085),
    // so zeta = (2 / 3) z sqrt(z) is formed in double-float (sqrt's residual from fma) and reduced against 2 pi held as
    // hi + lo. Past float range (z beyond ~5e25) the amplitude z^-1/4 / sqrt(pi) is below 1e-6 and the phase meaningless.
    float s = sqrtf(z);
    float ss = s * s;
    float sLo = ((z - ss) - tp_two_prod_err(s, s, ss)) / (2.0f * s);
    float2 zetaDf = tp_df_mul(tp_df_mul(tp_df(z), make_float2(s, sLo)), make_float2(0.6666666865348816f, -1.9868215e-8f));
    if (tp_isinf(zetaDf.x)) { return 0.0f; }
    float turns = rintf(zetaDf.x / 6.283185307179586f);
    float2 phase = tp_df_add(zetaDf, tp_df_neg(tp_df_mul(tp_df(turns), make_float2(6.2831854820251465f, -1.7484555e-7f))));
    phase = tp_df_add(phase, make_float2(0.7853981852531433f, -2.1855695e-8f));
    float sinPhase = sinf(phase.x) + cosf(phase.x) * phase.y;
    float cosPhase = cosf(phase.x) - sinf(phase.x) * phase.y;
    return (sinPhase * even - cosPhase * odd) / (1.7724538509055160f * sqrtf(sqrtf(z)));}

__device__ __forceinline__ float tp_igamma_series(float a, float x)
{
    float ap = a;
    float sum = 1.0f / a;
    float inc = sum;
    for (int k = 0; k < 1000; k++)
    {
        ap = ap + 1.0f;
        inc = inc * x / ap;
        sum = sum + inc;
        if (fabsf(inc) < fabsf(sum) * 1e-8f) { break; }
    }
    return sum * tp_gamma_prefix(a, x);
}

__device__ __forceinline__ float tp_igamma_continued_fraction(float a, float x)
{
    float b = x + 1.0f - a;
    float c = 1.0f / 1e-30f;
    float d = 1.0f / b;
    float h = d;
    for (int k = 1; k < 1000; k++)
    {
        float kf = (float)k;
        float an = -kf * (kf - a);
        b = b + 2.0f;
        d = an * d + b;
        if (fabsf(d) < 1e-30f) { d = 1e-30f; }
        c = b + an / c;
        if (fabsf(c) < 1e-30f) { c = 1e-30f; }
        d = 1.0f / d;
        float ratio = d * c;
        h = h * ratio;
        if (fabsf(ratio - 1.0f) < 1e-8f) { break; }
    }
    return tp_gamma_prefix(a, x) * h;
}
";

    // Unary kernels: y[t] = op(x[t]; p0, p1). Each body assigns y[t] from v (and p0 / p1).
    private static readonly (string Name, string Body)[] Unary =
    {
        ("tp_celu", "if (v > 0.0f) { y[t] = v; } else { y[t] = p0 * (expf(v / p0) - 1.0f); }"),
        ("tp_hardtanh", "y[t] = tp_clamp_nan(v, p0, p1);"),
        ("tp_logsigmoid", "if (tp_isnan(v)) { y[t] = v; } else { y[t] = tp_pick(v < 0.0f, v, 0.0f) - tp_log1p(expf(-fabsf(v))); }"),
        ("tp_softsign", "y[t] = v / (1.0f + fabsf(v));"),
        ("tp_logit", "float c = tp_clamp_nan(v, p0, p1); if (tp_isnan(c)) { y[t] = c; } else if ((__float_as_uint(c) & 0x80000000u) == 0u && !tp_is_zero(c) && c < 1.0f) { y[t] = tp_log(c) - tp_log1p(-c); } else { y[t] = logf(c / (1.0f - c)); }"),
        ("tp_sinc", "if (v == 0.0f) { y[t] = 1.0f; } else if (tp_isinf(v)) { y[t] = 0.0f; } else { y[t] = tp_sinpi(v) / (3.14159265358979f * v); }"),
        ("tp_erf", "y[t] = tp_erf_f(v);"),
        ("tp_bessel_j0", "y[t] = tp_bessel_j(0, v);"),
        ("tp_bessel_j1", "y[t] = tp_bessel_j(1, v);"),
        ("tp_bessel_y0", "y[t] = tp_bessel_y(0, v);"),
        ("tp_bessel_y1", "y[t] = tp_bessel_y(1, v);"),
        ("tp_bessel_i0", "y[t] = tp_bessel_i(0, v);"),
        ("tp_bessel_i1", "y[t] = tp_bessel_i(1, v);"),
        ("tp_erfcx", "y[t] = tp_erfcx_f(v);"),
        ("tp_ndtr", "y[t] = tp_ndtr_f(v);"),
        ("tp_ndtri", "y[t] = tp_ndtri_f(v);"),
        ("tp_entr", "if (tp_isnan(v)) { y[t] = v; } else if (v > 0.0f) { y[t] = -v * logf(v); } else if (v == 0.0f) { y[t] = 0.0f; } else { y[t] = tp_ninf(); }"),
        ("tp_signbit", "y[t] = tp_pick(tp_signbit_f(v), 1.0f, 0.0f);"),
        ("tp_isposinf", "y[t] = tp_pick(tp_isinf(v) && v > 0.0f, 1.0f, 0.0f);"),
        ("tp_isneginf", "y[t] = tp_pick(tp_isinf(v) && v < 0.0f, 1.0f, 0.0f);"),
        ("tp_log_ndtr", "y[t] = tp_log_ndtr_f(v);"),
        ("tp_bessel_k0", "y[t] = tp_bessel_k(0.0f, v);"),
        ("tp_bessel_k1", "y[t] = tp_bessel_k(1.0f, v);"),
        ("tp_bessel_k0e", "y[t] = tp_bessel_k_scaled(0.0f, v);"),
        ("tp_bessel_k1e", "y[t] = tp_bessel_k_scaled(1.0f, v);"),
        ("tp_spherical_bessel_j0", "if (tp_isnan(v)) { y[t] = v; } else if (tp_isinf(v)) { y[t] = 0.0f; } else if (fabsf(v) < 0.5f) { float w = v * v; y[t] = 1.0f + w * (-0.16666666666666666f + w * (8.333333333333333e-3f + w * (-1.984126984126984e-4f + w * 2.7557319223985893e-6f))); } else { y[t] = sinf(v) / v; }"),
        ("tp_airy_ai", "y[t] = tp_airy_ai_f(v);"),
        ("tp_angle", "if (tp_isnan(v)) { y[t] = v; } else if (v < 0.0f) { y[t] = 3.14159265358979f; } else { y[t] = 0.0f; }"),
        ("tp_nonzero", "y[t] = tp_pick(v != 0.0f, 1.0f, 0.0f);"),
        // mvlgamma (p = p0 >= 1): NaN at or below (p - 1) / 2, else p (p - 1) / 4 log(pi) + sum_j lgamma(x - j / 2).
        ("tp_mvlgamma", "int q = (int)p0; if (v <= (float)(q - 1) / 2.0f) { y[t] = tp_nan(); return; } float s = (float)q * ((float)q - 1.0f) / 4.0f * 1.1447298858494002f; for (int j = 0; j < q; j++) { s = s + tp_lgamma(v - (float)j / 2.0f); } y[t] = s;"),
    };

    private static string Elementwise
    {
        get
        {
            var sb = new StringBuilder();
            foreach (var (name, body) in Unary)
            {
                sb.Append("extern \"C\" __global__ __launch_bounds__(256) void ").Append(name)
                  .Append("(const float* x, float* y, int n, float p0, float p1)\n{\n    int t = bx_gid();\n    if (t >= n) { return; }\n    float v = x[t];\n    ")
                  .Append(body).Append("\n}\n\n");
            }
            return sb.Append(BinaryAndPolynomial).Append(Generators).ToString();
        }
    }

    private const string BinaryAndPolynomial = @"
// fmax / fmin: the NaN-ignoring max / min of CUDA's fmaxf / fminf.
extern ""C"" __global__ __launch_bounds__(256) void tp_fmax(const float* a, const float* b, float* y, int n)
{
    int t = bx_gid();
    if (t >= n) { return; }
    float u = a[t];
    float w = b[t];
    if (tp_isnan(u)) { y[t] = w; } else if (tp_isnan(w)) { y[t] = u; } else { y[t] = tp_pick(u > w, u, w); }
}

extern ""C"" __global__ __launch_bounds__(256) void tp_fmin(const float* a, const float* b, float* y, int n)
{
    int t = bx_gid();
    if (t >= n) { return; }
    float u = a[t];
    float w = b[t];
    if (tp_isnan(u)) { y[t] = w; } else if (tp_isnan(w)) { y[t] = u; } else { y[t] = tp_pick(u < w, u, w); }
}

extern ""C"" __global__ __launch_bounds__(256) void tp_heaviside(const float* a, const float* b, float* y, int n)
{
    int t = bx_gid();
    if (t >= n) { return; }
    float u = a[t];
    if (tp_isnan(u)) { y[t] = u; } else if (u < 0.0f) { y[t] = 0.0f; } else if (u > 0.0f) { y[t] = 1.0f; } else { y[t] = b[t]; }
}

extern ""C"" __global__ __launch_bounds__(256) void tp_greater_equal(const float* a, const float* b, float* y, int n)
{
    int t = bx_gid();
    if (t >= n) { return; }
    y[t] = tp_pick(a[t] >= b[t], 1.0f, 0.0f);
}

extern ""C"" __global__ __launch_bounds__(256) void tp_less_equal(const float* a, const float* b, float* y, int n)
{
    int t = bx_gid();
    if (t >= n) { return; }
    y[t] = tp_pick(a[t] <= b[t], 1.0f, 0.0f);
}

extern ""C"" __global__ __launch_bounds__(256) void tp_igamma(const float* a, const float* b, float* y, int n)
{
    int t = bx_gid();
    if (t >= n) { return; }
    float s = a[t];
    float v = b[t];
    if (tp_isnan(s) || tp_isnan(v) || s <= 0.0f || v < 0.0f) { y[t] = tp_nan(); }
    else if (v == 0.0f) { y[t] = 0.0f; }
    else if (tp_isinf(v)) { y[t] = 1.0f; }
    else if (v < s + 1.0f) { y[t] = tp_igamma_series(s, v); }
    else { y[t] = 1.0f - tp_igamma_continued_fraction(s, v); }
}

extern ""C"" __global__ __launch_bounds__(256) void tp_igammac(const float* a, const float* b, float* y, int n)
{
    int t = bx_gid();
    if (t >= n) { return; }
    float s = a[t];
    float v = b[t];
    if (tp_isnan(s) || tp_isnan(v) || s <= 0.0f || v < 0.0f) { y[t] = tp_nan(); }
    else if (v == 0.0f) { y[t] = 1.0f; }
    else if (tp_isinf(v)) { y[t] = 0.0f; }
    else if (v < s + 1.0f) { y[t] = 1.0f - tp_igamma_series(s, v); }
    else { y[t] = tp_igamma_continued_fraction(s, v); }
}

// floor(a / b) as the CPU's Math.Floor((double)a / b): the float quotient can round across an integer (0.3f / 0.1f is
// 3.0f, the double quotient 2.99999997), so its floor is corrected by the exact remainder fma(-q, b, a).
extern ""C"" __global__ __launch_bounds__(256) void tp_floor_divide(const float* a, const float* b, float* y, int n)
{
    int t = bx_gid();
    if (t >= n) { return; }
    float u = a[t];
    float w = b[t];
    float q = floorf(u / w);
    if (tp_isnan(q) || tp_isinf(q) || w == 0.0f) { y[t] = q; return; }
    float qw = q * w;
    float r = (u - qw) - tp_two_prod_err(q, w, qw);
    if ((w > 0.0f && r < 0.0f) || (w < 0.0f && r > 0.0f)) { q = q - 1.0f; }
    else if ((w > 0.0f && r >= w) || (w < 0.0f && r <= w)) { q = q + 1.0f; }
    y[t] = q;
}

// Orthogonal polynomials by the CPU's recurrence. kind: 0 ChebyshevT, 1 U, 2 V, 3 W, 4 HermiteH, 5 HermiteHe,
// 6 LaguerreL, 7 LegendreP; shifted maps x to 2x - 1. The host keeps infinite / > 2^20 degrees on the CPU.
extern ""C"" __global__ __launch_bounds__(256) void tp_polynomial(const float* x, const float* degree, float* y, int n, int kind, int shifted)
{
    int t = bx_gid();
    if (t >= n) { return; }
    float d = degree[t];
    if (tp_isnan(d)) { y[t] = tp_nan(); return; }
    float tr = truncf(d);
    if (tr < 0.0f) { y[t] = 0.0f; return; }
    if (tr > 1048576.0f) { y[t] = tp_nan(); return; }
    int deg = (int)tr;
    float v = x[t];
    if (tp_isnan(v)) { y[t] = tp_nan(); return; }
    // The recurrence runs in double-float: in plain float its rounding grows with the degree past the 2e-6 the CPU's
    // double recurrence is held to.
    float2 vv = tp_df(v);
    if (shifted != 0) { vv = tp_df_add(tp_df(2.0f * v), tp_df(-1.0f)); }
    if (deg == 0) { y[t] = 1.0f; return; }
    float2 twoV = make_float2(2.0f * vv.x, 2.0f * vv.y);
    float2 previous = tp_df(1.0f);
    float2 current = vv;
    if (kind == 1 || kind == 4) { current = twoV; }
    else if (kind == 2) { current = tp_df_add(twoV, tp_df(-1.0f)); }
    else if (kind == 3) { current = tp_df_add(twoV, tp_df(1.0f)); }
    else if (kind == 6) { current = tp_df_add(tp_df(1.0f), tp_df_neg(vv)); }
    for (int k = 1; k < deg; k++)
    {
        float kk = (float)k;
        float2 next = tp_df_add(tp_df_mul(twoV, current), tp_df_neg(previous));
        if (kind == 4) { next = tp_df_add(tp_df_mul(twoV, current), tp_df_neg(tp_df_mul(tp_df(2.0f * kk), previous))); }
        else if (kind == 5) { next = tp_df_add(tp_df_mul(vv, current), tp_df_neg(tp_df_mul(tp_df(kk), previous))); }
        else if (kind == 6)
        {
            float2 a = tp_df_mul(tp_df_add(tp_df(2.0f * kk + 1.0f), tp_df_neg(vv)), current);
            next = tp_df_div(tp_df_add(a, tp_df_neg(tp_df_mul(tp_df(kk), previous))), tp_df(kk + 1.0f));
        }
        else if (kind == 7)
        {
            float2 a = tp_df_mul(tp_df_mul(tp_df(2.0f * kk + 1.0f), vv), current);
            next = tp_df_div(tp_df_add(a, tp_df_neg(tp_df_mul(tp_df(kk), previous))), tp_df(kk + 1.0f));
        }
        previous = current;
        current = next;
    }
    y[t] = current.x + current.y;
}
";

    // Generators. A double the host holds travels as a float pair hi + lo (PortableTorchParity.Split), so arange and
    // logspace keep the CPU's double positions; everything else rounds once to float, as the CPU's FromDoubles does.
    private const string Generators = @"
extern ""C"" __global__ __launch_bounds__(256) void tp_range(float* y, int n, float startHi, float startLo, float spanHi, float spanLo, float denHi, float denLo)
{
    int t = bx_gid();
    if (t >= n) { return; }
    float2 v = tp_df_add(make_float2(startHi, startLo), tp_df_div(tp_df_mul(tp_df_int(t), make_float2(spanHi, spanLo)), make_float2(denHi, denLo)));
    y[t] = v.x + v.y;
}

// e^r for |r| <= ln2 / 2 by its Taylor series (float-exact to the last bit at this range).
__device__ __forceinline__ float tp_exp_small(float r)
{
    float term = 1.0f;
    float sum = 1.0f;
    for (int k = 1; k < 12; k++)
    {
        term = term * r / (float)k;
        sum = sum + term;
    }
    return sum;
}

// base^e = 2^n e^r with e ln(base) = n ln2 + r reduced in double-float, so the result keeps float's last bit where
// the GPU's pow (exp2 of e log2 base, each to ~1e-6) does not. lnBase arrives as hi + lo from the host's double.
extern ""C"" __global__ __launch_bounds__(256) void tp_logspace(float* y, int n, float startHi, float startLo, float spanHi, float spanLo, float denHi, float denLo, float lnBaseHi, float lnBaseLo, int single)
{
    int t = bx_gid();
    if (t >= n) { return; }
    float2 e = make_float2(startHi, startLo);
    if (single == 0) { e = tp_df_add(e, tp_df_div(tp_df_mul(tp_df_int(t), make_float2(spanHi, spanLo)), make_float2(denHi, denLo))); }
    float2 l = tp_df_mul(e, make_float2(lnBaseHi, lnBaseLo));
    if (l.x > 89.0f) { y[t] = tp_inf(); return; }
    if (l.x < -104.0f) { y[t] = 0.0f; return; }
    float turns = rintf(l.x / 0.6931471805599453f);
    float2 r = tp_df_add(l, tp_df_neg(tp_df_mul(tp_df(turns), make_float2(0.6931471824645996f, -1.904654323148236e-9f))));
    y[t] = ldexpf(tp_exp_small(r.x + r.y), (int)turns);
}

// Windows over t = i / (m - 1): 0 Hann, 1 Hamming(alpha, beta), 2 Blackman, 3 Bartlett, 4 Kaiser(beta = alpha).
extern ""C"" __global__ __launch_bounds__(256) void tp_window(float* y, int n, int kind, int m, float alpha, float beta)
{
    int t = bx_gid();
    if (t >= n) { return; }
    if (n == 1) { y[t] = 1.0f; return; }
    float u = (float)t / (float)(m - 1);
    float v = 0.0f;
    if (kind == 0) { v = 0.5f - 0.5f * cosf(6.283185307179586f * u); }
    else if (kind == 1) { v = alpha - beta * cosf(6.283185307179586f * u); }
    else if (kind == 2) { v = 0.42f - 0.5f * cosf(6.283185307179586f * u) + 0.08f * cosf(12.566370614359172f * u); }
    else if (kind == 3) { v = 1.0f - fabsf(2.0f * u - 1.0f); }
    else
    {
        float r = 2.0f * u - 1.0f;
        // I0(a s) / I0(a) = e^(a (s - 1)) I0e(a s) / I0e(a), with a (s - 1) = -a r^2 / (1 + s): no cancellation, where
        // e^(a s - a) from two large exponents would multiply float's error by a.
        float s = sqrtf(fmaxf(0.0f, 1.0f - r * r));
        float a = fabsf(alpha);
        v = expf(-a * r * r / (1.0f + s)) * tp_bessel_i_scaled(0, a * s) / tp_bessel_i_scaled(0, a);
    }
    y[t] = v;
}

// The first and last column a tril / triu row keeps.
__device__ __forceinline__ int tp_tri_lo(int r, int col, int offset, int lower)
{
    int edge = r + offset;
    if (lower != 0) { return 0; }
    if (edge > 0) { return edge; }
    return 0;
}

__device__ __forceinline__ int tp_tri_hi(int r, int col, int offset, int lower)
{
    int edge = r + offset;
    if (lower != 0)
    {
        if (edge < col - 1) { return edge; }
        return col - 1;
    }
    return col - 1;
}

// tril / triu indices as a [2, total] tensor, one thread per row.
extern ""C"" __global__ __launch_bounds__(256) void tp_tri_indices(float* y, int row, int col, int offset, int lower, int total)
{
    int t = bx_gid();
    if (t >= row) { return; }
    int at = 0;
    for (int r = 0; r < t; r++)
    {
        int lo = tp_tri_lo(r, col, offset, lower);
        int hi = tp_tri_hi(r, col, offset, lower);
        if (hi >= lo) { at = at + hi - lo + 1; }
    }
    int lo = tp_tri_lo(t, col, offset, lower);
    int hi = tp_tri_hi(t, col, offset, lower);
    for (int j = lo; j <= hi; j++)
    {
        y[at] = (float)t;
        y[total + at] = (float)j;
        at = at + 1;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void tp_diagflat(const float* x, float* y, int total, int size, int offset, int n)
{
    int t = bx_gid();
    if (t >= total) { return; }
    int r = t / size;
    int c = t % size;
    int k = -1;
    if (offset >= 0) { if (c - r == offset) { k = r; } }
    else { if (r - c == -offset) { k = c; } }
    if (k >= 0 && k < n) { y[t] = x[k]; } else { y[t] = 0.0f; }
}

// diagonal_scatter: g[0] other rank, g[1..8] other extents, g[9..16] other strides, g[17..21] stride1, stride2,
// start1, start2, diagonal.
extern ""C"" __global__ __launch_bounds__(256) void tp_diagonal_scatter(const float* src, float* y, const int* g, int count)
{
    int t = bx_gid();
    if (t >= count) { return; }
    int diagonal = g[21];
    int o = t / diagonal;
    int d = t % diagonal;
    int base = 0;
    int rest = o;
    for (int j = g[0] - 1; j >= 0; j--)
    {
        base = base + (rest % g[1 + j]) * g[9 + j];
        rest = rest / g[1 + j];
    }
    y[base + (d + g[19]) * g[17] + (d + g[20]) * g[18]] = src[t];
}

// quantile over sorted rows (ascending, NaN last). mode: 0 linear, 1 lower, 2 higher, 3 nearest (ties to even),
// 4 midpoint. The position q (count - 1) is formed in double-float from q's hi + lo, so its floor, ceiling and ties
// match the CPU's double.
extern ""C"" __global__ __launch_bounds__(256) void tp_quantile_rows(const float* sorted, float* y, int rows, int length, float qHi, float qLo, int mode, int ignoreNan)
{
    int r = bx_gid();
    if (r >= rows) { return; }
    int baseRow = r * length;
    int lo = 0;
    int hi = length;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (tp_isnan(sorted[baseRow + mid])) { hi = mid; } else { lo = mid + 1; }
    }
    int count = lo;
    if (count == 0 || (count < length && ignoreNan == 0)) { y[r] = tp_nan(); return; }
    float2 position = tp_df_mul(make_float2(qHi, qLo), tp_df_int(count - 1));
    float below = floorf(position.x);
    if (position.x == below && position.y < 0.0f) { below = below - 1.0f; }
    float above = ceilf(position.x);
    if (position.x == above && position.y > 0.0f) { above = above + 1.0f; }
    float w = (position.x - below) + position.y;
    if (mode == 1) { above = below; w = 0.0f; }
    else if (mode == 2) { below = above; w = 0.0f; }
    else if (mode == 3)
    {
        float nearest = below;
        if (w > 0.5f) { nearest = below + 1.0f; }
        else if (w == 0.5f && below - 2.0f * floorf(below / 2.0f) != 0.0f) { nearest = below + 1.0f; }
        below = nearest;
        above = nearest;
        w = 0.0f;
    }
    else if (mode == 4)
    {
        if (below == above) { w = 0.0f; } else { w = 0.5f; }
    }
    y[r] = (1.0f - w) * sorted[baseRow + (int)below] + w * sorted[baseRow + (int)above];
}
";

    // #1125 / #1128: the random samplers, integer ops, scan and index kernels. Draws come from the CUDA kernels' Philox4x32-10
    // counters (element, 0, draw, 0x5EED) under the seed's key, as a float with 48 random bits; a sampler's log(1 - u)
    // is log1p(-u), so a draw near 0 keeps its tail. Invalid parameters lower a 32-bit rank (element << 3 | code, bit 31
    // for multinomial's exhaustion code) with atomicMin, and tp_flag_merge folds it into the engine's 64-bit flag.
    private const string Sampling = @"
__device__ const float TP_STIRLING[10] = { 0.08106146679532726f, 0.04134069595540929f, 0.02767792568499834f, 0.02079067210376509f, 0.01664469118982119f, 0.01387612882307075f, 0.01189670994589177f, 0.01041126526197209f, 0.009255462182712733f, 0.008330563433362871f };

__device__ __forceinline__ unsigned int tp_mul_hi(unsigned int a, unsigned int b)
{
    unsigned int al = a & 0xFFFFu;
    unsigned int ah = a >> 16u;
    unsigned int bl = b & 0xFFFFu;
    unsigned int bh = b >> 16u;
    unsigned int lh = al * bh;
    unsigned int hl = ah * bl;
    unsigned int mid = ((al * bl) >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    return ah * bh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}

// Words 0 and 1 of Philox4x32-10 on the counter (element, 0, draw, 0x5EED) under the key (key0, key1).
__device__ __forceinline__ uint2 tp_philox(unsigned int key0, unsigned int key1, unsigned int element, unsigned int draw)
{
    unsigned int c0 = element;
    unsigned int c1 = 0u;
    unsigned int c2 = draw;
    unsigned int c3 = 0x5EEDu;
    unsigned int k0 = key0;
    unsigned int k1 = key1;
    for (int r = 0; r < 10; r++)
    {
        unsigned int lo0 = 0xD2511F53u * c0;
        unsigned int hi0 = tp_mul_hi(0xD2511F53u, c0);
        unsigned int lo1 = 0xCD9E8D57u * c2;
        unsigned int hi1 = tp_mul_hi(0xCD9E8D57u, c2);
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
        k0 = k0 + 0x9E3779B9u;
        k1 = k1 + 0xBB67AE85u;
    }
    return make_uint2(c0, c1);
}

// A uniform float in [0, 1): 24 bits from word 0, refined by 24 more from word 1 while the sum stays below 1.
__device__ __forceinline__ float tp_uniform(unsigned int key0, unsigned int key1, unsigned int element, unsigned int draw)
{
    uint2 bits = tp_philox(key0, key1, element, draw);
    float hi = (float)(bits.x >> 8u) * 5.9604644775390625e-8f;
    float s = hi + (float)(bits.y >> 8u) * 3.552713678800501e-15f;
    if (s < 1.0f) { return s; }
    return hi;
}

// A standard normal from draws draw and draw + 1 (Box-Muller, as the CUDA kernels).
__device__ __forceinline__ float tp_normal(unsigned int key0, unsigned int key1, unsigned int element, unsigned int draw)
{
    float u1 = tp_uniform(key0, key1, element, draw);
    float u2 = tp_uniform(key0, key1, element, draw + 1u);
    return sqrtf(-2.0f * tp_log1p(-u1)) * cosf(6.283185307179586f * u2);
}

__device__ __forceinline__ unsigned int tp_flag_key(int element, int code) { return ((unsigned int)element << 3u) | (unsigned int)code; }

extern ""C"" __global__ __launch_bounds__(256) void tp_flag_reset(unsigned int* flag, int n)
{
    int t = bx_gid();
    if (t >= n) { return; }
    flag[t] = 0xFFFFFFFFu;
}

// Lowers the 64-bit flag (low word, high word) to the rank's (element << 3 | code); rank bit 31 is the 2^58 element
// bit multinomial ranks its exhaustion code by.
extern ""C"" __global__ __launch_bounds__(256) void tp_flag_merge(const unsigned int* rank, unsigned int* flag, int n)
{
    if (bx_gid() >= n) { return; }
    unsigned int key = rank[0];
    if (key == 0xFFFFFFFFu) { return; }
    unsigned int lo = key & 0x7FFFFFFFu;
    unsigned int hi = 0u;
    if ((key & 0x80000000u) != 0u) { hi = 0x20000000u; }
    if (hi < flag[1] || (hi == flag[1] && lo < flag[0]))
    {
        flag[0] = lo;
        flag[1] = hi;
    }
}

// 0 uniform(a, b), 1 cauchy(median a, sigma b), 2 exponential(rate a), 3 geometric(p a), 4 log-normal(a, b),
// 5 standard normal, 6 randint(low a, span b).
extern ""C"" __global__ __launch_bounds__(256) void tp_sample(float* y, int n, int kind, float a, float b, unsigned int seedLo, unsigned int seedHi)
{
    int i = bx_gid();
    if (i >= n) { return; }
    unsigned int e = (unsigned int)i;
    float v = 0.0f;
    if (kind == 0) { v = a + (b - a) * tp_uniform(seedLo, seedHi, e, 0u); }
    else if (kind == 1) { v = a + b * tanf(3.14159265358979f * (tp_uniform(seedLo, seedHi, e, 0u) - 0.5f)); }
    else if (kind == 2) { v = -tp_log1p(-tp_uniform(seedLo, seedHi, e, 0u)) / a; }
    else if (kind == 3)
    {
        if (a == 1.0f) { v = 1.0f; }
        else { v = fmaxf(1.0f, ceilf(tp_log1p(-tp_uniform(seedLo, seedHi, e, 0u)) / tp_log1p(-a))); }
    }
    else if (kind == 4) { v = expf(a + b * tp_normal(seedLo, seedHi, e, 0u)); }
    else if (kind == 5) { v = tp_normal(seedLo, seedHi, e, 0u); }
    else { v = a + fminf(floorf(tp_uniform(seedLo, seedHi, e, 0u) * b), b - 1.0f); }
    y[i] = v;
}

extern ""C"" __global__ __launch_bounds__(256) void tp_bernoulli(const float* prob, float* y, unsigned int* rank, int n, unsigned int seedLo, unsigned int seedHi)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float q = prob[i];
    if (tp_isnan(q) || q < 0.0f || q > 1.0f) { atomicMin(&rank[0], tp_flag_key(i, 1)); return; }
    if (tp_uniform(seedLo, seedHi, (unsigned int)i, 0u) < q) { y[i] = 1.0f; } else { y[i] = 0.0f; }
}

// Poisson: Knuth's product method below rate 10, Hormann's PTRS above; code 1 for a negative or NaN rate.
extern ""C"" __global__ __launch_bounds__(256) void tp_poisson(const float* rates, float* y, unsigned int* rank, int n, unsigned int seedLo, unsigned int seedHi)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float rate = rates[i];
    if (tp_isnan(rate) || rate < 0.0f) { atomicMin(&rank[0], tp_flag_key(i, 1)); return; }
    if (rate == 0.0f) { y[i] = 0.0f; return; }
    unsigned int e = (unsigned int)i;
    unsigned int draw = 0u;
    if (rate < 10.0f)
    {
        float limit = expf(-rate);
        float product = tp_uniform(seedLo, seedHi, e, draw);
        draw = draw + 1u;
        int count = 0;
        while (product > limit)
        {
            count = count + 1;
            product = product * tp_uniform(seedLo, seedHi, e, draw);
            draw = draw + 1u;
        }
        y[i] = (float)count;
        return;
    }
    float slam = sqrtf(rate);
    float logLam = logf(rate);
    float bb = 0.931f + 2.53f * slam;
    float aa = -0.059f + 0.02483f * bb;
    float invAlpha = 1.1239f + 1.1328f / (bb - 3.4f);
    float vr = 0.9277f - 3.6224f / (bb - 2.0f);
    while (true)
    {
        float u = tp_uniform(seedLo, seedHi, e, draw) - 0.5f;
        float v = tp_uniform(seedLo, seedHi, e, draw + 1u);
        draw = draw + 2u;
        float us = 0.5f - fabsf(u);
        float k = floorf((2.0f * aa / us + bb) * u + rate + 0.43f);
        if (us >= 0.07f && v <= vr) { y[i] = k; return; }
        if (k < 0.0f || (us < 0.013f && v > us)) { continue; }
        if (logf(v) + logf(invAlpha) - logf(aa / (us * us) + bb) <= -rate + k * logLam - tp_lgamma(k + 1.0f)) { y[i] = k; return; }
    }
}

__device__ __forceinline__ float tp_stirling_tail(float k)
{
    if (k <= 9.0f) { return TP_STIRLING[(int)k]; }
    float kp1sq = (k + 1.0f) * (k + 1.0f);
    return (1.0f / 12.0f - (1.0f / 360.0f - 1.0f / 1260.0f / kp1sq) / kp1sq) / (k + 1.0f);
}

// Binomial: geometric waiting times while n min(p, 1-p) < 10, Hormann's BTRS above. Code 1: a count that is not a
// whole number within the long range; code 2: count < 0, or p outside [0, 1] / NaN.
extern ""C"" __global__ __launch_bounds__(256) void tp_binomial(const float* counts, const float* probs, float* y, unsigned int* rank, int n, unsigned int seedLo, unsigned int seedHi)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float count = counts[i];
    float q0 = probs[i];
    if (tp_isnan(count) || tp_isinf(count) || count != floorf(count) || count >= 9.2233720368547758e18f) { atomicMin(&rank[0], tp_flag_key(i, 1)); return; }
    if (count < 0.0f || tp_isnan(q0) || q0 < 0.0f || q0 > 1.0f) { atomicMin(&rank[0], tp_flag_key(i, 2)); return; }
    if (count == 0.0f || q0 == 0.0f) { y[i] = 0.0f; return; }
    if (q0 == 1.0f) { y[i] = count; return; }
    unsigned int e = (unsigned int)i;
    unsigned int draw = 0u;
    bool flip = q0 > 0.5f;
    float q = q0;
    if (flip) { q = 1.0f - q0; }
    float drawn = 0.0f;
    if (count * q >= 10.0f)
    {
        float stddev = sqrtf(count * q * (1.0f - q));
        float bb = 1.15f + 2.53f * stddev;
        float aa = -0.0873f + 0.0248f * bb + 0.01f * q;
        float c = count * q + 0.5f;
        float vR = 0.92f - 4.2f / bb;
        float ratio = q / (1.0f - q);
        float alpha = (2.83f + 5.1f / bb) * stddev;
        float m = floorf((count + 1.0f) * q);
        while (true)
        {
            float u = tp_uniform(seedLo, seedHi, e, draw) - 0.5f;
            float v = tp_uniform(seedLo, seedHi, e, draw + 1u);
            draw = draw + 2u;
            float us = 0.5f - fabsf(u);
            float k = floorf((2.0f * aa / us + bb) * u + c);
            if (k < 0.0f || k > count) { continue; }
            if (us >= 0.07f && v <= vR) { drawn = k; break; }
            float lv = logf(v * alpha / (aa / (us * us) + bb));
            float upper = (m + 0.5f) * logf((m + 1.0f) / (ratio * (count - m + 1.0f)))
                + (count + 1.0f) * logf((count - m + 1.0f) / (count - k + 1.0f))
                + (k + 0.5f) * logf(ratio * (count - k + 1.0f) / (k + 1.0f))
                + tp_stirling_tail(m) + tp_stirling_tail(count - m) - tp_stirling_tail(k) - tp_stirling_tail(count - k);
            if (lv <= upper) { drawn = k; break; }
        }
    }
    else
    {
        float logQ = tp_log1p(-q);
        float successes = 0.0f;
        float position = 0.0f;
        while (true)
        {
            float stride = floorf(tp_log1p(-tp_uniform(seedLo, seedHi, e, draw)) / logQ) + 1.0f;
            draw = draw + 1u;
            if (!(stride <= count - position)) { break; }
            position = position + stride;
            successes = successes + 1.0f;
        }
        drawn = successes;
    }
    if (flip) { y[i] = count - drawn; } else { y[i] = drawn; }
}

// Multinomial: one thread per row walks numSamples draws (removing each pick without replacement), as the CPU, with
// the running prefix (with replacement) or the remaining weights (without) in the row's scratch.
extern ""C"" __global__ __launch_bounds__(256) void tp_multinomial(const float* weights, float* scratch, float* y, unsigned int* rank, int rows, int categories, int numSamples, int replacement, unsigned int seedLo, unsigned int seedHi)
{
    int row = bx_gid();
    if (row >= rows) { return; }
    int w0 = row * categories;
    for (int c = 0; c < categories; c++)
    {
        float v = weights[w0 + c];
        if (tp_isnan(v) || v < 0.0f || tp_isinf(v)) { atomicMin(&rank[0], tp_flag_key(w0 + c, 1)); return; }
    }
    unsigned int e = (unsigned int)row;
    unsigned int draw = 0u;
    if (replacement != 0)
    {
        float total = 0.0f;
        int last = -1;
        for (int c = 0; c < categories; c++)
        {
            total = total + weights[w0 + c];
            scratch[w0 + c] = total;
            if (weights[w0 + c] > 0.0f) { last = c; }
        }
        if (!(total > 0.0f)) { atomicMin(&rank[0], 0x80000000u | tp_flag_key(w0, 2)); return; }
        for (int k = 0; k < numSamples; k++)
        {
            float u = tp_uniform(seedLo, seedHi, e, draw) * total;
            draw = draw + 1u;
            int lo = 0;
            int hi = categories;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (u < scratch[w0 + mid]) { hi = mid; } else { lo = mid + 1; }
            }
            if (lo < categories) { y[row * numSamples + k] = (float)lo; } else { y[row * numSamples + k] = (float)last; }
        }
        return;
    }
    for (int c = 0; c < categories; c++) { scratch[w0 + c] = weights[w0 + c]; }
    for (int k = 0; k < numSamples; k++)
    {
        float total = 0.0f;
        for (int c = 0; c < categories; c++) { total = total + scratch[w0 + c]; }
        if (!(total > 0.0f)) { atomicMin(&rank[0], 0x80000000u | tp_flag_key(w0, 2)); return; }
        float u = tp_uniform(seedLo, seedHi, e, draw) * total;
        draw = draw + 1u;
        float running = 0.0f;
        int chosen = -1;
        for (int c = categories - 1; c >= 0; c--) { if (scratch[w0 + c] > 0.0f) { chosen = c; break; } }
        for (int c = 0; c < categories; c++)
        {
            running = running + scratch[w0 + c];
            if (u < running && scratch[w0 + c] > 0.0f) { chosen = c; break; }
        }
        y[row * numSamples + k] = (float)chosen;
        scratch[w0 + chosen] = 0.0f;
    }
}

// Alpha dropout's keep mask, one draw per block: scale a or 0, offset b (kept) or gone = a sat + b (dropped).
extern ""C"" __global__ __launch_bounds__(256) void tp_dropout_affine(float* scale, float* offset, int n, int block, float dropP, float a, float gone, float b, unsigned int seedLo, unsigned int seedHi)
{
    int i = bx_gid();
    if (i >= n) { return; }
    if (tp_uniform(seedLo, seedHi, (unsigned int)(i / block), 0u) >= dropP) { scale[i] = a; offset[i] = b; }
    else { scale[i] = 0.0f; offset[i] = gone; }
}

// Channel dropout writes the scale only (its offset aliases the scale, which WebGPU cannot bind twice).
extern ""C"" __global__ __launch_bounds__(256) void tp_dropout_scale(float* scale, int n, int block, float dropP, float a, unsigned int seedLo, unsigned int seedHi)
{
    int i = bx_gid();
    if (i >= n) { return; }
    if (tp_uniform(seedLo, seedHi, (unsigned int)(i / block), 0u) >= dropP) { scale[i] = a; } else { scale[i] = 0.0f; }
}

extern ""C"" __global__ __launch_bounds__(256) void tp_rrelu(const float* x, float* y, int n, float lower, float upper, float middle, int training, unsigned int seedLo, unsigned int seedHi)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float v = x[i];
    float slope = 1.0f;
    if (!(v > 0.0f))
    {
        if (training == 0) { slope = middle; }
        else { slope = lower + (upper - lower) * tp_uniform(seedLo, seedHi, (unsigned int)i, 0u); }
    }
    y[i] = slope * v;
}

extern ""C"" __global__ __launch_bounds__(256) void tp_random_keys(float* y, int n, unsigned int seedLo, unsigned int seedHi)
{
    int i = bx_gid();
    if (i >= n) { return; }
    y[i] = tp_uniform(seedLo, seedHi, (unsigned int)i, 0u);
}
";
    private const string Indexing = @"
// ---- integer ops on raw int32 / int64 storage, as the CPU's IntegerBinary ----

__device__ __forceinline__ unsigned int tp_abs32(int x)
{
    if (x < 0) { return 0u - (unsigned int)x; }
    return (unsigned int)x;
}

// Binary gcd of two magnitudes.
__device__ __forceinline__ unsigned int tp_gcd32(unsigned int a0, unsigned int b0)
{
    unsigned int a = a0;
    unsigned int b = b0;
    if (a == 0u) { return b; }
    if (b == 0u) { return a; }
    unsigned int shift = 0u;
    while (((a | b) & 1u) == 0u) { a = a >> 1u; b = b >> 1u; shift = shift + 1u; }
    while ((a & 1u) == 0u) { a = a >> 1u; }
    while (b != 0u)
    {
        while ((b & 1u) == 0u) { b = b >> 1u; }
        if (a > b) { unsigned int s = a; a = b; b = s; }
        b = b - a;
    }
    return a << shift;
}

// op: 0 and, 1 or, 2 xor, 3 left shift, 4 right shift, 5 gcd, 6 lcm, 7 floor divide (code 1 for a zero divisor), on the
// sign-extended operands in 64 bits and wrapped to int32, which the low words alone determine.
extern ""C"" __global__ __launch_bounds__(256) void tp_int_binary(const int* a, const int* b, int* y, unsigned int* rank, int n, int op)
{
    int i = bx_gid();
    if (i >= n) { return; }
    int x = a[i];
    int v = b[i];
    int r = 0;
    if (op == 0) { r = x & v; }
    else if (op == 1) { r = x | v; }
    else if (op == 2) { r = x ^ v; }
    else if (op == 3)
    {
        int s = v & 63;
        if (v < 64 && s < 32) { r = (int)((unsigned int)x << (unsigned int)s); }
    }
    else if (op == 4)
    {
        int s = v & 63;
        if (v >= 64 || s >= 31) { r = x >> 31u; } else { r = x >> (unsigned int)s; }
    }
    else if (op == 5) { r = (int)tp_gcd32(tp_abs32(x), tp_abs32(v)); }
    else if (op == 6)
    {
        unsigned int g = tp_gcd32(tp_abs32(x), tp_abs32(v));
        if (g != 0u)
        {
            unsigned int q = tp_abs32(x) / g;
            unsigned int m = q * (unsigned int)v;
            if (x < 0) { m = (0u - q) * (unsigned int)v; }
            if (q != 0u && v != 0 && ((x < 0) != (v < 0))) { m = 0u - m; }
            r = (int)m;
        }
    }
    else
    {
        if (v == 0) { atomicMin(&rank[0], tp_flag_key(i, 1)); return; }
        unsigned int ax = tp_abs32(x);
        unsigned int av = tp_abs32(v);
        unsigned int q = ax / av;
        if ((x < 0) == (v < 0)) { r = (int)q; }
        else
        {
            unsigned int up = 0u;
            if (ax % av != 0u) { up = 1u; }
            r = (int)(0u - (q + up));
        }
    }
    y[i] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void tp_int_not(const int* a, int* y, int n)
{
    int i = bx_gid();
    if (i >= n) { return; }
    y[i] = ~a[i];
}

// 64-bit integers as (low word, high word).
__device__ __forceinline__ bool tp_i64_negative(uint2 a) { return (a.y & 0x80000000u) != 0u; }
__device__ __forceinline__ bool tp_u64_is_zero(uint2 a) { return a.x == 0u && a.y == 0u; }
__device__ __forceinline__ bool tp_u64_less(uint2 a, uint2 b) { return a.y < b.y || (a.y == b.y && a.x < b.x); }

__device__ __forceinline__ uint2 tp_u64_neg(uint2 a)
{
    unsigned int lo = (~a.x) + 1u;
    unsigned int hi = ~a.y;
    if (lo == 0u) { hi = hi + 1u; }
    return make_uint2(lo, hi);
}

__device__ __forceinline__ uint2 tp_u64_magnitude(uint2 a)
{
    if (tp_i64_negative(a)) { return tp_u64_neg(a); }
    return a;
}

__device__ __forceinline__ uint2 tp_u64_add(uint2 a, uint2 b)
{
    unsigned int lo = a.x + b.x;
    unsigned int hi = a.y + b.y;
    if (lo < a.x) { hi = hi + 1u; }
    return make_uint2(lo, hi);
}

__device__ __forceinline__ uint2 tp_u64_sub(uint2 a, uint2 b)
{
    unsigned int lo = a.x - b.x;
    unsigned int hi = a.y - b.y;
    if (a.x < b.x) { hi = hi - 1u; }
    return make_uint2(lo, hi);
}

// The low 64 bits of a * b.
__device__ __forceinline__ uint2 tp_u64_mul(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, tp_mul_hi(a.x, b.x) + a.x * b.y + a.y * b.x);
}

// Shifts by s in [0, 63].
__device__ __forceinline__ uint2 tp_u64_shl(uint2 a, unsigned int s)
{
    if (s == 0u) { return a; }
    if (s >= 32u) { return make_uint2(0u, a.x << (s - 32u)); }
    return make_uint2(a.x << s, (a.y << s) | (a.x >> (32u - s)));
}

__device__ __forceinline__ uint2 tp_u64_shr(uint2 a, unsigned int s)
{
    if (s == 0u) { return a; }
    if (s >= 32u) { return make_uint2(a.y >> (s - 32u), 0u); }
    return make_uint2((a.x >> s) | (a.y << (32u - s)), a.y >> s);
}

__device__ __forceinline__ uint2 tp_i64_sar(uint2 a, unsigned int s)
{
    if (tp_i64_negative(a))
    {
        uint2 r = tp_u64_shr(make_uint2(~a.x, ~a.y), s);
        return make_uint2(~r.x, ~r.y);
    }
    return tp_u64_shr(a, s);
}

// a / b for b != 0 and a, b at most 2^63 (restoring division).
__device__ __forceinline__ uint2 tp_u64_div(uint2 a, uint2 b)
{
    if (a.y == 0u && b.y == 0u) { return make_uint2(a.x / b.x, 0u); }
    uint2 q = make_uint2(0u, 0u);
    uint2 r = make_uint2(0u, 0u);
    for (int k = 63; k >= 0; k--)
    {
        r = tp_u64_shl(r, 1u);
        if ((tp_u64_shr(a, (unsigned int)k).x & 1u) != 0u) { r.x = r.x | 1u; }
        if (!tp_u64_less(r, b))
        {
            r = tp_u64_sub(r, b);
            if (k >= 32) { q.y = q.y | (1u << (unsigned int)(k - 32)); } else { q.x = q.x | (1u << (unsigned int)k); }
        }
    }
    return q;
}

__device__ __forceinline__ uint2 tp_gcd64(uint2 a0, uint2 b0)
{
    uint2 a = a0;
    uint2 b = b0;
    if (tp_u64_is_zero(a)) { return b; }
    if (tp_u64_is_zero(b)) { return a; }
    unsigned int shift = 0u;
    while (((a.x | b.x) & 1u) == 0u) { a = tp_u64_shr(a, 1u); b = tp_u64_shr(b, 1u); shift = shift + 1u; }
    while ((a.x & 1u) == 0u) { a = tp_u64_shr(a, 1u); }
    while (!tp_u64_is_zero(b))
    {
        while ((b.x & 1u) == 0u) { b = tp_u64_shr(b, 1u); }
        if (tp_u64_less(b, a)) { uint2 s = a; a = b; b = s; }
        b = tp_u64_sub(b, a);
    }
    return tp_u64_shl(a, shift);
}

// Narrows to the element type t: 0 long, 1 ulong, 2 short, 3 ushort, 4 sbyte, 5 byte, 6 uint.
__device__ __forceinline__ uint2 tp_i64_narrow(uint2 r, int t)
{
    if (t == 2)
    {
        int v = ((int)(r.x << 16u)) >> 16u;
        return make_uint2((unsigned int)v, (unsigned int)(v >> 31u));
    }
    if (t == 3) { return make_uint2(r.x & 0xFFFFu, 0u); }
    if (t == 4)
    {
        int v = ((int)(r.x << 24u)) >> 24u;
        return make_uint2((unsigned int)v, (unsigned int)(v >> 31u));
    }
    if (t == 5) { return make_uint2(r.x & 0xFFu, 0u); }
    if (t == 6) { return make_uint2(r.x, 0u); }
    return r;
}

// #1135's int64 ops: signed 64-bit math narrowed to the element type. Codes: 1 a zero divisor, 2 the CPU's
// OverflowException (Math.Abs of long.MinValue in gcd / lcm, long.MinValue / -1).
extern ""C"" __global__ __launch_bounds__(256) void tp_int64_binary(const unsigned int* a, const unsigned int* b, unsigned int* y, unsigned int* rank, int n, int op, int typeCode)
{
    int i = bx_gid();
    if (i >= n) { return; }
    uint2 x = make_uint2(a[2 * i], a[2 * i + 1]);
    uint2 v = make_uint2(b[2 * i], b[2 * i + 1]);
    bool xMin = x.x == 0u && x.y == 0x80000000u;
    bool vMin = v.x == 0u && v.y == 0x80000000u;
    uint2 r = make_uint2(0u, 0u);
    if (op == 0) { r = make_uint2(x.x & v.x, x.y & v.y); }
    else if (op == 1) { r = make_uint2(x.x | v.x, x.y | v.y); }
    else if (op == 2) { r = make_uint2(x.x ^ v.x, x.y ^ v.y); }
    else if (op == 3 || op == 4)
    {
        bool big = !tp_i64_negative(v) && (v.y != 0u || v.x >= 64u);
        unsigned int s = v.x & 63u;
        if (op == 3) { if (!big) { r = tp_u64_shl(x, s); } }
        else if (big) { r = tp_i64_sar(x, 63u); }
        else { r = tp_i64_sar(x, s); }
    }
    else if (op == 5)
    {
        if (xMin || vMin) { atomicMin(&rank[0], tp_flag_key(i, 2)); return; }
        r = tp_gcd64(tp_u64_magnitude(x), tp_u64_magnitude(v));
    }
    else if (op == 6)
    {
        if (xMin || vMin) { atomicMin(&rank[0], tp_flag_key(i, 2)); return; }
        uint2 g = tp_gcd64(tp_u64_magnitude(x), tp_u64_magnitude(v));
        uint2 q = make_uint2(0u, 0u);
        if (!tp_u64_is_zero(g)) { q = tp_u64_div(tp_u64_magnitude(x), g); }
        if (tp_i64_negative(x)) { q = tp_u64_neg(q); }
        uint2 m = tp_u64_mul(q, v);
        if (m.x == 0u && m.y == 0x80000000u) { atomicMin(&rank[0], tp_flag_key(i, 2)); return; }
        r = tp_u64_magnitude(m);
    }
    else
    {
        if (tp_u64_is_zero(v)) { atomicMin(&rank[0], tp_flag_key(i, 1)); return; }
        if (xMin && v.x == 0xFFFFFFFFu && v.y == 0xFFFFFFFFu) { atomicMin(&rank[0], tp_flag_key(i, 2)); return; }
        uint2 ax = tp_u64_magnitude(x);
        uint2 av = tp_u64_magnitude(v);
        uint2 q = tp_u64_div(ax, av);
        if (tp_i64_negative(x) == tp_i64_negative(v)) { r = q; }
        else
        {
            uint2 rem = tp_u64_sub(ax, tp_u64_mul(q, av));
            if (!tp_u64_is_zero(rem)) { q = tp_u64_add(q, make_uint2(1u, 0u)); }
            r = tp_u64_neg(q);
        }
    }
    r = tp_i64_narrow(r, typeCode);
    y[2 * i] = r.x;
    y[2 * i + 1] = r.y;
}

extern ""C"" __global__ __launch_bounds__(256) void tp_int64_not(const unsigned int* a, unsigned int* y, int n, int typeCode)
{
    int i = bx_gid();
    if (i >= n) { return; }
    uint2 r = tp_i64_narrow(make_uint2(~a[2 * i], ~a[2 * i + 1]), typeCode);
    y[2 * i] = r.x;
    y[2 * i + 1] = r.y;
}

// One Hillis-Steele pass of the inclusive prefix sum (exact in float for counts below 2^24).
extern ""C"" __global__ __launch_bounds__(256) void tp_scan_step(const float* src, float* dst, int n, int offset)
{
    int i = bx_gid();
    if (i >= n) { return; }
    if (i >= offset) { dst[i] = src[i] + src[i - offset]; } else { dst[i] = src[i]; }
}

// ---- nonzero_static / unravel_index / unique_dim / grid_sample 3-D ----

// shape[0] = rank, shape[1..8] = extents.
extern ""C"" __global__ __launch_bounds__(256) void tp_nonzero_static_scatter(const float* x, const float* positions, float* y, const int* shape, int n, int size)
{
    int i = bx_gid();
    if (i >= n) { return; }
    if (x[i] == 0.0f) { return; }
    int row = (int)positions[i] - 1;
    if (row >= size) { return; }
    int axes = shape[0];
    int rest = i;
    for (int k = axes - 1; k >= 0; k--)
    {
        y[row * axes + k] = (float)(rest % shape[1 + k]);
        rest = rest / shape[1 + k];
    }
}

extern ""C"" __global__ __launch_bounds__(256) void tp_unravel_axis(const float* indices, float* y, unsigned int* rank, int n, int total, int stride, int extent, int validate)
{
    int i = bx_gid();
    if (i >= n) { return; }
    float t = truncf(indices[i]);
    if (tp_isnan(t) || t < -2147483648.0f || t >= 2147483648.0f || (int)t < -total || (int)t >= total)
    {
        if (validate != 0) { atomicMin(&rank[0], tp_flag_key(i, 1)); }
        return;
    }
    int f = (int)t;
    if (f < 0) { f = f + total; }
    y[i] = (float)((f / stride) % extent);
}

// unique_dim's slice order: element by element with double.CompareTo (NaN below every number, equal to NaN); -1 / 0 / 1
// for slices sa and sb, inlined in each kernel (a helper cannot read a storage buffer every module declares).
extern ""C"" __global__ __launch_bounds__(256) void tp_slice_bitonic_step(const float* data, float* order, int rowLen, int k, int j, int length)
{
    int i = bx_gid();
    if (i >= rowLen) { return; }
    int ixj = i ^ j;
    if (ixj <= i) { return; }
    float fa = order[i];
    float fb = order[ixj];
    int sa = (int)fa;
    int sb = (int)fb;
    bool bBeforeA = false;
    bool aBeforeB = false;
    if (sa < 0 || sb < 0)
    {
        bBeforeA = sa < 0 && sb >= 0;
        aBeforeB = sb < 0 && sa >= 0;
    }
    else
    {
        int c = 0;
        for (int m = 0; m < length; m++)
        {
            float va = data[sa * length + m];
            float vb = data[sb * length + m];
            bool na = tp_isnan(va);
            bool nb = tp_isnan(vb);
            if (na || nb)
            {
                if (na && nb) { continue; }
                if (na) { c = -1; } else { c = 1; }
                break;
            }
            if (va < vb) { c = -1; break; }
            if (va > vb) { c = 1; break; }
        }
        bBeforeA = c > 0 || (c == 0 && sb < sa);
        aBeforeB = c < 0 || (c == 0 && sa < sb);
    }
    bool swap = aBeforeB;
    if ((i & k) == 0) { swap = bBeforeA; }
    if (swap)
    {
        order[i] = fb;
        order[ixj] = fa;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void tp_slice_first_flags(const float* data, const float* order, float* flags, int n, int length)
{
    int i = bx_gid();
    if (i >= n) { return; }
    if (i == 0) { flags[i] = 1.0f; return; }
    int sa = (int)order[i];
    int sb = (int)order[i - 1];
    int c = 0;
    for (int m = 0; m < length; m++)
    {
        float va = data[sa * length + m];
        float vb = data[sb * length + m];
        bool na = tp_isnan(va);
        bool nb = tp_isnan(vb);
        if (na || nb)
        {
            if (na && nb) { continue; }
            c = 1;
            break;
        }
        if (va != vb) { c = 1; break; }
    }
    if (c != 0) { flags[i] = 1.0f; } else { flags[i] = 0.0f; }
}

extern ""C"" __global__ __launch_bounds__(256) void tp_slice_gather_unique(const float* data, const float* order, const float* flags, const float* positions, float* y, int total, int length)
{
    int idx = bx_gid();
    if (idx >= total) { return; }
    int s = idx / length;
    int k = idx % length;
    if (flags[s] == 0.0f) { return; }
    int row = (int)positions[s] - 1;
    y[row * length + k] = data[(int)order[s] * length + k];
}

// a mod b for a >= 0, exactly: each step subtracts the largest b 2^k <= r, which Sterbenz's lemma makes exact.
__device__ __forceinline__ float tp_fmod(float a, float b)
{
    if (tp_isnan(a) || tp_isinf(a)) { return tp_nan(); }
    float r = a;
    while (r >= b)
    {
        float t = b;
        while (t * 2.0f <= r) { t = t * 2.0f; }
        r = r - t;
    }
    return r;
}

// (hi + lo) mod b in [0, b) for a double-float with hi >= 0.
__device__ __forceinline__ float tp_df_fmod(float2 a, float b)
{
    float r = tp_fmod(a.x, b);
    if (a.y >= 0.0f) { r = r + tp_fmod(a.y, b); }
    else { r = r - tp_fmod(-a.y, b); }
    if (r < 0.0f) { r = r + b; }
    if (r >= b) { r = r - b; }
    return r;
}

// The source coordinate normalized * scale + shift, as a double-float: the CPU forms it in double, and a coordinate
// far outside the volume must keep its exact position for reflection to land where the CPU's does.
__device__ __forceinline__ float2 tp_grid_unnormalize(float normalized, int size, int alignCorners)
{
    float scale = (float)size / 2.0f;
    if (alignCorners != 0) { scale = (float)(size - 1) / 2.0f; }
    float prod = normalized * scale;
    float2 shift = tp_df_int(size - 1);
    return tp_df_add(make_float2(prod, tp_two_prod_err(normalized, scale, prod)), make_float2(shift.x * 0.5f, shift.y * 0.5f));
}

// Reflects a coordinate into [twiceLow / 2, twiceHigh / 2] (the CPU's reflection: fold by the span, mirror odd folds).
__device__ __forceinline__ float tp_grid_reflect(float2 coord0, int twiceLow, int twiceHigh)
{
    if (twiceLow == twiceHigh) { return 0.0f; }
    float mn = (float)twiceLow / 2.0f;
    float span = (float)(twiceHigh - twiceLow) / 2.0f;
    float2 coord = tp_df_add(coord0, tp_df(-mn));
    if (coord.x < 0.0f) { coord = tp_df_neg(coord); }
    float m = tp_df_fmod(coord, 2.0f * span);
    if (m < span) { return m + mn; }
    return span - (m - span) + mn;
}

// The factor reflection applies to the coordinate's slope: 0 for a degenerate span, else +-1.
__device__ __forceinline__ float tp_grid_reflect_sign(float2 coord0, int twiceLow, int twiceHigh)
{
    if (twiceLow == twiceHigh) { return 0.0f; }
    float mn = (float)twiceLow / 2.0f;
    float span = (float)(twiceHigh - twiceLow) / 2.0f;
    float2 coord = tp_df_add(coord0, tp_df(-mn));
    float sgn = 1.0f;
    if (coord.x < 0.0f)
    {
        sgn = -1.0f;
        coord = tp_df_neg(coord);
    }
    if (tp_df_fmod(coord, 2.0f * span) < span) { return sgn; }
    return -sgn;
}

__device__ __forceinline__ float tp_grid_source(float normalized, int size, int padding, int alignCorners)
{
    float2 c = tp_grid_unnormalize(normalized, size, alignCorners);
    float coord = c.x + c.y;
    if (padding == 2)
    {
        if (alignCorners != 0) { coord = tp_grid_reflect(c, 0, 2 * (size - 1)); }
        else { coord = tp_grid_reflect(c, -1, 2 * size - 1); }
    }
    if (padding >= 1)
    {
        if (coord <= 0.0f) { return 0.0f; }
        if (coord >= (float)(size - 1)) { return (float)(size - 1); }
    }
    return coord;
}

// d source / d normalized for tp_grid_source.
__device__ __forceinline__ float tp_grid_slope(float normalized, int size, int padding, int alignCorners)
{
    float2 c = tp_grid_unnormalize(normalized, size, alignCorners);
    float coord = c.x + c.y;
    float slope = (float)size / 2.0f;
    if (alignCorners != 0) { slope = (float)(size - 1) / 2.0f; }
    if (padding == 2)
    {
        if (alignCorners != 0)
        {
            slope = slope * tp_grid_reflect_sign(c, 0, 2 * (size - 1));
            coord = tp_grid_reflect(c, 0, 2 * (size - 1));
        }
        else
        {
            slope = slope * tp_grid_reflect_sign(c, -1, 2 * size - 1);
            coord = tp_grid_reflect(c, -1, 2 * size - 1);
        }
    }
    if (padding >= 1 && (coord <= 0.0f || coord >= (float)(size - 1))) { return 0.0f; }
    return slope;
}
// A voxel index as the CPU's VoxelIndex: NaN and out-of-int-range coordinates land far outside the volume.
__device__ __forceinline__ int tp_grid_index(float v)
{
    if (tp_isnan(v) || v < -2147483648.0f || v >= 2147483648.0f) { return -4; }
    return (int)v;
}

extern ""C"" __global__ __launch_bounds__(256) void tp_grid_sample_3d(const float* x, const float* grid, float* y, int batches, int channels, int depth, int height, int width, int points, int mode, int padding, int alignCorners)
{
    int idx = bx_gid();
    if (idx >= batches * points) { return; }
    int bi = idx / points;
    int q = idx % points;
    float cx = tp_grid_source(grid[idx * 3], width, padding, alignCorners);
    float cy = tp_grid_source(grid[idx * 3 + 1], height, padding, alignCorners);
    float cz = tp_grid_source(grid[idx * 3 + 2], depth, padding, alignCorners);
    int volume = depth * height * width;
    float fx0 = floorf(cx);
    float fy0 = floorf(cy);
    float fz0 = floorf(cz);
    int x0 = tp_grid_index(fx0);
    int y0 = tp_grid_index(fy0);
    int z0 = tp_grid_index(fz0);
    float fx = cx - fx0;
    float fy = cy - fy0;
    float fz = cz - fz0;
    for (int ch = 0; ch < channels; ch++)
    {
        int plane = (bi * channels + ch) * volume;
        float acc = 0.0f;
        if (mode == 1)
        {
            int ix = tp_grid_index(rintf(cx));
            int iy = tp_grid_index(rintf(cy));
            int iz = tp_grid_index(rintf(cz));
            if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth) { acc = x[plane + (iz * height + iy) * width + ix]; }
        }
        else
        {
            for (int oz = 0; oz < 2; oz++)
            {
                for (int oy = 0; oy < 2; oy++)
                {
                    for (int ox = 0; ox < 2; ox++)
                    {
                        int ix = x0 + ox;
                        int iy = y0 + oy;
                        int iz = z0 + oz;
                        if (ix < 0 || ix >= width || iy < 0 || iy >= height || iz < 0 || iz >= depth) { continue; }
                        float wx = 1.0f - fx;
                        if (ox == 1) { wx = fx; }
                        float wy = 1.0f - fy;
                        if (oy == 1) { wy = fy; }
                        float wz = 1.0f - fz;
                        if (oz == 1) { wz = fz; }
                        acc = acc + wx * wy * wz * x[plane + (iz * height + iy) * width + ix];
                    }
                }
            }
        }
        y[(bi * channels + ch) * points + q] = acc;
    }
}

// grid_sample 3-D adjoints, as the CPU's GridSample3DBackward: dx gathers w * dy from every point through an atomic
// float add (points share voxels), dgrid is owned by its point's thread.
extern ""C"" __global__ __launch_bounds__(256) void tp_grid_sample_3d_backward(const float* x, const float* grid, const float* dy, unsigned int* dx, float* dgrid, int batches, int channels, int depth, int height, int width, int points, int mode, int padding, int alignCorners)
{
    int idx = bx_gid();
    if (idx >= batches * points) { return; }
    int bi = idx / points;
    int q = idx % points;
    float gx = grid[idx * 3];
    float gy = grid[idx * 3 + 1];
    float gz = grid[idx * 3 + 2];
    float cx = tp_grid_source(gx, width, padding, alignCorners);
    float cy = tp_grid_source(gy, height, padding, alignCorners);
    float cz = tp_grid_source(gz, depth, padding, alignCorners);
    float sx = tp_grid_slope(gx, width, padding, alignCorners);
    float sy = tp_grid_slope(gy, height, padding, alignCorners);
    float sz = tp_grid_slope(gz, depth, padding, alignCorners);
    int volume = depth * height * width;
    float fx0 = floorf(cx);
    float fy0 = floorf(cy);
    float fz0 = floorf(cz);
    int x0 = tp_grid_index(fx0);
    int y0 = tp_grid_index(fy0);
    int z0 = tp_grid_index(fz0);
    float fx = cx - fx0;
    float fy = cy - fy0;
    float fz = cz - fz0;
    float dgx = 0.0f;
    float dgy = 0.0f;
    float dgz = 0.0f;
    for (int ch = 0; ch < channels; ch++)
    {
        int plane = (bi * channels + ch) * volume;
        float up = dy[(bi * channels + ch) * points + q];
        if (mode == 1)
        {
            int ix = tp_grid_index(rintf(cx));
            int iy = tp_grid_index(rintf(cy));
            int iz = tp_grid_index(rintf(cz));
            if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth)
            {
                int at = plane + (iz * height + iy) * width + ix;
                atomicAddFloat(&dx[at], up);
                float v = x[at] * up;
                dgx = dgx + v * 0.0f;
                dgy = dgy + v * 0.0f;
                dgz = dgz + v * 0.0f;
            }
        }
        else
        {
            for (int oz = 0; oz < 2; oz++)
            {
                for (int oy = 0; oy < 2; oy++)
                {
                    for (int ox = 0; ox < 2; ox++)
                    {
                        int ix = x0 + ox;
                        int iy = y0 + oy;
                        int iz = z0 + oz;
                        if (ix < 0 || ix >= width || iy < 0 || iy >= height || iz < 0 || iz >= depth) { continue; }
                        float wx = 1.0f - fx;
                        float sgx = -1.0f;
                        if (ox == 1) { wx = fx; sgx = 1.0f; }
                        float wy = 1.0f - fy;
                        float sgy = -1.0f;
                        if (oy == 1) { wy = fy; sgy = 1.0f; }
                        float wz = 1.0f - fz;
                        float sgz = -1.0f;
                        if (oz == 1) { wz = fz; sgz = 1.0f; }
                        int at = plane + (iz * height + iy) * width + ix;
                        atomicAddFloat(&dx[at], wx * wy * wz * up);
                        float v = x[at] * up;
                        dgx = dgx + v * (sgx * wy * wz * sx);
                        dgy = dgy + v * (sgy * wx * wz * sy);
                        dgz = dgz + v * (sgz * wx * wy * sz);
                    }
                }
            }
        }
    }
    dgrid[idx * 3] = dgx;
    dgrid[idx * 3 + 1] = dgy;
    dgrid[idx * 3 + 2] = dgz;
}
";
}