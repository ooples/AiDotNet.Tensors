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
    public static string GetSource() => Tables() + Helpers + SpecialFunctions + Elementwise;

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
__device__ __forceinline__ bool tp_signbit(float v) { return (__float_as_uint(v) & 0x80000000u) != 0u; }
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

// sin(pi x) with exact argument reduction (SpecialFunctions.SinPi).
__device__ __forceinline__ float tp_sinpi(float x)
{
    if (tp_isnan(x) || tp_isinf(x)) { return tp_nan(); }
    float r = x - 2.0f * rintf(x * 0.5f);
    float a = fabsf(r);
    if (a == 0.0f || a == 1.0f) { return 0.0f; }
    if (a > 0.5f) { a = 1.0f - a; }
    float s = sinf(3.14159265358979f * a);
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

__device__ __forceinline__ float tp_erf(float x)
{
    if (tp_isnan(x)) { return x; }
    if (tp_isinf(x)) { return tp_pick(x > 0.0f, 1.0f, -1.0f); }
    if (fabsf(x) < 0.5f) { return tp_erf_series(x); }
    float tail = tp_erfc(fabsf(x));
    if (x > 0.0f) { return 1.0f - tail; }
    return tail - 1.0f;
}

__device__ __forceinline__ float tp_erfcx(float x)
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

__device__ __forceinline__ float tp_ndtr(float x) { return 0.5f * tp_erfc(-x / 1.4142135623730951f); }

__device__ __forceinline__ float tp_log_ndtr(float x)
{
    if (tp_isnan(x)) { return x; }
    if (tp_isinf(x) && x < 0.0f) { return tp_ninf(); }
    if (x > 0.0f) { return tp_log1p(-tp_ndtr(-x)); }
    if (x > -1.0f) { return logf(tp_ndtr(x)); }
    return logf(0.5f * tp_erfcx(-x / 1.4142135623730951f)) - 0.5f * x * x;
}

// Acklam's rational approximation refined by two Halley steps against ndtr.
__device__ __forceinline__ float tp_ndtri(float p)
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
        float e = tp_ndtr(x) - p;
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

__device__ __forceinline__ float tp_airy_ai(float x)
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
        ("tp_erf", "y[t] = tp_erf(v);"),
        ("tp_bessel_j0", "y[t] = tp_bessel_j(0, v);"),
        ("tp_bessel_j1", "y[t] = tp_bessel_j(1, v);"),
        ("tp_bessel_y0", "y[t] = tp_bessel_y(0, v);"),
        ("tp_bessel_y1", "y[t] = tp_bessel_y(1, v);"),
        ("tp_bessel_i0", "y[t] = tp_bessel_i(0, v);"),
        ("tp_bessel_i1", "y[t] = tp_bessel_i(1, v);"),
        ("tp_erfcx", "y[t] = tp_erfcx(v);"),
        ("tp_ndtr", "y[t] = tp_ndtr(v);"),
        ("tp_ndtri", "y[t] = tp_ndtri(v);"),
        ("tp_entr", "if (tp_isnan(v)) { y[t] = v; } else if (v > 0.0f) { y[t] = -v * logf(v); } else if (v == 0.0f) { y[t] = 0.0f; } else { y[t] = tp_ninf(); }"),
        ("tp_signbit", "y[t] = tp_pick(tp_signbit(v), 1.0f, 0.0f);"),
        ("tp_isposinf", "y[t] = tp_pick(tp_isinf(v) && v > 0.0f, 1.0f, 0.0f);"),
        ("tp_isneginf", "y[t] = tp_pick(tp_isinf(v) && v < 0.0f, 1.0f, 0.0f);"),
        ("tp_log_ndtr", "y[t] = tp_log_ndtr(v);"),
        ("tp_bessel_k0", "y[t] = tp_bessel_k(0.0f, v);"),
        ("tp_bessel_k1", "y[t] = tp_bessel_k(1.0f, v);"),
        ("tp_bessel_k0e", "y[t] = tp_bessel_k_scaled(0.0f, v);"),
        ("tp_bessel_k1e", "y[t] = tp_bessel_k_scaled(1.0f, v);"),
        ("tp_spherical_bessel_j0", "if (tp_isnan(v)) { y[t] = v; } else if (tp_isinf(v)) { y[t] = 0.0f; } else if (fabsf(v) < 1e-4f) { y[t] = 1.0f - v * v / 6.0f; } else { y[t] = sinf(v) / v; }"),
        ("tp_airy_ai", "y[t] = tp_airy_ai(v);"),
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
            return sb.Append(BinaryAndPolynomial).ToString();
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
}
