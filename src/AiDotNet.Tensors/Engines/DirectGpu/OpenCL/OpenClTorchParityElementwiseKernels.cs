// Copyright (c) AiDotNet. All rights reserved.
// OpenCL C port of the PyTorch-parity elementwise kernels (#1126), function for function with
// CUDA/Kernels/CudaTorchParityElementwiseKernels: the same argument order, branches and NaN semantics.
//
// Every special function is evaluated in DOUBLE and rounded once to float, as the CPU reference does. OpenCL's float
// builtins are only bounded to a few ulp (erf to 16), which is too loose against a double reference, and OpenCL has no
// Bessel, erfcx or normal-quantile builtins, so those are ports of SpecialFunctions' series, asymptotic expansions and
// quadratures over the same Gauss-Legendre nodes. Requires cl_khr_fp64 and cl_khr_int64_base_atomics (the 64-bit
// validation flag); a device without them fails to build this module and the ops keep their host fallback.
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>OpenCL source for the PyTorch-parity elementwise kernels.</summary>
internal static class OpenClTorchParityElementwiseKernels
{
    /// <summary>Work-group size of the block scan (AMD and most OpenCL devices cap work groups at 256).</summary>
    internal const int ScanBlock = 256;

    public static string GetSource() => @"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define PARITY_INDEX int i = (int)get_global_id(0); if (i >= n) return;
#define PARITY_PI 3.141592653589793
#define PARITY_SQRT_PI 1.7724538509055160273
#define PARITY_SQRT2 1.4142135623730950488
#define PARITY_EXP_UNDERFLOW 745.0
#define PARITY_LARGE_ARGUMENT 50.0
#define PARITY_SCAN_BLOCK " + ScanBlockText + @"
" + GaussLegendreTables() + @"

// ---- double special functions: ports of Helpers/SpecialFunctions ----

// erf(x) = 2/sqrt(pi) sum (-1)^n x^(2n+1) / (n! (2n+1)), |x| < 2.5.
double parity_erf_series(double x)
{
    double x2 = x * x, term = x, sum = x;
    for (int k = 1; k < 200; k++)
    {
        term *= -x2 / k;
        double add = term / (2 * k + 1);
        sum += add;
        if (fabs(add) < 1e-17 * fabs(sum)) break;
    }
    return 2.0 / PARITY_SQRT_PI * sum;
}

// sqrt(pi) exp(x^2) erfc(x) by the modified-Lentz continued fraction, x >= 2.5.
double parity_erfc_continued_fraction(double x)
{
    const double tiny = 1e-300;
    double f = x, c = x, d = 0.0;
    for (int k = 1; k < 500; k++)
    {
        double a = k / 2.0;
        d = x + a * d;
        d = fabs(d) < tiny ? tiny : d;
        c = x + a / c;
        c = fabs(c) < tiny ? tiny : c;
        d = 1.0 / d;
        double delta = c * d;
        f *= delta;
        if (fabs(delta - 1.0) < 1e-16) break;
    }
    return 1.0 / f;
}

double parity_erfc_large(double x) { return exp(-x * x) * parity_erfc_continued_fraction(x) / PARITY_SQRT_PI; }

// erfc on [0.5, 2.5): erfc(2.5) + (2/sqrt(pi)) * integral_x^2.5 e^{-t^2} dt (32-node Gauss-Legendre).
double parity_erfc_mid(double x)
{
    double mid = 0.5 * (x + 2.5), halfWidth = 0.5 * (2.5 - x), sum = 0.0;
    for (int k = 0; k < 32; k++) { double t = mid + halfWidth * PARITY_GL32_X[k]; sum += PARITY_GL32_W[k] * exp(-t * t); }
    return parity_erfc_large(2.5) + 2.0 / PARITY_SQRT_PI * sum * halfWidth;
}

double parity_erfc_d(double x)
{
    if (isnan(x)) return x;
    if (isinf(x)) return x > 0.0 ? 0.0 : 2.0;
    if (x >= 2.5) return parity_erfc_large(x);
    if (x >= 0.5) return parity_erfc_mid(x);
    if (x <= -0.5)
    {
        double ax = -x;
        return 2.0 - (ax >= 2.5 ? parity_erfc_large(ax) : parity_erfc_mid(ax));
    }
    return 1.0 - parity_erf_series(x);
}

double parity_erf_d(double x)
{
    if (isnan(x)) return x;
    if (isinf(x)) return x > 0.0 ? 1.0 : -1.0;
    if (fabs(x) < 0.5) return parity_erf_series(x);
    double tail = parity_erfc_d(fabs(x));
    return x > 0.0 ? 1.0 - tail : tail - 1.0;
}

double parity_erfcx_d(double x)
{
    if (isnan(x)) return x;
    if (isinf(x) && x > 0.0) return 0.0;
    if (x >= 2.5) return parity_erfc_continued_fraction(x) / PARITY_SQRT_PI;
    if (x >= 0.5) return exp(x * x) * parity_erfc_mid(x);
    if (x > -0.5) return exp(x * x) * (1.0 - parity_erf_series(x));
    if (x < -26.7) return INFINITY;
    double ax = -x, positive = ax >= 2.5 ? parity_erfc_continued_fraction(ax) / PARITY_SQRT_PI : exp(ax * ax) * parity_erfc_mid(ax);
    return 2.0 * exp(x * x) - positive;
}

double parity_ndtr_d(double x) { return 0.5 * parity_erfc_d(-x / PARITY_SQRT2); }

double parity_log1p_d(double y) { double u = 1.0 + y; return u == 1.0 ? y : log(u) * y / (u - 1.0); }

double parity_log_ndtr_d(double x)
{
    if (isnan(x)) return x;
    if (isinf(x) && x < 0.0) return -INFINITY;
    if (x > 0.0) return parity_log1p_d(-parity_ndtr_d(-x));
    if (x > -1.0) return log(parity_ndtr_d(x));
    return log(0.5 * parity_erfcx_d(-x / PARITY_SQRT2)) - 0.5 * x * x;
}

// Acklam's rational approximation refined by two Halley steps against ndtr.
double parity_ndtri_d(double p)
{
    if (isnan(p) || p < 0.0 || p > 1.0) return NAN;
    if (p == 0.0) return -INFINITY;
    if (p == 1.0) return INFINITY;
    const double a0 = -3.969683028665376e+01, a1 = 2.209460984245205e+02, a2 = -2.759285104469687e+02,
        a3 = 1.383577518672690e+02, a4 = -3.066479806614716e+01, a5 = 2.506628277459239e+00;
    const double b0 = -5.447609879822406e+01, b1 = 1.615858368580409e+02, b2 = -1.556989798598866e+02,
        b3 = 6.680131188771972e+01, b4 = -1.328068155288572e+01;
    const double c0 = -7.784894002430293e-03, c1 = -3.223964580411365e-01, c2 = -2.400758277161838e+00,
        c3 = -2.549732539343734e+00, c4 = 4.374664141464968e+00, c5 = 2.938163982698783e+00;
    const double d0 = 7.784695709041462e-03, d1 = 3.224671290700398e-01, d2 = 2.445134137142996e+00, d3 = 3.754408661907416e+00;
    double x;
    if (p < 0.02425)
    {
        double q = sqrt(-2.0 * log(p));
        x = (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0);
    }
    else if (p > 1.0 - 0.02425)
    {
        double q = sqrt(-2.0 * log(1.0 - p));
        x = -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0);
    }
    else
    {
        double q = p - 0.5, r = q * q;
        x = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q / (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0);
    }
    for (int k = 0; k < 2; k++)
    {
        double e = parity_ndtr_d(x) - p;
        double u = e * sqrt(2.0 * PARITY_PI) * exp(x * x / 2.0);
        x -= u / (1.0 + x * u / 2.0);
    }
    return x;
}

// ---- Bessel functions of integer order (SpecialFunctions.BesselJ / BesselY / BesselIScaled) ----

double parity_bessel_series(int order, double x, int alternating)
{
    double halfX = x / 2.0, term = 1.0;
    for (int k = 1; k <= order; k++) term *= halfX / k;
    double sum = term, q = halfX * halfX * (alternating ? -1.0 : 1.0);
    for (int k = 1; k < 60; k++)
    {
        term *= q / (k * (double)(k + order));
        sum += term;
        if (fabs(term) <= 1e-17 * fabs(sum)) break;
    }
    return sum;
}

// The large-x series of e^x K (alternating = 0) or sqrt(2 pi x) e^-x I (alternating = 1), summed to its smallest term.
double parity_modified_bessel_asymptotic_signed(double mu, double x, int alternating)
{
    double term = 1.0, sum = 1.0, last = 1.7976931348623157e308;
    for (int k = 1; k < 200; k++)
    {
        term *= (mu - (2.0 * k - 1.0) * (2.0 * k - 1.0)) / (8.0 * k * x) * (alternating ? -1.0 : 1.0);
        if (fabs(term) > last) break;
        last = fabs(term);
        sum += term;
        if (fabs(term) < 1e-17 * fabs(sum)) break;
    }
    return sum;
}

// Hankel's asymptotic expansion: J (which = 0) or Y (which = 1) for x > 25.
double parity_hankel_asymptotic(int order, double x, int which)
{
    double mu = 4.0 * order * order, p = 1.0, q = 0.0, term = 1.0, lastP = 1.7976931348623157e308;
    for (int k = 1; k < 60; k++)
    {
        term *= (mu - (2 * k - 1) * (2 * k - 1)) / (k * 8.0 * x);
        if (fabs(term) > lastP) break;
        lastP = fabs(term);
        if (k % 2 == 1) q += ((k / 2) % 2 == 0 ? 1.0 : -1.0) * term;
        else p += ((k / 2) % 2 == 0 ? 1.0 : -1.0) * term;
        if (fabs(term) < 1e-17) break;
    }
    double phi = (order / 2.0 + 0.25) * PARITY_PI, s = sqrt(2.0 / (PARITY_PI * x));
    double cx = cos(x), sx = sin(x), cp = cos(phi), sp = sin(phi);
    double cosChi = cx * cp + sx * sp, sinChi = sx * cp - cx * sp;
    return which == 0 ? s * (p * cosChi - q * sinChi) : s * (p * sinChi + q * cosChi);
}

double parity_bessel_j(int order, double x)
{
    if (isnan(x)) return x;
    double ax = fabs(x);
    if (isinf(ax)) return 0.0;
    if (ax < 2.0) return parity_bessel_series(order, x, 1);
    if (ax > 25.0) return (order % 2 == 1 && x < 0.0 ? -1.0 : 1.0) * parity_hankel_asymptotic(order, ax, 0);
    int points = 2 * (int)ceil(ax) + 48;
    double sum = 0.0;
    for (int k = 0; k < points; k++)
    {
        double tau = 2.0 * PARITY_PI * k / points;
        sum += cos(order * tau - x * sin(tau));
    }
    return sum / points;
}

double parity_bessel_y(int order, double x)
{
    if (isnan(x) || x < 0.0) return NAN;
    if (x == 0.0) return -INFINITY;
    if (isinf(x)) return 0.0;
    if (x > 25.0) return parity_hankel_asymptotic(order, x, 1);
    double first = 0.0;
    for (int k = 0; k < 96; k++)
    {
        double tau = 0.5 * PARITY_PI + 0.5 * PARITY_PI * PARITY_GL96_X[k];
        first += PARITY_GL96_W[k] * sin(x * sin(tau) - order * tau);
    }
    first *= 0.5 * PARITY_PI;
    double sign = order % 2 == 0 ? 1.0 : -1.0, second = 0.0;
    for (int panel = 0; panel < 2000; panel++)
    {
        if (x * sinh((double)panel) - order * panel > PARITY_EXP_UNDERFLOW) break;
        double part = 0.0;
        for (int k = 0; k < 32; k++)
        {
            double t = panel + 0.5 + 0.5 * PARITY_GL32_X[k];
            part += PARITY_GL32_W[k] * (exp(order * t) + sign * exp(-order * t)) * exp(-x * sinh(t));
        }
        second += 0.5 * part;
    }
    return (first - second) / PARITY_PI;
}

double parity_bessel_i_scaled(int order, double x)
{
    if (isnan(x)) return x;
    double ax = fabs(x), sign = order % 2 == 1 && x < 0.0 ? -1.0 : 1.0;
    if (ax < 2.0) return sign * exp(-ax) * parity_bessel_series(order, ax, 0);
    if (ax > PARITY_LARGE_ARGUMENT) return sign * parity_modified_bessel_asymptotic_signed(4.0 * order * order, ax, 1) / sqrt(2.0 * PARITY_PI * ax);
    int points = 40 + (int)ceil(sqrt(80.0 * ax));
    double sum = 0.0;
    for (int k = 0; k < points; k++)
    {
        double tau = 2.0 * PARITY_PI * k / points;
        sum += exp(ax * (cos(tau) - 1.0)) * cos(order * tau);
    }
    return sign * sum / points;
}

double parity_bessel_i(int order, double x)
{
    if (isinf(x)) return order % 2 == 1 && x < 0.0 ? -INFINITY : INFINITY;
    return parity_bessel_i_scaled(order, x) * exp(fabs(x));
}

// e^x K_nu(x): the asymptotic series above x = 50, else the h = 0.02 trapezoid of e^{-x (cosh t - 1)} cosh(nu t).
double parity_bessel_k_scaled(double nu, double x)
{
    if (isnan(x) || x < 0.0) return NAN;
    if (x == 0.0) return INFINITY;
    if (x > PARITY_LARGE_ARGUMENT) return sqrt(PARITY_PI / (2.0 * x)) * parity_modified_bessel_asymptotic_signed(4.0 * nu * nu, x, 0);
    const double h = 0.02;
    double sum = 0.5;
    for (int k = 1; k < 100000; k++)
    {
        double t = k * h;
        if (x * (cosh(t) - 1.0) - fabs(nu) * t > PARITY_EXP_UNDERFLOW) break;
        sum += exp(-x * (cosh(t) - 1.0)) * cosh(nu * t);
    }
    return sum * h;
}

double parity_bessel_k(double nu, double x) { return (isinf(x) && x > 0.0) ? 0.0 : parity_bessel_k_scaled(nu, x) * exp(-x); }
// J_nu(x) for non-integer nu (Schlafli's integral) over the CPU's Gauss-Legendre nodes.
double parity_bessel_j_fractional(double nu, double x)
{
    double first = 0.0;
    for (int k = 0; k < 96; k++)
    {
        double tau = 0.5 * PARITY_PI + 0.5 * PARITY_PI * PARITY_GL96_X[k];
        first += PARITY_GL96_W[k] * cos(nu * tau - x * sin(tau));
    }
    first *= 0.5 * PARITY_PI;
    double second = 0.0;
    for (int panel = 0; panel < 2000; panel++)
    {
        if (x * sinh((double)panel) + nu * panel > PARITY_EXP_UNDERFLOW) break;
        double part = 0.0;
        for (int k = 0; k < 32; k++)
        {
            double t = panel + 0.5 + 0.5 * PARITY_GL32_X[k];
            part += PARITY_GL32_W[k] * exp(-x * sinh(t) - nu * t);
        }
        second += 0.5 * part;
    }
    return (first - sin(nu * PARITY_PI) * second) / PARITY_PI;
}

// a * b = p + e exactly (FMA).
void parity_two_product(double a, double b, double* p, double* e) { *p = a * b; *e = fma(a, b, -*p); }

// zeta = (2/3) z sqrt(z) as an unevaluated double-double hi + lo.
void parity_zeta_double_double(double z, double* hi, double* lo)
{
    double root = sqrt(z), sq, sqErr;
    parity_two_product(root, root, &sq, &sqErr);
    double rootLo = ((z - sq) - sqErr) / (2.0 * root);
    double m, mErr;
    parity_two_product(z, root, &m, &mErr);
    double mLo = mErr + z * rootLo;
    const double twoThirds = 2.0 / 3.0;
    double three, threeErr;
    parity_two_product(3.0, twoThirds, &three, &threeErr);
    double twoThirdsLo = ((2.0 - three) - threeErr) / 3.0;
    double p, pErr;
    parity_two_product(m, twoThirds, &p, &pErr);
    double pLo = pErr + m * twoThirdsLo + mLo * twoThirds;
    *hi = p + pLo;
    *lo = pLo - (*hi - p);
}

double parity_airy_ai_d(double x)
{
    if (isnan(x)) return x;
    if (isinf(x)) return 0.0;
    if (x >= -5.0 && x <= 1.0)
    {
        double x3 = x * x * x, f = 1.0, g = x, tf = 1.0, tg = x;
        for (int k = 1; k < 200; k++)
        {
            tf *= x3 / ((3.0 * k - 1.0) * (3.0 * k));
            tg *= x3 / ((3.0 * k) * (3.0 * k + 1.0));
            f += tf;
            g += tg;
            if (fabs(tf) + fabs(tg) < 1e-18 * (fabs(f) + fabs(g))) break;
        }
        return 0.355028053887817239 * f - 0.258819403792806798 * g;
    }
    double z = fabs(x), zeta = 2.0 / 3.0 * z * sqrt(z);
    if (x > 0.0) return sqrt(x / 3.0) / PARITY_PI * parity_bessel_k(1.0 / 3.0, zeta);
    if (z <= 15.0) return sqrt(z) / 3.0 * (parity_bessel_j_fractional(1.0 / 3.0, zeta) + parity_bessel_j_fractional(-1.0 / 3.0, zeta));
    double u = 1.0, even = 1.0, odd = 0.0, power = 1.0, last = 1.7976931348623157e308;
    for (int k = 1; k < 60; k++)
    {
        u *= (6.0 * k - 5.0) * (6.0 * k - 3.0) * (6.0 * k - 1.0) / ((2.0 * k - 1.0) * 216.0 * k);
        power /= zeta;
        double term = u * power;
        if (term > last) break;
        last = term;
        int pair = k / 2;
        double s = (pair % 2 == 0 ? 1.0 : -1.0) * term;
        if (k % 2 == 0) even += s; else odd += s;
        if (term < 1e-17) break;
    }
    double hi, lo;
    parity_zeta_double_double(z, &hi, &lo);
    double sh = sin(hi), ch = cos(hi), sl = sin(lo + PARITY_PI / 4.0), cl = cos(lo + PARITY_PI / 4.0);
    double sinPhase = sh * cl + ch * sl, cosPhase = ch * cl - sh * sl;
    return (sinPhase * even - cosPhase * odd) / (sqrt(PARITY_PI) * pow(z, 0.25));
}

double parity_igamma_series(double a, double x)
{
    double ap = a, sum = 1.0 / a, inc = sum;
    for (int k = 0; k < 1000; k++)
    {
        ap += 1.0;
        inc *= x / ap;
        sum += inc;
        if (fabs(inc) < fabs(sum) * 1e-16) break;
    }
    return sum * exp(-x + a * log(x) - lgamma(a));
}

double parity_igamma_continued_fraction(double a, double x)
{
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int k = 1; k < 1000; k++)
    {
        double an = -k * (k - a);
        b += 2.0;
        d = an * d + b;
        if (fabs(d) < tiny) d = tiny;
        c = b + an / c;
        if (fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        double ratio = d * c;
        h *= ratio;
        if (fabs(ratio - 1.0) < 1e-16) break;
    }
    return exp(-x + a * log(x) - lgamma(a)) * h;
}

// sin(pi x) with exact argument reduction (SpecialFunctions.SinPi).
double parity_sinpi_d(double x)
{
    if (isnan(x) || isinf(x)) return NAN;
    double r = x - 2.0 * rint(x * 0.5);
    double a = fabs(r);
    if (a == 0.0 || a == 1.0) return 0.0;
    if (a > 0.5) a = 1.0 - a;
    return (r < 0.0 ? -1.0 : 1.0) * sin(PARITY_PI * a);
}
// ---- unary kernels: y[i] = op(x[i]; p0, p1), evaluated in double ----

#define PARITY_UNARY_D(name, expr) \
__kernel void name(__global const float* restrict x, __global float* restrict y, int n, float p0, float p1) \
{ PARITY_INDEX double v = (double)x[i]; double a0 = (double)p0; double a1 = (double)p1; y[i] = (float)(expr); }

// min(max(v, lo), hi) with NaN propagating from v OR either bound (Math.Max / Math.Min semantics).
double parity_clamp_nan(double v, double lo, double hi) { return (isnan(v) || isnan(lo) || isnan(hi)) ? NAN : (v < lo ? lo : (v > hi ? hi : v)); }

// celu: x > 0 ? x : alpha * (exp(x / alpha) - 1)
PARITY_UNARY_D(parity_celu, v > 0.0 ? v : a0 * (exp(v / a0) - 1.0))
PARITY_UNARY_D(parity_hardtanh, parity_clamp_nan(v, a0, a1))
// logsigmoid: min(x, 0) - log(1 + exp(-|x|)), NaN propagating
PARITY_UNARY_D(parity_logsigmoid, isnan(v) ? v : (v < 0.0 ? v : 0.0) - log1p(exp(-fabs(v))))
PARITY_UNARY_D(parity_softsign, v / (1.0 + fabs(v)))
// logit: c = clamp(x, lo, hi) (lo / hi are -inf / +inf without eps), log(c / (1 - c))
PARITY_UNARY_D(parity_logit, log(parity_clamp_nan(v, a0, a1) / (1.0 - parity_clamp_nan(v, a0, a1))))
// sinc: x == 0 ? 1 : isinf(x) ? 0 : sin(pi x) / (pi x)
PARITY_UNARY_D(parity_sinc, v == 0.0 ? 1.0 : (isinf(v) ? 0.0 : parity_sinpi_d(v) / (PARITY_PI * v)))
PARITY_UNARY_D(parity_erf, parity_erf_d(v))
PARITY_UNARY_D(parity_bessel_j0, parity_bessel_j(0, v))
PARITY_UNARY_D(parity_bessel_j1, parity_bessel_j(1, v))
PARITY_UNARY_D(parity_bessel_y0, parity_bessel_y(0, v))
PARITY_UNARY_D(parity_bessel_y1, parity_bessel_y(1, v))
PARITY_UNARY_D(parity_bessel_i0, parity_bessel_i(0, v))
PARITY_UNARY_D(parity_bessel_i1, parity_bessel_i(1, v))
PARITY_UNARY_D(parity_erfcx, parity_erfcx_d(v))
PARITY_UNARY_D(parity_ndtr, parity_ndtr_d(v))
PARITY_UNARY_D(parity_ndtri, parity_ndtri_d(v))
PARITY_UNARY_D(parity_entr, isnan(v) ? v : (v > 0.0 ? -v * log(v) : (v == 0.0 ? 0.0 : -INFINITY)))
PARITY_UNARY_D(parity_signbit, signbit(v) ? 1.0 : 0.0)
PARITY_UNARY_D(parity_isposinf, (isinf(v) && v > 0.0) ? 1.0 : 0.0)
PARITY_UNARY_D(parity_isneginf, (isinf(v) && v < 0.0) ? 1.0 : 0.0)
PARITY_UNARY_D(parity_log_ndtr, parity_log_ndtr_d(v))
PARITY_UNARY_D(parity_bessel_k0, parity_bessel_k(0.0, v))
PARITY_UNARY_D(parity_bessel_k1, parity_bessel_k(1.0, v))
PARITY_UNARY_D(parity_bessel_k0e, parity_bessel_k_scaled(0.0, v))
PARITY_UNARY_D(parity_bessel_k1e, parity_bessel_k_scaled(1.0, v))
PARITY_UNARY_D(parity_spherical_bessel_j0, isnan(v) ? v : isinf(v) ? 0.0 : fabs(v) < 1e-4 ? 1.0 - v * v / 6.0 : sin(v) / v)
PARITY_UNARY_D(parity_airy_ai, parity_airy_ai_d(v))
// angle of a real number: NaN stays NaN, negatives give pi, everything else (including -0) 0.
PARITY_UNARY_D(parity_angle, isnan(v) ? v : (v < 0.0 ? PARITY_PI : 0.0))
// all / any indicator: 1 where the element is non-zero (NaN counts as non-zero), else 0.
PARITY_UNARY_D(parity_nonzero, v != 0.0 ? 1.0 : 0.0)

// mvlgamma (p = p0 >= 1): NaN at or below (p - 1) / 2, else p (p - 1) / 4 log(pi) + sum_j lgamma(x - j / 2).
__kernel void parity_mvlgamma(__global const float* restrict x, __global float* restrict y, int n, float p0, float p1)
{
    PARITY_INDEX
    double v = (double)x[i];
    int p = (int)p0;
    if (v <= (p - 1) / 2.0) { y[i] = NAN; return; }
    double sum = p * (p - 1.0) / 4.0 * log(PARITY_PI);
    for (int j = 0; j < p; j++) sum += lgamma(v - j / 2.0);
    y[i] = (float)sum;
}

// ---- binary kernels (same shape) ----

// fmax / fmin: NaN-ignoring max / min -- exactly OpenCL's fmax / fmin.
__kernel void parity_fmax(__global const float* restrict a, __global const float* restrict b, __global float* restrict y, int n)
{ PARITY_INDEX y[i] = fmax(a[i], b[i]); }

__kernel void parity_fmin(__global const float* restrict a, __global const float* restrict b, __global float* restrict y, int n)
{ PARITY_INDEX y[i] = fmin(a[i], b[i]); }

// heaviside: isnan(x) ? x : x < 0 ? 0 : x > 0 ? 1 : values
__kernel void parity_heaviside(__global const float* restrict x, __global const float* restrict v, __global float* restrict y, int n)
{
    PARITY_INDEX
    float t = x[i];
    y[i] = isnan(t) ? t : (t < 0.0f ? 0.0f : (t > 0.0f ? 1.0f : v[i]));
}

// greater_equal / less_equal: 1 or 0 (a comparison with NaN is false, as in .NET)
__kernel void parity_greater_equal(__global const float* restrict a, __global const float* restrict b, __global float* restrict y, int n)
{ PARITY_INDEX y[i] = a[i] >= b[i] ? 1.0f : 0.0f; }

__kernel void parity_less_equal(__global const float* restrict a, __global const float* restrict b, __global float* restrict y, int n)
{ PARITY_INDEX y[i] = a[i] <= b[i] ? 1.0f : 0.0f; }

__kernel void parity_igamma(__global const float* restrict as, __global const float* restrict xs, __global float* restrict y, int n)
{
    PARITY_INDEX
    double a = (double)as[i], v = (double)xs[i], r;
    if (isnan(a) || isnan(v) || a <= 0.0 || v < 0.0) r = NAN;
    else if (v == 0.0) r = 0.0;
    else if (isinf(v)) r = 1.0;
    else r = v < a + 1.0 ? parity_igamma_series(a, v) : 1.0 - parity_igamma_continued_fraction(a, v);
    y[i] = (float)r;
}

__kernel void parity_igammac(__global const float* restrict as, __global const float* restrict xs, __global float* restrict y, int n)
{
    PARITY_INDEX
    double a = (double)as[i], v = (double)xs[i], r;
    if (isnan(a) || isnan(v) || a <= 0.0 || v < 0.0) r = NAN;
    else if (v == 0.0) r = 1.0;
    else if (isinf(v)) r = 0.0;
    else r = v < a + 1.0 ? 1.0 - parity_igamma_series(a, v) : parity_igamma_continued_fraction(a, v);
    y[i] = (float)r;
}

// floor_divide on floats: floor(a / b) in double, as the CPU's Math.Floor(x / y).
__kernel void parity_floor_divide(__global const float* restrict a, __global const float* restrict b, __global float* restrict y, int n)
{ PARITY_INDEX y[i] = (float)floor((double)a[i] / (double)b[i]); }

// Orthogonal polynomials: the CPU's recurrence in double. kind: 0 ChebyshevT, 1 U, 2 V, 3 W, 4 HermiteH,
// 5 HermiteHe, 6 LaguerreL, 7 LegendreP; shifted maps x to 2x - 1. The host keeps infinite / > 2^20 degrees on the CPU.
__kernel void parity_polynomial(__global const float* restrict x, __global const float* restrict degree, __global float* restrict y,
    int n, int kind, int shifted)
{
    PARITY_INDEX
    double d = (double)degree[i];
    if (isnan(d)) { y[i] = NAN; return; }
    double t = trunc(d);
    if (t < 0.0) { y[i] = 0.0f; return; }
    if (t > 1048576.0) { y[i] = NAN; return; }
    long deg = (long)t;
    double v = (double)x[i];
    if (isnan(v)) { y[i] = NAN; return; }
    if (shifted) v = 2.0 * v - 1.0;
    double previous = 1.0, current;
    switch (kind)
    {
        case 0: current = v; break;
        case 1: current = 2.0 * v; break;
        case 2: current = 2.0 * v - 1.0; break;
        case 3: current = 2.0 * v + 1.0; break;
        case 4: current = 2.0 * v; break;
        case 5: current = v; break;
        case 6: current = 1.0 - v; break;
        default: current = v; break;
    }
    if (deg == 0) { y[i] = 1.0f; return; }
    for (long k = 1; k < deg; k++)
    {
        double kk = (double)k, next;
        switch (kind)
        {
            case 4: next = 2.0 * v * current - 2.0 * kk * previous; break;
            case 5: next = v * current - kk * previous; break;
            case 6: next = ((2.0 * kk + 1.0 - v) * current - kk * previous) / (kk + 1.0); break;
            case 7: next = ((2.0 * kk + 1.0) * v * current - kk * previous) / (kk + 1.0); break;
            default: next = 2.0 * v * current - previous; break;
        }
        previous = current;
        current = next;
    }
    y[i] = (float)current;
}
// ---- generators: the CPU's FromDoubles lambdas in double, rounded once to float ----

__kernel void parity_range(__global float* restrict y, int n, double start, double span, double denominator)
{ PARITY_INDEX y[i] = (float)(start + i * span / denominator); }

__kernel void parity_logspace(__global float* restrict y, int n, double start, double span, double denominator, double base, int single)
{ PARITY_INDEX y[i] = (float)pow(base, single ? start : start + i * span / denominator); }

// Windows over t = i / (m - 1): 0 Hann, 1 Hamming(alpha, beta), 2 Blackman, 3 Bartlett, 4 Kaiser(beta = alpha).
__kernel void parity_window(__global float* restrict y, int n, int kind, int m, double alpha, double beta)
{
    PARITY_INDEX
    if (n == 1) { y[i] = 1.0f; return; }
    double t = i / (double)(m - 1), v;
    switch (kind)
    {
        case 0: v = 0.5 - 0.5 * cos(2.0 * PARITY_PI * t); break;
        case 1: v = alpha - beta * cos(2.0 * PARITY_PI * t); break;
        case 2: v = 0.42 - 0.5 * cos(2.0 * PARITY_PI * t) + 0.08 * cos(4.0 * PARITY_PI * t); break;
        case 3: v = 1.0 - fabs(2.0 * t - 1.0); break;
        default:
        {
            double r = 2.0 * t - 1.0;
            v = parity_bessel_i(0, alpha * sqrt(fmax(0.0, 1.0 - r * r))) / parity_bessel_i(0, alpha);
            break;
        }
    }
    y[i] = (float)v;
}

void parity_tri_columns(int r, int col, int offset, int lower, long* lo, long* hi)
{
    long edge = (long)r + offset;
    if (lower) { *lo = 0; *hi = edge < col - 1 ? edge : col - 1; }
    else { *lo = edge > 0 ? edge : 0; *hi = col - 1; }
}

// tril / triu indices as a [2, total] tensor, one work item per row.
__kernel void parity_tri_indices(__global float* restrict y, int row, int col, int offset, int lower, int total)
{
    int i = (int)get_global_id(0);
    if (i >= row) return;
    long lo, hi, at = 0;
    for (int r = 0; r < i; r++)
    {
        parity_tri_columns(r, col, offset, lower, &lo, &hi);
        if (hi >= lo) at += hi - lo + 1;
    }
    parity_tri_columns(i, col, offset, lower, &lo, &hi);
    for (long j = lo; j <= hi; j++, at++)
    {
        y[at] = (float)i;
        y[total + at] = (float)j;
    }
}

__kernel void parity_diagflat(__global const float* restrict x, __global float* restrict y, int total, int size, int offset, int n)
{
    int idx = (int)get_global_id(0);
    if (idx >= total) return;
    int r = idx / size, c = idx % size;
    int k = offset >= 0 ? (c - r == offset ? r : -1) : (r - c == -offset ? c : -1);
    y[idx] = (k >= 0 && k < n) ? x[k] : 0.0f;
}

// diagonal_scatter: geometry[0] = other rank, [1..8] other extents, [9..16] other strides,
// [17..21] stride1, stride2, start1, start2, diagonal (CUDA passes the same as a by-value struct).
__kernel void parity_diagonal_scatter(__global const float* restrict src, __global float* restrict y, int count,
    __global const int* restrict g)
{
    int idx = (int)get_global_id(0);
    if (idx >= count) return;
    int diagonal = g[21];
    int o = idx / diagonal, t = idx % diagonal;
    int base = 0, rest = o;
    for (int j = g[0] - 1; j >= 0; j--) { base += rest % g[1 + j] * g[9 + j]; rest /= g[1 + j]; }
    y[base + (t + g[19]) * g[17] + (t + g[20]) * g[18]] = src[idx];
}

// quantile over sorted rows (ascending, NaN last). mode: 0 linear, 1 lower, 2 higher, 3 nearest (ties to even),
// 4 midpoint. Without ignoreNan a row holding a NaN gives NaN; a row with no values gives NaN.
__kernel void parity_quantile_rows(__global const float* restrict sorted, __global float* restrict y, int rows, int length,
    double q, int mode, int ignoreNan)
{
    int r = (int)get_global_id(0);
    if (r >= rows) return;
    __global const float* s = sorted + (long)r * length;
    int lo = 0, hi = length;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (isnan(s[mid])) hi = mid; else lo = mid + 1; }
    int count = lo;
    if (count == 0 || (count < length && !ignoreNan)) { y[r] = NAN; return; }
    double position = q * (count - 1);
    int below = (int)floor(position), above = (int)ceil(position);
    double w = position - below;
    switch (mode)
    {
        case 1: above = below; w = 0.0; break;
        case 2: below = above; w = 0.0; break;
        case 3: below = above = (int)rint(position); w = 0.0; break;
        case 4: w = below == above ? 0.0 : 0.5; break;
        default: break;
    }
    y[r] = (float)((1.0 - w) * (double)s[below] + w * (double)s[above]);
}

// ---- random sampling: counter-based Philox4x32-10, bit-identical to the CUDA kernels' streams ----

void parity_philox_round(uint* c, uint k0, uint k1)
{
    uint lo0 = 0xD2511F53u * c[0], hi0 = mul_hi(0xD2511F53u, c[0]);
    uint lo1 = 0xCD9E8D57u * c[2], hi1 = mul_hi(0xCD9E8D57u, c[2]);
    uint n0 = hi1 ^ c[1] ^ k0, n2 = hi0 ^ c[3] ^ k1;
    c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
}

typedef struct { uint key0, key1; ulong element; uint draw; } ParityRng;

ParityRng parity_rng(ulong seed, ulong element)
{
    ParityRng r; r.key0 = (uint)seed; r.key1 = (uint)(seed >> 32); r.element = element; r.draw = 0; return r;
}

// A uniform double in [0, 1) with 53 random bits.
double parity_uniform(ParityRng* r)
{
    uint c[4] = { (uint)r->element, (uint)(r->element >> 32), r->draw++, 0x5EEDu };
    uint k0 = r->key0, k1 = r->key1;
    for (int round = 0; round < 10; round++)
    {
        parity_philox_round(c, k0, k1);
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return ((c[0] >> 5) * 67108864.0 + (c[1] >> 6)) * (1.0 / 9007199254740992.0);
}

double parity_standard_normal(ParityRng* r)
{
    double u1 = parity_uniform(r), u2 = parity_uniform(r);
    return sqrt(-2.0 * log(1.0 - u1)) * cos(2.0 * PARITY_PI * u2);
}

// Lowers the 64-bit flag to (index << 3 | code); the host raises the CPU's exception for the first offender.
void parity_flag(__global ulong* flag, ulong element, int code) { atom_min(flag, (element << 3) | (ulong)code); }

__kernel void parity_flag_reset(__global ulong* flag)
{
    if (get_global_id(0) == 0) *flag = 0xFFFFFFFFFFFFFFFFul;
}

// 0 uniform(a, b), 1 cauchy(median a, sigma b), 2 exponential(rate a), 3 geometric(p a), 4 log-normal(a, b),
// 5 standard normal, 6 randint(low a, span b).
__kernel void parity_sample(__global float* restrict y, int n, int kind, double a, double b, ulong seed)
{
    PARITY_INDEX
    ParityRng r = parity_rng(seed, (ulong)i);
    double v;
    switch (kind)
    {
        case 0: v = a + (b - a) * parity_uniform(&r); break;
        case 1: v = a + b * tan(PARITY_PI * (parity_uniform(&r) - 0.5)); break;
        case 2: v = -log(1.0 - parity_uniform(&r)) / a; break;
        case 3: v = a == 1.0 ? 1.0 : fmax(1.0, ceil(log(1.0 - parity_uniform(&r)) / log1p(-a))); break;
        case 4: v = exp(a + b * parity_standard_normal(&r)); break;
        case 5: v = parity_standard_normal(&r); break;
        default: v = a + fmin(floor(parity_uniform(&r) * b), b - 1.0); break;
    }
    y[i] = (float)v;
}

__kernel void parity_bernoulli(__global const float* restrict p, __global float* restrict y, int n, ulong seed, __global ulong* flag)
{
    PARITY_INDEX
    double q = (double)p[i];
    if (!(q >= 0.0 && q <= 1.0)) { parity_flag(flag, (ulong)i, 1); return; }
    ParityRng r = parity_rng(seed, (ulong)i);
    y[i] = parity_uniform(&r) < q ? 1.0f : 0.0f;
}

// Poisson: Knuth's product method below rate 10, Hormann's PTRS above; code 1 for a negative or NaN rate.
__kernel void parity_poisson(__global const float* restrict rates, __global float* restrict y, int n, ulong seed, __global ulong* flag)
{
    PARITY_INDEX
    double rate = (double)rates[i];
    if (!(rate >= 0.0)) { parity_flag(flag, (ulong)i, 1); return; }
    if (rate == 0.0) { y[i] = 0.0f; return; }
    ParityRng r = parity_rng(seed, (ulong)i);
    if (rate < 10.0)
    {
        double limit = exp(-rate), product = parity_uniform(&r);
        int k = 0;
        while (product > limit) { k++; product *= parity_uniform(&r); }
        y[i] = (float)k;
        return;
    }
    double slam = sqrt(rate), logLam = log(rate);
    double bb = 0.931 + 2.53 * slam, aa = -0.059 + 0.02483 * bb, invAlpha = 1.1239 + 1.1328 / (bb - 3.4), vr = 0.9277 - 3.6224 / (bb - 2.0);
    while (true)
    {
        double u = parity_uniform(&r) - 0.5, v = parity_uniform(&r), us = 0.5 - fabs(u);
        double k = floor((2.0 * aa / us + bb) * u + rate + 0.43);
        if (us >= 0.07 && v <= vr) { y[i] = (float)k; return; }
        if (k < 0.0 || (us < 0.013 && v > us)) continue;
        if (log(v) + log(invAlpha) - log(aa / (us * us) + bb) <= -rate + k * logLam - lgamma(k + 1.0)) { y[i] = (float)k; return; }
    }
}

double parity_stirling_tail(double k)
{
    const double table[10] = {
        0.08106146679532726, 0.04134069595540929, 0.02767792568499834, 0.02079067210376509, 0.01664469118982119,
        0.01387612882307075, 0.01189670994589177, 0.01041126526197209, 0.009255462182712733, 0.008330563433362871 };
    if (k <= 9.0) return table[(int)k];
    double kp1sq = (k + 1.0) * (k + 1.0);
    return (1.0 / 12.0 - (1.0 / 360.0 - 1.0 / 1260.0 / kp1sq) / kp1sq) / (k + 1.0);
}
// Binomial: geometric waiting times while n min(p, 1-p) < 10, Hormann's BTRS above. Code 1: a count that is not a
// whole number within the long range; code 2: count < 0, or p outside [0, 1] / NaN.
__kernel void parity_binomial(__global const float* restrict counts, __global const float* restrict probs, __global float* restrict y,
    int total, ulong seed, __global ulong* flag)
{
    int i = (int)get_global_id(0);
    if (i >= total) return;
    double count = (double)counts[i], p = (double)probs[i];
    if (isnan(count) || isinf(count) || count != floor(count) || count >= 9.2233720368547758E18) { parity_flag(flag, (ulong)i, 1); return; }
    long n = (long)count;
    if (n < 0 || !(p >= 0.0 && p <= 1.0)) { parity_flag(flag, (ulong)i, 2); return; }
    if (n == 0 || p == 0.0) { y[i] = 0.0f; return; }
    if (p == 1.0) { y[i] = (float)n; return; }
    ParityRng r = parity_rng(seed, (ulong)i);
    bool flip = p > 0.5;
    double q = flip ? 1.0 - p : p, draw;
    if (n * q >= 10.0)
    {
        double stddev = sqrt(n * q * (1.0 - q)), bb = 1.15 + 2.53 * stddev, aa = -0.0873 + 0.0248 * bb + 0.01 * q;
        double c = n * q + 0.5, vR = 0.92 - 4.2 / bb, ratio = q / (1.0 - q), alpha = (2.83 + 5.1 / bb) * stddev;
        double m = floor((n + 1) * q);
        while (true)
        {
            double u = parity_uniform(&r) - 0.5, v = parity_uniform(&r), us = 0.5 - fabs(u);
            double k = floor((2.0 * aa / us + bb) * u + c);
            if (k < 0.0 || k > (double)n) continue;
            if (us >= 0.07 && v <= vR) { draw = k; break; }
            v = log(v * alpha / (aa / (us * us) + bb));
            double upper = (m + 0.5) * log((m + 1.0) / (ratio * (n - m + 1.0)))
                + (n + 1.0) * log((n - m + 1.0) / (n - k + 1.0))
                + (k + 0.5) * log(ratio * (n - k + 1.0) / (k + 1.0))
                + parity_stirling_tail(m) + parity_stirling_tail(n - m) - parity_stirling_tail(k) - parity_stirling_tail(n - k);
            if (v <= upper) { draw = k; break; }
        }
    }
    else
    {
        double logQ = log1p(-q);
        long successes = 0, position = 0;
        while (true)
        {
            double step = floor(log(1.0 - parity_uniform(&r)) / logQ) + 1.0;
            if (!(step <= (double)(n - position))) break;
            position += (long)step;
            successes++;
        }
        draw = (double)successes;
    }
    y[i] = (float)(flip ? n - draw : draw);
}

// Multinomial: one work item per row walks numSamples draws (removing each pick without replacement), as the CPU.
// Code 1: a negative, NaN or infinite weight; code 2 (ranked after every weight error, bit 58): no positive weight left.
__kernel void parity_multinomial(__global const float* restrict weights, __global float* restrict scratch, __global float* restrict y,
    int rows, int categories, int numSamples, int replacement, ulong seed, __global ulong* flag)
{
    int row = (int)get_global_id(0);
    if (row >= rows) return;
    __global const float* w = weights + (long)row * categories;
    for (int c = 0; c < categories; c++)
    {
        double v = (double)w[c];
        if (!(v >= 0.0) || isinf(v)) { parity_flag(flag, (ulong)row * categories + c, 1); return; }
    }
    ParityRng r = parity_rng(seed, (ulong)row);
    if (replacement)
    {
        __global double* prefix = ((__global double*)scratch) + (long)row * categories;
        double total = 0.0;
        int last = -1;
        for (int c = 0; c < categories; c++) { total += (double)w[c]; prefix[c] = total; if (w[c] > 0.0f) last = c; }
        if (!(total > 0.0)) { parity_flag(flag, (1ul << 58) | ((ulong)row * categories), 2); return; }
        for (int k = 0; k < numSamples; k++)
        {
            double u = parity_uniform(&r) * total;
            int lo = 0, hi = categories;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (u < prefix[mid]) hi = mid; else lo = mid + 1; }
            y[(long)row * numSamples + k] = (float)(lo < categories ? lo : last);
        }
        return;
    }
    __global float* s = scratch + (long)row * categories;
    for (int c = 0; c < categories; c++) s[c] = w[c];
    for (int k = 0; k < numSamples; k++)
    {
        double total = 0.0;
        for (int c = 0; c < categories; c++) total += (double)s[c];
        if (!(total > 0.0)) { parity_flag(flag, (1ul << 58) | ((ulong)row * categories), 2); return; }
        double u = parity_uniform(&r) * total, running = 0.0;
        int chosen = -1;
        for (int c = categories - 1; c >= 0; c--) if (s[c] > 0.0f) { chosen = c; break; }
        for (int c = 0; c < categories; c++)
        {
            running += (double)s[c];
            if (u < running && s[c] > 0.0f) { chosen = c; break; }
        }
        y[(long)row * numSamples + k] = (float)chosen;
        s[chosen] = 0.0f;
    }
}

// Dropout keep mask, one draw per block of `block` elements: alpha dropout writes (a * keep, a * sat * (1 - keep) + b),
// channel dropout writes scale only (offset may alias scale).
__kernel void parity_dropout_affine(__global float* scale, __global float* offset, int n, int block, double p, double a, double sat,
    double b, int alpha, ulong seed)
{
    PARITY_INDEX
    ParityRng r = parity_rng(seed, (ulong)(i / block));
    double keep = parity_uniform(&r) >= p ? 1.0 : 0.0;
    if (alpha) { scale[i] = (float)(a * keep); offset[i] = (float)(a * sat * (1.0 - keep) + b); }
    else scale[i] = (float)(keep * a);
}

__kernel void parity_rrelu(__global const float* restrict x, __global float* restrict y, int n, double lower, double upper, int training,
    ulong seed)
{
    PARITY_INDEX
    double v = (double)x[i];
    double slope;
    if (v > 0.0) slope = 1.0;
    else if (!training) slope = (lower + upper) / 2.0;
    else { ParityRng r = parity_rng(seed, (ulong)i); slope = lower + (upper - lower) * parity_uniform(&r); }
    y[i] = (float)(slope * v);
}

__kernel void parity_random_keys(__global float* restrict y, int n, ulong seed)
{
    PARITY_INDEX
    ParityRng r = parity_rng(seed, (ulong)i);
    y[i] = (float)parity_uniform(&r);
}

// ---- nonzero_static / unravel_index / unique_dim / grid_sample 3-D ----

// shape[0] = rank, shape[1..8] = extents.
__kernel void parity_nonzero_static_scatter(__global const float* restrict x, __global const float* restrict positions,
    __global float* restrict y, int n, int size, __global const int* restrict shape)
{
    PARITY_INDEX
    if (x[i] == 0.0f) return;
    int row = (int)positions[i] - 1;
    if (row >= size) return;
    int rank = shape[0], rest = i;
    for (int k = rank - 1; k >= 0; k--) { y[row * rank + k] = (float)(rest % shape[1 + k]); rest /= shape[1 + k]; }
}

__kernel void parity_unravel_axis(__global const float* restrict indices, __global float* restrict y, int n, long total, long stride,
    long extent, int validate, __global ulong* flag)
{
    PARITY_INDEX
    double t = trunc((double)indices[i]);
    if (isnan(t) || t < -(double)total || t >= (double)total)
    {
        if (validate) parity_flag(flag, (ulong)i, 1);
        return;
    }
    long f = (long)t;
    y[i] = (float)(((f + total) % total) / stride % extent);
}

// unique_dim's slice order: element by element with double.CompareTo (NaN below every number, equal to NaN).
int parity_slice_compare(__global const float* data, int a, int b, int length)
{
    __global const float* sa = data + (long)a * length;
    __global const float* sb = data + (long)b * length;
    for (int k = 0; k < length; k++)
    {
        float x = sa[k], y = sb[k];
        int nx = isnan(x), ny = isnan(y);
        if (nx || ny) { if (nx && ny) continue; return nx ? -1 : 1; }
        if (x < y) return -1;
        if (x > y) return 1;
    }
    return 0;
}

__kernel void parity_slice_bitonic_step(__global const float* restrict data, __global float* restrict order, int rowLen, int k, int j, int length)
{
    int i = (int)get_global_id(0);
    if (i >= rowLen) return;
    int ixj = i ^ j;
    if (ixj <= i) return;
    float fa = order[i], fb = order[ixj];
    int a = (int)fa, b = (int)fb;
    int bBeforeA, aBeforeB;
    if (a < 0 || b < 0) { bBeforeA = a < 0 && b >= 0; aBeforeB = b < 0 && a >= 0; }
    else
    {
        int c = parity_slice_compare(data, a, b, length);
        bBeforeA = c > 0 || (c == 0 && b < a);
        aBeforeB = c < 0 || (c == 0 && a < b);
    }
    int up = (i & k) == 0;
    if (up ? bBeforeA : aBeforeB) { order[i] = fb; order[ixj] = fa; }
}

__kernel void parity_slice_first_flags(__global const float* restrict data, __global const float* restrict order, __global float* restrict flags,
    int n, int length)
{
    PARITY_INDEX
    flags[i] = (i == 0 || parity_slice_compare(data, (int)order[i], (int)order[i - 1], length) != 0) ? 1.0f : 0.0f;
}

__kernel void parity_slice_gather_unique(__global const float* restrict data, __global const float* restrict order,
    __global const float* restrict flags, __global const float* restrict positions, __global float* restrict y, int total, int length)
{
    int idx = (int)get_global_id(0);
    if (idx >= total) return;
    int s = idx / length, k = idx % length;
    if (flags[s] == 0.0f) return;
    int row = (int)positions[s] - 1;
    y[(long)row * length + k] = data[(long)(int)order[s] * length + k];
}
double parity_grid_reflect(double coord, int twiceLow, int twiceHigh)
{
    if (twiceLow == twiceHigh) return 0.0;
    double mn = twiceLow / 2.0, span = (twiceHigh - twiceLow) / 2.0;
    coord -= mn;
    if (coord < 0.0) coord = -coord;
    double extra = fmod(coord, span), flips = floor(coord / span);
    return fmod(flips, 2.0) == 0.0 ? extra + mn : span - extra + mn;
}

double parity_grid_source(double normalized, int size, int padding, int align)
{
    double coord = align ? (normalized + 1.0) / 2.0 * (size - 1) : ((normalized + 1.0) * size - 1.0) / 2.0;
    if (padding == 2) coord = align ? parity_grid_reflect(coord, 0, 2 * (size - 1)) : parity_grid_reflect(coord, -1, 2 * size - 1);
    if (padding >= 1)
    {
        if (coord <= 0.0) return 0.0;
        if (coord >= size - 1) return (double)(size - 1);
    }
    return coord;
}

// A voxel index as the CPU's VoxelIndex: NaN and out-of-int-range coordinates land far outside the volume.
long parity_grid_index(double v) { return (isnan(v) || v < -2147483648.0 || v >= 2147483648.0) ? -4 : (long)v; }

__kernel void parity_grid_sample_3d(__global const float* restrict x, __global const float* restrict grid, __global float* restrict y,
    int batches, int channels, int depth, int height, int width, int points, int mode, int padding, int align)
{
    int idx = (int)get_global_id(0);
    if (idx >= batches * points) return;
    int b = idx / points, q = idx % points;
    __global const float* g = grid + (long)idx * 3;
    double cx = parity_grid_source((double)g[0], width, padding, align);
    double cy = parity_grid_source((double)g[1], height, padding, align);
    double cz = parity_grid_source((double)g[2], depth, padding, align);
    int volume = depth * height * width;
    int offsets[8]; double weights[8]; int taps = 0;
    if (mode == 1)
    {
        long ix = parity_grid_index(rint(cx)), iy = parity_grid_index(rint(cy)), iz = parity_grid_index(rint(cz));
        if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth)
        { offsets[0] = (int)((iz * height + iy) * width + ix); weights[0] = 1.0; taps = 1; }
    }
    else
    {
        double fx0 = floor(cx), fy0 = floor(cy), fz0 = floor(cz);
        long x0 = parity_grid_index(fx0), y0 = parity_grid_index(fy0), z0 = parity_grid_index(fz0);
        double fx = cx - fx0, fy = cy - fy0, fz = cz - fz0;
        for (int oz = 0; oz < 2; oz++)
            for (int oy = 0; oy < 2; oy++)
                for (int ox = 0; ox < 2; ox++)
                {
                    long ix = x0 + ox, iy = y0 + oy, iz = z0 + oz;
                    if (ix < 0 || ix >= width || iy < 0 || iy >= height || iz < 0 || iz >= depth) continue;
                    double wx = ox == 1 ? fx : 1.0 - fx, wy = oy == 1 ? fy : 1.0 - fy, wz = oz == 1 ? fz : 1.0 - fz;
                    offsets[taps] = (int)((iz * height + iy) * width + ix);
                    weights[taps] = wx * wy * wz;
                    taps++;
                }
    }
    for (int ch = 0; ch < channels; ch++)
    {
        __global const float* plane = x + ((long)b * channels + ch) * volume;
        double acc = 0.0;
        for (int t = 0; t < taps; t++) acc += weights[t] * (double)plane[offsets[t]];
        y[((long)b * channels + ch) * points + q] = (float)acc;
    }
}

// Inclusive prefix sum: a PARITY_SCAN_BLOCK-wide Hillis-Steele scan per work group, the group totals scanned the same
// way (recursively, on the host side), then each group's carry added. Exact in float for counts below 2^24.
__kernel __attribute__((reqd_work_group_size(PARITY_SCAN_BLOCK, 1, 1)))
void parity_scan_block(__global const float* restrict input, __global float* restrict output, __global float* restrict sums, int n, int writeSums)
{
    __local float s[PARITY_SCAN_BLOCK];
    int t = (int)get_local_id(0), i = (int)get_group_id(0) * PARITY_SCAN_BLOCK + t;
    s[t] = i < n ? input[i] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = 1; offset < PARITY_SCAN_BLOCK; offset <<= 1)
    {
        float v = t >= offset ? s[t - offset] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        s[t] += v;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < n) output[i] = s[t];
    if (writeSums && t == PARITY_SCAN_BLOCK - 1) sums[get_group_id(0)] = s[PARITY_SCAN_BLOCK - 1];
}

__kernel void parity_scan_add(__global float* restrict output, __global const float* restrict carries, int n)
{
    PARITY_INDEX
    int block = i / PARITY_SCAN_BLOCK;
    if (block > 0) output[i] += carries[block - 1];
}

// Integer ops on raw int32 storage, as the CPU's IntegerBinary: 64-bit arithmetic on the sign-extended operands,
// wrapped to int32. op: 0 and, 1 or, 2 xor, 3 left shift, 4 right shift, 5 gcd, 6 lcm, 7 floor divide (code 1 for a
// zero divisor). Shifts follow C#'s long shifts: a count of 64 or more gives 0 (left) or the sign fill (right).
long parity_gcd(long x, long y)
{
    x = x < 0 ? -x : x;
    y = y < 0 ? -y : y;
    while (y != 0) { long t = x % y; x = y; y = t; }
    return x;
}

__kernel void parity_int_binary(__global const int* restrict a, __global const int* restrict b, __global int* restrict y, int n, int op,
    __global ulong* flag)
{
    PARITY_INDEX
    long x = a[i], v = b[i], r;
    switch (op)
    {
        case 0: r = x & v; break;
        case 1: r = x | v; break;
        case 2: r = x ^ v; break;
        case 3: r = v >= 64 ? 0 : (long)((ulong)x << (v & 63)); break;
        case 4: r = v >= 64 ? (x < 0 ? -1 : 0) : x >> (v & 63); break;
        case 5: r = parity_gcd(x, v); break;
        case 6: { long g = parity_gcd(x, v); long m = g == 0 ? 0 : x / g * v; r = m < 0 ? -m : m; break; }
        default:
            if (v == 0) { parity_flag(flag, (ulong)i, 1); return; }
            r = x / v;
            if (x % v != 0 && ((x < 0) != (v < 0))) r -= 1;
            break;
    }
    y[i] = (int)r;
}

__kernel void parity_int_not(__global const int* restrict a, __global int* restrict y, int n)
{ PARITY_INDEX y[i] = ~a[i]; }

// #1135: the same ops on the raw int64 layout (two words per element) that long, ulong, short, ushort, sbyte, byte and
// uint tensors use on the device, as the CUDA kernels: signed 64-bit math, narrowed to the element type t (0 long,
// 1 ulong, 2 short, 3 ushort, 4 sbyte, 5 byte, 6 uint). Flag codes: 1 a zero divisor, 2 the CPU's OverflowException
// (Math.Abs of long.MinValue in gcd / lcm, long.MinValue / -1). gcd is binary (shifts and subtraction) on the magnitudes,
// as on Vulkan, whose driver's 64-bit % proved unreliable.
long parity_int_narrow(long r, int t)
{
    switch (t)
    {
        case 2: return (long)(short)r;
        case 3: return r & 0xFFFFL;
        case 4: return (long)(char)r;
        case 5: return r & 0xFFL;
        case 6: return r & 0xFFFFFFFFL;
        default: return r;
    }
}

ulong parity_magnitude(long x) { return x < 0 ? (ulong)(-x) : (ulong)x; }

ulong parity_gcd_u(ulong a, ulong b)
{
    if (a == 0) return b;
    if (b == 0) return a;
    int shift = 0;
    while (((a | b) & 1UL) == 0) { a >>= 1; b >>= 1; shift++; }
    while ((a & 1UL) == 0) a >>= 1;
    while (b != 0)
    {
        while ((b & 1UL) == 0) b >>= 1;
        if (a > b) { ulong s = a; a = b; b = s; }
        b -= a;
    }
    return a << shift;
}

__kernel void parity_int64_binary(__global const long* restrict a, __global const long* restrict b, __global long* restrict y, int n,
    int op, int t, __global ulong* flag)
{
    PARITY_INDEX
    const long minValue = (long)0x8000000000000000UL;
    long x = a[i], v = b[i], r;
    switch (op)
    {
        case 0: r = x & v; break;
        case 1: r = x | v; break;
        case 2: r = x ^ v; break;
        case 3: r = v >= 64 ? 0 : (long)((ulong)x << (v & 63)); break;
        case 4: r = v >= 64 ? (x < 0 ? -1 : 0) : x >> (v & 63); break;
        case 5:
            if (x == minValue || v == minValue) { parity_flag(flag, (ulong)i, 2); return; }
            r = (long)parity_gcd_u(parity_magnitude(x), parity_magnitude(v));
            break;
        case 6:
        {
            if (x == minValue || v == minValue) { parity_flag(flag, (ulong)i, 2); return; }
            ulong g = parity_gcd_u(parity_magnitude(x), parity_magnitude(v));
            ulong q = g == 0 ? 0 : parity_magnitude(x) / g;
            long m = (long)((x < 0 ? (ulong)(-(long)q) : q) * (ulong)v);   // wraps, as C#
            if (m == minValue) { parity_flag(flag, (ulong)i, 2); return; }
            r = m < 0 ? -m : m;
            break;
        }
        default:
        {
            if (v == 0) { parity_flag(flag, (ulong)i, 1); return; }
            if (x == minValue && v == -1) { parity_flag(flag, (ulong)i, 2); return; }
            ulong ax = parity_magnitude(x), av = parity_magnitude(v), q = ax / av;
            r = ((x < 0) == (v < 0)) ? (long)q : -((long)q + (ax % av != 0 ? 1 : 0));
            break;
        }
    }
    y[i] = parity_int_narrow(r, t);
}

__kernel void parity_int64_not(__global const long* restrict a, __global long* restrict y, int n, int t)
{ PARITY_INDEX y[i] = parity_int_narrow(~a[i], t); }

// grid_sample 3-D adjoints, as the CPU's GridSample3DBackward; dx gathers w * dy from every point through a float
// compare-and-swap add (points share voxels), dgrid is owned by its point's work item.
double parity_grid_reflect_slope(double coord, int twiceLow, int twiceHigh, double* slope)
{
    if (twiceLow == twiceHigh) { *slope = 0.0; return 0.0; }
    double mn = twiceLow / 2.0, span = (twiceHigh - twiceLow) / 2.0, sign = 1.0;
    coord -= mn;
    if (coord < 0.0) { sign = -1.0; coord = -coord; }
    double extra = fmod(coord, span), flips = floor(coord / span);
    if (fmod(flips, 2.0) == 0.0) { *slope *= sign; return extra + mn; }
    *slope *= -sign;
    return span - extra + mn;
}

double parity_grid_source_slope(double normalized, int size, int padding, int align, double* slope)
{
    double coord = align ? (normalized + 1.0) / 2.0 * (size - 1) : ((normalized + 1.0) * size - 1.0) / 2.0;
    *slope = align ? (size - 1) / 2.0 : size / 2.0;
    if (padding == 2)
        coord = align ? parity_grid_reflect_slope(coord, 0, 2 * (size - 1), slope) : parity_grid_reflect_slope(coord, -1, 2 * size - 1, slope);
    if (padding >= 1)
    {
        if (coord <= 0.0) { *slope = 0.0; return 0.0; }
        if (coord >= size - 1) { *slope = 0.0; return (double)(size - 1); }
    }
    return coord;
}

void parity_atomic_add_float(__global float* address, float value)
{
    union { uint u; float f; } expected, desired;
    do
    {
        expected.f = *address;
        desired.f = expected.f + value;
    } while (atomic_cmpxchg((volatile __global uint*)address, expected.u, desired.u) != expected.u);
}

__kernel void parity_grid_sample_3d_backward(__global const float* restrict x, __global const float* restrict grid,
    __global const float* restrict dy, __global float* dx, __global float* restrict dgrid, int batches, int channels, int depth,
    int height, int width, int points, int mode, int padding, int align)
{
    int idx = (int)get_global_id(0);
    if (idx >= batches * points) return;
    int b = idx / points, q = idx % points;
    __global const float* g = grid + (long)idx * 3;
    double sx, sy, sz;
    double cx = parity_grid_source_slope((double)g[0], width, padding, align, &sx);
    double cy = parity_grid_source_slope((double)g[1], height, padding, align, &sy);
    double cz = parity_grid_source_slope((double)g[2], depth, padding, align, &sz);
    int volume = depth * height * width;
    int offsets[8]; double weights[8], gradX[8], gradY[8], gradZ[8]; int taps = 0;
    if (mode == 1)
    {
        long ix = parity_grid_index(rint(cx)), iy = parity_grid_index(rint(cy)), iz = parity_grid_index(rint(cz));
        if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth)
        {
            offsets[0] = (int)((iz * height + iy) * width + ix); weights[0] = 1.0;
            gradX[0] = gradY[0] = gradZ[0] = 0.0; taps = 1;
        }
    }
    else
    {
        double fx0 = floor(cx), fy0 = floor(cy), fz0 = floor(cz);
        long x0 = parity_grid_index(fx0), y0 = parity_grid_index(fy0), z0 = parity_grid_index(fz0);
        double fx = cx - fx0, fy = cy - fy0, fz = cz - fz0;
        for (int oz = 0; oz < 2; oz++)
            for (int oy = 0; oy < 2; oy++)
                for (int ox = 0; ox < 2; ox++)
                {
                    long ix = x0 + ox, iy = y0 + oy, iz = z0 + oz;
                    if (ix < 0 || ix >= width || iy < 0 || iy >= height || iz < 0 || iz >= depth) continue;
                    double wx = ox == 1 ? fx : 1.0 - fx, wy = oy == 1 ? fy : 1.0 - fy, wz = oz == 1 ? fz : 1.0 - fz;
                    offsets[taps] = (int)((iz * height + iy) * width + ix);
                    weights[taps] = wx * wy * wz;
                    gradX[taps] = (ox == 1 ? 1.0 : -1.0) * wy * wz * sx;
                    gradY[taps] = (oy == 1 ? 1.0 : -1.0) * wx * wz * sy;
                    gradZ[taps] = (oz == 1 ? 1.0 : -1.0) * wx * wy * sz;
                    taps++;
                }
    }
    double dgx = 0.0, dgy = 0.0, dgz = 0.0;
    for (int ch = 0; ch < channels; ch++)
    {
        long plane = ((long)b * channels + ch) * volume;
        double up = (double)dy[((long)b * channels + ch) * points + q];
        for (int t = 0; t < taps; t++)
        {
            parity_atomic_add_float(dx + plane + offsets[t], (float)(weights[t] * up));
            double v = (double)x[plane + offsets[t]] * up;
            dgx += v * gradX[t]; dgy += v * gradY[t]; dgz += v * gradZ[t];
        }
    }
    dgrid[(long)idx * 3] = (float)dgx;
    dgrid[(long)idx * 3 + 1] = (float)dgy;
    dgrid[(long)idx * 3 + 2] = (float)dgz;
}
";

    private static readonly string ScanBlockText = ScanBlock.ToString(System.Globalization.CultureInfo.InvariantCulture);

    // The CPU quadratures' Gauss-Legendre rules (96 nodes on [0, pi], 32 per unit tail panel), emitted as program-scope
    // constants at round-trip precision so the device sums over the same nodes.
    private static string GaussLegendreTables()
    {
        var sb = new System.Text.StringBuilder();
        foreach (int n in new[] { 96, 32 })
        {
            var (x, w) = AiDotNet.Tensors.Helpers.SpecialFunctions.GaussLegendreRule(n);
            sb.Append("__constant double PARITY_GL").Append(n).Append("_X[] = { ").Append(Join(x)).Append(" };\n");
            sb.Append("__constant double PARITY_GL").Append(n).Append("_W[] = { ").Append(Join(w)).Append(" };\n");
        }
        return sb.ToString();
    }

    private static string Join(double[] values)
        => string.Join(", ", System.Linq.Enumerable.Select(values, v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture)));

    public static string[] GetKernelNames() => new[]
    {
        "parity_celu", "parity_hardtanh", "parity_logsigmoid", "parity_softsign", "parity_logit", "parity_sinc",
        "parity_erf", "parity_fmax", "parity_fmin", "parity_heaviside", "parity_greater_equal", "parity_less_equal",
        "parity_bessel_j0", "parity_bessel_j1", "parity_bessel_y0", "parity_bessel_y1", "parity_bessel_i0", "parity_bessel_i1",
        "parity_erfcx", "parity_ndtr", "parity_ndtri", "parity_entr", "parity_signbit", "parity_isposinf", "parity_isneginf",
        "parity_polynomial", "parity_log_ndtr", "parity_bessel_k0", "parity_bessel_k1", "parity_bessel_k0e", "parity_bessel_k1e",
        "parity_spherical_bessel_j0", "parity_airy_ai", "parity_mvlgamma", "parity_igamma", "parity_igammac",
        "parity_range", "parity_logspace", "parity_window", "parity_tri_indices",
        "parity_angle", "parity_nonzero", "parity_floor_divide", "parity_diagflat", "parity_diagonal_scatter", "parity_quantile_rows",
        "parity_flag_reset", "parity_sample", "parity_bernoulli", "parity_poisson", "parity_binomial", "parity_multinomial",
        "parity_dropout_affine", "parity_rrelu", "parity_random_keys",
        "parity_nonzero_static_scatter", "parity_unravel_axis", "parity_slice_bitonic_step", "parity_slice_first_flags",
        "parity_slice_gather_unique", "parity_grid_sample_3d", "parity_scan_block", "parity_scan_add", "parity_int_binary",
        "parity_int_not", "parity_grid_sample_3d_backward", "parity_int64_binary", "parity_int64_not",
    };
}
#endif