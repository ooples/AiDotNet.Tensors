// Copyright (c) AiDotNet. All rights reserved.
// Device kernels for PyTorch-parity elementwise ops that computed on the host (#1112).
//
// Each kernel mirrors the CPU reference lambda in CpuEngine.TorchMisc / TorchSpecial exactly, including the NaN
// behaviour of the .NET Math calls it replaces: Math.Max / Math.Min PROPAGATE a NaN, whereas CUDA's fmaxf / fminf
// return the non-NaN operand, so the clamps below test for NaN explicitly. Compiled WITHOUT --use_fast_math: erff,
// log1pf, expf and sinpif must be the accurate libm versions to agree with the CPU's double-precision reference.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>CUDA source for the PyTorch-parity elementwise kernels.</summary>
    internal static class CudaTorchParityElementwiseKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

#define PARITY_INDEX int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
" + GaussLegendreTables() + @"

// celu: x > 0 ? x : alpha * (exp(x / alpha) - 1)
extern ""C"" __global__ __launch_bounds__(256) void parity_celu(const float* __restrict__ x, float* __restrict__ y, int n, float alpha, float unused)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = v > 0.0f ? v : alpha * (expf(v / alpha) - 1.0f);
}

// hardtanh: min(max(x, lo), hi), NaN propagating from x OR either bound (Math.Max / Math.Min semantics)
extern ""C"" __global__ __launch_bounds__(256) void parity_hardtanh(const float* __restrict__ x, float* __restrict__ y, int n, float lo, float hi)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = (isnan(v) || isnan(lo) || isnan(hi)) ? NAN : (v < lo ? lo : (v > hi ? hi : v));
}

// logsigmoid: min(x, 0) - log(1 + exp(-|x|)) (the stable form the CPU uses), NaN propagating
extern ""C"" __global__ __launch_bounds__(256) void parity_logsigmoid(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = isnan(v) ? v : (v < 0.0f ? v : 0.0f) - log1pf(expf(-fabsf(v)));
}

// softsign: x / (1 + |x|)
extern ""C"" __global__ __launch_bounds__(256) void parity_softsign(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = v / (1.0f + fabsf(v));
}

// logit: c = min(max(x, lo), hi) (lo/hi are -inf/+inf without eps), log(c / (1 - c)), NaN propagating from x or a bound
extern ""C"" __global__ __launch_bounds__(256) void parity_logit(const float* __restrict__ x, float* __restrict__ y, int n, float lo, float hi)
{
    PARITY_INDEX
    float v = x[i];
    float c = (isnan(v) || isnan(lo) || isnan(hi)) ? NAN : (v < lo ? lo : (v > hi ? hi : v));
    y[i] = logf(c / (1.0f - c));
}

// sinc: x == 0 ? 1 : isinf(x) ? 0 : sin(pi x) / (pi x)
extern ""C"" __global__ __launch_bounds__(256) void parity_sinc(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = v == 0.0f ? 1.0f : (isinf(v) ? 0.0f : sinpif(v) / (3.14159265358979323846f * v));
}

// erf
extern ""C"" __global__ __launch_bounds__(256) void parity_erf(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    y[i] = erff(x[i]);
}

// fmax / fmin: NaN-ignoring max/min (isnan(a) ? b : isnan(b) ? a : max) -- exactly CUDA's fmaxf / fminf
extern ""C"" __global__ __launch_bounds__(256) void parity_fmax(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n)
{
    PARITY_INDEX
    y[i] = fmaxf(a[i], b[i]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_fmin(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n)
{
    PARITY_INDEX
    y[i] = fminf(a[i], b[i]);
}

// heaviside: isnan(x) ? x : x < 0 ? 0 : x > 0 ? 1 : values
extern ""C"" __global__ __launch_bounds__(256) void parity_heaviside(const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ y, int n)
{
    PARITY_INDEX
    float t = x[i];
    y[i] = isnan(t) ? t : (t < 0.0f ? 0.0f : (t > 0.0f ? 1.0f : v[i]));
}

// greater_equal / less_equal: 1 or 0 (a comparison with NaN is false, as in .NET)
extern ""C"" __global__ __launch_bounds__(256) void parity_greater_equal(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n)
{
    PARITY_INDEX
    y[i] = a[i] >= b[i] ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity_less_equal(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n)
{
    PARITY_INDEX
    y[i] = a[i] <= b[i] ? 1.0f : 0.0f;
}

// Special functions and indicators (#1112 tier 4): CUDA libm's j0f / j1f / y0f / y1f / cyl_bessel_i0f / i1f /
// erfcxf / normcdff / normcdfinvf share the CPU references' edge values (Y_n: NaN for x < 0, -inf at 0, 0 at +inf;
// J_n: 0 at +-inf; ndtri: NaN outside [0, 1], -+inf at 0 / 1). entr mirrors its CPU lambda.
extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_j0(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = j0f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_j1(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = j1f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_y0(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = y0f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_y1(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = y1f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_i0(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = cyl_bessel_i0f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_i1(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = cyl_bessel_i1f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_erfcx(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = erfcxf(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_ndtr(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = normcdff(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_ndtri(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = normcdfinvf(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_entr(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = isnan(v) ? v : (v > 0.0f ? -v * logf(v) : (v == 0.0f ? 0.0f : -INFINITY));
}

extern ""C"" __global__ __launch_bounds__(256) void parity_signbit(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = signbit(v) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity_isposinf(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = (isinf(v) && v > 0.0f) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity_isneginf(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = (isinf(v) && v < 0.0f) ? 1.0f : 0.0f;
}

// Orthogonal polynomials (#1112 tier 4): the CPU's SpecialFunctions.Polynomial recurrence, run in DOUBLE (float drifts
// at high degree). kind: 0 ChebyshevT, 1 U, 2 V, 3 W, 4 HermiteH, 5 HermiteHe, 6 LaguerreL, 7 LegendreP; shifted maps
// x to 2x - 1. Degree: NaN -> NaN; truncated toward zero; negative -> 0 (checked before a NaN x, as on the CPU). The
// host rejects infinite or > 2^20 degrees before launching (the CPU throws for them).
extern ""C"" __global__ __launch_bounds__(256) void parity_polynomial(const float* __restrict__ x, const float* __restrict__ degree,
    float* __restrict__ y, int n, int kind, int shifted)
{
    PARITY_INDEX
    double d = (double)degree[i];
    if (isnan(d)) { y[i] = NAN; return; }
    double t = trunc(d);
    if (t < 0.0) { y[i] = 0.0f; return; }
    // The host keeps infinite / > 2^20 degrees on the CPU path (which throws); never loop unbounded if one slips through.
    if (t > 1048576.0) { y[i] = NAN; return; }
    long long deg = (long long)t;
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
    for (long long k = 1; k < deg; k++)
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

// Special functions with no CUDA libm counterpart (#1112 tier 4): device ports, in double, of the CPU's
// SpecialFunctions -- the same branches, series, quadratures and cutoffs -- so they agree with it to float rounding.
#define PARITY_EXP_UNDERFLOW 745.0
#define PARITY_LARGE_ARGUMENT 50.0
#define PARITY_PI 3.141592653589793

__device__ double parity_ndtr_d(double x) { return 0.5 * erfc(-x / 1.4142135623730951); }

__device__ double parity_log_ndtr(double x)
{
    if (isnan(x)) return x;
    if (isinf(x) && x < 0.0) return -INFINITY;
    if (x > 0.0) return log1p(-parity_ndtr_d(-x));
    if (x > -1.0) return log(parity_ndtr_d(x));
    return log(0.5 * erfcx(-x / 1.4142135623730951)) - 0.5 * x * x;
}

// sum c_k(mu) / x^k, c_k = c_{k-1} (mu - (2k-1)^2) / (8k): the large-x series of e^x K, summed to its smallest term.
__device__ double parity_modified_bessel_asymptotic(double mu, double x)
{
    double term = 1.0, sum = 1.0, last = 1.7976931348623157e308;
    for (int k = 1; k < 200; k++)
    {
        term *= (mu - (2.0 * k - 1.0) * (2.0 * k - 1.0)) / (8.0 * k * x);
        if (fabs(term) > last) break;
        last = fabs(term);
        sum += term;
        if (fabs(term) < 1e-17 * fabs(sum)) break;
    }
    return sum;
}

// e^x K_nu(x): the asymptotic series above x = 50, else the trapezoid (h = 0.02) of the even, double-exponentially
// decaying integrand e^{-x (cosh t - 1)} cosh(nu t) over [0, inf).
__device__ double parity_bessel_k_scaled(double nu, double x)
{
    if (isnan(x) || x < 0.0) return NAN;
    if (x == 0.0) return INFINITY;
    if (x > PARITY_LARGE_ARGUMENT) return sqrt(PARITY_PI / (2.0 * x)) * parity_modified_bessel_asymptotic(4.0 * nu * nu, x);
    const double h = 0.02;
    double sum = 0.5;   // f(0) = 1, halved
    for (int i = 1; i < 100000; i++)
    {
        double t = i * h;
        if (x * (cosh(t) - 1.0) - fabs(nu) * t > PARITY_EXP_UNDERFLOW) break;
        sum += exp(-x * (cosh(t) - 1.0)) * cosh(nu * t);
    }
    return sum * h;
}

__device__ double parity_bessel_k(double nu, double x)
{
    return (isinf(x) && x > 0.0) ? 0.0 : parity_bessel_k_scaled(nu, x) * exp(-x);
}

// J_nu(x) for non-integer nu and 0 < x <= ~40 (Schlafli's integral): 96-node Gauss-Legendre on [0, pi], minus
// sin(nu pi) times unit panels of 32-node Gauss-Legendre over the decaying tail -- the CPU's nodes and weights.
__device__ double parity_bessel_j_fractional(double nu, double x)
{
    double first = 0.0;
    for (int i = 0; i < 96; i++)
    {
        double tau = 0.5 * PARITY_PI + 0.5 * PARITY_PI * PARITY_GL96_X[i];
        first += PARITY_GL96_W[i] * cos(nu * tau - x * sin(tau));
    }
    first *= 0.5 * PARITY_PI;
    double second = 0.0;
    for (int panel = 0; panel < 2000; panel++)
    {
        if (x * sinh((double)panel) + nu * panel > PARITY_EXP_UNDERFLOW) break;
        double part = 0.0;
        for (int i = 0; i < 32; i++)
        {
            double t = panel + 0.5 + 0.5 * PARITY_GL32_X[i];
            part += PARITY_GL32_W[i] * exp(-x * sinh(t) - nu * t);
        }
        second += 0.5 * part;
    }
    return (first - sin(nu * PARITY_PI) * second) / PARITY_PI;
}

// a * b = p + e exactly (by FMA; the CPU uses Dekker's split, which is exact too).
__device__ void parity_two_product(double a, double b, double* p, double* e) { *p = a * b; *e = fma(a, b, -*p); }

// zeta = (2/3) z sqrt(z) as an unevaluated double-double hi + lo.
__device__ void parity_zeta_double_double(double z, double* hi, double* lo)
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

__device__ double parity_airy_ai(double x)
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
        int half = k / 2;
        double s = (half % 2 == 0 ? 1.0 : -1.0) * term;
        if (k % 2 == 0) even += s; else odd += s;
        if (term < 1e-17) break;
    }
    double hi, lo;
    parity_zeta_double_double(z, &hi, &lo);
    double sh = sin(hi), ch = cos(hi), sl = sin(lo + PARITY_PI / 4.0), cl = cos(lo + PARITY_PI / 4.0);
    double sinPhase = sh * cl + ch * sl, cosPhase = ch * cl - sh * sl;
    return (sinPhase * even - cosPhase * odd) / (sqrt(PARITY_PI) * pow(z, 0.25));
}

// The regularized incomplete gamma P(a, x) / Q(a, x): the series below x = a + 1, the Lentz continued fraction above.
__device__ double parity_igamma_series(double a, double x)
{
    double ap = a, sum = 1.0 / a, del = sum;
    for (int k = 0; k < 1000; k++)
    {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (fabs(del) < fabs(sum) * 1e-16) break;
    }
    return sum * exp(-x + a * log(x) - lgamma(a));
}

__device__ double parity_igamma_continued_fraction(double a, double x)
{
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 1000; i++)
    {
        double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (fabs(d) < tiny) d = tiny;
        c = b + an / c;
        if (fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (fabs(del - 1.0) < 1e-16) break;
    }
    return exp(-x + a * log(x) - lgamma(a)) * h;
}

#define PARITY_UNARY_D(name, expr) \
extern ""C"" __global__ __launch_bounds__(256) void name(const float* __restrict__ x, float* __restrict__ y, int n, float p0, float p1) \
{ PARITY_INDEX double v = (double)x[i]; y[i] = (float)(expr); }

PARITY_UNARY_D(parity_log_ndtr, parity_log_ndtr(v))
PARITY_UNARY_D(parity_bessel_k0, parity_bessel_k(0.0, v))
PARITY_UNARY_D(parity_bessel_k1, parity_bessel_k(1.0, v))
PARITY_UNARY_D(parity_bessel_k0e, parity_bessel_k_scaled(0.0, v))
PARITY_UNARY_D(parity_bessel_k1e, parity_bessel_k_scaled(1.0, v))
PARITY_UNARY_D(parity_spherical_bessel_j0, isnan(v) ? v : isinf(v) ? 0.0 : fabs(v) < 1e-4 ? 1.0 - v * v / 6.0 : sin(v) / v)
PARITY_UNARY_D(parity_airy_ai, parity_airy_ai(v))

// mvlgamma (p = p0 >= 1): NaN at or below (p - 1) / 2, else p (p - 1) / 4 log(pi) + sum_j lgamma(x - j / 2).
extern ""C"" __global__ __launch_bounds__(256) void parity_mvlgamma(const float* __restrict__ x, float* __restrict__ y, int n, float p0, float p1)
{
    PARITY_INDEX
    double v = (double)x[i];
    int p = (int)p0;
    if (v <= (p - 1) / 2.0) { y[i] = NAN; return; }
    double sum = p * (p - 1.0) / 4.0 * log(PARITY_PI);
    for (int j = 0; j < p; j++) sum += lgamma(v - j / 2.0);
    y[i] = (float)sum;
}

extern ""C"" __global__ __launch_bounds__(256) void parity_igamma(const float* __restrict__ as, const float* __restrict__ xs, float* __restrict__ y, int n)
{
    PARITY_INDEX
    double a = (double)as[i], v = (double)xs[i], r;
    if (isnan(a) || isnan(v) || a <= 0.0 || v < 0.0) r = NAN;
    else if (v == 0.0) r = 0.0;
    else if (isinf(v)) r = 1.0;
    else r = v < a + 1.0 ? parity_igamma_series(a, v) : 1.0 - parity_igamma_continued_fraction(a, v);
    y[i] = (float)r;
}

extern ""C"" __global__ __launch_bounds__(256) void parity_igammac(const float* __restrict__ as, const float* __restrict__ xs, float* __restrict__ y, int n)
{
    PARITY_INDEX
    double a = (double)as[i], v = (double)xs[i], r;
    if (isnan(a) || isnan(v) || a <= 0.0 || v < 0.0) r = NAN;
    else if (v == 0.0) r = 1.0;
    else if (isinf(v)) r = 0.0;
    else r = v < a + 1.0 ? 1.0 - parity_igamma_series(a, v) : parity_igamma_continued_fraction(a, v);
    y[i] = (float)r;
}

// Generators (#1112 tier 5): creation ops built on the device instead of on the host and uploaded. Each evaluates
// the CPU's FromDoubles lambda in double with the same operation order, then rounds once to float.

// arange / range (span = step, denominator = 1, exact) and linspace-style points: start + i * span / denominator.
extern ""C"" __global__ __launch_bounds__(256) void parity_range(float* __restrict__ y, int n, double start, double span, double denominator)
{
    PARITY_INDEX
    y[i] = (float)(start + i * span / denominator);
}

// logspace: base ^ (single ? start : start + i * span / denominator).
extern ""C"" __global__ __launch_bounds__(256) void parity_logspace(float* __restrict__ y, int n, double start, double span,
    double denominator, double base, int single)
{
    PARITY_INDEX
    y[i] = (float)pow(base, single ? start : start + i * span / denominator);
}

// Windows over t = i / (m - 1): 0 Hann, 1 Hamming(alpha, beta), 2 Blackman, 3 Bartlett, 4 Kaiser(beta = alpha).
// A one-point window is [1].
extern ""C"" __global__ __launch_bounds__(256) void parity_window(float* __restrict__ y, int n, int kind, int m, double alpha, double beta)
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
            v = cyl_bessel_i0(alpha * sqrt(fmax(0.0, 1.0 - r * r))) / cyl_bessel_i0(alpha);
            break;
        }
    }
    y[i] = (float)v;
}

// Columns kept in row r of a row x col grid: lower keeps j - r <= offset, upper keeps j - r >= offset.
__device__ void parity_tri_columns(int r, int col, int offset, int lower, long long* lo, long long* hi)
{
    long long edge = (long long)r + offset;
    if (lower) { *lo = 0; *hi = edge < col - 1 ? edge : col - 1; }
    else { *lo = edge > 0 ? edge : 0; *hi = col - 1; }
}

// tril / triu indices as a [2, total] tensor (row indices, then column indices), one thread per row; each row's
// output offset is the kept-count of the rows before it.
extern ""C"" __global__ __launch_bounds__(256) void parity_tri_indices(float* __restrict__ y, int row, int col, int offset, int lower, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row) return;
    long long lo, hi, at = 0;
    for (int r = 0; r < i; r++)
    {
        parity_tri_columns(r, col, offset, lower, &lo, &hi);
        if (hi >= lo) at += hi - lo + 1;
    }
    parity_tri_columns(i, col, offset, lower, &lo, &hi);
    for (long long j = lo; j <= hi; j++, at++)
    {
        y[at] = (float)i;
        y[total + at] = (float)j;
    }
}

// Index, reduction-feeding and order-statistic ops (#1112 tier 6).

// angle of a real number: NaN stays NaN, negatives (including -inf) give pi, everything else (including -0) 0.
PARITY_UNARY_D(parity_angle, isnan(v) ? v : (v < 0.0 ? PARITY_PI : 0.0))
// all / any indicator: 1 where the element is non-zero (NaN counts as non-zero), else 0.
PARITY_UNARY_D(parity_nonzero, v != 0.0 ? 1.0 : 0.0)

// floor_divide on floats: floor(a / b) in double, as the CPU's Math.Floor(x / y).
extern ""C"" __global__ __launch_bounds__(256) void parity_floor_divide(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n)
{
    PARITY_INDEX
    y[i] = (float)floor((double)a[i] / (double)b[i]);
}

// diagflat: a size x size matrix with the n input values on diagonal `offset`, zeros elsewhere.
extern ""C"" __global__ __launch_bounds__(256) void parity_diagflat(const float* __restrict__ x, float* __restrict__ y, int total, int size, int offset, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int r = idx / size, c = idx % size;
    int k = offset >= 0 ? (c - r == offset ? r : -1) : (r - c == -offset ? c : -1);
    y[idx] = (k >= 0 && k < n) ? x[k] : 0.0f;
}

// diagonal_scatter: writes src (torch.diagonal's layout: the other axes in order, then the diagonal) into the copy of
// the input already in y. Geometry by value: the other axes' extents and strides, the two diagonal strides and starts.
struct ParityDiagonalGeometry
{
    int otherRank;
    int otherExtent[8];
    int otherStride[8];
    int stride1, stride2, start1, start2, diagonal;
};

extern ""C"" __global__ __launch_bounds__(256) void parity_diagonal_scatter(const float* __restrict__ src, float* __restrict__ y, int count,
    ParityDiagonalGeometry g)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int o = idx / g.diagonal, t = idx % g.diagonal;
    int base = 0, rest = o;
    for (int j = g.otherRank - 1; j >= 0; j--) { base += rest % g.otherExtent[j] * g.otherStride[j]; rest /= g.otherExtent[j]; }
    y[base + (t + g.start1) * g.stride1 + (t + g.start2) * g.stride2] = src[idx];
}

// quantile over sorted rows (ascending, NaN last): per row, the CPU's order statistics and interpolation.
// mode: 0 linear, 1 lower, 2 higher, 3 nearest (ties to even), 4 midpoint. Without ignoreNan a row holding a NaN
// gives NaN; with it the NaNs are dropped first. A row with no values gives NaN.
extern ""C"" __global__ __launch_bounds__(256) void parity_quantile_rows(const float* __restrict__ sorted, float* __restrict__ y,
    int rows, int length, double q, int mode, int ignoreNan)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* s = sorted + (long long)r * length;
    int lo = 0, hi = length;   // the first NaN (NaN sorts last)
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
";
        }

        // The CPU quadratures' Gauss-Legendre rules (96 nodes on [0, pi], 32 per unit tail panel), emitted as device
        // constants at round-trip precision so the device sums over the same nodes.
        private static string GaussLegendreTables()
        {
            var sb = new System.Text.StringBuilder();
            foreach (int n in new[] { 96, 32 })
            {
                var (x, w) = AiDotNet.Tensors.Helpers.SpecialFunctions.GaussLegendreRule(n);
                sb.Append("__constant__ double PARITY_GL").Append(n).Append("_X[] = { ").Append(Join(x)).Append(" };\n");
                sb.Append("__constant__ double PARITY_GL").Append(n).Append("_W[] = { ").Append(Join(w)).Append(" };\n");
            }
            return sb.ToString();
        }

        private static string Join(double[] values)
            => string.Join(", ", System.Linq.Enumerable.Select(values, v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture)));

        public static string[] GetKernelNames() => new[]
        {
            "parity_celu", "parity_hardtanh", "parity_logsigmoid", "parity_softsign", "parity_logit", "parity_sinc",
            "parity_erf", "parity_fmax", "parity_fmin", "parity_heaviside", "parity_greater_equal", "parity_less_equal",
            "parity_bessel_j0", "parity_bessel_j1", "parity_bessel_y0", "parity_bessel_y1", "parity_bessel_i0", "parity_bessel_i1", "parity_erfcx", "parity_ndtr", "parity_ndtri", "parity_entr", "parity_signbit", "parity_isposinf", "parity_isneginf",
            "parity_polynomial", "parity_log_ndtr", "parity_bessel_k0", "parity_bessel_k1", "parity_bessel_k0e", "parity_bessel_k1e",
            "parity_spherical_bessel_j0", "parity_airy_ai", "parity_mvlgamma", "parity_igamma", "parity_igammac",
            "parity_range", "parity_logspace", "parity_window", "parity_tri_indices",
            "parity_angle", "parity_nonzero", "parity_floor_divide", "parity_diagflat", "parity_diagonal_scatter", "parity_quantile_rows",
        };
    }
}
