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

// Random sampling (#1112 tier 7): a counter-based Philox4x32-10 generator, as torch.cuda uses. Element i draws its
// k-th uniform from counter (i, k), so a seed gives the same tensor on every run and launch shape, and rejection
// samplers can draw as many values as they need. The streams differ from the CPU's System.Random (as torch's CPU and
// CUDA generators differ); the distributions are the CPU's, sampled by the same algorithms.
//
// Invalid per-element parameters do not produce values: the kernel lowers a 64-bit flag to (index << 3 | code) with
// atomicMin, and the host reads it back to raise the CPU's exception for the first offending element.

__device__ __forceinline__ void parity_philox_round(unsigned int* c, unsigned int k0, unsigned int k1)
{
    unsigned int lo0 = 0xD2511F53u * c[0], hi0 = __umulhi(0xD2511F53u, c[0]);
    unsigned int lo1 = 0xCD9E8D57u * c[2], hi1 = __umulhi(0xCD9E8D57u, c[2]);
    unsigned int n0 = hi1 ^ c[1] ^ k0, n2 = hi0 ^ c[3] ^ k1;
    c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
}

struct ParityRng { unsigned int key0, key1; unsigned long long element; unsigned int draw; };

__device__ __forceinline__ ParityRng parity_rng(unsigned long long seed, unsigned long long element)
{
    ParityRng r; r.key0 = (unsigned int)seed; r.key1 = (unsigned int)(seed >> 32); r.element = element; r.draw = 0; return r;
}

// A uniform double in [0, 1) with 53 random bits.
__device__ double parity_uniform(ParityRng* r)
{
    unsigned int c[4] = { (unsigned int)r->element, (unsigned int)(r->element >> 32), r->draw++, 0x5EEDu };
    unsigned int k0 = r->key0, k1 = r->key1;
    for (int round = 0; round < 10; round++)
    {
        parity_philox_round(c, k0, k1);
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return ((c[0] >> 5) * 67108864.0 + (c[1] >> 6)) * (1.0 / 9007199254740992.0);
}

// Box-Muller, as the CPU's StandardNormal: sqrt(-2 log(1 - U1)) cos(2 pi U2).
__device__ double parity_standard_normal(ParityRng* r)
{
    double u1 = parity_uniform(r), u2 = parity_uniform(r);
    return sqrt(-2.0 * log(1.0 - u1)) * cos(2.0 * PARITY_PI * u2);
}

__device__ __forceinline__ void parity_flag(unsigned long long* flag, unsigned long long element, int code)
{
    atomicMin(flag, (element << 3) | (unsigned long long)code);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_flag_reset(unsigned long long* flag)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) *flag = 0xFFFFFFFFFFFFFFFFull;
}

// Shape-only distributions: 0 uniform(a = low, b = high), 1 cauchy(a = median, b = sigma), 2 exponential(a = rate),
// 3 geometric(a = p), 4 log-normal(a = mean, b = std), 5 standard normal, 6 randint(a = low, b = span).
extern ""C"" __global__ __launch_bounds__(256) void parity_sample(float* __restrict__ y, int n, int kind, double a, double b,
    unsigned long long seed)
{
    PARITY_INDEX
    ParityRng r = parity_rng(seed, (unsigned long long)i);
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

// Bernoulli: 1 with probability p; code 1 for p outside [0, 1] or NaN.
extern ""C"" __global__ __launch_bounds__(256) void parity_bernoulli(const float* __restrict__ p, float* __restrict__ y, int n,
    unsigned long long seed, unsigned long long* flag)
{
    PARITY_INDEX
    double q = (double)p[i];
    if (!(q >= 0.0 && q <= 1.0)) { parity_flag(flag, (unsigned long long)i, 1); return; }
    ParityRng r = parity_rng(seed, (unsigned long long)i);
    y[i] = parity_uniform(&r) < q ? 1.0f : 0.0f;
}

// Poisson: Knuth's product method below rate 10, Hormann's PTRS above; code 1 for a negative or NaN rate.
extern ""C"" __global__ __launch_bounds__(256) void parity_poisson(const float* __restrict__ rates, float* __restrict__ y, int n,
    unsigned long long seed, unsigned long long* flag)
{
    PARITY_INDEX
    double rate = (double)rates[i];
    if (!(rate >= 0.0)) { parity_flag(flag, (unsigned long long)i, 1); return; }
    if (rate == 0.0) { y[i] = 0.0f; return; }
    ParityRng r = parity_rng(seed, (unsigned long long)i);
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

__device__ double parity_stirling_tail(double k)
{
    const double table[10] = {
        0.08106146679532726, 0.04134069595540929, 0.02767792568499834, 0.02079067210376509, 0.01664469118982119,
        0.01387612882307075, 0.01189670994589177, 0.01041126526197209, 0.009255462182712733, 0.008330563433362871 };
    if (k <= 9.0) return table[(int)k];
    double kp1sq = (k + 1.0) * (k + 1.0);
    return (1.0 / 12.0 - (1.0 / 360.0 - 1.0 / 1260.0 / kp1sq) / kp1sq) / (k + 1.0);
}

// Binomial: geometric waiting times while n min(p, 1-p) < 10, Hormann's BTRS above (as the CPU and PyTorch).
// Code 1: count not a whole number of trials within the long range; code 2: count < 0, or p outside [0, 1] / NaN.
extern ""C"" __global__ __launch_bounds__(256) void parity_binomial(const float* __restrict__ counts, const float* __restrict__ probs,
    float* __restrict__ y, int total, unsigned long long seed, unsigned long long* flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    double count = (double)counts[i], p = (double)probs[i];
    if (isnan(count) || isinf(count) || count != floor(count) || count >= 9.2233720368547758E18) { parity_flag(flag, (unsigned long long)i, 1); return; }
    long long n = (long long)count;
    if (n < 0 || !(p >= 0.0 && p <= 1.0)) { parity_flag(flag, (unsigned long long)i, 2); return; }
    if (n == 0 || p == 0.0) { y[i] = 0.0f; return; }
    if (p == 1.0) { y[i] = (float)n; return; }
    ParityRng r = parity_rng(seed, (unsigned long long)i);
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
        // log1p: for q below ~1e-16, log(1 - q) rounds to 0. Each step is checked against the remaining trials
        // before it is added, so an enormous (or infinite) step ends the walk instead of overflowing the position.
        double logQ = log1p(-q);
        long long successes = 0, position = 0;
        while (true)
        {
            double step = floor(log(1.0 - parity_uniform(&r)) / logQ) + 1.0;
            if (!(step <= (double)(n - position))) break;
            position += (long long)step;
            successes++;
        }
        draw = (double)successes;
    }
    y[i] = (float)(flip ? n - draw : draw);
}

// Multinomial: one thread per row walks numSamples draws over its categories (removing each pick without
// replacement), as the CPU. The row's weights are copied into scratch (rows x categories). Code 1: a negative,
// NaN or infinite weight (flagged at its element); code 2: a row with no positive weight left (flagged at the row).
extern ""C"" __global__ __launch_bounds__(256) void parity_multinomial(const float* __restrict__ weights, float* __restrict__ scratch,
    float* __restrict__ y, int rows, int categories, int numSamples, int replacement, unsigned long long seed, unsigned long long* flag)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* w = weights + (long long)row * categories;
    for (int c = 0; c < categories; c++)
    {
        double v = (double)w[c];
        if (!(v >= 0.0) || isinf(v)) { parity_flag(flag, (unsigned long long)row * categories + c, 1); return; }
    }
    ParityRng r = parity_rng(seed, (unsigned long long)row);
    if (replacement)
    {
        // The weights never change, so the CPU's walk is a search of the double prefix sums (accumulated in the
        // CPU's order): the first category whose running total exceeds u, which is always a positive-weight one.
        // O(categories + numSamples log categories) instead of O(numSamples x categories). scratch holds doubles.
        double* prefix = ((double*)scratch) + (long long)row * categories;
        double total = 0.0;
        int last = -1;
        for (int c = 0; c < categories; c++) { total += (double)w[c]; prefix[c] = total; if (w[c] > 0.0f) last = c; }
        if (!(total > 0.0)) { parity_flag(flag, (1ull << 58) | ((unsigned long long)row * categories), 2); return; }
        for (int k = 0; k < numSamples; k++)
        {
            double u = parity_uniform(&r) * total;
            int lo = 0, hi = categories;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (u < prefix[mid]) hi = mid; else lo = mid + 1; }
            y[(long long)row * numSamples + k] = (float)(lo < categories ? lo : last);
        }
        return;
    }
    float* s = scratch + (long long)row * categories;
    for (int c = 0; c < categories; c++) s[c] = w[c];
    for (int k = 0; k < numSamples; k++)
    {
        double total = 0.0;
        for (int c = 0; c < categories; c++) total += (double)s[c];
        // Ranked after every weight error (bit 58), as the CPU validates all weights before it samples.
        if (!(total > 0.0)) { parity_flag(flag, (1ull << 58) | ((unsigned long long)row * categories), 2); return; }
        double u = parity_uniform(&r) * total, running = 0.0;
        int chosen = -1;
        for (int c = categories - 1; c >= 0; c--) if (s[c] > 0.0f) { chosen = c; break; }
        for (int c = 0; c < categories; c++)
        {
            running += (double)s[c];
            if (u < running && s[c] > 0.0f) { chosen = c; break; }
        }
        y[(long long)row * numSamples + k] = (float)chosen;
        if (!replacement) s[chosen] = 0.0f;
    }
}

// Dropout keep mask, one draw per block of `block` elements (block = 1 elementwise): keep = (U >= p). Writes the
// per-element scale and offset of out = x * scale + offset: alpha dropout uses (a * keep, a * sat * (1 - keep) + b),
// channel dropout (keep * inverseKeep, 0).
extern ""C"" __global__ __launch_bounds__(256) void parity_dropout_affine(float* __restrict__ scale, float* __restrict__ offset, int n,
    int block, double p, double a, double sat, double b, int alpha, unsigned long long seed)
{
    PARITY_INDEX
    ParityRng r = parity_rng(seed, (unsigned long long)(i / block));
    double keep = parity_uniform(&r) >= p ? 1.0 : 0.0;
    if (alpha) { scale[i] = (float)(a * keep); offset[i] = (float)(a * sat * (1.0 - keep) + b); }
    else { scale[i] = (float)(keep * a); offset[i] = 0.0f; }
}

// rrelu: slope 1 above zero; below, the eval slope (training = 0) or a uniform draw in [lower, upper].
extern ""C"" __global__ __launch_bounds__(256) void parity_rrelu(const float* __restrict__ x, float* __restrict__ y, int n,
    double lower, double upper, int training, unsigned long long seed)
{
    PARITY_INDEX
    double v = (double)x[i];
    double slope;
    if (v > 0.0) slope = 1.0;
    else if (!training) slope = (lower + upper) / 2.0;
    else { ParityRng r = parity_rng(seed, (unsigned long long)i); slope = lower + (upper - lower) * parity_uniform(&r); }
    y[i] = (float)(slope * v);
}

// randperm keys: one uniform per position, sorted to give a uniformly random permutation (ties keep index order).
extern ""C"" __global__ __launch_bounds__(256) void parity_random_keys(float* __restrict__ y, int n, unsigned long long seed)
{
    PARITY_INDEX
    ParityRng r = parity_rng(seed, (unsigned long long)i);
    y[i] = (float)parity_uniform(&r);
}

// nonzero_static / unravel_index / unique_dim / grid_sample 3-D (#1112 tier 8).

struct ParityShape8 { int rank; int extent[8]; };

// nonzero_static: an element that is non-zero (NaN counts; -0 does not) and whose inclusive-scan position is within
// `size` writes its coordinates to row (position - 1) of the [size, rank] output (pre-filled with the fill value).
extern ""C"" __global__ __launch_bounds__(256) void parity_nonzero_static_scatter(const float* __restrict__ x, const float* __restrict__ positions,
    float* __restrict__ y, int n, int size, ParityShape8 shape)
{
    PARITY_INDEX
    if (x[i] == 0.0f) return;
    int row = (int)positions[i] - 1;
    if (row >= size) return;
    int rest = i;
    for (int k = shape.rank - 1; k >= 0; k--) { y[row * shape.rank + k] = (float)(rest % shape.extent[k]); rest /= shape.extent[k]; }
}

// unravel_index, one launch per output axis: ((flat + total) % total) / stride % extent. The first launch also flags
// (code 1) an index outside [-total, total), or a NaN (which the CPU's (long) cast also rejects).
extern ""C"" __global__ __launch_bounds__(256) void parity_unravel_axis(const float* __restrict__ indices, float* __restrict__ y, int n,
    long long total, long long stride, long long extent, int validate, unsigned long long* flag)
{
    PARITY_INDEX
    double v = (double)indices[i];
    if (isnan(v) || v < -(double)total || v >= (double)total)
    {
        if (validate) parity_flag(flag, (unsigned long long)i, 1);
        return;
    }
    long long f = (long long)v;
    if (f < -total || f >= total) { if (validate) parity_flag(flag, (unsigned long long)i, 1); return; }
    y[i] = (float)(((f + total) % total) / stride % extent);
}

// unique_dim's slice order, as the CPU's comparer: element by element with double.CompareTo, which ranks NaN below
// every number and equal to NaN, and -0 equal to 0. Padding slots (index < 0) sort after every slice; equal slices
// keep index order.
__device__ int parity_slice_compare(const float* __restrict__ data, int a, int b, int length)
{
    const float* sa = data + (long long)a * length;
    const float* sb = data + (long long)b * length;
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

extern ""C"" __global__ __launch_bounds__(256) void parity_slice_bitonic_step(const float* __restrict__ data, float* __restrict__ order,
    int rowLen, int k, int j, int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowLen) return;
    int ixj = i ^ j;
    if (ixj <= i) return;
    float fa = order[i], fb = order[ixj];
    int a = (int)fa, b = (int)fb;
    // ""b sorts before a"" / ""a sorts before b"" under the slice order.
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

// 1 where a sorted slice differs from the one before it (the first slice always), else 0.
extern ""C"" __global__ __launch_bounds__(256) void parity_slice_first_flags(const float* __restrict__ data, const float* __restrict__ order,
    float* __restrict__ flags, int n, int length)
{
    PARITY_INDEX
    flags[i] = (i == 0 || parity_slice_compare(data, (int)order[i], (int)order[i - 1], length) != 0) ? 1.0f : 0.0f;
}

// Copies each first-of-a-run slice (in sorted order) to row (position - 1) of the [unique, length] output.
extern ""C"" __global__ __launch_bounds__(256) void parity_slice_gather_unique(const float* __restrict__ data, const float* __restrict__ order,
    const float* __restrict__ flags, const float* __restrict__ positions, float* __restrict__ y, int total, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int s = idx / length, k = idx % length;
    if (flags[s] == 0.0f) return;
    int row = (int)positions[s] - 1;
    y[(long long)row * length + k] = data[(long long)(int)order[s] * length + k];
}

// grid_sample 3-D, as the CPU's GridSampler3D: unnormalize, clip or reflect, then trilinear (8 taps) or nearest
// (round half to even), skipping out-of-range voxels. mode 0 bilinear, 1 nearest; padding 0 zeros, 1 border,
// 2 reflection. One thread per (batch, point) locates its taps once and sums every channel.
__device__ double parity_grid_reflect(double coord, int twiceLow, int twiceHigh)
{
    if (twiceLow == twiceHigh) return 0.0;
    double mn = twiceLow / 2.0, span = (twiceHigh - twiceLow) / 2.0;
    coord -= mn;
    if (coord < 0.0) coord = -coord;
    double extra = fmod(coord, span), flips = floor(coord / span);
    return fmod(flips, 2.0) == 0.0 ? extra + mn : span - extra + mn;
}

__device__ double parity_grid_source(double normalized, int size, int padding, int align)
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

// A voxel index from a coordinate as the CPU's (int) cast, where NaN and out-of-range values become int.MinValue: here
// a sentinel far enough below zero that neither it nor its +1 neighbour lands inside the volume.
__device__ __forceinline__ long long parity_grid_index(double v) { return (isnan(v) || v < -2147483648.0 || v >= 2147483648.0) ? -4 : (long long)v; }

extern ""C"" __global__ __launch_bounds__(256) void parity_grid_sample_3d(const float* __restrict__ x, const float* __restrict__ grid,
    float* __restrict__ y, int batches, int channels, int depth, int height, int width, int points, int mode, int padding, int align)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batches * points) return;
    int b = idx / points, q = idx % points;
    const float* g = grid + (long long)idx * 3;
    double cx = parity_grid_source((double)g[0], width, padding, align);
    double cy = parity_grid_source((double)g[1], height, padding, align);
    double cz = parity_grid_source((double)g[2], depth, padding, align);
    int volume = depth * height * width;
    int offsets[8]; double weights[8]; int taps = 0;
    if (mode == 1)
    {
        long long ix = parity_grid_index(rint(cx)), iy = parity_grid_index(rint(cy)), iz = parity_grid_index(rint(cz));
        if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth)
        { offsets[0] = (int)((iz * height + iy) * width + ix); weights[0] = 1.0; taps = 1; }
    }
    else
    {
        double fx0 = floor(cx), fy0 = floor(cy), fz0 = floor(cz);
        long long x0 = parity_grid_index(fx0), y0 = parity_grid_index(fy0), z0 = parity_grid_index(fz0);
        double fx = cx - fx0, fy = cy - fy0, fz = cz - fz0;
        for (int oz = 0; oz < 2; oz++)
            for (int oy = 0; oy < 2; oy++)
                for (int ox = 0; ox < 2; ox++)
                {
                    long long ix = x0 + ox, iy = y0 + oy, iz = z0 + oz;
                    if (ix < 0 || ix >= width || iy < 0 || iy >= height || iz < 0 || iz >= depth) continue;
                    double wx = ox == 1 ? fx : 1.0 - fx, wy = oy == 1 ? fy : 1.0 - fy, wz = oz == 1 ? fz : 1.0 - fz;
                    offsets[taps] = (int)((iz * height + iy) * width + ix);
                    weights[taps] = wx * wy * wz;
                    taps++;
                }
    }
    for (int ch = 0; ch < channels; ch++)
    {
        const float* plane = x + ((long long)b * channels + ch) * volume;
        double acc = 0.0;
        for (int t = 0; t < taps; t++) acc += weights[t] * (double)plane[offsets[t]];
        y[((long long)b * channels + ch) * points + q] = (float)acc;
    }
}

// Inclusive prefix sum (#1112 tier 8): a 1024-element Hillis-Steele scan per block, the block totals scanned the same
// way (recursively, on the host side), then each block's carry added. Exact in float for counts below 2^24.
extern ""C"" __global__ __launch_bounds__(1024) void parity_scan_block(const float* __restrict__ input, float* __restrict__ output,
    float* __restrict__ sums, int n)
{
    __shared__ float s[1024];
    int t = threadIdx.x, i = blockIdx.x * 1024 + t;
    s[t] = i < n ? input[i] : 0.0f;
    __syncthreads();
    for (int offset = 1; offset < 1024; offset <<= 1)
    {
        float v = t >= offset ? s[t - offset] : 0.0f;
        __syncthreads();
        s[t] += v;
        __syncthreads();
    }
    if (i < n) output[i] = s[t];
    if (sums != 0 && t == 1023) sums[blockIdx.x] = s[1023];
}

extern ""C"" __global__ __launch_bounds__(256) void parity_scan_add(float* __restrict__ output, const float* __restrict__ carries, int n)
{
    PARITY_INDEX
    int block = i / 1024;
    if (block > 0) output[i] += carries[block - 1];
}

// Integer ops on raw int32 storage (#1112 tier 9), as the CPU's IntegerBinary: each op runs in 64-bit arithmetic on
// the sign-extended operands and wraps to int32. op: 0 and, 1 or, 2 xor, 3 left shift, 4 right shift, 5 gcd, 6 lcm,
// 7 floor divide (code 1 for a zero divisor: the CPU's DivideByZeroException). Shifts follow C#'s long shifts: a
// count of 64 or more gives 0 (left) or the sign fill (right); otherwise the count is masked to its low 6 bits.
__device__ long long parity_gcd(long long x, long long y)
{
    x = x < 0 ? -x : x;
    y = y < 0 ? -y : y;
    while (y != 0) { long long t = x % y; x = y; y = t; }
    return x;
}

extern ""C"" __global__ __launch_bounds__(256) void parity_int_binary(const int* __restrict__ a, const int* __restrict__ b,
    int* __restrict__ y, int n, int op, unsigned long long* flag)
{
    PARITY_INDEX
    long long x = a[i], v = b[i], r;
    switch (op)
    {
        case 0: r = x & v; break;
        case 1: r = x | v; break;
        case 2: r = x ^ v; break;
        case 3: r = v >= 64 ? 0 : (long long)((unsigned long long)x << (v & 63)); break;
        case 4: r = v >= 64 ? (x < 0 ? -1 : 0) : x >> (v & 63); break;
        case 5: r = parity_gcd(x, v); break;
        case 6: { long long g = parity_gcd(x, v); long long m = g == 0 ? 0 : x / g * v; r = m < 0 ? -m : m; break; }
        default:
            if (v == 0) { parity_flag(flag, (unsigned long long)i, 1); return; }
            r = x / v;
            if (x % v != 0 && ((x < 0) != (v < 0))) r -= 1;
            break;
    }
    y[i] = (int)r;
}

extern ""C"" __global__ __launch_bounds__(256) void parity_int_not(const int* __restrict__ a, int* __restrict__ y, int n)
{
    PARITY_INDEX
    y[i] = ~a[i];
}

// grid_sample 3-D adjoints, as the CPU's GridSample3DBackward: per (batch, point) the taps, their weights and each
// weight's derivative with respect to the normalized coordinates (d voxel / d normalized: (size - 1) / 2 aligned,
// size / 2 otherwise; 0 where border / reflection clipping holds the coordinate; sign-flipped by each reflection).
// dx gathers w * dy from every point (atomics: points share voxels); dgrid is owned by its point's thread.
__device__ double parity_grid_reflect_slope(double coord, int twiceLow, int twiceHigh, double* slope)
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

__device__ double parity_grid_source_slope(double normalized, int size, int padding, int align, double* slope)
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

extern ""C"" __global__ __launch_bounds__(256) void parity_grid_sample_3d_backward(const float* __restrict__ x, const float* __restrict__ grid,
    const float* __restrict__ dy, float* __restrict__ dx, float* __restrict__ dgrid, int batches, int channels, int depth, int height,
    int width, int points, int mode, int padding, int align)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batches * points) return;
    int b = idx / points, q = idx % points;
    const float* g = grid + (long long)idx * 3;
    double sx, sy, sz;
    double cx = parity_grid_source_slope((double)g[0], width, padding, align, &sx);
    double cy = parity_grid_source_slope((double)g[1], height, padding, align, &sy);
    double cz = parity_grid_source_slope((double)g[2], depth, padding, align, &sz);
    int volume = depth * height * width;
    int offsets[8]; double weights[8], gradX[8], gradY[8], gradZ[8]; int taps = 0;
    if (mode == 1)
    {
        long long ix = parity_grid_index(rint(cx)), iy = parity_grid_index(rint(cy)), iz = parity_grid_index(rint(cz));
        if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth)
        {
            offsets[0] = (int)((iz * height + iy) * width + ix); weights[0] = 1.0;
            gradX[0] = gradY[0] = gradZ[0] = 0.0; taps = 1;
        }
    }
    else
    {
        double fx0 = floor(cx), fy0 = floor(cy), fz0 = floor(cz);
        long long x0 = parity_grid_index(fx0), y0 = parity_grid_index(fy0), z0 = parity_grid_index(fz0);
        double fx = cx - fx0, fy = cy - fy0, fz = cz - fz0;
        for (int oz = 0; oz < 2; oz++)
            for (int oy = 0; oy < 2; oy++)
                for (int ox = 0; ox < 2; ox++)
                {
                    long long ix = x0 + ox, iy = y0 + oy, iz = z0 + oz;
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
        long long plane = ((long long)b * channels + ch) * volume;
        double up = (double)dy[((long long)b * channels + ch) * points + q];
        for (int t = 0; t < taps; t++)
        {
            atomicAdd(dx + plane + offsets[t], (float)(weights[t] * up));
            double v = (double)x[plane + offsets[t]] * up;
            dgx += v * gradX[t]; dgy += v * gradY[t]; dgz += v * gradZ[t];
        }
    }
    dgrid[(long long)idx * 3] = (float)dgx;
    dgrid[(long long)idx * 3 + 1] = (float)dgy;
    dgrid[(long long)idx * 3 + 2] = (float)dgz;
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
            "parity_flag_reset", "parity_sample", "parity_bernoulli", "parity_poisson", "parity_binomial", "parity_multinomial",
            "parity_dropout_affine", "parity_rrelu", "parity_random_keys",
            "parity_nonzero_static_scatter", "parity_unravel_axis", "parity_slice_bitonic_step", "parity_slice_first_flags",
            "parity_slice_gather_unique", "parity_grid_sample_3d", "parity_scan_block", "parity_scan_add", "parity_int_binary", "parity_int_not", "parity_grid_sample_3d_backward",
        };
    }
}
