// Copyright (c) AiDotNet. All rights reserved.
// GLSL (Vulkan) port of the PyTorch-parity elementwise kernels (#1127), kernel for kernel with
// CUDA/Kernels/CudaTorchParityElementwiseKernels and OpenCL/OpenClTorchParityElementwiseKernels.
//
// GLSL has double arithmetic, sqrt, fma, floor and frexp / ldexp, but no double exp, log or trigonometry, and its float
// transcendentals carry loose error bounds (sin only to an absolute 2^-11 under SPIR-V's precision rules). The prelude
// below therefore implements those functions in double -- range reduction plus series that converge to double
// precision on the reduced interval -- and the special functions are ports of SpecialFunctions on top of them, so the
// results agree with the CPU's double reference to float rounding. Needs shaderFloat64 and shaderInt64; the kernels
// that report invalid arguments also need 64-bit buffer atomics. A device without a feature fails that pipeline and
// the op keeps its recorded host fallback.
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>GLSL compute shaders for the PyTorch-parity elementwise ops (one shader per kernel).</summary>
internal static class VulkanTorchParityElementwiseKernels
{
    /// <summary>Work-group size of every kernel, including the block scan.</summary>
    internal const int LocalSize = 256;

    private static readonly System.TimeSpan RegexTimeout = System.TimeSpan.FromSeconds(1);

    // An unsuffixed floating literal in GLSL is a float; every one in the prelude and kernel bodies below is meant as
    // a double, so they are suffixed lf when a shader is assembled (a literal already carrying f or lf is left alone).
    private static readonly Regex FloatingLiteral = new Regex(
        @"(?<![\w.])(\d+\.\d*(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])", RegexOptions.CultureInvariant, RegexTimeout);

    private static string AsDoubleLiterals(string source) => FloatingLiteral.Replace(source, m => m.Value + "lf");

    private const string Header = @"#version 450
#extension GL_ARB_gpu_shader_int64 : require
";

    private const string AtomicsHeader = @"#extension GL_EXT_shader_atomic_int64 : require
";

    /// <summary>
    /// Assembles one kernel: header, buffer bindings, the push-constant block (instance <c>pc</c>), the double prelude
    /// and the body (which defines <c>main</c>).
    /// </summary>
    private static string Kernel(string bindings, string pushConstants, string body, bool atomics = false)
        => Header + (atomics ? AtomicsHeader : string.Empty)
           + "layout(local_size_x = " + LocalSize.ToString(System.Globalization.CultureInfo.InvariantCulture) + ") in;\n"
           + bindings
           + "layout(push_constant) uniform P { " + pushConstants + " } pc;\n"
           + Prelude.Value + AsDoubleLiterals(body);

    private static readonly System.Lazy<string> Prelude = new System.Lazy<string>(() => AsDoubleLiterals(GaussLegendreTables() + LibmSource + SpecialSource));

    // The CPU quadratures' Gauss-Legendre rules (96 nodes on [0, pi], 32 per unit tail panel) as const arrays, written
    // at round-trip precision (AsDoubleLiterals makes them doubles).
    private static string GaussLegendreTables()
    {
        var sb = new System.Text.StringBuilder();
        foreach (int n in new[] { 96, 32 })
        {
            var (x, w) = AiDotNet.Tensors.Helpers.SpecialFunctions.GaussLegendreRule(n);
            string count = n.ToString(System.Globalization.CultureInfo.InvariantCulture);
            sb.Append("const double PARITY_GL").Append(count).Append("_X[").Append(count).Append("] = double[").Append(count).Append("](").Append(Join(x)).Append(");\n");
            sb.Append("const double PARITY_GL").Append(count).Append("_W[").Append(count).Append("] = double[").Append(count).Append("](").Append(Join(w)).Append(");\n");
        }
        return sb.ToString();
    }

    // "R" can print an exponent-only form (1E-05); give every value a decimal point so AsDoubleLiterals sees it.
    private static string Join(double[] values)
        => string.Join(", ", System.Linq.Enumerable.Select(values, v =>
        {
            string s = v.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
            if (s.IndexOf('E') >= 0 && s.IndexOf('.') < 0) s = s.Replace("E", ".0E");
            else if (s.IndexOf('.') < 0) s += ".0";
            return s;
        }));

    private const string LibmSource = @"
// ---- double libm ----
#define PARITY_PI 3.141592653589793
#define PARITY_SQRT_PI 1.7724538509055160273
#define PARITY_SQRT2 1.4142135623730950488
#define PARITY_EXP_UNDERFLOW 745.0
#define PARITY_LARGE_ARGUMENT 50.0

double d_nan() { return packDouble2x32(uvec2(0u, 0x7FF80000u)); }
double d_inf() { return packDouble2x32(uvec2(0u, 0x7FF00000u)); }
float f_nan() { return uintBitsToFloat(0x7FC00000u); }
bool d_signbit(double x) { return (unpackDouble2x32(x).y >> 31) != 0u; }

// Float bits to double, keeping subnormals: a plain float load may flush them to zero (AMD does), which turned
// logit / ndtri of a tiny input into -inf.
double f_in(uint bits)
{
    if (((bits >> 23) & 0xFFu) != 0u) return double(uintBitsToFloat(bits));
    double magnitude = double(bits & 0x7FFFFFu) * 1.401298464324817e-45;   // m * 2^-149, exact
    return (bits >> 31) != 0u ? -magnitude : magnitude;
}

// Double to float bits, rounding to nearest even into the subnormal range instead of flushing it to zero.
uint f_out(double v)
{
    if (isnan(v)) return 0x7FC00000u;
    uint sign = d_signbit(v) ? 0x80000000u : 0u;
    double a = abs(v);
    // Below the smallest normal float: the mantissa is a * 2^149 rounded (2^23 is exactly the smallest normal's bits).
    if (a < 1.1754943508222875e-38) return sign | uint(roundEven(a * 7.1362384635297994e44));
    return floatBitsToUint(float(v));
}

// e^x: x = k ln2 + r, |r| <= ln2 / 2, Taylor to r^16 (error below 1e-17 relative), then ldexp.
double d_exp(double x)
{
    if (isnan(x)) return x;
    if (x > 709.782712893384) return d_inf();
    if (x < -745.1332191019412) return 0.0;
    double k = roundEven(x * 1.4426950408889634);
    double r = (x - k * 0.6931471803691238) - k * 1.9082149292705877e-10;
    double term = 1.0, sum = 1.0;
    for (int i = 1; i <= 16; i++) { term *= r / double(i); sum += term; }
    int ki = int(k);
    // ldexp in two steps keeps the scale finite near the subnormal end.
    if (ki < -1000) return ldexp(ldexp(sum, ki + 1000), -1000);
    return ldexp(sum, ki);
}

// log x: x = m 2^e with m in [sqrt(1/2), sqrt(2)), log m = 2 atanh((m - 1) / (m + 1)) by its series.
double d_log(double x)
{
    if (isnan(x) || x < 0.0) return d_nan();
    if (x == 0.0) return -d_inf();
    if (isinf(x)) return x;
    int e;
    double m = frexp(x, e);
    if (m < 0.7071067811865476) { m *= 2.0; e -= 1; }
    double f = (m - 1.0) / (m + 1.0), s = f * f, term = f, sum = f;
    for (int i = 1; i <= 12; i++) { term *= s; sum += term / double(2 * i + 1); }
    return double(e) * 0.6931471803691238 + (2.0 * sum + double(e) * 1.9082149292705877e-10);
}

// log(1 + y), accurate for small y (Goldberg's correction, as SpecialFunctions.Log1p).
double d_log1p(double y) { double u = 1.0 + y; return u == 1.0 ? y : d_log(u) * y / (u - 1.0); }

// sin / cos: x = n pi/2 + r (three-part Cody-Waite pi/2), Taylor on |r| <= pi/4. Past |x| ~ 1e9 the reduction loses
// digits, but every caller then scales the result by a factor at most 1/|x|, so only boundedness matters there.
void d_reduce(double x, out double r, out int quadrant)
{
    double n = roundEven(x * 0.6366197723675814);
    r = ((x - n * 1.57079632673412561417) - n * 6.07710050630396597660e-11) - n * 2.02226624871116645580e-21;
    // Past |x| ~ 1e15 no double reduction means anything; return a bounded value (callers scale it by at most 1/|x|).
    // A reduced r a rounding step outside [-pi/4, pi/4] keeps its quadrant: the series are still exact there.
    if (abs(n) > 1e15) { r = 0.0; n = 0.0; }
    quadrant = int(mod(n, 4.0));
}

double d_sin_poly(double r)
{
    double r2 = r * r, term = r, sum = r;
    for (int i = 1; i <= 9; i++) { term *= -r2 / double((2 * i) * (2 * i + 1)); sum += term; }
    return sum;
}

double d_cos_poly(double r)
{
    double r2 = r * r, term = 1.0, sum = 1.0;
    for (int i = 1; i <= 9; i++) { term *= -r2 / double((2 * i - 1) * (2 * i)); sum += term; }
    return sum;
}

double d_sin(double x)
{
    if (isnan(x) || isinf(x)) return d_nan();
    double r; int q;
    d_reduce(x, r, q);
    if (q == 0) return d_sin_poly(r);
    if (q == 1) return d_cos_poly(r);
    if (q == 2) return -d_sin_poly(r);
    return -d_cos_poly(r);
}

double d_cos(double x)
{
    if (isnan(x) || isinf(x)) return d_nan();
    double r; int q;
    d_reduce(x, r, q);
    if (q == 0) return d_cos_poly(r);
    if (q == 1) return -d_sin_poly(r);
    if (q == 2) return -d_cos_poly(r);
    return d_sin_poly(r);
}

double d_tan(double x) { return d_sin(x) / d_cos(x); }
double d_sinh(double x) { double e = d_exp(x); return 0.5 * (e - 1.0 / e); }
double d_cosh(double x) { double e = d_exp(x); return 0.5 * (e + 1.0 / e); }

// x^y for the generators (logspace): exp(y log |x|), with the sign of a negative base raised to an integer power.
double d_pow(double x, double y)
{
    if (y == 0.0) return 1.0;
    if (isnan(x) || isnan(y)) return d_nan();
    if (x == 0.0) return y > 0.0 ? 0.0 : d_inf();
    if (x < 0.0)
    {
        if (y != floor(y)) return d_nan();
        double v = d_exp(y * d_log(-x));
        return mod(abs(y), 2.0) == 1.0 ? -v : v;
    }
    return d_exp(y * d_log(x));
}

double d_fmod(double x, double y) { return x - y * trunc(x / y); }

// log|Gamma(x)| (SpecialFunctions.LogGamma: Lanczos g = 7, reflection below 1/2, unrolled since GLSL forbids recursion).
double d_lanczos_lgamma(double x)
{
    x -= 1.0;
    double a = 0.99999999999980993, t = x + 7.5;
    a += 676.5203681218851 / (x + 1.0);
    a += -1259.1392167224028 / (x + 2.0);
    a += 771.32342877765313 / (x + 3.0);
    a += -176.61502916214059 / (x + 4.0);
    a += 12.507343278686905 / (x + 5.0);
    a += -0.13857109526572012 / (x + 6.0);
    a += 9.9843695780195716e-6 / (x + 7.0);
    a += 1.5056327351493116e-7 / (x + 8.0);
    return 0.5 * d_log(2.0 * PARITY_PI) + (x + 0.5) * d_log(t) - t + d_log(a);
}

double d_lgamma(double x)
{
    if (isnan(x)) return x;
    if (isinf(x)) return d_inf();
    if (x <= 0.0 && floor(x) == x) return d_inf();
    if (x < 0.5) return d_log(PARITY_PI / abs(d_sin(PARITY_PI * x))) - d_lanczos_lgamma(1.0 - x);
    return d_lanczos_lgamma(x);
}
";
    private const string SpecialSource = @"
// ---- special functions: ports of Helpers/SpecialFunctions ----

double sf_erf_series(double x)
{
    double x2 = x * x, term = x, sum = x;
    for (int k = 1; k < 200; k++)
    {
        term *= -x2 / double(k);
        double add = term / double(2 * k + 1);
        sum += add;
        if (abs(add) < 1e-17 * abs(sum)) break;
    }
    return 2.0 / PARITY_SQRT_PI * sum;
}

double sf_erfc_continued_fraction(double x)
{
    const double tiny = 1e-300;
    double f = x, c = x, d = 0.0;
    for (int k = 1; k < 500; k++)
    {
        double a = double(k) / 2.0;
        d = x + a * d;
        d = abs(d) < tiny ? tiny : d;
        c = x + a / c;
        c = abs(c) < tiny ? tiny : c;
        d = 1.0 / d;
        double delta = c * d;
        f *= delta;
        if (abs(delta - 1.0) < 1e-16) break;
    }
    return 1.0 / f;
}

double sf_erfc_large(double x) { return d_exp(-x * x) * sf_erfc_continued_fraction(x) / PARITY_SQRT_PI; }

double sf_erfc_mid(double x)
{
    double mid = 0.5 * (x + 2.5), halfWidth = 0.5 * (2.5 - x), sum = 0.0;
    for (int k = 0; k < 32; k++) { double t = mid + halfWidth * PARITY_GL32_X[k]; sum += PARITY_GL32_W[k] * d_exp(-t * t); }
    return sf_erfc_large(2.5) + 2.0 / PARITY_SQRT_PI * sum * halfWidth;
}

double sf_erfc(double x)
{
    if (isnan(x)) return x;
    if (isinf(x)) return x > 0.0 ? 0.0 : 2.0;
    if (x >= 2.5) return sf_erfc_large(x);
    if (x >= 0.5) return sf_erfc_mid(x);
    if (x <= -0.5) { double ax = -x; return 2.0 - (ax >= 2.5 ? sf_erfc_large(ax) : sf_erfc_mid(ax)); }
    return 1.0 - sf_erf_series(x);
}

double sf_erf(double x)
{
    if (isnan(x)) return x;
    if (isinf(x)) return x > 0.0 ? 1.0 : -1.0;
    if (abs(x) < 0.5) return sf_erf_series(x);
    double tail = sf_erfc(abs(x));
    return x > 0.0 ? 1.0 - tail : tail - 1.0;
}

double sf_erfcx(double x)
{
    if (isnan(x)) return x;
    if (isinf(x) && x > 0.0) return 0.0;
    if (x >= 2.5) return sf_erfc_continued_fraction(x) / PARITY_SQRT_PI;
    if (x >= 0.5) return d_exp(x * x) * sf_erfc_mid(x);
    if (x > -0.5) return d_exp(x * x) * (1.0 - sf_erf_series(x));
    if (x < -26.7) return d_inf();
    double ax = -x, positive = ax >= 2.5 ? sf_erfc_continued_fraction(ax) / PARITY_SQRT_PI : d_exp(ax * ax) * sf_erfc_mid(ax);
    return 2.0 * d_exp(x * x) - positive;
}

double sf_ndtr(double x) { return 0.5 * sf_erfc(-x / PARITY_SQRT2); }

double sf_log_ndtr(double x)
{
    if (isnan(x)) return x;
    if (isinf(x) && x < 0.0) return -d_inf();
    if (x > 0.0) return d_log1p(-sf_ndtr(-x));
    if (x > -1.0) return d_log(sf_ndtr(x));
    return d_log(0.5 * sf_erfcx(-x / PARITY_SQRT2)) - 0.5 * x * x;
}

double sf_ndtri(double p)
{
    if (isnan(p) || p < 0.0 || p > 1.0) return d_nan();
    if (p == 0.0) return -d_inf();
    if (p == 1.0) return d_inf();
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
        double q = sqrt(-2.0 * d_log(p));
        x = (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0);
    }
    else if (p > 1.0 - 0.02425)
    {
        double q = sqrt(-2.0 * d_log(1.0 - p));
        x = -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0);
    }
    else
    {
        double q = p - 0.5, r = q * q;
        x = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q / (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0);
    }
    for (int k = 0; k < 2; k++)
    {
        double e = sf_ndtr(x) - p;
        double u = e * sqrt(2.0 * PARITY_PI) * d_exp(x * x / 2.0);
        x -= u / (1.0 + x * u / 2.0);
    }
    return x;
}

double sf_bessel_series(int order, double x, bool alternating)
{
    double halfX = x / 2.0, term = 1.0;
    for (int k = 1; k <= order; k++) term *= halfX / double(k);
    double sum = term, q = halfX * halfX * (alternating ? -1.0 : 1.0);
    for (int k = 1; k < 60; k++)
    {
        term *= q / (double(k) * double(k + order));
        sum += term;
        if (abs(term) <= 1e-17 * abs(sum)) break;
    }
    return sum;
}

double sf_modified_bessel_asymptotic(double mu, double x, bool alternating)
{
    double term = 1.0, sum = 1.0, last = 1.7976931348623157e308;
    for (int k = 1; k < 200; k++)
    {
        term *= (mu - (2.0 * double(k) - 1.0) * (2.0 * double(k) - 1.0)) / (8.0 * double(k) * x) * (alternating ? -1.0 : 1.0);
        if (abs(term) > last) break;
        last = abs(term);
        sum += term;
        if (abs(term) < 1e-17 * abs(sum)) break;
    }
    return sum;
}

// Hankel's asymptotic expansion: J (which = 0) or Y (which = 1) for x > 25.
double sf_hankel_asymptotic(int order, double x, int which)
{
    double mu = 4.0 * double(order * order), p = 1.0, q = 0.0, term = 1.0, lastP = 1.7976931348623157e308;
    for (int k = 1; k < 60; k++)
    {
        term *= (mu - double((2 * k - 1) * (2 * k - 1))) / (double(k) * 8.0 * x);
        if (abs(term) > lastP) break;
        lastP = abs(term);
        if (k % 2 == 1) q += ((k / 2) % 2 == 0 ? 1.0 : -1.0) * term;
        else p += ((k / 2) % 2 == 0 ? 1.0 : -1.0) * term;
        if (abs(term) < 1e-17) break;
    }
    double phi = (double(order) / 2.0 + 0.25) * PARITY_PI, s = sqrt(2.0 / (PARITY_PI * x));
    double cx = d_cos(x), sx = d_sin(x), cp = d_cos(phi), sp = d_sin(phi);
    double cosChi = cx * cp + sx * sp, sinChi = sx * cp - cx * sp;
    return which == 0 ? s * (p * cosChi - q * sinChi) : s * (p * sinChi + q * cosChi);
}

double sf_bessel_j(int order, double x)
{
    if (isnan(x)) return x;
    double ax = abs(x);
    if (isinf(ax)) return 0.0;
    if (ax < 2.0) return sf_bessel_series(order, x, true);
    if (ax > 25.0) return (order % 2 == 1 && x < 0.0 ? -1.0 : 1.0) * sf_hankel_asymptotic(order, ax, 0);
    int points = 2 * int(ceil(ax)) + 48;
    double sum = 0.0;
    for (int k = 0; k < points; k++)
    {
        double tau = 2.0 * PARITY_PI * double(k) / double(points);
        sum += d_cos(double(order) * tau - x * d_sin(tau));
    }
    return sum / double(points);
}

double sf_bessel_y(int order, double x)
{
    if (isnan(x) || x < 0.0) return d_nan();
    if (x == 0.0) return -d_inf();
    if (isinf(x)) return 0.0;
    if (x > 25.0) return sf_hankel_asymptotic(order, x, 1);
    double first = 0.0;
    for (int k = 0; k < 96; k++)
    {
        double tau = 0.5 * PARITY_PI + 0.5 * PARITY_PI * PARITY_GL96_X[k];
        first += PARITY_GL96_W[k] * d_sin(x * d_sin(tau) - double(order) * tau);
    }
    first *= 0.5 * PARITY_PI;
    double sign = order % 2 == 0 ? 1.0 : -1.0, second = 0.0;
    for (int panel = 0; panel < 2000; panel++)
    {
        if (x * d_sinh(double(panel)) - double(order * panel) > PARITY_EXP_UNDERFLOW) break;
        double part = 0.0;
        for (int k = 0; k < 32; k++)
        {
            double t = double(panel) + 0.5 + 0.5 * PARITY_GL32_X[k];
            part += PARITY_GL32_W[k] * (d_exp(double(order) * t) + sign * d_exp(-double(order) * t)) * d_exp(-x * d_sinh(t));
        }
        second += 0.5 * part;
    }
    return (first - second) / PARITY_PI;
}

double sf_bessel_i_scaled(int order, double x)
{
    if (isnan(x)) return x;
    double ax = abs(x), sign = order % 2 == 1 && x < 0.0 ? -1.0 : 1.0;
    if (ax < 2.0) return sign * d_exp(-ax) * sf_bessel_series(order, ax, false);
    if (ax > PARITY_LARGE_ARGUMENT) return sign * sf_modified_bessel_asymptotic(4.0 * double(order * order), ax, true) / sqrt(2.0 * PARITY_PI * ax);
    int points = 40 + int(ceil(sqrt(80.0 * ax)));
    double sum = 0.0;
    for (int k = 0; k < points; k++)
    {
        double tau = 2.0 * PARITY_PI * double(k) / double(points);
        sum += d_exp(ax * (d_cos(tau) - 1.0)) * d_cos(double(order) * tau);
    }
    return sign * sum / double(points);
}

double sf_bessel_i(int order, double x)
{
    if (isinf(x)) return order % 2 == 1 && x < 0.0 ? -d_inf() : d_inf();
    return sf_bessel_i_scaled(order, x) * d_exp(abs(x));
}

double sf_bessel_k_scaled(double nu, double x)
{
    if (isnan(x) || x < 0.0) return d_nan();
    if (x == 0.0) return d_inf();
    if (x > PARITY_LARGE_ARGUMENT) return sqrt(PARITY_PI / (2.0 * x)) * sf_modified_bessel_asymptotic(4.0 * nu * nu, x, false);
    const double h = 0.02;
    double sum = 0.5;
    for (int k = 1; k < 100000; k++)
    {
        double t = double(k) * h;
        double ct = d_cosh(t) - 1.0;
        if (x * ct - abs(nu) * t > PARITY_EXP_UNDERFLOW) break;
        sum += d_exp(-x * ct) * d_cosh(nu * t);
    }
    return sum * h;
}

double sf_bessel_k(double nu, double x) { return (isinf(x) && x > 0.0) ? 0.0 : sf_bessel_k_scaled(nu, x) * d_exp(-x); }

double sf_bessel_j_fractional(double nu, double x)
{
    double first = 0.0;
    for (int k = 0; k < 96; k++)
    {
        double tau = 0.5 * PARITY_PI + 0.5 * PARITY_PI * PARITY_GL96_X[k];
        first += PARITY_GL96_W[k] * d_cos(nu * tau - x * d_sin(tau));
    }
    first *= 0.5 * PARITY_PI;
    double second = 0.0;
    for (int panel = 0; panel < 2000; panel++)
    {
        if (x * d_sinh(double(panel)) + nu * double(panel) > PARITY_EXP_UNDERFLOW) break;
        double part = 0.0;
        for (int k = 0; k < 32; k++)
        {
            double t = double(panel) + 0.5 + 0.5 * PARITY_GL32_X[k];
            part += PARITY_GL32_W[k] * d_exp(-x * d_sinh(t) - nu * t);
        }
        second += 0.5 * part;
    }
    return (first - d_sin(nu * PARITY_PI) * second) / PARITY_PI;
}

void sf_two_product(double a, double b, out double p, out double e) { p = a * b; e = fma(a, b, -p); }

void sf_zeta_double_double(double z, out double hi, out double lo)
{
    double root = sqrt(z), sq, sqErr;
    sf_two_product(root, root, sq, sqErr);
    double rootLo = ((z - sq) - sqErr) / (2.0 * root);
    double m, mErr;
    sf_two_product(z, root, m, mErr);
    double mLo = mErr + z * rootLo;
    const double twoThirds = 2.0 / 3.0;
    double three, threeErr;
    sf_two_product(3.0, twoThirds, three, threeErr);
    double twoThirdsLo = ((2.0 - three) - threeErr) / 3.0;
    double p, pErr;
    sf_two_product(m, twoThirds, p, pErr);
    double pLo = pErr + m * twoThirdsLo + mLo * twoThirds;
    hi = p + pLo;
    lo = pLo - (hi - p);
}

double sf_airy_ai(double x)
{
    if (isnan(x)) return x;
    if (isinf(x)) return 0.0;
    if (x >= -5.0 && x <= 1.0)
    {
        double x3 = x * x * x, f = 1.0, g = x, tf = 1.0, tg = x;
        for (int k = 1; k < 200; k++)
        {
            double kk = double(k);
            tf *= x3 / ((3.0 * kk - 1.0) * (3.0 * kk));
            tg *= x3 / ((3.0 * kk) * (3.0 * kk + 1.0));
            f += tf;
            g += tg;
            if (abs(tf) + abs(tg) < 1e-18 * (abs(f) + abs(g))) break;
        }
        return 0.355028053887817239 * f - 0.258819403792806798 * g;
    }
    double z = abs(x), zeta = 2.0 / 3.0 * z * sqrt(z);
    if (x > 0.0) return sqrt(x / 3.0) / PARITY_PI * sf_bessel_k(1.0 / 3.0, zeta);
    if (z <= 15.0) return sqrt(z) / 3.0 * (sf_bessel_j_fractional(1.0 / 3.0, zeta) + sf_bessel_j_fractional(-1.0 / 3.0, zeta));
    double u = 1.0, even = 1.0, odd = 0.0, power = 1.0, last = 1.7976931348623157e308;
    for (int k = 1; k < 60; k++)
    {
        double kk = double(k);
        u *= (6.0 * kk - 5.0) * (6.0 * kk - 3.0) * (6.0 * kk - 1.0) / ((2.0 * kk - 1.0) * 216.0 * kk);
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
    sf_zeta_double_double(z, hi, lo);
    double sh = d_sin(hi), ch = d_cos(hi), sl = d_sin(lo + PARITY_PI / 4.0), cl = d_cos(lo + PARITY_PI / 4.0);
    double sinPhase = sh * cl + ch * sl, cosPhase = ch * cl - sh * sl;
    return (sinPhase * even - cosPhase * odd) / (sqrt(PARITY_PI) * sqrt(sqrt(z)));
}

double sf_igamma_series(double a, double x)
{
    double ap = a, sum = 1.0 / a, inc = sum;
    for (int k = 0; k < 1000; k++)
    {
        ap += 1.0;
        inc *= x / ap;
        sum += inc;
        if (abs(inc) < abs(sum) * 1e-16) break;
    }
    return sum * d_exp(-x + a * d_log(x) - d_lgamma(a));
}

double sf_igamma_continued_fraction(double a, double x)
{
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int k = 1; k < 1000; k++)
    {
        double an = -double(k) * (double(k) - a);
        b += 2.0;
        d = an * d + b;
        if (abs(d) < tiny) d = tiny;
        c = b + an / c;
        if (abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        double ratio = d * c;
        h *= ratio;
        if (abs(ratio - 1.0) < 1e-16) break;
    }
    return d_exp(-x + a * d_log(x) - d_lgamma(a)) * h;
}

double sf_sinpi(double x)
{
    if (isnan(x) || isinf(x)) return d_nan();
    double r = x - 2.0 * roundEven(x * 0.5);
    double a = abs(r);
    if (a == 0.0 || a == 1.0) return 0.0;
    if (a > 0.5) a = 1.0 - a;
    return (r < 0.0 ? -1.0 : 1.0) * d_sin(PARITY_PI * a);
}

double sf_clamp_nan(double v, double lo, double hi) { return (isnan(v) || isnan(lo) || isnan(hi)) ? d_nan() : (v < lo ? lo : (v > hi ? hi : v)); }

// unpackDouble2x32 order: the low word first.
double pc_double(uvec2 bits) { return packDouble2x32(bits); }
";
    private const string UnaryBindings = @"
layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Y { float y[]; };
";

    // The special functions read and write raw float bits so subnormal inputs and results survive (see f_in / f_out).
    private const string UnaryBitsBindings = @"
layout(std430, set = 0, binding = 0) readonly buffer X { uint xbits[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Y { uint ybits[]; };
";

    private const string BinaryBindings = @"
layout(std430, set = 0, binding = 0) readonly buffer A { float a[]; };
layout(std430, set = 0, binding = 1) readonly buffer B { float b[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Y { float y[]; };
";

    private const string OutputBinding = @"
layout(std430, set = 0, binding = 0) writeonly buffer Y { float y[]; };
";

    private const string Index = "int i = int(gl_GlobalInvocationID.x); if (i >= pc.n) return;\n";

    // The unary kernels: y[i] = (expr)(v = x[i], a0 = p0, a1 = p1), evaluated in double.
    private static readonly System.Collections.Generic.Dictionary<string, string> UnaryExpressions =
        new System.Collections.Generic.Dictionary<string, string>
        {
            ["parity_celu"] = "v > 0.0 ? v : a0 * (d_exp(v / a0) - 1.0)",
            ["parity_hardtanh"] = "sf_clamp_nan(v, a0, a1)",
            ["parity_logsigmoid"] = "isnan(v) ? v : (v < 0.0 ? v : 0.0) - d_log1p(d_exp(-abs(v)))",
            ["parity_softsign"] = "v / (1.0 + abs(v))",
            ["parity_logit"] = "d_log(sf_clamp_nan(v, a0, a1) / (1.0 - sf_clamp_nan(v, a0, a1)))",
            ["parity_sinc"] = "v == 0.0 ? 1.0 : (isinf(v) ? 0.0 : sf_sinpi(v) / (PARITY_PI * v))",
            ["parity_erf"] = "sf_erf(v)",
            ["parity_bessel_j0"] = "sf_bessel_j(0, v)",
            ["parity_bessel_j1"] = "sf_bessel_j(1, v)",
            ["parity_bessel_y0"] = "sf_bessel_y(0, v)",
            ["parity_bessel_y1"] = "sf_bessel_y(1, v)",
            ["parity_bessel_i0"] = "sf_bessel_i(0, v)",
            ["parity_bessel_i1"] = "sf_bessel_i(1, v)",
            ["parity_erfcx"] = "sf_erfcx(v)",
            ["parity_ndtr"] = "sf_ndtr(v)",
            ["parity_ndtri"] = "sf_ndtri(v)",
            ["parity_entr"] = "isnan(v) ? v : (v > 0.0 ? -v * d_log(v) : (v == 0.0 ? 0.0 : -d_inf()))",
            ["parity_signbit"] = "d_signbit(v) ? 1.0 : 0.0",
            ["parity_isposinf"] = "(isinf(v) && v > 0.0) ? 1.0 : 0.0",
            ["parity_isneginf"] = "(isinf(v) && v < 0.0) ? 1.0 : 0.0",
            ["parity_log_ndtr"] = "sf_log_ndtr(v)",
            ["parity_bessel_k0"] = "sf_bessel_k(0.0, v)",
            ["parity_bessel_k1"] = "sf_bessel_k(1.0, v)",
            ["parity_bessel_k0e"] = "sf_bessel_k_scaled(0.0, v)",
            ["parity_bessel_k1e"] = "sf_bessel_k_scaled(1.0, v)",
            ["parity_spherical_bessel_j0"] = "isnan(v) ? v : isinf(v) ? 0.0 : abs(v) < 1e-4 ? 1.0 - v * v / 6.0 : d_sin(v) / v",
            ["parity_airy_ai"] = "sf_airy_ai(v)",
            ["parity_angle"] = "isnan(v) ? v : (v < 0.0 ? PARITY_PI : 0.0)",
            ["parity_nonzero"] = "v != 0.0 ? 1.0 : 0.0",
        };

    private static string UnaryKernel(string expression) => Kernel(UnaryBitsBindings, "int n; float p0; float p1;",
        "void main() { " + Index + "double v = f_in(xbits[i]); double a0 = double(pc.p0); double a1 = double(pc.p1); ybits[i] = f_out(" + expression + "); }\n");

    private static readonly string Mvlgamma = Kernel(UnaryBitsBindings, "int n; float p0; float p1;", @"
void main()
{
    " + Index + @"
    double v = f_in(xbits[i]);
    int p = int(pc.p0);
    if (v <= double(p - 1) / 2.0) { ybits[i] = 0x7FC00000u; return; }
    double sum = double(p) * (double(p) - 1.0) / 4.0 * d_log(PARITY_PI);
    for (int j = 0; j < p; j++) sum += d_lgamma(v - double(j) / 2.0);
    ybits[i] = f_out(sum);
}
");

    private static string BinaryKernel(string body) => Kernel(BinaryBindings, "int n;", "void main() { " + Index + body + " }\n");

    // fmax / fmin ignore a NaN operand; GLSL's max / min leave that undefined, so the choice is explicit.
    private static readonly System.Collections.Generic.Dictionary<string, string> BinaryBodies =
        new System.Collections.Generic.Dictionary<string, string>
        {
            ["parity_fmax"] = "float p = a[i], q = b[i]; y[i] = isnan(p) ? q : (isnan(q) ? p : max(p, q));",
            ["parity_fmin"] = "float p = a[i], q = b[i]; y[i] = isnan(p) ? q : (isnan(q) ? p : min(p, q));",
            ["parity_heaviside"] = "float t = a[i]; y[i] = isnan(t) ? t : (t < 0.0f ? 0.0f : (t > 0.0f ? 1.0f : b[i]));",
            ["parity_greater_equal"] = "y[i] = a[i] >= b[i] ? 1.0f : 0.0f;",
            ["parity_less_equal"] = "y[i] = a[i] <= b[i] ? 1.0f : 0.0f;",
            ["parity_floor_divide"] = "y[i] = float(floor(double(a[i]) / double(b[i])));",
            ["parity_igamma"] = @"double av = double(a[i]), v = double(b[i]), r;
                if (isnan(av) || isnan(v) || av <= 0.0 || v < 0.0) r = d_nan();
                else if (v == 0.0) r = 0.0;
                else if (isinf(v)) r = 1.0;
                else r = v < av + 1.0 ? sf_igamma_series(av, v) : 1.0 - sf_igamma_continued_fraction(av, v);
                y[i] = float(r);",
            ["parity_igammac"] = @"double av = double(a[i]), v = double(b[i]), r;
                if (isnan(av) || isnan(v) || av <= 0.0 || v < 0.0) r = d_nan();
                else if (v == 0.0) r = 1.0;
                else if (isinf(v)) r = 0.0;
                else r = v < av + 1.0 ? 1.0 - sf_igamma_series(av, v) : sf_igamma_continued_fraction(av, v);
                y[i] = float(r);",
        };

    private static readonly string Polynomial = Kernel(BinaryBindings, "int n; int kind; int shifted;", @"
void main()
{
    " + Index + @"
    double d = double(b[i]);
    if (isnan(d)) { y[i] = f_nan(); return; }
    double t = trunc(d);
    if (t < 0.0) { y[i] = 0.0f; return; }
    if (t > 1048576.0) { y[i] = f_nan(); return; }
    int deg = int(t);
    double v = double(a[i]);
    if (isnan(v)) { y[i] = f_nan(); return; }
    if (pc.shifted != 0) v = 2.0 * v - 1.0;
    double previous = 1.0, current;
    switch (pc.kind)
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
    for (int k = 1; k < deg; k++)
    {
        double kk = double(k), next;
        switch (pc.kind)
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
    y[i] = float(current);
}
");
    // ---- generators. Doubles and 64-bit values travel as lo / hi uint pairs, and every pair sits FIRST in its push block,
    // at an 8-byte-aligned offset: AMD's compiler merges a pair into one aligned 64-bit load, so a pair at an odd word
    // offset read the 8 bytes starting one word early. ----

    private static readonly string Range = Kernel(OutputBinding, "uint startLo; uint startHi; uint spanLo; uint spanHi; uint denLo; uint denHi; int n;", @"
void main()
{
    " + Index + @"
    double start = pc_double(uvec2(pc.startLo, pc.startHi)), span = pc_double(uvec2(pc.spanLo, pc.spanHi)), den = pc_double(uvec2(pc.denLo, pc.denHi));
    y[i] = float(start + double(i) * span / den);
}
");

    private static readonly string Logspace = Kernel(OutputBinding,
        "uint startLo; uint startHi; uint spanLo; uint spanHi; uint denLo; uint denHi; uint baseLo; uint baseHi; int n; int single;", @"
void main()
{
    " + Index + @"
    double start = pc_double(uvec2(pc.startLo, pc.startHi)), span = pc_double(uvec2(pc.spanLo, pc.spanHi)), den = pc_double(uvec2(pc.denLo, pc.denHi));
    double base = pc_double(uvec2(pc.baseLo, pc.baseHi));
    y[i] = float(d_pow(base, pc.single != 0 ? start : start + double(i) * span / den));
}
");

    private static readonly string Window = Kernel(OutputBinding, "uint alphaLo; uint alphaHi; uint betaLo; uint betaHi; int n; int kind; int m;", @"
void main()
{
    " + Index + @"
    if (pc.n == 1) { y[i] = 1.0f; return; }
    double alpha = pc_double(uvec2(pc.alphaLo, pc.alphaHi)), beta = pc_double(uvec2(pc.betaLo, pc.betaHi));
    double t = double(i) / double(pc.m - 1), v;
    switch (pc.kind)
    {
        case 0: v = 0.5 - 0.5 * d_cos(2.0 * PARITY_PI * t); break;
        case 1: v = alpha - beta * d_cos(2.0 * PARITY_PI * t); break;
        case 2: v = 0.42 - 0.5 * d_cos(2.0 * PARITY_PI * t) + 0.08 * d_cos(4.0 * PARITY_PI * t); break;
        case 3: v = 1.0 - abs(2.0 * t - 1.0); break;
        default:
        {
            double r = 2.0 * t - 1.0;
            v = sf_bessel_i(0, alpha * sqrt(max(0.0, 1.0 - r * r))) / sf_bessel_i(0, alpha);
            break;
        }
    }
    y[i] = float(v);
}
");

    private static readonly string TriIndices = Kernel(OutputBinding, "int row; int col; int offset; int lower; int total;", @"
void tri_columns(int r, out int lo, out int hi)
{
    int edge = r + pc.offset;
    if (pc.lower != 0) { lo = 0; hi = min(edge, pc.col - 1); }
    else { lo = max(edge, 0); hi = pc.col - 1; }
}
void main()
{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.row) return;
    int lo, hi, at = 0;
    for (int r = 0; r < i; r++) { tri_columns(r, lo, hi); if (hi >= lo) at += hi - lo + 1; }
    tri_columns(i, lo, hi);
    for (int j = lo; j <= hi; j++, at++) { y[at] = float(i); y[pc.total + at] = float(j); }
}
");

    private static readonly string Diagflat = Kernel(UnaryBindings, "int total; int size; int offset; int n;", @"
void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= pc.total) return;
    int r = idx / pc.size, c = idx % pc.size;
    int k = pc.offset >= 0 ? (c - r == pc.offset ? r : -1) : (r - c == -pc.offset ? c : -1);
    y[idx] = (k >= 0 && k < pc.n) ? x[k] : 0.0f;
}
");

    // geometry: [0] other rank, [1..8] extents, [9..16] strides, [17..21] stride1, stride2, start1, start2, diagonal.
    private static readonly string DiagonalScatter = Kernel(@"
layout(std430, set = 0, binding = 0) readonly buffer S { float src[]; };
layout(std430, set = 0, binding = 1) buffer Y { float y[]; };
layout(std430, set = 0, binding = 2) readonly buffer G { int g[]; };
", "int count;", @"
void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= pc.count) return;
    int diagonal = g[21];
    int o = idx / diagonal, t = idx % diagonal;
    int base = 0, rest = o;
    for (int j = g[0] - 1; j >= 0; j--) { base += rest % g[1 + j] * g[9 + j]; rest /= g[1 + j]; }
    y[base + (t + g[19]) * g[17] + (t + g[20]) * g[18]] = src[idx];
}
");

    private static readonly string QuantileRows = Kernel(UnaryBindings, "uint qLo; uint qHi; int rows; int rowLength; int mode; int ignoreNan;", @"
void main()
{
    int r = int(gl_GlobalInvocationID.x);
    if (r >= pc.rows) return;
    int baseIndex = r * pc.rowLength;
    int lo = 0, hi = pc.rowLength;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (isnan(x[baseIndex + mid])) hi = mid; else lo = mid + 1; }
    int count = lo;
    if (count == 0 || (count < pc.rowLength && pc.ignoreNan == 0)) { y[r] = f_nan(); return; }
    double position = pc_double(uvec2(pc.qLo, pc.qHi)) * double(count - 1);
    int below = int(floor(position)), above = int(ceil(position));
    double w = position - double(below);
    switch (pc.mode)
    {
        case 1: above = below; w = 0.0; break;
        case 2: below = above; w = 0.0; break;
        case 3: below = int(roundEven(position)); above = below; w = 0.0; break;
        case 4: w = below == above ? 0.0 : 0.5; break;
        default: break;
    }
    y[r] = float((1.0 - w) * double(x[baseIndex + below]) + w * double(x[baseIndex + above]));
}
");
    // ---- random sampling: counter-based Philox4x32-10, bit-identical to the CUDA and OpenCL streams ----

    private const string RandomSource = @"
uint rng_key0, rng_key1, rng_draw;
uint rng_element_lo, rng_element_hi;

void rng_init(uint seedLo, uint seedHi, uint element) { rng_key0 = seedLo; rng_key1 = seedHi; rng_element_lo = element; rng_element_hi = 0u; rng_draw = 0u; }

double rng_uniform()
{
    uint c0 = rng_element_lo, c1 = rng_element_hi, c2 = rng_draw, c3 = 0x5EEDu;
    rng_draw += 1u;
    uint k0 = rng_key0, k1 = rng_key1;
    for (int round = 0; round < 10; round++)
    {
        uint hi0, lo0, hi1, lo1;
        umulExtended(0xD2511F53u, c0, hi0, lo0);
        umulExtended(0xCD9E8D57u, c2, hi1, lo1);
        uint n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return (double(c0 >> 5) * 67108864.0 + double(c1 >> 6)) * (1.0 / 9007199254740992.0);
}

double rng_standard_normal()
{
    double u1 = rng_uniform(), u2 = rng_uniform();
    return sqrt(-2.0 * d_log(1.0 - u1)) * d_cos(2.0 * PARITY_PI * u2);
}
";

    // The 64-bit validation flag, lowered to (index << 3 | code) with atomicMin.
    private const string FlagBinding = "layout(std430, set = 0, binding = {0}) buffer F {{ uint64_t flag; }};\n";

    private static string Flag(int binding) => string.Format(System.Globalization.CultureInfo.InvariantCulture, FlagBinding, binding);

    private const string FlagSource = @"
void raise_flag(uint64_t element, int code) { atomicMin(flag, (element << 3) | uint64_t(code)); }
";

    private static string RandomKernel(string bindings, string pushConstants, string body, bool atomics = false)
        => Kernel(bindings, pushConstants, RandomSource + (atomics ? FlagSource : string.Empty) + body, atomics);

    private static readonly string FlagReset = Kernel(Flag(0), "int n;", "void main() { if (gl_GlobalInvocationID.x == 0u) flag = 0xFFFFFFFFFFFFFFFFul; }\n", atomics: true);

    private static readonly string Sample = RandomKernel(OutputBinding,
        "uint aLo; uint aHi; uint bLo; uint bHi; uint seedLo; uint seedHi; int n; int kind;", @"
void main()
{
    " + Index + @"
    double a = pc_double(uvec2(pc.aLo, pc.aHi)), b = pc_double(uvec2(pc.bLo, pc.bHi));
    rng_init(pc.seedLo, pc.seedHi, uint(i));
    double v;
    switch (pc.kind)
    {
        case 0: v = a + (b - a) * rng_uniform(); break;
        case 1: v = a + b * d_tan(PARITY_PI * (rng_uniform() - 0.5)); break;
        case 2: v = -d_log(1.0 - rng_uniform()) / a; break;
        case 3: v = a == 1.0 ? 1.0 : max(1.0, ceil(d_log(1.0 - rng_uniform()) / d_log1p(-a))); break;
        case 4: v = d_exp(a + b * rng_standard_normal()); break;
        case 5: v = rng_standard_normal(); break;
        default: v = a + min(floor(rng_uniform() * b), b - 1.0); break;
    }
    y[i] = float(v);
}
");

    private static readonly string Bernoulli = RandomKernel(UnaryBindings + Flag(2), "uint seedLo; uint seedHi; int n;", @"
void main()
{
    " + Index + @"
    double q = double(x[i]);
    if (!(q >= 0.0 && q <= 1.0)) { raise_flag(uint64_t(i), 1); return; }
    rng_init(pc.seedLo, pc.seedHi, uint(i));
    y[i] = rng_uniform() < q ? 1.0f : 0.0f;
}
", atomics: true);

    private static readonly string Poisson = RandomKernel(UnaryBindings + Flag(2), "uint seedLo; uint seedHi; int n;", @"
void main()
{
    " + Index + @"
    double rate = double(x[i]);
    if (!(rate >= 0.0)) { raise_flag(uint64_t(i), 1); return; }
    if (rate == 0.0) { y[i] = 0.0f; return; }
    rng_init(pc.seedLo, pc.seedHi, uint(i));
    if (rate < 10.0)
    {
        double limit = d_exp(-rate), product = rng_uniform();
        int k = 0;
        while (product > limit) { k++; product *= rng_uniform(); }
        y[i] = float(k);
        return;
    }
    double slam = sqrt(rate), logLam = d_log(rate);
    double bb = 0.931 + 2.53 * slam, aa = -0.059 + 0.02483 * bb, invAlpha = 1.1239 + 1.1328 / (bb - 3.4), vr = 0.9277 - 3.6224 / (bb - 2.0);
    while (true)
    {
        double u = rng_uniform() - 0.5, v = rng_uniform(), us = 0.5 - abs(u);
        double k = floor((2.0 * aa / us + bb) * u + rate + 0.43);
        if (us >= 0.07 && v <= vr) { y[i] = float(k); return; }
        if (k < 0.0 || (us < 0.013 && v > us)) continue;
        if (d_log(v) + d_log(invAlpha) - d_log(aa / (us * us) + bb) <= -rate + k * logLam - d_lgamma(k + 1.0)) { y[i] = float(k); return; }
    }
}
", atomics: true);

    private static readonly string Binomial = RandomKernel(BinaryBindings + Flag(3), "uint seedLo; uint seedHi; int n;", @"
double stirling_tail(double k)
{
    const double table[10] = double[10](0.08106146679532726, 0.04134069595540929, 0.02767792568499834, 0.02079067210376509,
        0.01664469118982119, 0.01387612882307075, 0.01189670994589177, 0.01041126526197209, 0.009255462182712733, 0.008330563433362871);
    if (k <= 9.0) return table[int(k)];
    double kp1sq = (k + 1.0) * (k + 1.0);
    return (1.0 / 12.0 - (1.0 / 360.0 - 1.0 / 1260.0 / kp1sq) / kp1sq) / (k + 1.0);
}
void main()
{
    " + Index + @"
    double count = double(a[i]), p = double(b[i]);
    if (isnan(count) || isinf(count) || count != floor(count) || count >= 9.2233720368547758E18) { raise_flag(uint64_t(i), 1); return; }
    int64_t n = int64_t(count);
    if (n < 0l || !(p >= 0.0 && p <= 1.0)) { raise_flag(uint64_t(i), 2); return; }
    if (n == 0l || p == 0.0) { y[i] = 0.0f; return; }
    if (p == 1.0) { y[i] = float(n); return; }
    rng_init(pc.seedLo, pc.seedHi, uint(i));
    bool flip = p > 0.5;
    double q = flip ? 1.0 - p : p, draw, dn = double(n);
    if (dn * q >= 10.0)
    {
        double stddev = sqrt(dn * q * (1.0 - q)), bb = 1.15 + 2.53 * stddev, aa = -0.0873 + 0.0248 * bb + 0.01 * q;
        double c = dn * q + 0.5, vR = 0.92 - 4.2 / bb, ratio = q / (1.0 - q), alpha = (2.83 + 5.1 / bb) * stddev;
        double m = floor((dn + 1.0) * q);
        while (true)
        {
            double u = rng_uniform() - 0.5, v = rng_uniform(), us = 0.5 - abs(u);
            double k = floor((2.0 * aa / us + bb) * u + c);
            if (k < 0.0 || k > dn) continue;
            if (us >= 0.07 && v <= vR) { draw = k; break; }
            v = d_log(v * alpha / (aa / (us * us) + bb));
            double upper = (m + 0.5) * d_log((m + 1.0) / (ratio * (dn - m + 1.0)))
                + (dn + 1.0) * d_log((dn - m + 1.0) / (dn - k + 1.0))
                + (k + 0.5) * d_log(ratio * (dn - k + 1.0) / (k + 1.0))
                + stirling_tail(m) + stirling_tail(dn - m) - stirling_tail(k) - stirling_tail(dn - k);
            if (v <= upper) { draw = k; break; }
        }
    }
    else
    {
        double logQ = d_log1p(-q);
        int64_t successes = 0l, position = 0l;
        while (true)
        {
            double step = floor(d_log(1.0 - rng_uniform()) / logQ) + 1.0;
            if (!(step <= double(n - position))) break;
            position += int64_t(step);
            successes++;
        }
        draw = double(successes);
    }
    y[i] = float(flip ? dn - draw : draw);
}
", atomics: true);
    // scratch is raw 32-bit words: floats without replacement, double prefix sums (two words each) with it.
    private static readonly string Multinomial = RandomKernel(@"
layout(std430, set = 0, binding = 0) readonly buffer W { float w[]; };
layout(std430, set = 0, binding = 1) buffer S { uint scratch[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Y { float y[]; };
" + Flag(3), "uint seedLo; uint seedHi; int rows; int categories; int numSamples; int replacement;", @"
void main()
{
    int row = int(gl_GlobalInvocationID.x);
    if (row >= pc.rows) return;
    int baseIndex = row * pc.categories;
    for (int c = 0; c < pc.categories; c++)
    {
        double v = double(w[baseIndex + c]);
        if (!(v >= 0.0) || isinf(v)) { raise_flag(uint64_t(baseIndex + c), 1); return; }
    }
    rng_init(pc.seedLo, pc.seedHi, uint(row));
    if (pc.replacement != 0)
    {
        double total = 0.0;
        int last = -1;
        for (int c = 0; c < pc.categories; c++)
        {
            total += double(w[baseIndex + c]);
            uvec2 bits = unpackDouble2x32(total);
            scratch[2 * (baseIndex + c)] = bits.x; scratch[2 * (baseIndex + c) + 1] = bits.y;
            if (w[baseIndex + c] > 0.0f) last = c;
        }
        if (!(total > 0.0)) { raise_flag((1ul << 58) | uint64_t(baseIndex), 2); return; }
        for (int k = 0; k < pc.numSamples; k++)
        {
            double u = rng_uniform() * total;
            int lo = 0, hi = pc.categories;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                double prefix = packDouble2x32(uvec2(scratch[2 * (baseIndex + mid)], scratch[2 * (baseIndex + mid) + 1]));
                if (u < prefix) hi = mid; else lo = mid + 1;
            }
            y[row * pc.numSamples + k] = float(lo < pc.categories ? lo : last);
        }
        return;
    }
    for (int c = 0; c < pc.categories; c++) scratch[baseIndex + c] = floatBitsToUint(w[baseIndex + c]);
    for (int k = 0; k < pc.numSamples; k++)
    {
        double total = 0.0;
        for (int c = 0; c < pc.categories; c++) total += double(uintBitsToFloat(scratch[baseIndex + c]));
        if (!(total > 0.0)) { raise_flag((1ul << 58) | uint64_t(baseIndex), 2); return; }
        double u = rng_uniform() * total, running = 0.0;
        int chosen = -1;
        for (int c = pc.categories - 1; c >= 0; c--) if (uintBitsToFloat(scratch[baseIndex + c]) > 0.0f) { chosen = c; break; }
        for (int c = 0; c < pc.categories; c++)
        {
            float s = uintBitsToFloat(scratch[baseIndex + c]);
            running += double(s);
            if (u < running && s > 0.0f) { chosen = c; break; }
        }
        y[row * pc.numSamples + k] = float(chosen);
        scratch[baseIndex + chosen] = floatBitsToUint(0.0f);
    }
}
", atomics: true);

    // Channel dropout (alpha = 0) writes scale only; the engine may pass the scale buffer as offset too.
    private static readonly string DropoutAffine = RandomKernel(@"
layout(std430, set = 0, binding = 0) buffer Sc { float scale[]; };
layout(std430, set = 0, binding = 1) buffer Of { float offsetOut[]; };
", "uint pLo; uint pHi; uint aLo; uint aHi; uint satLo; uint satHi; uint bLo; uint bHi; uint seedLo; uint seedHi; int n; int block; int alpha;", @"
void main()
{
    " + Index + @"
    rng_init(pc.seedLo, pc.seedHi, uint(i / pc.block));
    double p = pc_double(uvec2(pc.pLo, pc.pHi)), a = pc_double(uvec2(pc.aLo, pc.aHi));
    double sat = pc_double(uvec2(pc.satLo, pc.satHi)), b = pc_double(uvec2(pc.bLo, pc.bHi));
    double keep = rng_uniform() >= p ? 1.0 : 0.0;
    if (pc.alpha != 0) { scale[i] = float(a * keep); offsetOut[i] = float(a * sat * (1.0 - keep) + b); }
    else scale[i] = float(keep * a);
}
");

    private static readonly string Rrelu = RandomKernel(UnaryBindings, "uint lowerLo; uint lowerHi; uint upperLo; uint upperHi; uint seedLo; uint seedHi; int n; int training;", @"
void main()
{
    " + Index + @"
    double v = double(x[i]);
    double lower = pc_double(uvec2(pc.lowerLo, pc.lowerHi)), upper = pc_double(uvec2(pc.upperLo, pc.upperHi));
    double slope;
    if (v > 0.0) slope = 1.0;
    else if (pc.training == 0) slope = (lower + upper) / 2.0;
    else { rng_init(pc.seedLo, pc.seedHi, uint(i)); slope = lower + (upper - lower) * rng_uniform(); }
    y[i] = float(slope * v);
}
");

    private static readonly string RandomKeys = RandomKernel(OutputBinding, "uint seedLo; uint seedHi; int n;", @"
void main()
{
    " + Index + @"
    rng_init(pc.seedLo, pc.seedHi, uint(i));
    y[i] = float(rng_uniform());
}
");

    // ---- nonzero_static / unravel_index / unique_dim / grid_sample 3-D / scan / integer ops ----

    // shape: [0] rank, [1..8] extents.
    private static readonly string NonzeroStaticScatter = Kernel(@"
layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 1) readonly buffer Pos { float positions[]; };
layout(std430, set = 0, binding = 2) buffer Y { float y[]; };
layout(std430, set = 0, binding = 3) readonly buffer Shp { int shape[]; };
", "int n; int size;", @"
void main()
{
    " + Index + @"
    if (x[i] == 0.0f) return;
    int row = int(positions[i]) - 1;
    if (row >= pc.size) return;
    int rank = shape[0], rest = i;
    for (int k = rank - 1; k >= 0; k--) { y[row * rank + k] = float(rest % shape[1 + k]); rest /= shape[1 + k]; }
}
");

    private static readonly string UnravelAxis = Kernel(UnaryBindings + Flag(2),
        "uint totalLo; uint totalHi; uint strideLo; uint strideHi; uint extentLo; uint extentHi; int n; int validate;", FlagSource + @"
void main()
{
    " + Index + @"
    int64_t total = int64_t(packUint2x32(uvec2(pc.totalLo, pc.totalHi)));
    int64_t stride = int64_t(packUint2x32(uvec2(pc.strideLo, pc.strideHi)));
    int64_t extent = int64_t(packUint2x32(uvec2(pc.extentLo, pc.extentHi)));
    double t = trunc(double(x[i]));
    if (isnan(t) || t < -double(total) || t >= double(total))
    {
        if (pc.validate != 0) raise_flag(uint64_t(i), 1);
        return;
    }
    // f + total is non-negative here, so the integer division and remainder below see no negative operand.
    int64_t f = int64_t(t) + total;
    int64_t wrapped = f - (f / total) * total;
    int64_t q = wrapped / stride;
    y[i] = float(q - (q / extent) * extent);
}
", atomics: true);
    private const string SliceCompare = @"
// unique_dim's slice order: element by element with double.CompareTo (NaN below every number, equal to NaN).
int slice_compare(int a, int b, int width)
{
    for (int k = 0; k < width; k++)
    {
        float p = data[a * width + k], q = data[b * width + k];
        bool np = isnan(p), nq = isnan(q);
        if (np || nq) { if (np && nq) continue; return np ? -1 : 1; }
        if (p < q) return -1;
        if (p > q) return 1;
    }
    return 0;
}
";

    private static readonly string SliceBitonicStep = Kernel(@"
layout(std430, set = 0, binding = 0) readonly buffer D { float data[]; };
layout(std430, set = 0, binding = 1) buffer O { float order[]; };
", "int rowLen; int k; int j; int rowLength;", SliceCompare + @"
void main()
{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.rowLen) return;
    int ixj = i ^ pc.j;
    if (ixj <= i) return;
    float fa = order[i], fb = order[ixj];
    int a = int(fa), b = int(fb);
    bool bBeforeA, aBeforeB;
    if (a < 0 || b < 0) { bBeforeA = a < 0 && b >= 0; aBeforeB = b < 0 && a >= 0; }
    else
    {
        int c = slice_compare(a, b, pc.rowLength);
        bBeforeA = c > 0 || (c == 0 && b < a);
        aBeforeB = c < 0 || (c == 0 && a < b);
    }
    bool up = (i & pc.k) == 0;
    if (up ? bBeforeA : aBeforeB) { order[i] = fb; order[ixj] = fa; }
}
");

    private static readonly string SliceFirstFlags = Kernel(@"
layout(std430, set = 0, binding = 0) readonly buffer D { float data[]; };
layout(std430, set = 0, binding = 1) readonly buffer O { float order[]; };
layout(std430, set = 0, binding = 2) writeonly buffer F { float flags[]; };
", "int n; int rowLength;", SliceCompare + @"
void main()
{
    " + Index + @"
    flags[i] = (i == 0 || slice_compare(int(order[i]), int(order[i - 1]), pc.rowLength) != 0) ? 1.0f : 0.0f;
}
");

    private static readonly string SliceGatherUnique = Kernel(@"
layout(std430, set = 0, binding = 0) readonly buffer D { float data[]; };
layout(std430, set = 0, binding = 1) readonly buffer O { float order[]; };
layout(std430, set = 0, binding = 2) readonly buffer F { float flags[]; };
layout(std430, set = 0, binding = 3) readonly buffer Pos { float positions[]; };
layout(std430, set = 0, binding = 4) buffer Y { float y[]; };
", "int total; int rowLength;", @"
void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= pc.total) return;
    int s = idx / pc.rowLength, k = idx % pc.rowLength;
    if (flags[s] == 0.0f) return;
    int row = int(positions[s]) - 1;
    y[row * pc.rowLength + k] = data[int(order[s]) * pc.rowLength + k];
}
");

    private const string GridSource = @"
double grid_reflect_slope(double coord, int twiceLow, int twiceHigh, inout double slope)
{
    if (twiceLow == twiceHigh) { slope = 0.0; return 0.0; }
    double mn = double(twiceLow) / 2.0, span = double(twiceHigh - twiceLow) / 2.0, sign = 1.0;
    coord -= mn;
    if (coord < 0.0) { sign = -1.0; coord = -coord; }
    double extra = d_fmod(coord, span), flips = floor(coord / span);
    if (d_fmod(flips, 2.0) == 0.0) { slope *= sign; return extra + mn; }
    slope *= -sign;
    return span - extra + mn;
}

// The unnormalized, clipped or reflected source coordinate, and d coordinate / d normalized in slope.
double grid_source(double normalized, int size, int padding, bool align, out double slope)
{
    double coord = align ? (normalized + 1.0) / 2.0 * double(size - 1) : ((normalized + 1.0) * double(size) - 1.0) / 2.0;
    slope = align ? double(size - 1) / 2.0 : double(size) / 2.0;
    if (padding == 2)
        coord = align ? grid_reflect_slope(coord, 0, 2 * (size - 1), slope) : grid_reflect_slope(coord, -1, 2 * size - 1, slope);
    if (padding >= 1)
    {
        if (coord <= 0.0) { slope = 0.0; return 0.0; }
        if (coord >= double(size - 1)) { slope = 0.0; return double(size - 1); }
    }
    return coord;
}

// A voxel index as the CPU's VoxelIndex: NaN and out-of-int-range coordinates land far outside the volume.
int64_t grid_index(double v) { return (isnan(v) || v < -2147483648.0 || v >= 2147483648.0) ? -4l : int64_t(v); }

int taps;
int tapOffset[8];
double tapWeight[8], tapGradX[8], tapGradY[8], tapGradZ[8];

void grid_taps(double cx, double cy, double cz, double sx, double sy, double sz)
{
    taps = 0;
    if (pc.mode == 1)
    {
        int64_t ix = grid_index(roundEven(cx)), iy = grid_index(roundEven(cy)), iz = grid_index(roundEven(cz));
        if (ix >= 0l && ix < int64_t(pc.width) && iy >= 0l && iy < int64_t(pc.height) && iz >= 0l && iz < int64_t(pc.depth))
        {
            tapOffset[0] = int((iz * int64_t(pc.height) + iy) * int64_t(pc.width) + ix); tapWeight[0] = 1.0;
            tapGradX[0] = 0.0; tapGradY[0] = 0.0; tapGradZ[0] = 0.0; taps = 1;
        }
        return;
    }
    double fx0 = floor(cx), fy0 = floor(cy), fz0 = floor(cz);
    int64_t x0 = grid_index(fx0), y0 = grid_index(fy0), z0 = grid_index(fz0);
    double fx = cx - fx0, fy = cy - fy0, fz = cz - fz0;
    for (int oz = 0; oz < 2; oz++)
        for (int oy = 0; oy < 2; oy++)
            for (int ox = 0; ox < 2; ox++)
            {
                int64_t ix = x0 + int64_t(ox), iy = y0 + int64_t(oy), iz = z0 + int64_t(oz);
                if (ix < 0l || ix >= int64_t(pc.width) || iy < 0l || iy >= int64_t(pc.height) || iz < 0l || iz >= int64_t(pc.depth)) continue;
                double wx = ox == 1 ? fx : 1.0 - fx, wy = oy == 1 ? fy : 1.0 - fy, wz = oz == 1 ? fz : 1.0 - fz;
                tapOffset[taps] = int((iz * int64_t(pc.height) + iy) * int64_t(pc.width) + ix);
                tapWeight[taps] = wx * wy * wz;
                tapGradX[taps] = (ox == 1 ? 1.0 : -1.0) * wy * wz * sx;
                tapGradY[taps] = (oy == 1 ? 1.0 : -1.0) * wx * wz * sy;
                tapGradZ[taps] = (oz == 1 ? 1.0 : -1.0) * wx * wy * sz;
                taps++;
            }
}
";

    private const string GridPush = "int batches; int channels; int depth; int height; int width; int points; int mode; int padding; int align;";

    private static readonly string GridSample3D = Kernel(@"
layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 1) readonly buffer G { float grid[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Y { float y[]; };
", GridPush, GridSource + @"
void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= pc.batches * pc.points) return;
    int b = idx / pc.points, q = idx % pc.points;
    bool align = pc.align != 0;
    double sx, sy, sz;
    double cx = grid_source(double(grid[idx * 3]), pc.width, pc.padding, align, sx);
    double cy = grid_source(double(grid[idx * 3 + 1]), pc.height, pc.padding, align, sy);
    double cz = grid_source(double(grid[idx * 3 + 2]), pc.depth, pc.padding, align, sz);
    grid_taps(cx, cy, cz, sx, sy, sz);
    int volume = pc.depth * pc.height * pc.width;
    for (int ch = 0; ch < pc.channels; ch++)
    {
        int plane = (b * pc.channels + ch) * volume;
        double acc = 0.0;
        for (int t = 0; t < taps; t++) acc += tapWeight[t] * double(x[plane + tapOffset[t]]);
        y[(b * pc.channels + ch) * pc.points + q] = float(acc);
    }
}
");

    // dx gathers w * dy from every point through a compare-and-swap float add (points share voxels).
    private static readonly string GridSample3DBackward = Kernel(@"
layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 1) readonly buffer G { float grid[]; };
layout(std430, set = 0, binding = 2) readonly buffer Dy { float dy[]; };
layout(std430, set = 0, binding = 3) buffer Dx { uint dx[]; };
layout(std430, set = 0, binding = 4) writeonly buffer Dg { float dgrid[]; };
", GridPush, GridSource + @"
void add_dx(int index, float value)
{
    uint expected = dx[index];
    while (true)
    {
        uint desired = floatBitsToUint(uintBitsToFloat(expected) + value);
        uint seen = atomicCompSwap(dx[index], expected, desired);
        if (seen == expected) break;
        expected = seen;
    }
}
void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= pc.batches * pc.points) return;
    int b = idx / pc.points, q = idx % pc.points;
    bool align = pc.align != 0;
    double sx, sy, sz;
    double cx = grid_source(double(grid[idx * 3]), pc.width, pc.padding, align, sx);
    double cy = grid_source(double(grid[idx * 3 + 1]), pc.height, pc.padding, align, sy);
    double cz = grid_source(double(grid[idx * 3 + 2]), pc.depth, pc.padding, align, sz);
    grid_taps(cx, cy, cz, sx, sy, sz);
    int volume = pc.depth * pc.height * pc.width;
    double dgx = 0.0, dgy = 0.0, dgz = 0.0;
    for (int ch = 0; ch < pc.channels; ch++)
    {
        int plane = (b * pc.channels + ch) * volume;
        double up = double(dy[(b * pc.channels + ch) * pc.points + q]);
        for (int t = 0; t < taps; t++)
        {
            add_dx(plane + tapOffset[t], float(tapWeight[t] * up));
            double v = double(x[plane + tapOffset[t]]) * up;
            dgx += v * tapGradX[t]; dgy += v * tapGradY[t]; dgz += v * tapGradZ[t];
        }
    }
    dgrid[idx * 3] = float(dgx);
    dgrid[idx * 3 + 1] = float(dgy);
    dgrid[idx * 3 + 2] = float(dgz);
}
");
    // Inclusive prefix sum: a LocalSize-wide Hillis-Steele scan per work group, the group totals scanned the same way
    // (recursively, on the host side), then each group's carry added. Exact in float for counts below 2^24.
    private static readonly string ScanBlock = Kernel(@"
layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 1) buffer Y { float y[]; };
layout(std430, set = 0, binding = 2) buffer Sums { float sums[]; };
", "int n; int writeSums;", @"
shared float s[gl_WorkGroupSize.x];
void main()
{
    int t = int(gl_LocalInvocationID.x), i = int(gl_WorkGroupID.x * gl_WorkGroupSize.x) + t;
    s[t] = i < pc.n ? x[i] : 0.0f;
    barrier();
    for (int offset = 1; offset < int(gl_WorkGroupSize.x); offset <<= 1)
    {
        float v = t >= offset ? s[t - offset] : 0.0f;
        barrier();
        s[t] += v;
        barrier();
    }
    if (i < pc.n) y[i] = s[t];
    if (pc.writeSums != 0 && t == int(gl_WorkGroupSize.x) - 1) sums[gl_WorkGroupID.x] = s[t];
}
");

    private static readonly string ScanAdd = Kernel(@"
layout(std430, set = 0, binding = 0) buffer Y { float y[]; };
layout(std430, set = 0, binding = 1) readonly buffer C { float carries[]; };
", "int n;", @"
void main()
{
    " + Index + @"
    int block = i / int(gl_WorkGroupSize.x);
    if (block > 0) y[i] += carries[block - 1];
}
");

    // Integer ops on raw int32 storage, as the CPU's IntegerBinary: 64-bit arithmetic on the sign-extended operands,
    // wrapped to int32. op: 0 and, 1 or, 2 xor, 3 left shift, 4 right shift, 5 gcd, 6 lcm, 7 floor divide (code 1 for a
    // zero divisor). GLSL leaves / and % undefined for negative operands, so quotients are formed on magnitudes.
    private static readonly string IntBinary = Kernel(@"
layout(std430, set = 0, binding = 0) readonly buffer A { int a[]; };
layout(std430, set = 0, binding = 1) readonly buffer B { int b[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Y { int y[]; };
" + Flag(3), "int n; int op;", FlagSource + @"
int64_t magnitude(int64_t v) { return v < 0l ? -v : v; }
// Quotient truncated toward zero, as C#.
int64_t truncated_divide(int64_t x, int64_t v)
{
    int64_t q = int64_t(uint64_t(magnitude(x)) / uint64_t(magnitude(v)));
    return (x < 0l) != (v < 0l) ? -q : q;
}
int64_t gcd64(int64_t x, int64_t y)
{
    uint64_t p = uint64_t(magnitude(x)), q = uint64_t(magnitude(y));
    while (q != 0ul) { uint64_t t = p % q; p = q; q = t; }
    return int64_t(p);
}
void main()
{
    " + Index + @"
    int64_t x = int64_t(a[i]), v = int64_t(b[i]), r;
    switch (pc.op)
    {
        case 0: r = x & v; break;
        case 1: r = x | v; break;
        case 2: r = x ^ v; break;
        case 3: r = v >= 64l ? 0l : int64_t(uint64_t(x) << uint64_t(v & 63l)); break;
        case 4: r = v >= 64l ? (x < 0l ? -1l : 0l) : x >> (v & 63l); break;
        case 5: r = gcd64(x, v); break;
        case 6: { int64_t g = gcd64(x, v); int64_t m = g == 0l ? 0l : truncated_divide(x, g) * v; r = magnitude(m); break; }
        default:
            if (v == 0l) { raise_flag(uint64_t(i), 1); return; }
            r = truncated_divide(x, v);
            if (x - r * v != 0l && ((x < 0l) != (v < 0l))) r -= 1l;
            break;
    }
    y[i] = int(r);
}
", atomics: true);

    private static readonly string IntNot = Kernel(@"
layout(std430, set = 0, binding = 0) readonly buffer A { int a[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Y { int y[]; };
", "int n;", "void main() { " + Index + "y[i] = ~a[i]; }\n");

    private static readonly System.Lazy<System.Collections.Generic.Dictionary<string, string>> Sources =
        new System.Lazy<System.Collections.Generic.Dictionary<string, string>>(BuildSources);

    private static System.Collections.Generic.Dictionary<string, string> BuildSources()
    {
        var map = new System.Collections.Generic.Dictionary<string, string>(System.StringComparer.Ordinal);
        foreach (var pair in UnaryExpressions) map[pair.Key] = UnaryKernel(pair.Value);
        foreach (var pair in BinaryBodies) map[pair.Key] = BinaryKernel(pair.Value);
        map["parity_mvlgamma"] = Mvlgamma;
        map["parity_polynomial"] = Polynomial;
        map["parity_range"] = Range;
        map["parity_logspace"] = Logspace;
        map["parity_window"] = Window;
        map["parity_tri_indices"] = TriIndices;
        map["parity_diagflat"] = Diagflat;
        map["parity_diagonal_scatter"] = DiagonalScatter;
        map["parity_quantile_rows"] = QuantileRows;
        map["parity_flag_reset"] = FlagReset;
        map["parity_sample"] = Sample;
        map["parity_bernoulli"] = Bernoulli;
        map["parity_poisson"] = Poisson;
        map["parity_binomial"] = Binomial;
        map["parity_multinomial"] = Multinomial;
        map["parity_dropout_affine"] = DropoutAffine;
        map["parity_rrelu"] = Rrelu;
        map["parity_random_keys"] = RandomKeys;
        map["parity_nonzero_static_scatter"] = NonzeroStaticScatter;
        map["parity_unravel_axis"] = UnravelAxis;
        map["parity_slice_bitonic_step"] = SliceBitonicStep;
        map["parity_slice_first_flags"] = SliceFirstFlags;
        map["parity_slice_gather_unique"] = SliceGatherUnique;
        map["parity_grid_sample_3d"] = GridSample3D;
        map["parity_grid_sample_3d_backward"] = GridSample3DBackward;
        map["parity_scan_block"] = ScanBlock;
        map["parity_scan_add"] = ScanAdd;
        map["parity_int_binary"] = IntBinary;
        map["parity_int_not"] = IntNot;
        return map;
    }

    /// <summary>The GLSL source of the kernel with the CUDA / OpenCL name <paramref name="name"/>.</summary>
    internal static string Source(string name)
        => Sources.Value.TryGetValue(name, out var source)
            ? source
            : throw new System.ArgumentOutOfRangeException(nameof(name), "No Vulkan torch-parity kernel named " + name + ".");

    /// <summary>Every kernel name (the same set as the CUDA and OpenCL modules).</summary>
    internal static System.Collections.Generic.IEnumerable<string> KernelNames => Sources.Value.Keys;
}