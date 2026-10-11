// Copyright (c) AiDotNet. All rights reserved.
// Double-precision math for the Vulkan PyTorch-parity kernels (#1127).
//
// The CPU reference evaluates these ops in double (Helpers/SpecialFunctions and the Math calls around it) and rounds
// once to float. GLSL has exp / log / sin / cos / pow for float only, so this is a double library in GLSL: the fdlibm
// elementary functions (within 1-2 ulp of .NET's, with Payne-Hanek reduction so sin / cos stay exact for huge
// arguments) and a branch-for-branch port of SpecialFunctions. Shaders using it need the shaderFloat64 feature.
using System;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>GLSL source for the double-precision library behind the Vulkan PyTorch-parity kernels.</summary>
internal static class VulkanTorchParityMath
{
    private static readonly Regex FloatLiteral = new Regex(
        @"(?<![\w.])(\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])", RegexOptions.CultureInvariant);

    /// <summary>
    /// Marks every floating-point literal in <paramref name="glsl"/> as a double (the <c>lf</c> suffix). The kernels are
    /// written with bare literals; without the suffix GLSL reads them as float and silently drops half the digits.
    /// </summary>
    public static string Doubles(string glsl) => FloatLiteral.Replace(glsl, "$1lf");

    private static readonly Lazy<string> TablesSource = new Lazy<string>(BuildTables);
    private static readonly Lazy<string> ElementarySource = new Lazy<string>(() => Doubles(Elementary));
    private static readonly Lazy<string> SpecialSource = new Lazy<string>(() => Doubles(Special));

    /// <summary>Constant tables: the bits of 2/pi and the CPU quadratures' Gauss-Legendre rules.</summary>
    public static string Tables => TablesSource.Value;

    /// <summary>d_exp, d_log, d_sin, d_cos, d_pow and friends. Needs <see cref="Tables"/> before it.</summary>
    public static string ElementaryFunctions => ElementarySource.Value;

    /// <summary>The sf_* special functions. Needs <see cref="ElementaryFunctions"/> before it.</summary>
    public static string SpecialFunctionsSource => SpecialSource.Value;

    /// <summary>Number of 32-bit words of 2/pi the shaders carry (enough for any double argument).</summary>
    internal const int TwoOverPiWordCount = 40;

    /// <summary>The first <paramref name="words"/> 32-bit words of the binary expansion of 2/pi.</summary>
    internal static uint[] TwoOverPiWords(int words)
    {
        // pi by Machin's formula in fixed point, with guard bits.
        int bits = words * 32 + 128;
        BigInteger one = BigInteger.One << bits;
        BigInteger pi = 16 * ArcTanInverse(5, one) - 4 * ArcTanInverse(239, one);
        BigInteger twoOverPi = (one << (bits + 1)) / pi;
        var result = new uint[words];
        for (int i = 0; i < words; i++) result[i] = (uint)((twoOverPi >> (bits - 32 * (i + 1))) & 0xFFFFFFFF);
        return result;
    }

    // atan(1 / inverse) scaled by `one`.
    private static BigInteger ArcTanInverse(int inverse, BigInteger one)
    {
        BigInteger power = one / inverse, sum = power, square = (BigInteger)inverse * inverse;
        for (int k = 3; ; k += 2)
        {
            power /= square;
            if (power.IsZero) return sum;
            sum += (k & 2) != 0 ? -(power / k) : power / k;
        }
    }

    private static string BuildTables()
    {
        var sb = new StringBuilder();
        var words = TwoOverPiWords(TwoOverPiWordCount);
        sb.Append("const uint TWO_OVER_PI[").Append(words.Length).Append("] = uint[").Append(words.Length).Append("](");
        for (int i = 0; i < words.Length; i++) sb.Append(i == 0 ? "" : ", ").Append("0x").Append(words[i].ToString("X8", CultureInfo.InvariantCulture)).Append('u');
        sb.Append(");\n");
        foreach (int n in new[] { 96, 32 })
        {
            var (x, w) = SpecialFunctions.GaussLegendreRule(n);
            AppendDoubles(sb, "SF_GL" + n + "_X", x);
            AppendDoubles(sb, "SF_GL" + n + "_W", w);
        }
        return sb.ToString();
    }

    private static void AppendDoubles(StringBuilder sb, string name, double[] values)
    {
        sb.Append("const double ").Append(name).Append('[').Append(values.Length).Append("] = double[").Append(values.Length).Append("](");
        for (int i = 0; i < values.Length; i++)
        {
            string text = values[i].ToString("R", CultureInfo.InvariantCulture);
            if (text.IndexOf('.') < 0 && text.IndexOf('E') < 0) text += ".0";
            sb.Append(i == 0 ? "" : ", ").Append(text).Append("lf");
        }
        sb.Append(");\n");
    }

    private const string Elementary = @"
// ---- double-precision elementary functions ----------------------------------------------------------------------
// GLSL's exp / log / sin / cos / pow take float only, and the special functions below are evaluated in double (as on
// the CPU), so these are the fdlibm algorithms on doubles. NaN / infinity are built and tested by bit pattern, so a
// driver's relaxed float handling cannot change them.
double d_from_bits(uint hi, uint lo) { return packDouble2x32(uvec2(lo, hi)); }
uint d_hi(double x) { return unpackDouble2x32(x).y; }
uint d_lo(double x) { return unpackDouble2x32(x).x; }
double d_inf() { return d_from_bits(0x7FF00000u, 0u); }
double d_nan() { return d_from_bits(0x7FF80000u, 0u); }
bool d_isnan(double x) { uint h = d_hi(x) & 0x7FFFFFFFu; return h > 0x7FF00000u || (h == 0x7FF00000u && d_lo(x) != 0u); }
bool d_isinf(double x) { return (d_hi(x) & 0x7FFFFFFFu) == 0x7FF00000u && d_lo(x) == 0u; }
bool d_signbit(double x) { return (d_hi(x) & 0x80000000u) != 0u; }
// 2^k for k in [-1022, 1023].
double d_pow2(int k) { return d_from_bits(uint(k + 1023) << 20, 0u); }
// x * 2^k, in two steps so results in the subnormal range round once and large k cannot overflow early.
double d_scale(double x, int k)
{
    if (k > 1000) { x *= d_pow2(1000); k -= 1000; if (k > 1000) k = 1000; }
    else if (k < -1000) { x *= d_pow2(-1000); k += 1000; if (k < -1000) k = -1000; }
    return x * d_pow2(k);
}

double d_exp(double x)
{
    if (d_isnan(x)) return x;
    if (x > 709.782712893383973096) return d_inf();
    if (x < -745.13321910194110842) return 0.0;
    double k = floor(x * 1.44269504088896338700 + 0.5);
    double hi = x - k * 6.93147180369123816490e-01;
    double lo = k * 1.90821492927058770002e-10;
    double r = hi - lo;
    double t = r * r;
    double c = r - t * (1.66666666666666019037e-01 + t * (-2.77777777770155933842e-03 + t * (6.61375632143793436117e-05
        + t * (-1.65339022054652515390e-06 + t * 4.13813679705723846039e-08))));
    double y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi);
    return d_scale(y, int(k));
}

double d_log(double x)
{
    if (d_isnan(x)) return x;
    if (x < 0.0) return d_nan();
    if (x == 0.0) return -d_inf();
    if (d_isinf(x)) return x;
    int k = 0;
    if (d_hi(x) < 0x00100000u) { x *= 18014398509481984.0; k = -54; }   // subnormal: scale by 2^54
    uint hx = d_hi(x);
    k += int(hx >> 20) - 1023;
    hx &= 0x000FFFFFu;
    uint i = (hx + 0x95F64u) & 0x100000u;
    x = d_from_bits(hx | (i ^ 0x3FF00000u), d_lo(x));   // x in [sqrt(2)/2, sqrt(2))
    k += int(i >> 20);
    double f = x - 1.0;
    double s = f / (2.0 + f);
    double dk = double(k);
    double z = s * s;
    double w = z * z;
    double t1 = w * (3.999999999940941908e-01 + w * (2.222219843214978396e-01 + w * 1.531383769920937332e-01));
    double t2 = z * (6.666666666666735130e-01 + w * (2.857142874366239149e-01 + w * (1.818357216161805012e-01
        + w * 1.479819860511658591e-01)));
    double R = t2 + t1;
    double hfsq = 0.5 * f * f;
    return dk * 6.93147180369123816490e-01 - ((hfsq - (s * (hfsq + R) + dk * 1.90821492927058770002e-10)) - f);
}

// log(1 + y), as the CPU's SpecialFunctions.Log1p (Goldberg's correction).
double d_log1p(double y)
{
    // precise: (1 + y) - 1 must not be simplified to y.
    precise double u = 1.0 + y;
    precise double d = u - 1.0;
    return u == 1.0 ? y : d_log(u) * y / d;
}

// sin and cos on [-pi/4, pi/4] with a tail y (fdlibm __kernel_sin / __kernel_cos).
double d_kernel_sin(double x, double y)
{
    double z = x * x, w = z * z;
    double r = 8.33333333332248946124e-03 + z * (-1.98412698298579493134e-04 + z * 2.75573137070700676789e-06)
        + z * w * (-2.50507602534068634195e-08 + z * 1.58969099521155010221e-10);
    double v = z * x;
    return x - ((z * (0.5 * y - v * r) - y) - v * -1.66666666666666324348e-01);
}

double d_kernel_cos(double x, double y)
{
    double z = x * x, w = z * z;
    double r = z * (4.16666666666666019037e-02 + z * (-1.38888888888741095749e-03 + z * 2.48015872894767294178e-05))
        + w * w * (-2.75573143513906633035e-07 + z * (2.08757232129817482790e-09 + z * -1.13596475577881948265e-11));
    double hz = 0.5 * z;
    w = 1.0 - hz;
    return w + (((1.0 - w) - hz) + (z * r - x * y));
}

// a * b = p + e exactly (Dekker / Veltkamp; `precise` keeps the compiler from contracting or reassociating it).
void d_two_product(double a, double b, out double p, out double e)
{
    precise double t = 134217729.0 * a;
    precise double ah = t - (t - a);
    precise double al = a - ah;
    precise double u = 134217729.0 * b;
    precise double bh = u - (u - b);
    precise double bl = b - bh;
    precise double pp = a * b;
    precise double ee = ((ah * bh - pp) + ah * bl + al * bh) + al * bl;
    p = pp;
    e = ee;
}

// The 32 bits of the word array w (most significant word first) that start `bit` bits in.
uint d_bits_at(uint w[10], int bit)
{
    int word = bit >> 5, shift = bit & 31;
    uint hi = word < 10 ? w[word] : 0u;
    if (shift == 0) return hi;
    uint lo = word + 1 < 10 ? w[word + 1] : 0u;
    return (hi << shift) | (lo >> (32 - shift));
}

// x = n * pi/2 + (y0 + y1) with |y0 + y1| <= pi/4; returns n mod 4. Cody-Waite with a three-part pi/2 while
// n * pio2_1 is exact (|x| < 2^20 pi/2), Payne-Hanek on the bits of 2/pi beyond.
int d_rem_pio2(double x, out double y0, out double y1)
{
    double ax = abs(x);
    if (ax < 1647099.0)
    {
        // pi/2 = pio2_1 + pio2_2 + pio2_3 + pio2_3t (33 bits each, so fn * pio2_k is exact); the rounding error of each
        // subtraction is carried into the tail.
        precise double fn = floor(x * 6.36619772367581382433e-01 + 0.5);
        precise double t = x - fn * 1.57079632673412561417e+00;
        precise double w = fn * 6.07710050630396597660e-11;
        precise double u = t - w;
        precise double uerr = (t - u) - w;
        precise double w3 = fn * 2.02226624871116645580e-21;
        precise double r = u - w3;
        precise double tail = fn * 8.47842766036889956997e-32 - (((u - r) - w3) + uerr);
        precise double z0 = r - tail;
        y0 = z0;
        y1 = (r - z0) - tail;
        return int(fn) & 3;
    }
    // ax = m * 2^e with m a 53-bit integer.
    uint hx = d_hi(ax), lx = d_lo(ax);
    int e = int(hx >> 20) - 1075;
    uint m1 = (hx & 0x000FFFFFu) | 0x00100000u, m0 = lx;
    // Words of 2/pi from the first whose product with m is not a multiple of 4.
    int first = e >= 2 ? (e - 2) / 32 : 0;
    uint prod[10];
    for (int j = 0; j < 10; j++) prod[j] = 0u;
    // prod (10 words, most significant first) = m * TWO_OVER_PI[first .. first + 7], modulo 2^320.
    for (int j = 7; j >= 0; j--)
    {
        uint a = TWO_OVER_PI[first + j];
        uint lo0, hi0, lo1, hi1;
        umulExtended(a, m0, hi0, lo0);
        umulExtended(a, m1, hi1, lo1);
        // add lo0 at word j + 2, (hi0 + lo1) at j + 1, hi1 at j
        uint carry;
        uint s = uaddCarry(prod[j + 2], lo0, carry); prod[j + 2] = s;
        uint c2;
        s = uaddCarry(prod[j + 1], hi0, c2); uint c3; s = uaddCarry(s, lo1, c3); uint c4; s = uaddCarry(s, carry, c4);
        prod[j + 1] = s;
        uint up = c2 + c3 + c4;
        uint c5;
        s = uaddCarry(prod[j], hi1, c5); uint c6; s = uaddCarry(s, up, c6); prod[j] = s;
        uint over = c5 + c6;
        for (int q = j - 1; q >= 0 && over != 0u; q--) { uint c7; prod[q] = uaddCarry(prod[q], over, c7); over = c7; }
    }
    // prod's lowest bit weighs 2^(e - 32 (first + 8)), so its first `point` bits (from the top) are the integer part:
    // take that part's low two bits and 128 fraction bits.
    int point = 320 + (e - 32 * (first + 8));
    int n = int(d_bits_at(prod, point - 2) >> 30);
    uint f0 = d_bits_at(prod, point), f1 = d_bits_at(prod, point + 32), f2 = d_bits_at(prod, point + 64), f3 = d_bits_at(prod, point + 96);
    bool negative = (f0 & 0x80000000u) != 0u;
    if (negative)
    {
        // fraction >= 1/2: round n up and take 1 - fraction.
        n = (n + 1) & 3;
        f0 = ~f0; f1 = ~f1; f2 = ~f2; f3 = ~f3;
        uint ca, cb, cc, cd;
        f3 = uaddCarry(f3, 1u, ca); f2 = uaddCarry(f2, ca, cb); f1 = uaddCarry(f1, cb, cc); f0 = uaddCarry(f0, cc, cd);
    }
    // fraction = hi + lo as doubles (hi + lo exact to 2^-128).
    double a0 = double(f0) * d_pow2(-32), a1 = double(f1) * d_pow2(-64), a2 = double(f2) * d_pow2(-96), a3 = double(f3) * d_pow2(-128);
    precise double hi = a0 + a1;
    precise double err = (a0 - hi) + a1;
    precise double lo = err + (a2 + a3);
    // times pi/2 = PH + PL
    double p, pe;
    d_two_product(hi, 1.57079632679489655800e+00, p, pe);
    precise double tail = pe + hi * 6.12323399573676603587e-17 + lo * 1.57079632679489655800e+00;
    precise double r0 = p + tail;
    precise double r1 = tail - (r0 - p);
    if (negative) { r0 = -r0; r1 = -r1; }
    if (x < 0.0) { r0 = -r0; r1 = -r1; n = (4 - n) & 3; }
    y0 = r0;
    y1 = r1;
    return n;
}

double d_sin(double x)
{
    if (d_isnan(x) || d_isinf(x)) return d_nan();
    double y0, y1;
    int n = d_rem_pio2(x, y0, y1);
    if (n == 0) return d_kernel_sin(y0, y1);
    if (n == 1) return d_kernel_cos(y0, y1);
    if (n == 2) return -d_kernel_sin(y0, y1);
    return -d_kernel_cos(y0, y1);
}

double d_cos(double x)
{
    if (d_isnan(x) || d_isinf(x)) return d_nan();
    double y0, y1;
    int n = d_rem_pio2(x, y0, y1);
    if (n == 0) return d_kernel_cos(y0, y1);
    if (n == 1) return -d_kernel_sin(y0, y1);
    if (n == 2) return -d_kernel_cos(y0, y1);
    return d_kernel_sin(y0, y1);
}

double d_tan(double x) { return d_sin(x) / d_cos(x); }

double d_cosh(double x)
{
    double e = d_exp(abs(x));
    return 0.5 * e + 0.5 / e;
}

double d_sinh(double x)
{
    double ax = abs(x);
    if (ax < 0.5)
    {
        // Taylor series: e^x - e^-x cancels here.
        double x2 = x * x, term = x, sum = x;
        for (int k = 1; k < 12; k++) { term *= x2 / double((2 * k) * (2 * k + 1)); sum += term; }
        return sum;
    }
    double e = d_exp(ax);
    double r = 0.5 * e - 0.5 / e;
    return x < 0.0 ? -r : r;
}

// x^y for the cases the kernels use (x >= 0, or a negative x with an integer y).
double d_pow(double x, double y)
{
    if (y == 0.0 || x == 1.0) return 1.0;
    if (d_isnan(x) || d_isnan(y)) return d_nan();
    if (x == 0.0)
    {
        bool odd = !d_isinf(y) && floor(y) == y && abs(y) < 9007199254740992.0 && (y * 0.5 != floor(y * 0.5));
        double zero = odd && d_signbit(x) ? -0.0 : 0.0;
        return y > 0.0 ? zero : (odd && d_signbit(x) ? -d_inf() : d_inf());
    }
    if (x < 0.0)
    {
        if (d_isinf(y)) { double a = -x; return a == 1.0 ? 1.0 : ((a > 1.0) == (y > 0.0) ? d_inf() : 0.0); }
        if (floor(y) != y) return d_nan();
        bool odd = abs(y) < 9007199254740992.0 && (y * 0.5 != floor(y * 0.5));
        double r = d_exp(y * d_log(-x));
        return odd ? -r : r;
    }
    return d_exp(y * d_log(x));
}

// Round half to even, as C's rint / .NET's Math.Round.
double d_rint(double x)
{
    if (abs(x) >= 4503599627370496.0 || d_isnan(x)) return x;
    double f = floor(x), d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1.0;
    return (f * 0.5 == floor(f * 0.5)) ? f : f + 1.0;
}

double d_trunc(double x) { return x < 0.0 ? ceil(x) : floor(x); }

// The double equal to the float with these bits. Exact for every float, subnormals included: a device's own
// float -> double conversion may flush a subnormal to zero.
double d_from_float_bits(uint bits)
{
    uint sgn = bits & 0x80000000u, e = (bits >> 23) & 0xFFu, m = bits & 0x7FFFFFu;
    if (e == 0xFFu) return d_from_bits(sgn | 0x7FF00000u | (m >> 3), m << 29);
    if (e == 0u)
    {
        double tiny = double(m) * d_pow2(-149);
        return sgn != 0u ? -tiny : tiny;
    }
    return d_from_bits(sgn | ((e + 896u) << 20) | (m >> 3), m << 29);
}

// The bits of the float nearest x (ties to even), with overflow to infinity and gradual underflow.
uint d_to_float_bits(double x)
{
    uint sgn = d_hi(x) & 0x80000000u;
    if (d_isnan(x)) return sgn | 0x7FC00000u;
    double a = abs(x);
    if (a == 0.0) return sgn;
    if (d_isinf(a)) return sgn | 0x7F800000u;
    int e = int((d_hi(a) >> 20) & 0x7FFu) - 1023;
    if (e > 127) return sgn | 0x7F800000u;
    if (e < -150) return sgn;
    if (e < -126)
    {
        // Subnormal float: a whole number of 2^-149 units (the top of the range rounds up to the smallest normal).
        return sgn | uint(d_rint(a * d_pow2(149)));
    }
    // Significand in [2^23, 2^24); a round up to 2^24 carries into the exponent.
    double r = d_rint(d_scale(a, 23 - e));
    if (r >= 16777216.0) { r = 8388608.0; e += 1; }
    if (e > 127) return sgn | 0x7F800000u;
    return sgn | (uint(e + 127) << 23) | (uint(r) - 0x800000u);
}

// C's fmod for b > 0: exact, by binary long division (each subtraction is exact).
double d_fmod(double a, double b)
{
    if (d_isnan(a) || d_isinf(a)) return d_nan();
    double r = abs(a);
    if (r >= b)
    {
        int steps = int((d_hi(r) >> 20) & 0x7FFu) - int((d_hi(b) >> 20) & 0x7FFu);
        double scaled = d_scale(b, steps);
        for (int k = steps; k >= 0; k--)
        {
            if (r >= scaled) r -= scaled;
            scaled *= 0.5;
        }
    }
    return a < 0.0 ? -r : r;
}
";

    private const string Special = @"
// ---- special functions: ports of Helpers/SpecialFunctions (the CPU reference), branch for branch --------------------
const double SF_PI = 3.141592653589793;
const double SF_SQRT_PI = 1.7724538509055160273;
const double SF_SQRT2 = 1.4142135623730950488;
const double SF_EXP_UNDERFLOW = 745.0;
const double SF_LARGE_ARGUMENT = 50.0;
const double SF_MAX = 1.7976931348623157e308;

// erf(x) = 2/sqrt(pi) sum (-1)^n x^(2n+1) / (n! (2n+1)).
double sf_erf_series(double x)
{
    double x2 = x * x, term = x, sum = x;
    for (int n = 1; n < 200; n++)
    {
        term *= -x2 / double(n);
        double add = term / double(2 * n + 1);
        sum += add;
        if (abs(add) < 1e-17 * abs(sum)) break;
    }
    return 2.0 / SF_SQRT_PI * sum;
}

// sqrt(pi) exp(x^2) erfc(x) as a continued fraction (modified Lentz), x >= 2.5.
double sf_erfc_continued_fraction(double x)
{
    double f = x, c = x, d = 0.0;
    for (int n = 1; n < 500; n++)
    {
        double a = double(n) / 2.0;
        d = x + a * d;
        d = abs(d) < 1e-300 ? 1e-300 : d;
        c = x + a / c;
        c = abs(c) < 1e-300 ? 1e-300 : c;
        d = 1.0 / d;
        double delta = c * d;
        f *= delta;
        if (abs(delta - 1.0) < 1e-16) break;
    }
    return 1.0 / f;
}

double sf_erfc_large(double x) { return d_exp(-x * x) * sf_erfc_continued_fraction(x) / SF_SQRT_PI; }

// erfc on [0.5, 2.5): erfc(2.5) plus (2/sqrt(pi)) times the 32-node Gauss-Legendre integral of e^{-t^2} over [x, 2.5].
double sf_erfc_mid(double x)
{
    double mid = 0.5 * (x + 2.5), halfWidth = 0.5 * (2.5 - x), sum = 0.0;
    for (int i = 0; i < 32; i++)
    {
        double t = mid + halfWidth * SF_GL32_X[i];
        sum += SF_GL32_W[i] * d_exp(-t * t);
    }
    return sf_erfc_large(2.5) + 2.0 / SF_SQRT_PI * (sum * halfWidth);
}

// erfc(x) for x >= 0.5.
double sf_erfc_upper(double x) { return x >= 2.5 ? sf_erfc_large(x) : sf_erfc_mid(x); }

double sf_erfc(double x)
{
    if (d_isnan(x)) return x;
    if (d_isinf(x)) return x > 0.0 ? 0.0 : 2.0;
    if (x >= 0.5) return sf_erfc_upper(x);
    if (x <= -0.5) return 2.0 - sf_erfc_upper(-x);
    return 1.0 - sf_erf_series(x);
}

double sf_erf(double x)
{
    if (d_isnan(x)) return x;
    if (d_isinf(x)) return x > 0.0 ? 1.0 : -1.0;
    if (abs(x) < 0.5) return sf_erf_series(x);
    double tail = sf_erfc_upper(abs(x));
    return x > 0.0 ? 1.0 - tail : tail - 1.0;
}

// erfcx(x) for x >= 0.5.
double sf_erfcx_upper(double x)
{
    return x >= 2.5 ? sf_erfc_continued_fraction(x) / SF_SQRT_PI : d_exp(x * x) * sf_erfc_mid(x);
}

double sf_erfcx(double x)
{
    if (d_isnan(x)) return x;
    if (d_isinf(x) && x > 0.0) return 0.0;
    if (x >= 0.5) return sf_erfcx_upper(x);
    if (x > -0.5) return d_exp(x * x) * (1.0 - sf_erf_series(x));
    if (x < -26.7) return d_inf();
    return 2.0 * d_exp(x * x) - sf_erfcx_upper(-x);
}

double sf_ndtr(double x) { return 0.5 * sf_erfc(-x / SF_SQRT2); }

double sf_log_ndtr(double x)
{
    if (d_isnan(x)) return x;
    if (d_isinf(x) && x < 0.0) return -d_inf();
    if (x > 0.0) return d_log1p(-sf_ndtr(-x));
    if (x > -1.0) return d_log(sf_ndtr(x));
    return d_log(0.5 * sf_erfcx(-x / SF_SQRT2)) - 0.5 * x * x;
}

// The standard normal quantile: Acklam's rational approximation refined by two Halley steps.
double sf_ndtri(double p)
{
    if (d_isnan(p) || p < 0.0 || p > 1.0) return d_nan();
    if (p == 0.0) return -d_inf();
    if (p == 1.0) return d_inf();
    double x;
    if (p < 0.02425 || p > 1.0 - 0.02425)
    {
        double q = sqrt(-2.0 * d_log(p < 0.02425 ? p : 1.0 - p));
        x = (((((-7.784894002430293e-03 * q + -3.223964580411365e-01) * q + -2.400758277161838e+00) * q + -2.549732539343734e+00) * q
            + 4.374664141464968e+00) * q + 2.938163982698783e+00)
            / ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e+00) * q + 3.754408661907416e+00) * q + 1.0);
        if (p >= 0.02425) x = -x;
    }
    else
    {
        double q = p - 0.5, r = q * q;
        x = (((((-3.969683028665376e+01 * r + 2.209460984245205e+02) * r + -2.759285104469687e+02) * r + 1.383577518672690e+02) * r
            + -3.066479806614716e+01) * r + 2.506628277459239e+00) * q
            / (((((-5.447609879822406e+01 * r + 1.615858368580409e+02) * r + -1.556989798598866e+02) * r + 6.680131188771972e+01) * r
            + -1.328068155288572e+01) * r + 1.0);
    }
    for (int i = 0; i < 2; i++)
    {
        double e = sf_ndtr(x) - p;
        double u = e * sqrt(2.0 * SF_PI) * d_exp(x * x / 2.0);
        x -= u / (1.0 + x * u / 2.0);
    }
    return x;
}

// log|Gamma(x)| for x >= 0.5 (Lanczos, g = 7).
double sf_lgamma_upper(double x)
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
    return 0.5 * d_log(2.0 * SF_PI) + (x + 0.5) * d_log(t) - t + d_log(a);
}

double sf_lgamma(double x)
{
    if (d_isnan(x)) return x;
    if (d_isinf(x)) return d_inf();
    if (x <= 0.0 && floor(x) == x) return d_inf();
    if (x < 0.5) return d_log(SF_PI / abs(d_sin(SF_PI * x))) - sf_lgamma_upper(1.0 - x);
    return sf_lgamma_upper(x);
}

double sf_igamma_series(double a, double x)
{
    double ap = a, sum = 1.0 / a, del = sum;
    for (int k = 0; k < 1000; k++)
    {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (abs(del) < abs(sum) * 1e-16) break;
    }
    return sum * d_exp(-x + a * d_log(x) - sf_lgamma(a));
}

double sf_igamma_continued_fraction(double a, double x)
{
    double b = x + 1.0 - a, c = 1.0 / 1e-300, d = 1.0 / b, h = d;
    for (int i = 1; i < 1000; i++)
    {
        double an = -double(i) * (double(i) - a);
        b += 2.0;
        d = an * d + b;
        if (abs(d) < 1e-300) d = 1e-300;
        c = b + an / c;
        if (abs(c) < 1e-300) c = 1e-300;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (abs(del - 1.0) < 1e-16) break;
    }
    return d_exp(-x + a * d_log(x) - sf_lgamma(a)) * h;
}

double sf_igamma(double a, double x)
{
    if (d_isnan(a) || d_isnan(x) || a <= 0.0 || x < 0.0) return d_nan();
    if (x == 0.0) return 0.0;
    if (d_isinf(x)) return 1.0;
    return x < a + 1.0 ? sf_igamma_series(a, x) : 1.0 - sf_igamma_continued_fraction(a, x);
}

double sf_igammac(double a, double x)
{
    if (d_isnan(a) || d_isnan(x) || a <= 0.0 || x < 0.0) return d_nan();
    if (x == 0.0) return 1.0;
    if (d_isinf(x)) return 0.0;
    return x < a + 1.0 ? 1.0 - sf_igamma_series(a, x) : sf_igamma_continued_fraction(a, x);
}

double sf_mvlgamma(double x, int p)
{
    double sum = double(p * (p - 1)) / 4.0 * d_log(SF_PI);
    for (int j = 0; j < p; j++) sum += sf_lgamma(x - double(j) / 2.0);
    return sum;
}

// sum (+-1)^k (x/2)^(2k+n) / (k! (k+n)!): the power series of J_n (alternating) or I_n, used for |x| < 2.
double sf_bessel_series(int n, double x, bool alternating)
{
    double halfX = x / 2.0, term = 1.0;
    for (int k = 1; k <= n; k++) term *= halfX / double(k);
    double sum = term, q = halfX * halfX * (alternating ? -1.0 : 1.0);
    for (int k = 1; k < 60; k++)
    {
        term *= q / (double(k) * double(k + n));
        sum += term;
        if (abs(term) <= 1e-17 * abs(sum)) break;
    }
    return sum;
}

// sum c_k(mu) / x^k, c_k = c_{k-1} (mu - (2k-1)^2) / (8k): the large-x series of e^-x I (alternating) and e^x K.
double sf_modified_bessel_asymptotic(double mu, double x, bool alternating)
{
    double term = 1.0, sum = 1.0, last = SF_MAX;
    for (int k = 1; k < 200; k++)
    {
        double dk = double(k);
        term *= (mu - (2.0 * dk - 1.0) * (2.0 * dk - 1.0)) / (8.0 * dk * x) * (alternating ? -1.0 : 1.0);
        if (abs(term) > last) break;
        last = abs(term);
        sum += term;
        if (abs(term) < 1e-17 * abs(sum)) break;
    }
    return sum;
}

// Hankel's asymptotic expansion, integer n >= 0, x large: J_n and Y_n.
void sf_hankel_asymptotic(int n, double x, out double J, out double Y)
{
    double mu = 4.0 * double(n * n), p = 1.0, q = 0.0, term = 1.0, lastP = SF_MAX;
    for (int k = 1; k < 60; k++)
    {
        term *= (mu - double((2 * k - 1) * (2 * k - 1))) / (double(k) * 8.0 * x);
        if (abs(term) > lastP) break;
        lastP = abs(term);
        double signedTerm = ((k / 2) % 2 == 0 ? 1.0 : -1.0) * term;
        if (k % 2 == 1) q += signedTerm; else p += signedTerm;
        if (abs(term) < 1e-17) break;
    }
    double phi = (double(n) / 2.0 + 0.25) * SF_PI, s = sqrt(2.0 / (SF_PI * x));
    double cx = d_cos(x), sx = d_sin(x), cp = d_cos(phi), sp = d_sin(phi);
    double cosChi = cx * cp + sx * sp, sinChi = sx * cp - cx * sp;
    J = s * (p * cosChi - q * sinChi);
    Y = s * (p * sinChi + q * cosChi);
}

double sf_bessel_j(int n, double x)
{
    if (d_isnan(x)) return x;
    double ax = abs(x);
    if (d_isinf(ax)) return 0.0;
    if (ax < 2.0) return sf_bessel_series(n, x, true);
    if (ax > 25.0)
    {
        double J, Y;
        sf_hankel_asymptotic(n, ax, J, Y);
        return (n % 2 == 1 && x < 0.0 ? -1.0 : 1.0) * J;
    }
    // J_n(x) = (1/2pi) int_0^2pi cos(n tau - x sin tau) d tau: the equally spaced mean of a periodic analytic integrand.
    int points = 2 * int(ceil(ax)) + 48;
    double sum = 0.0;
    for (int i = 0; i < points; i++)
    {
        double tau = 2.0 * SF_PI * double(i) / double(points);
        sum += d_cos(double(n) * tau - x * d_sin(tau));
    }
    return sum / double(points);
}

double sf_bessel_y(int n, double x)
{
    if (d_isnan(x) || x < 0.0) return d_nan();
    if (x == 0.0) return -d_inf();
    if (d_isinf(x)) return 0.0;
    if (x > 25.0)
    {
        double J, Y;
        sf_hankel_asymptotic(n, x, J, Y);
        return Y;
    }
    // Y_n(x) = (1/pi) int_0^pi sin(x sin tau - n tau) d tau - (1/pi) int_0^inf (e^{nt} + (-1)^n e^{-nt}) e^{-x sinh t} dt.
    double dn = double(n), first = 0.0;
    for (int i = 0; i < 96; i++)
    {
        double tau = 0.5 * SF_PI + 0.5 * SF_PI * SF_GL96_X[i];
        first += SF_GL96_W[i] * d_sin(x * d_sin(tau) - dn * tau);
    }
    first *= 0.5 * SF_PI;
    double sgn = n % 2 == 0 ? 1.0 : -1.0, second = 0.0;
    for (int panel = 0; panel < 2000; panel++)
    {
        double start = double(panel);
        if (x * d_sinh(start) - dn * start > SF_EXP_UNDERFLOW) break;
        double part = 0.0;
        for (int i = 0; i < 32; i++)
        {
            double t = start + 0.5 + 0.5 * SF_GL32_X[i];
            part += SF_GL32_W[i] * ((d_exp(dn * t) + sgn * d_exp(-dn * t)) * d_exp(-x * d_sinh(t)));
        }
        second += part * 0.5;
    }
    return (first - second) / SF_PI;
}

// e^{-|x|} I_n(x), integer n.
double sf_bessel_i_scaled(int n, double x)
{
    if (d_isnan(x)) return x;
    double ax = abs(x);
    double sgn = n % 2 == 1 && x < 0.0 ? -1.0 : 1.0;
    if (ax < 2.0) return sgn * d_exp(-ax) * sf_bessel_series(n, ax, false);
    if (ax > SF_LARGE_ARGUMENT) return sgn * sf_modified_bessel_asymptotic(4.0 * double(n * n), ax, true) / sqrt(2.0 * SF_PI * ax);
    int points = 40 + int(ceil(sqrt(80.0 * ax)));
    double sum = 0.0;
    for (int i = 0; i < points; i++)
    {
        double tau = 2.0 * SF_PI * double(i) / double(points);
        sum += d_exp(ax * (d_cos(tau) - 1.0)) * d_cos(double(n) * tau);
    }
    return sgn * sum / double(points);
}

double sf_bessel_i(int n, double x)
{
    if (d_isinf(x)) return n % 2 == 1 && x < 0.0 ? -d_inf() : d_inf();
    return sf_bessel_i_scaled(n, x) * d_exp(abs(x));
}

// e^x K_nu(x): the asymptotic series above x = 50, else the trapezoid (h = 0.02) of e^{-x (cosh t - 1)} cosh(nu t).
double sf_bessel_k_scaled(double nu, double x)
{
    if (d_isnan(x) || x < 0.0) return d_nan();
    if (x == 0.0) return d_inf();
    if (x > SF_LARGE_ARGUMENT) return sqrt(SF_PI / (2.0 * x)) * sf_modified_bessel_asymptotic(4.0 * nu * nu, x, false);
    double sum = 0.5;   // f(0) = 1, halved
    for (int i = 1; i < 100000; i++)
    {
        double t = double(i) * 0.02;
        double grow = x * (d_cosh(t) - 1.0);
        if (grow - abs(nu) * t > SF_EXP_UNDERFLOW) break;
        sum += d_exp(-grow) * d_cosh(nu * t);
    }
    return sum * 0.02;
}

double sf_bessel_k(double nu, double x)
{
    return (d_isinf(x) && x > 0.0) ? 0.0 : sf_bessel_k_scaled(nu, x) * d_exp(-x);
}

double sf_spherical_bessel_j0(double x)
{
    if (d_isnan(x)) return x;
    if (d_isinf(x)) return 0.0;
    if (abs(x) < 1e-4) return 1.0 - x * x / 6.0;
    return d_sin(x) / x;
}

// J_nu(x) for non-integer nu and 0 < x <= ~40 (Schlafli's integral).
double sf_bessel_j_fractional(double nu, double x)
{
    double first = 0.0;
    for (int i = 0; i < 96; i++)
    {
        double tau = 0.5 * SF_PI + 0.5 * SF_PI * SF_GL96_X[i];
        first += SF_GL96_W[i] * d_cos(nu * tau - x * d_sin(tau));
    }
    first *= 0.5 * SF_PI;
    double second = 0.0;
    for (int panel = 0; panel < 2000; panel++)
    {
        double start = double(panel);
        if (x * d_sinh(start) + nu * start > SF_EXP_UNDERFLOW) break;
        double part = 0.0;
        for (int i = 0; i < 32; i++)
        {
            double t = start + 0.5 + 0.5 * SF_GL32_X[i];
            part += SF_GL32_W[i] * d_exp(-x * d_sinh(t) - nu * t);
        }
        second += part * 0.5;
    }
    return (first - d_sin(nu * SF_PI) * second) / SF_PI;
}

// zeta = (2/3) z sqrt(z) as an unevaluated double-double hi + lo.
void sf_zeta_double_double(double z, out double hi, out double lo)
{
    double root = sqrt(z), sq, sqErr;
    d_two_product(root, root, sq, sqErr);
    precise double rootLo = ((z - sq) - sqErr) / (2.0 * root);
    double m, mErr;
    d_two_product(z, root, m, mErr);
    precise double mLo = mErr + z * rootLo;
    double twoThirds = 2.0 / 3.0;
    double three, threeErr;
    d_two_product(3.0, twoThirds, three, threeErr);
    precise double twoThirdsLo = ((2.0 - three) - threeErr) / 3.0;
    double p, pErr;
    d_two_product(m, twoThirds, p, pErr);
    precise double pLo = pErr + m * twoThirdsLo + mLo * twoThirds;
    precise double h = p + pLo;
    precise double l = pLo - (h - p);
    hi = h;
    lo = l;
}

double sf_airy_ai(double x)
{
    if (d_isnan(x)) return x;
    if (d_isinf(x)) return 0.0;
    if (x >= -5.0 && x <= 1.0)
    {
        double x3 = x * x * x, f = 1.0, g = x, tf = 1.0, tg = x;
        for (int k = 1; k < 200; k++)
        {
            double dk = double(k);
            tf *= x3 / ((3.0 * dk - 1.0) * (3.0 * dk));
            tg *= x3 / ((3.0 * dk) * (3.0 * dk + 1.0));
            f += tf;
            g += tg;
            if (abs(tf) + abs(tg) < 1e-18 * (abs(f) + abs(g))) break;
        }
        return 0.355028053887817239 * f - 0.258819403792806798 * g;
    }
    double z = abs(x), zeta = 2.0 / 3.0 * z * sqrt(z);
    if (x > 0.0) return sqrt(x / 3.0) / SF_PI * sf_bessel_k(1.0 / 3.0, zeta);
    if (z <= 15.0) return sqrt(z) / 3.0 * (sf_bessel_j_fractional(1.0 / 3.0, zeta) + sf_bessel_j_fractional(-1.0 / 3.0, zeta));
    double u = 1.0, even = 1.0, odd = 0.0, power = 1.0, last = SF_MAX;
    for (int k = 1; k < 60; k++)
    {
        double dk = double(k);
        u *= (6.0 * dk - 5.0) * (6.0 * dk - 3.0) * (6.0 * dk - 1.0) / ((2.0 * dk - 1.0) * 216.0 * dk);
        power /= zeta;
        double term = u * power;
        if (term > last) break;
        last = term;
        double signedTerm = ((k / 2) % 2 == 0 ? 1.0 : -1.0) * term;
        if (k % 2 == 0) even += signedTerm; else odd += signedTerm;
        if (term < 1e-17) break;
    }
    double hi, lo;
    sf_zeta_double_double(z, hi, lo);
    double sh = d_sin(hi), ch = d_cos(hi), sl = d_sin(lo + SF_PI / 4.0), cl = d_cos(lo + SF_PI / 4.0);
    double sinPhase = sh * cl + ch * sl, cosPhase = ch * cl - sh * sl;
    return (sinPhase * even - cosPhase * odd) / (sqrt(SF_PI) * sqrt(sqrt(z)));
}

// sin(pi x) with exact argument reduction: exactly 0 at every integer.
double sf_sin_pi(double x)
{
    if (d_isnan(x) || d_isinf(x)) return d_nan();
    double r = x - 2.0 * d_rint(x / 2.0);
    double a = abs(r);
    if (a == 0.0 || a == 1.0) return 0.0;
    if (a > 0.5) a = 1.0 - a;
    double s = d_sin(SF_PI * a);
    return r < 0.0 ? -s : s;
}

// The orthogonal-polynomial recurrences of SpecialFunctions.Polynomial. kind: 0 Chebyshev T, 1 U, 2 V, 3 W,
// 4 Hermite H, 5 Hermite He, 6 Laguerre L, 7 Legendre P.
double sf_polynomial(int kind, double v, int deg)
{
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
    if (deg == 0) return 1.0;
    for (int k = 1; k < deg; k++)
    {
        double kk = double(k), next;
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
    return current;
}
";
}
