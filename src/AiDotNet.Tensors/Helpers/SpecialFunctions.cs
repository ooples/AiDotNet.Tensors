using System;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Double-precision special functions behind the torch.special parity ops. Each uses a method that reaches close to
/// double precision without coefficient tables: series where they converge without cancellation, asymptotic
/// expansions far out, continued fractions, and the trapezoid rule on integral representations whose integrands are
/// periodic or decay doubly-exponentially (where it converges exponentially fast).
/// </summary>
internal static class SpecialFunctions
{
    private const double SqrtPi = 1.7724538509055160273;
    private const double Sqrt2 = 1.4142135623730950488;
    private const double EulerGamma = 0.57721566490153286061;

    // ---- error function family -------------------------------------------------------------------------------

    /// <summary>erf(x).</summary>
    public static double Erf(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (Math.Abs(x) < 2.5) return ErfSeries(x);
        return x > 0 ? 1.0 - ErfcLarge(x) : ErfcLarge(-x) - 1.0;
    }

    /// <summary>erfc(x) = 1 - erf(x), without cancellation for large x.</summary>
    public static double Erfc(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (x >= 2.5) return ErfcLarge(x);
        if (x <= -2.5) return 2.0 - ErfcLarge(-x);
        return 1.0 - ErfSeries(x);
    }

    /// <summary>The scaled complementary error function erfcx(x) = exp(x²)·erfc(x).</summary>
    public static double Erfcx(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (x >= 2.5) return ErfcContinuedFraction(x) / SqrtPi;
        if (x >= -2.5) return Math.Exp(x * x) * (1.0 - ErfSeries(x));
        if (x < -26.7) return double.PositiveInfinity;
        return 2.0 * Math.Exp(x * x) - Erfcx(-x);
    }

    // erf(x) = 2/√π Σ (-1)^n x^(2n+1) / (n! (2n+1)), used for |x| < 2.5 (terms stay below ~e^6.25).
    private static double ErfSeries(double x)
    {
        double x2 = x * x, term = x, sum = x;
        for (int n = 1; n < 200; n++)
        {
            term *= -x2 / n;
            double add = term / (2 * n + 1);
            sum += add;
            if (Math.Abs(add) < 1e-17 * Math.Abs(sum)) break;
        }
        return 2.0 / SqrtPi * sum;
    }

    private static double ErfcLarge(double x) => Math.Exp(-x * x) * ErfcContinuedFraction(x) / SqrtPi;

    // √π·exp(x²)·erfc(x) = 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...))))), modified Lentz, x ≥ 2.5.
    private static double ErfcContinuedFraction(double x)
    {
        const double Tiny = 1e-300;
        double f = x, c = x, d = 0;
        for (int n = 1; n < 500; n++)
        {
            double a = n / 2.0;
            d = x + a * d;
            d = Math.Abs(d) < Tiny ? Tiny : d;
            c = x + a / c;
            c = Math.Abs(c) < Tiny ? Tiny : c;
            d = 1.0 / d;
            double delta = c * d;
            f *= delta;
            if (Math.Abs(delta - 1.0) < 1e-16) break;
        }
        return 1.0 / f;
    }

    /// <summary>The standard normal CDF Φ(x).</summary>
    public static double Ndtr(double x) => 0.5 * Erfc(-x / Sqrt2);

    /// <summary>log Φ(x), accurate in the far left tail.</summary>
    public static double LogNdtr(double x)
    {
        if (x > -1.0) return Math.Log(Ndtr(x));
        // Φ(x) = ½·erfcx(-x/√2)·exp(-x²/2)
        return Math.Log(0.5 * Erfcx(-x / Sqrt2)) - 0.5 * x * x;
    }

    /// <summary>The standard normal quantile Φ⁻¹(p): Acklam's rational approximation refined by Halley steps.</summary>
    public static double Ndtri(double p)
    {
        if (double.IsNaN(p) || p < 0 || p > 1) return double.NaN;
        if (p == 0) return double.NegativeInfinity;
        if (p == 1) return double.PositiveInfinity;
        double[] a = { -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };
        double[] b = { -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 };
        double[] c = { -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };
        double[] d = { 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00 };
        double x;
        if (p < 0.02425)
        {
            double q = Math.Sqrt(-2 * Math.Log(p));
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        else if (p > 1 - 0.02425)
        {
            double q = Math.Sqrt(-2 * Math.Log(1 - p));
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        else
        {
            double q = p - 0.5, r = q * q;
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        }
        // Halley refinement against Φ.
        for (int i = 0; i < 2; i++)
        {
            double e = Ndtr(x) - p;
            double u = e * Math.Sqrt(2 * Math.PI) * Math.Exp(x * x / 2);
            x -= u / (1 + x * u / 2);
        }
        return x;
    }

    // ---- gamma family ------------------------------------------------------------------------------------------

    private static readonly double[] Lanczos =
    {
        0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    };

    /// <summary>log|Γ(x)| (Lanczos, g = 7, with reflection below ½).</summary>
    public static double LogGamma(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (x <= 0 && Math.Floor(x) == x) return double.PositiveInfinity;
        if (x < 0.5) return Math.Log(Math.PI / Math.Abs(Math.Sin(Math.PI * x))) - LogGamma(1 - x);
        x -= 1;
        double a = Lanczos[0], t = x + 7.5;
        for (int i = 1; i < 9; i++) a += Lanczos[i] / (x + i);
        return 0.5 * Math.Log(2 * Math.PI) + (x + 0.5) * Math.Log(t) - t + Math.Log(a);
    }

    /// <summary>ψ(x) = d/dx log Γ(x): recurrence up to x ≥ 6, then the asymptotic series.</summary>
    public static double Digamma(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (x <= 0 && Math.Floor(x) == x) return double.NaN;
        double result = 0;
        if (x < 0)
        {
            // ψ(1 - x) - ψ(x) = π·cot(πx)
            result -= Math.PI / Math.Tan(Math.PI * x);
            x = 1 - x;
        }
        while (x < 6) { result -= 1 / x; x += 1; }
        double f = 1 / (x * x);
        double series = f * (-1.0 / 12 + f * (1.0 / 120 + f * (-1.0 / 252 + f * (1.0 / 240 + f * (-1.0 / 132 + f * (691.0 / 32760 - f / 12))))));
        return result + Math.Log(x) - 0.5 / x + series;
    }

    /// <summary>The regularized lower incomplete gamma function P(a, x).</summary>
    public static double Igamma(double a, double x)
    {
        if (double.IsNaN(a) || double.IsNaN(x) || a <= 0 || x < 0) return double.NaN;
        if (x == 0) return 0;
        if (double.IsPositiveInfinity(x)) return 1;
        return x < a + 1 ? IgammaSeries(a, x) : 1.0 - IgammaContinuedFraction(a, x);
    }

    /// <summary>The regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).</summary>
    public static double Igammac(double a, double x)
    {
        if (double.IsNaN(a) || double.IsNaN(x) || a <= 0 || x < 0) return double.NaN;
        if (x == 0) return 1;
        if (double.IsPositiveInfinity(x)) return 0;
        return x < a + 1 ? 1.0 - IgammaSeries(a, x) : IgammaContinuedFraction(a, x);
    }

    private static double IgammaSeries(double a, double x)
    {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 1000; n++)
        {
            ap += 1;
            del *= x / ap;
            sum += del;
            if (Math.Abs(del) < Math.Abs(sum) * 1e-16) break;
        }
        return sum * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
    }

    private static double IgammaContinuedFraction(double a, double x)
    {
        const double Tiny = 1e-300;
        double b = x + 1 - a, c = 1 / Tiny, d = 1 / b, h = d;
        for (int i = 1; i < 1000; i++)
        {
            double an = -i * (i - a);
            b += 2;
            d = an * d + b;
            if (Math.Abs(d) < Tiny) d = Tiny;
            c = b + an / c;
            if (Math.Abs(c) < Tiny) c = Tiny;
            d = 1 / d;
            double del = d * c;
            h *= del;
            if (Math.Abs(del - 1) < 1e-16) break;
        }
        return Math.Exp(-x + a * Math.Log(x) - LogGamma(a)) * h;
    }

    /// <summary>The multivariate log-gamma function of dimension <paramref name="p"/>.</summary>
    public static double Mvlgamma(double x, int p)
    {
        double sum = p * (p - 1) / 4.0 * Math.Log(Math.PI);
        for (int j = 0; j < p; j++) sum += LogGamma(x - j / 2.0);
        return sum;
    }

    /// <summary>d/dx of <see cref="Mvlgamma"/>: Σⱼ ψ(x - j/2).</summary>
    public static double MvlgammaDerivative(double x, int p)
    {
        double sum = 0;
        for (int j = 0; j < p; j++) sum += Digamma(x - j / 2.0);
        return sum;
    }

    // ---- Bessel functions --------------------------------------------------------------------------------------

    // Hankel's asymptotic expansion for integer order n ≥ 0, x large: returns (J_n, Y_n).
    private static (double J, double Y) HankelAsymptotic(int n, double x)
    {
        double mu = 4.0 * n * n, p = 1, q = 0, term = 1;
        double lastP = double.MaxValue;
        for (int k = 1; k < 60; k++)
        {
            term *= (mu - (2 * k - 1) * (2 * k - 1)) / (k * 8.0 * x);
            if (Math.Abs(term) > lastP) break;   // the series is asymptotic: stop at its smallest term
            lastP = Math.Abs(term);
            if (k % 2 == 1) q += ((k / 2) % 2 == 0 ? 1 : -1) * term;
            else p += ((k / 2) % 2 == 0 ? 1 : -1) * term;
            if (Math.Abs(term) < 1e-17) break;
        }
        double chi = x - (n / 2.0 + 0.25) * Math.PI, s = Math.Sqrt(2 / (Math.PI * x));
        return (s * (p * Math.Cos(chi) - q * Math.Sin(chi)), s * (p * Math.Sin(chi) + q * Math.Cos(chi)));
    }

    /// <summary>The Bessel function of the first kind J_n(x), integer n.</summary>
    public static double BesselJ(int n, double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        double ax = Math.Abs(x);
        if (ax > 25) return (n % 2 == 1 && x < 0 ? -1 : 1) * HankelAsymptotic(n, ax).J;
        // J_n(x) = (1/2π)∫₀^{2π} cos(nτ - x sin τ) dτ: a periodic analytic integrand, so the equally spaced mean
        // converges exponentially (aliasing error ~ J_N(x)).
        int points = 2 * (int)Math.Ceiling(ax) + 48;
        double sum = 0;
        for (int i = 0; i < points; i++)
        {
            double tau = 2 * Math.PI * i / points;
            sum += Math.Cos(n * tau - x * Math.Sin(tau));
        }
        return sum / points;
    }

    /// <summary>The Bessel function of the second kind Y_n(x), integer n, x &gt; 0.</summary>
    public static double BesselY(int n, double x)
    {
        if (double.IsNaN(x) || x < 0) return double.NaN;
        if (x == 0) return double.NegativeInfinity;
        if (x > 25) return HankelAsymptotic(n, x).Y;
        // Y_n(x) = (1/π)∫₀^π sin(x sin τ - nτ) dτ - (1/π)∫₀^∞ (e^{nt} + (-1)ⁿe^{-nt}) e^{-x sinh t} dt.
        double first = GaussLegendre(0, Math.PI, 96, tau => Math.Sin(x * Math.Sin(tau) - n * tau));
        double sign = n % 2 == 0 ? 1 : -1;
        // Not even in t, so the trapezoid rule would only be O(h²) here: composite Gauss–Legendre instead.
        double second = DecayingGaussLegendre(t => (Math.Exp(n * t) + sign * Math.Exp(-n * t)) * Math.Exp(-x * Math.Sinh(t)),
            t => x * Math.Sinh(t) - n * t > 745);
        return (first - second) / Math.PI;
    }

    /// <summary>
    /// The modified Bessel function of the first kind, exponentially scaled: e^{-|x|}·I_n(x), integer n.
    /// </summary>
    public static double BesselIScaled(int n, double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        double ax = Math.Abs(x);
        // I_n(x) = (1/2π)∫₀^{2π} e^{x cos τ} cos(nτ) dτ; scaled by e^{-|x|} so large x cannot overflow. Fourier
        // coefficients beyond k decay like e^{-k²/2|x|}, so √(80|x|) + 40 points reach double precision.
        int points = 40 + (int)Math.Ceiling(Math.Sqrt(80 * ax));
        double sum = 0;
        for (int i = 0; i < points; i++)
        {
            double tau = 2 * Math.PI * i / points;
            sum += Math.Exp(ax * (Math.Cos(tau) - 1)) * Math.Cos(n * tau);
        }
        double value = sum / points;
        return n % 2 == 1 && x < 0 ? -value : value;
    }

    /// <summary>The modified Bessel function of the first kind I_n(x), integer n.</summary>
    public static double BesselI(int n, double x) => BesselIScaled(n, x) * Math.Exp(Math.Abs(x));

    /// <summary>
    /// The modified Bessel function of the second kind, exponentially scaled: e^{x}·K_ν(x), x &gt; 0, any real ν.
    /// </summary>
    public static double BesselKScaled(double nu, double x)
    {
        if (double.IsNaN(x) || x < 0) return double.NaN;
        if (x == 0) return double.PositiveInfinity;
        // e^{x}K_ν(x) = ∫₀^∞ e^{-x(cosh t - 1)} cosh(νt) dt; the integrand decays double-exponentially, so the
        // trapezoid rule converges exponentially in 1/h.
        return DecayingTrapezoid(t => Math.Exp(-x * (Math.Cosh(t) - 1)) * Math.Cosh(nu * t),
            t => x * (Math.Cosh(t) - 1) - Math.Abs(nu) * t > 745);
    }

    /// <summary>The modified Bessel function of the second kind K_ν(x), x &gt; 0.</summary>
    public static double BesselK(double nu, double x) => BesselKScaled(nu, x) * Math.Exp(-x);

    /// <summary>The spherical Bessel function j₀(x) = sin(x)/x.</summary>
    public static double SphericalBesselJ0(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (Math.Abs(x) < 1e-4) return 1 - x * x / 6;
        return Math.Sin(x) / x;
    }

    // ∫₀^∞ f for a smooth, decaying f: unit-width panels of 32-node Gauss–Legendre up to where 'beyond' holds.
    private static double DecayingGaussLegendre(Func<double, double> f, Func<double, bool> beyond)
    {
        double sum = 0;
        for (int panel = 0; panel < 2000; panel++)
        {
            if (beyond(panel)) break;
            sum += GaussLegendre(panel, panel + 1, 32, f);
        }
        return sum;
    }

    // Trapezoid rule on [0, ∞) for an EVEN integrand that decays double-exponentially (half the full-line rule, which
    // converges exponentially in 1/h), stopping where 'beyond' holds.
    private static double DecayingTrapezoid(Func<double, double> f, Func<double, bool> beyond)
    {
        const double H = 0.02;
        double sum = 0.5 * f(0);
        for (int i = 1; i < 100000; i++)
        {
            double t = i * H;
            if (beyond(t)) break;
            sum += f(t);
        }
        return sum * H;
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, (double[] X, double[] W)> GaussLegendreRules =
        new System.Collections.Concurrent.ConcurrentDictionary<int, (double[] X, double[] W)>();

    // Gauss–Legendre quadrature of f over [a, b] with n nodes (nodes by Newton's method on Pₙ, cached per n).
    private static double GaussLegendre(double a, double b, int n, Func<double, double> f)
    {
        var (nodes, weights) = GaussLegendreRules.GetOrAdd(n, ComputeGaussLegendre);
        double mid = 0.5 * (a + b), half = 0.5 * (b - a), sum = 0;
        for (int i = 0; i < n; i++) sum += weights[i] * f(mid + half * nodes[i]);
        return sum * half;
    }

    private static (double[] X, double[] W) ComputeGaussLegendre(int n)
    {
        var x = new double[n];
        var w = new double[n];
        for (int i = 0; i < n; i++)
        {
            double z = Math.Cos(Math.PI * (i + 0.75) / (n + 0.5)), pp = 0;
            for (int iter = 0; iter < 100; iter++)
            {
                double p1 = 1, p2 = 0;
                for (int j = 1; j <= n; j++)
                {
                    double p3 = p2;
                    p2 = p1;
                    p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
                }
                pp = n * (z * p1 - p2) / (z * z - 1);
                double dz = p1 / pp;
                z -= dz;
                if (Math.Abs(dz) < 1e-16) break;
            }
            x[i] = z;
            w[i] = 2 / ((1 - z * z) * pp * pp);
        }
        return (x, w);
    }

    /// <summary>J_ν(x) for real, non-integer ν and 0 &lt; x ≤ ~40 (Schläfli's integral).</summary>
    private static double BesselJFractional(double nu, double x)
    {
        double first = GaussLegendre(0, Math.PI, 96, tau => Math.Cos(nu * tau - x * Math.Sin(tau)));
        double second = DecayingGaussLegendre(t => Math.Exp(-x * Math.Sinh(t) - nu * t), t => x * Math.Sinh(t) + nu * t > 745);
        return (first - Math.Sin(nu * Math.PI) * second) / Math.PI;
    }

    // ---- Airy -------------------------------------------------------------------------------------------------

    private const double AiryC1 = 0.355028053887817239;   // Ai(0)
    private const double AiryC2 = 0.258819403792806798;   // -Ai'(0)

    /// <summary>The Airy function Ai(x).</summary>
    public static double AiryAi(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (double.IsPositiveInfinity(x)) return 0;
        if (double.IsNegativeInfinity(x)) return 0;
        if (x >= -5 && x <= 1)
        {
            // Maclaurin: Ai = c₁f - c₂g (terms stay below ~e^{7.5} on this interval).
            double x3 = x * x * x, f = 1, g = x, tf = 1, tg = x;
            for (int k = 1; k < 200; k++)
            {
                tf *= x3 / ((3.0 * k - 1) * (3.0 * k));
                tg *= x3 / ((3.0 * k) * (3.0 * k + 1));
                f += tf;
                g += tg;
                if (Math.Abs(tf) + Math.Abs(tg) < 1e-18 * (Math.Abs(f) + Math.Abs(g))) break;
            }
            return AiryC1 * f - AiryC2 * g;
        }
        double z = Math.Abs(x), zeta = 2.0 / 3.0 * z * Math.Sqrt(z);
        if (x > 0) return Math.Sqrt(x / 3) / Math.PI * BesselK(1.0 / 3.0, zeta);
        if (z <= 15) return Math.Sqrt(z) / 3 * (BesselJFractional(1.0 / 3.0, zeta) + BesselJFractional(-1.0 / 3.0, zeta));
        // DLMF 9.7.9: Ai(-z) ~ π^{-1/2} z^{-1/4}[sin(ζ+π/4) Σ(-1)ᵏu₂ₖζ^{-2k} - cos(ζ+π/4) Σ(-1)ᵏu₂ₖ₊₁ζ^{-2k-1}].
        double u = 1, even = 1, odd = 0, power = 1, last = double.MaxValue;
        for (int k = 1; k < 60; k++)
        {
            u *= (6.0 * k - 5) * (6.0 * k - 3) * (6.0 * k - 1) / ((2.0 * k - 1) * 216.0 * k);
            power /= zeta;
            double term = u * power;
            if (term > last) break;
            last = term;
            int half = k / 2;
            double signed = (half % 2 == 0 ? 1 : -1) * term;
            if (k % 2 == 0) even += signed; else odd += signed;
            if (term < 1e-17) break;
        }
        double phase = zeta + Math.PI / 4;
        return (Math.Sin(phase) * even - Math.Cos(phase) * odd) / (Math.Sqrt(Math.PI) * Math.Pow(z, 0.25));
    }

    /// <summary>Ai'(x), by Richardson-extrapolated central differences of <see cref="AiryAi"/>.</summary>
    public static double AiryAiDerivative(double x)
    {
        const double H = 1e-3;
        double d1 = (AiryAi(x + H) - AiryAi(x - H)) / (2 * H);
        double d2 = (AiryAi(x + H / 2) - AiryAi(x - H / 2)) / H;
        return (4 * d2 - d1) / 3;
    }

    // ---- orthogonal polynomials (integer degree n; n < 0 gives 0, as PyTorch does) -------------------------------

    /// <summary>The kinds of orthogonal polynomial evaluated by <see cref="Polynomial"/>.</summary>
    public enum PolynomialKind
    {
        ChebyshevT, ChebyshevU, ChebyshevV, ChebyshevW, HermiteH, HermiteHe, LaguerreL, LegendreP,
    }

    /// <summary>The degree-<paramref name="n"/> polynomial of <paramref name="kind"/> at <paramref name="x"/>.</summary>
    public static double Polynomial(PolynomialKind kind, double x, int n)
    {
        if (n < 0) return 0;
        if (double.IsNaN(x)) return double.NaN;
        double previous = 1, current;
        switch (kind)
        {
            case PolynomialKind.ChebyshevT: current = x; break;
            case PolynomialKind.ChebyshevU: current = 2 * x; break;
            case PolynomialKind.ChebyshevV: current = 2 * x - 1; break;
            case PolynomialKind.ChebyshevW: current = 2 * x + 1; break;
            case PolynomialKind.HermiteH: current = 2 * x; break;
            case PolynomialKind.HermiteHe: current = x; break;
            case PolynomialKind.LaguerreL: current = 1 - x; break;
            case PolynomialKind.LegendreP: current = x; break;
            default: throw new ArgumentOutOfRangeException(nameof(kind));
        }
        if (n == 0) return previous;
        for (int k = 1; k < n; k++)
        {
            double next = kind switch
            {
                PolynomialKind.HermiteH => 2 * x * current - 2 * k * previous,
                PolynomialKind.HermiteHe => x * current - k * previous,
                PolynomialKind.LaguerreL => ((2 * k + 1 - x) * current - k * previous) / (k + 1),
                PolynomialKind.LegendreP => ((2 * k + 1) * x * current - k * previous) / (k + 1),
                _ => 2 * x * current - previous,
            };
            previous = current;
            current = next;
        }
        return current;
    }
}
