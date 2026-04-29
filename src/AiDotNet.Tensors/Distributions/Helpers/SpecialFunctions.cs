using System;

namespace AiDotNet.Tensors.Distributions.Helpers;

/// <summary>
/// Special functions used by the distribution implementations: log-gamma, digamma,
/// trigamma, error function and its inverse, regularised incomplete gamma / beta.
///
/// Implementations are textbook approximations (Lanczos for log-gamma, Cody/Hastings
/// for erf, Stirling-asymptotic + recurrence for digamma) targeting roughly 1e-7
/// accuracy in single precision. They are deterministic, allocation-free, and
/// suitable for inclusion in CUDA-graph-safe optimizer/distribution paths.
/// </summary>
public static class SpecialFunctions
{
    /// <summary>log Γ(x) via the Lanczos approximation (g = 7).</summary>
    public static float Lgamma(float x)
    {
        if (x <= 0f) return float.PositiveInfinity;
        // Reflection formula for x < 0.5 (kept for completeness; we mostly hit x > 0).
        if (x < 0.5f)
        {
            // log(π / sin(πx)) - log(Γ(1 - x))
            return (float)(Math.Log(Math.PI / Math.Sin(Math.PI * x)) - Lgamma(1f - x));
        }
        double xd = x - 1.0;
        double[] g = LanczosCoefficients;
        double a = g[0];
        for (int i = 1; i < g.Length; i++) a += g[i] / (xd + i);
        double t = xd + g.Length - 1.5;
        return (float)(0.5 * Math.Log(2 * Math.PI) + (xd + 0.5) * Math.Log(t) - t + Math.Log(a));
    }

    private static readonly double[] LanczosCoefficients =
    {
        0.99999999999980993,
        676.5203681218851,
       -1259.1392167224028,
        771.32342877765313,
       -176.61502916214059,
        12.507343278686905,
       -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    };

    /// <summary>Γ(x).</summary>
    public static float Gamma(float x) => MathF.Exp(Lgamma(x));

    /// <summary>log B(α, β) = log Γ(α) + log Γ(β) − log Γ(α+β).</summary>
    public static float LogBeta(float a, float b) => Lgamma(a) + Lgamma(b) - Lgamma(a + b);

    /// <summary>Digamma ψ(x) = d/dx log Γ(x).</summary>
    public static float Digamma(float x)
    {
        // Recurrence + asymptotic series (Bernoulli-number expansion).
        double xd = x;
        double result = 0.0;
        // Push x up so we can use the asymptotic expansion accurately (need x ≥ 6).
        while (xd < 6.0)
        {
            result -= 1.0 / xd;
            xd += 1.0;
        }
        // ψ(x) ≈ log(x) − 1/(2x) − Σ B_{2k} / (2k x^{2k})
        double inv = 1.0 / xd;
        result += Math.Log(xd) - 0.5 * inv;
        double inv2 = inv * inv;
        result -= inv2 * (1.0 / 12.0
                       - inv2 * (1.0 / 120.0
                              - inv2 * (1.0 / 252.0
                                     - inv2 * (1.0 / 240.0
                                            - inv2 * (1.0 / 132.0)))));
        return (float)result;
    }

    /// <summary>Trigamma ψ′(x).</summary>
    public static float Trigamma(float x)
    {
        double xd = x;
        double result = 0.0;
        while (xd < 6.0)
        {
            result += 1.0 / (xd * xd);
            xd += 1.0;
        }
        // Asymptotic: ψ′(x) ≈ 1/x + 1/(2x²) + Σ B_{2k}/x^{2k+1}
        double inv = 1.0 / xd;
        double inv2 = inv * inv;
        result += inv + 0.5 * inv2
               + inv2 * inv * (1.0 / 6.0
                            - inv2 * (1.0 / 30.0
                                   - inv2 * (1.0 / 42.0)));
        return (float)result;
    }

    /// <summary>Error function erf(x). Abramowitz &amp; Stegun 7.1.26.</summary>
    public static float Erf(float x)
    {
        const float a1 = 0.254829592f;
        const float a2 = -0.284496736f;
        const float a3 = 1.421413741f;
        const float a4 = -1.453152027f;
        const float a5 = 1.061405429f;
        const float p  = 0.3275911f;
        int sign = x < 0 ? -1 : 1;
        float ax = MathF.Abs(x);
        float t = 1f / (1f + p * ax);
        float y = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-ax * ax);
        return sign * y;
    }

    /// <summary>Complementary error function erfc(x) = 1 − erf(x).</summary>
    public static float Erfc(float x) => 1f - Erf(x);

    /// <summary>Inverse error function erf⁻¹(p) for p ∈ (-1, 1). Winitzki rational approximation.</summary>
    public static float ErfInv(float p)
    {
        if (p <= -1f) return float.NegativeInfinity;
        if (p >= 1f) return float.PositiveInfinity;
        float ln1mp2 = MathF.Log(1f - p * p);
        float a = 0.147f;
        float twoOverPiA = 2f / (MathF.PI * a);
        float term = twoOverPiA + ln1mp2 * 0.5f;
        float sign = p < 0 ? -1f : 1f;
        return sign * MathF.Sqrt(MathF.Sqrt(term * term - ln1mp2 / a) - term);
    }

    /// <summary>Standard-normal CDF Φ(x).</summary>
    public static float NormalCdf(float x) => 0.5f * (1f + Erf(x * 0.70710678f));

    /// <summary>Standard-normal inverse CDF Φ⁻¹(p) for p ∈ (0, 1).</summary>
    public static float NormalIcdf(float p) => 1.41421356f * ErfInv(2f * p - 1f);

    /// <summary>Regularised lower incomplete gamma P(s, x). Series + continued-fraction switching.</summary>
    public static float GammaP(float s, float x)
    {
        if (x < 0f || s <= 0f) return float.NaN;
        if (x == 0f) return 0f;
        if (x < s + 1f)
        {
            // Power series.
            double term = 1.0 / s;
            double sum = term;
            for (int n = 1; n < 200; n++)
            {
                term *= x / (s + n);
                sum += term;
                if (Math.Abs(term) < Math.Abs(sum) * 1e-9) break;
            }
            return (float)(sum * Math.Exp(-x + s * Math.Log(x) - Lgamma(s)));
        }
        return 1f - GammaQ(s, x);
    }

    /// <summary>Regularised upper incomplete gamma Q(s, x) = 1 − P(s, x).</summary>
    public static float GammaQ(float s, float x)
    {
        if (x < 0f || s <= 0f) return float.NaN;
        if (x == 0f) return 1f;
        if (x < s + 1f) return 1f - GammaP(s, x);
        // Continued fraction (Lentz).
        double b = x + 1.0 - s;
        double c = 1.0 / 1e-30;
        double d = 1.0 / b;
        double h = d;
        for (int i = 1; i < 200; i++)
        {
            double an = -i * (i - s);
            b += 2.0;
            d = an * d + b;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = b + an / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1.0) < 1e-9) break;
        }
        return (float)(h * Math.Exp(-x + s * Math.Log(x) - Lgamma(s)));
    }

    /// <summary>Regularised incomplete beta I_x(a, b). Continued-fraction (Numerical Recipes 6.4).</summary>
    public static float BetaI(float a, float b, float x)
    {
        if (x <= 0f) return 0f;
        if (x >= 1f) return 1f;
        float bt = MathF.Exp(Lgamma(a + b) - Lgamma(a) - Lgamma(b) + a * MathF.Log(x) + b * MathF.Log(1f - x));
        if (x < (a + 1f) / (a + b + 2f))
            return (float)(bt * BetaCf(a, b, x) / a);
        return (float)(1f - bt * BetaCf(b, a, 1f - x) / b);
    }

    private static double BetaCf(double a, double b, double x)
    {
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (Math.Abs(d) < 1e-30) d = 1e-30;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= 200; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1.0) < 1e-9) break;
        }
        return h;
    }
}
