namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// ln|Γ(x)| evaluated in log space, so it stays finite where Γ(x) itself overflows (x &gt; 171.6).
/// </summary>
/// <remarks>
/// The Lanczos approximation (g = 7, nine coefficients) that <see cref="MathHelper.Gamma{T}"/> uses, with its
/// product taken as a sum of logarithms; the reflection formula covers x &lt; 0.5.
/// </remarks>
internal static class GammaFunction
{
    private static readonly double[] Coefficients =
    {
        0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    };

    private static readonly double HalfLogTwoPi = 0.5 * Math.Log(2 * Math.PI);

    /// <summary>ln|Γ(x)|; +∞ at the poles (zero and the negative integers).</summary>
    public static double LogGamma(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (double.IsPositiveInfinity(x)) return double.PositiveInfinity;
        if (x < 0.5)
        {
            double sin = Math.Sin(Math.PI * x);
            if (sin == 0) return double.PositiveInfinity;
            return Math.Log(Math.PI / Math.Abs(sin)) - LogGamma(1 - x);
        }
        x -= 1;
        double t = x + 7.5;
        double sum = Coefficients[0];
        for (int i = 8; i >= 1; i--) sum += Coefficients[i] / (x + i);
        return HalfLogTwoPi + (x + 0.5) * Math.Log(t) - t + Math.Log(sum);
    }
}
