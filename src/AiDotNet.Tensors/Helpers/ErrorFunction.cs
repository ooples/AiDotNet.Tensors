namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// The error function and its complement to double precision, after W. J. Cody, "Rational Chebyshev Approximations
/// for the Error Function", Math. Comp. 23 (1969), as implemented in his SPECFUN routine <c>CALERF</c>.
/// </summary>
/// <remarks>
/// Three rational approximations cover |x| ≤ 0.46875 (erf), 0.46875 &lt; |x| ≤ 4 and |x| &gt; 4 (erfc), each with
/// a maximum relative error near 1e-16. The complement is computed directly, so <c>Erfc</c> keeps its relative
/// precision in the tail where <c>1 − Erf</c> would cancel to zero.
/// </remarks>
internal static class ErrorFunction
{
    private const double Threshold = 0.46875;
    private const double InverseSqrtPi = 5.6418958354775628695e-1;
    // Below this erf(x) = 2x/√π to double precision; above this erfc(x) underflows.
    private const double Small = 1.11e-16;
    private const double Big = 26.543;

    private static readonly double[] A = { 3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02, 3.20937758913846947e03, 1.85777706184603153e-1 };
    private static readonly double[] B = { 2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03, 2.84423683343917062e03 };
    private static readonly double[] C =
    {
        5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01, 2.98635138197400131e02,
        8.81952221241769090e02, 1.71204761263407058e03, 2.05107837782607147e03, 1.23033935479799725e03,
        2.15311535474403846e-8,
    };
    private static readonly double[] D =
    {
        1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02, 1.62138957456669019e03,
        3.29079923573345963e03, 4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03,
    };
    private static readonly double[] P =
    {
        3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2,
        6.58749161529837803e-4, 1.63153871373020978e-2,
    };
    private static readonly double[] Q = { 2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3 };

    /// <summary>erf(x).</summary>
    public static double Erf(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        double y = Math.Abs(x);
        if (y <= Threshold) return Central(x);
        double tail = Tail(y);
        double result = (0.5 - tail) + 0.5;
        return x < 0 ? -result : result;
    }

    /// <summary>erfc(x) = 1 − erf(x).</summary>
    public static double Erfc(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        double y = Math.Abs(x);
        if (y <= Threshold) return 1.0 - Central(x);
        double tail = Tail(y);
        return x < 0 ? 2.0 - tail : tail;
    }

    // erf(x) for |x| ≤ 0.46875.
    private static double Central(double x)
    {
        double y = Math.Abs(x);
        double ysq = y > Small ? y * y : 0.0;
        double num = A[4] * ysq, den = ysq;
        for (int i = 0; i < 3; i++)
        {
            num = (num + A[i]) * ysq;
            den = (den + B[i]) * ysq;
        }
        return x * (num + A[3]) / (den + B[3]);
    }

    // erfc(y) for y > 0.46875.
    private static double Tail(double y)
    {
        double result;
        if (y <= 4.0)
        {
            double num = C[8] * y, den = y;
            for (int i = 0; i < 7; i++)
            {
                num = (num + C[i]) * y;
                den = (den + D[i]) * y;
            }
            result = (num + C[7]) / (den + D[7]);
        }
        else
        {
            if (y >= Big) return 0.0;
            double ysq = 1.0 / (y * y);
            double num = P[5] * ysq, den = ysq;
            for (int i = 0; i < 4; i++)
            {
                num = (num + P[i]) * ysq;
                den = (den + Q[i]) * ysq;
            }
            result = ysq * (num + P[4]) / (den + Q[4]);
            result = (InverseSqrtPi - result) / y;
        }
        // exp(−y²) split as exp(−s²)·exp(−(y − s)(y + s)) with s = y rounded down to 1/16, to keep its precision.
        double s = Math.Floor(y * 16.0) / 16.0;
        double del = (y - s) * (y + s);
        return Math.Exp(-s * s) * Math.Exp(-del) * result;
    }
}
