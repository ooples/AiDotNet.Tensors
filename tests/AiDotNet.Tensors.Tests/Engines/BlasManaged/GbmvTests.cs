using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class GbmvTests
{
    // Build the LAPACK band array (lda = kl+ku+1, column-major band) from a dense m×n A.
    private static double[] DenseToBand(double[] a, int m, int n, int kl, int ku)
    {
        int lda = kl + ku + 1;
        double[] ab = new double[lda * n];
        for (int j = 0; j < n; j++)
            for (int i = Math.Max(0, j - ku); i <= Math.Min(m - 1, j + kl); i++)
                ab[j * lda + (ku - j + i)] = a[i * n + j];
        return ab;
    }

    // Zero out elements of dense A outside the band so the dense reference matches.
    private static void MaskBand(double[] a, int m, int n, int kl, int ku)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (j < i - kl || j > i + ku) a[i * n + j] = 0.0;
    }

    private static double[] ReferenceGbmv(
        bool trans, int m, int n, double alpha, double[] aDenseBanded, double[] x, double beta, double[] y)
    {
        int lenY = trans ? n : m, lenX = trans ? m : n;
        var outY = (double[])y.Clone();
        for (int o = 0; o < lenY; o++)
        {
            double acc = 0;
            for (int p = 0; p < lenX; p++)
            {
                // non-trans: A(o,p); trans: A(p,o)
                double aval = trans ? aDenseBanded[p * n + o] : aDenseBanded[o * n + p];
                acc += aval * x[p];
            }
            outY[o] = alpha * acc + beta * y[o];
        }
        return outY;
    }

    [Fact]
    public void Gbmv_FP64_NoTrans_MatchesDenseReference()
    {
        const int m = 6, n = 6, kl = 1, ku = 2;
        var rng = new Random(42);
        double[] aDense = new double[m * n];
        for (int i = 0; i < aDense.Length; i++) aDense[i] = rng.NextDouble() * 2 - 1;
        MaskBand(aDense, m, n, kl, ku);
        double[] ab = DenseToBand(aDense, m, n, kl, ku);
        double[] x = new double[n];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextDouble() * 2 - 1;
        double[] y = new double[m];
        for (int i = 0; i < y.Length; i++) y[i] = rng.NextDouble() * 2 - 1;

        double[] expected = ReferenceGbmv(false, m, n, 1.0, aDense, x, 0.0, y);
        double[] actual = (double[])y.Clone();
        BlasManagedLib.Gbmv<double>(false, m, n, kl, ku, 1.0, ab, kl + ku + 1, x, 1, 0.0, actual, 1);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 10);
    }
}
