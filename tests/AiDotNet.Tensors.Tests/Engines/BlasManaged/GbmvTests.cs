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

    [Theory]
    [InlineData(false, 7, 5, 2, 1)]   // non-square m>n
    [InlineData(true,  7, 5, 2, 1)]   // trans, non-square
    [InlineData(false, 5, 8, 1, 3)]   // non-square n>m
    [InlineData(true,  5, 8, 1, 3)]
    [InlineData(false, 6, 6, 0, 0)]   // diagonal only
    public void Gbmv_FP64_Coverage_AlphaBeta(bool trans, int m, int n, int kl, int ku)
    {
        var rng = new Random(321);
        double[] aDense = new double[m * n];
        for (int i = 0; i < aDense.Length; i++) aDense[i] = rng.NextDouble() * 2 - 1;
        MaskBand(aDense, m, n, kl, ku);
        double[] ab = DenseToBand(aDense, m, n, kl, ku);
        int lenX = trans ? m : n, lenY = trans ? n : m;
        double[] x = new double[lenX];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextDouble() * 2 - 1;
        double[] y = new double[lenY];
        for (int i = 0; i < y.Length; i++) y[i] = rng.NextDouble() * 2 - 1;

        double alpha = 1.7, beta = -0.5;
        double[] expected = ReferenceGbmv(trans, m, n, alpha, aDense, x, beta, y);
        double[] actual = (double[])y.Clone();
        BlasManagedLib.Gbmv<double>(trans, m, n, kl, ku, alpha, ab, kl + ku + 1, x, 1, beta, actual, 1);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 9);
    }

    [Fact]
    public void Gbmv_FP64_NonUnitStrides_MatchesReference()
    {
        const int m = 5, n = 5, kl = 1, ku = 1;
        var rng = new Random(7);
        double[] aDense = new double[m * n];
        for (int i = 0; i < aDense.Length; i++) aDense[i] = rng.NextDouble() * 2 - 1;
        MaskBand(aDense, m, n, kl, ku);
        double[] ab = DenseToBand(aDense, m, n, kl, ku);

        const int incx = 2, incy = 3;
        double[] xCompact = new double[n];
        for (int i = 0; i < n; i++) xCompact[i] = rng.NextDouble() * 2 - 1;
        double[] yCompact = new double[m];
        for (int i = 0; i < m; i++) yCompact[i] = rng.NextDouble() * 2 - 1;

        // Strided buffers.
        double[] x = new double[n * incx];
        for (int i = 0; i < n; i++) x[i * incx] = xCompact[i];
        double[] y = new double[m * incy];
        for (int i = 0; i < m; i++) y[i * incy] = yCompact[i];

        double[] expectedCompact = ReferenceGbmv(false, m, n, 1.3, aDense, xCompact, 0.7, yCompact);
        BlasManagedLib.Gbmv<double>(false, m, n, kl, ku, 1.3, ab, kl + ku + 1, x, incx, 0.7, y, incy);
        for (int i = 0; i < m; i++) Assert.Equal(expectedCompact[i], y[i * incy], 9);
    }

    [Fact]
    public void Gbmv_FP32_NoTrans_MatchesReference()
    {
        const int m = 6, n = 6, kl = 1, ku = 2;
        var rng = new Random(11);
        double[] aDense = new double[m * n];
        for (int i = 0; i < aDense.Length; i++) aDense[i] = rng.NextDouble() * 2 - 1;
        MaskBand(aDense, m, n, kl, ku);
        double[] abD = DenseToBand(aDense, m, n, kl, ku);
        float[] ab = Array.ConvertAll(abD, v => (float)v);
        double[] xD = new double[n];
        for (int i = 0; i < n; i++) xD[i] = rng.NextDouble() * 2 - 1;
        float[] x = Array.ConvertAll(xD, v => (float)v);
        double[] yD = new double[m];
        for (int i = 0; i < m; i++) yD[i] = rng.NextDouble() * 2 - 1;
        float[] y = Array.ConvertAll(yD, v => (float)v);

        double[] expected = ReferenceGbmv(false, m, n, 1.0, aDense, xD, 0.0, yD);
        BlasManagedLib.Gbmv<float>(false, m, n, kl, ku, 1f, ab, kl + ku + 1, x, 1, 0f, y, 1);
        for (int i = 0; i < m; i++) Assert.Equal(expected[i], y[i], 3);
    }
}
