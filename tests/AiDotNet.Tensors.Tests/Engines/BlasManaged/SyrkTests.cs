using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class SyrkTests
{
    // Reference: C = alpha*op(A)*op(A)^T + beta*C, full dense, then caller checks triangle.
    // trans=false: A is n×k, op(A)=A. trans=true: A is k×n, op(A)=A^T (n×k effective).
    private static double[] ReferenceSyrk(
        bool trans, int n, int k, double alpha, double[] a, int lda, double beta, double[] c, int ldc)
    {
        var outC = (double[])c.Clone();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double dot = 0;
                for (int p = 0; p < k; p++)
                {
                    // op(A)[i,p] and op(A)[j,p]
                    double aip = trans ? a[p * lda + i] : a[i * lda + p];
                    double ajp = trans ? a[p * lda + j] : a[j * lda + p];
                    dot += aip * ajp;
                }
                outC[i * ldc + j] = alpha * dot + beta * c[i * ldc + j];
            }
        return outC;
    }

    [Fact]
    public void Syrk_FP64_LowerNoTrans_MatchesReferenceTriangle()
    {
        const int n = 5, k = 3;
        var rng = new Random(42);
        double[] a = new double[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        double[] c = new double[n * n];
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 2 - 1;

        double[] expectedFull = ReferenceSyrk(false, n, k, 1.0, a, k, 0.0, c, n);

        double[] actual = (double[])c.Clone();
        BlasManagedLib.Syrk<double>(Uplo.Lower, trans: false, n, k, 1.0, a, k, 0.0, actual, n);

        // Only the lower triangle (incl diagonal) must match; upper is untouched (= original c).
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (j <= i) Assert.Equal(expectedFull[i * n + j], actual[i * n + j], 10);
                else        Assert.Equal(c[i * n + j], actual[i * n + j], 10);
    }
}
