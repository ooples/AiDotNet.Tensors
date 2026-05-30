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

    public static System.Collections.Generic.IEnumerable<object[]> Matrix()
    {
        foreach (var uplo in new[] { Uplo.Upper, Uplo.Lower })
        foreach (var trans in new[] { false, true })
            yield return new object[] { uplo, trans };
    }

    [Theory]
    [MemberData(nameof(Matrix))]
    public void Syrk_FP64_Coverage_AlphaBeta_MatchesReference(Uplo uplo, bool trans)
    {
        const int n = 6, k = 4;
        var rng = new Random(321);
        // trans=false → A is n×k (lda=k); trans=true → A is k×n (lda=n).
        int aRows = trans ? k : n, aCols = trans ? n : k, lda = aCols;
        double[] a = new double[aRows * aCols];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        double[] c = new double[n * n];
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 2 - 1;

        double alpha = 1.7, beta = -0.5;
        double[] expectedFull = ReferenceSyrk(trans, n, k, alpha, a, lda, beta, c, n);
        double[] actual = (double[])c.Clone();
        BlasManagedLib.Syrk<double>(uplo, trans, n, k, alpha, a, lda, beta, actual, n);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                bool inTri = uplo == Uplo.Lower ? j <= i : j >= i;
                if (inTri) Assert.Equal(expectedFull[i * n + j], actual[i * n + j], 9);
                else       Assert.Equal(c[i * n + j], actual[i * n + j], 9);
            }
    }

    [Fact]
    public void Syrk_FP32_LowerNoTrans_MatchesReference()
    {
        const int n = 5, k = 4;
        var rng = new Random(11);
        float[] a = new float[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] c = new float[n * n];
        for (int i = 0; i < c.Length; i++) c[i] = (float)(rng.NextDouble() * 2 - 1);

        double[] a64 = Array.ConvertAll(a, x => (double)x);
        double[] c64 = Array.ConvertAll(c, x => (double)x);
        double[] expectedFull = ReferenceSyrk(false, n, k, 1.0, a64, k, 0.0, c64, n);

        float[] actual = (float[])c.Clone();
        BlasManagedLib.Syrk<float>(Uplo.Lower, false, n, k, 1f, a, k, 0f, actual, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                Assert.Equal(expectedFull[i * n + j], actual[i * n + j], 3);
    }
}
