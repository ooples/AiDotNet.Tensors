using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class TrsmTests
{
    // Naive reference: solve op(A)·X = alpha·B in place, row-major, left side.
    // Independent of production code so it is a genuine oracle.
    private static void ReferenceTrsmLeft(
        Side side, Uplo uplo, bool transA, Diag diag,
        int m, int n, double alpha,
        double[] a, int lda, double[] b, int ldb)
    {
        // Scale B by alpha first.
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                b[i * ldb + j] *= alpha;

        // Effective triangular access: A(r,c) with optional transpose.
        double A(int r, int c) => transA ? a[c * lda + r] : a[r * lda + c];

        bool lower = (uplo == Uplo.Lower) ^ transA; // transpose flips triangle
        if (side != Side.Left) throw new NotSupportedException("test covers Left only here");

        if (lower)
        {
            // Forward substitution: row 0..m-1
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = b[i * ldb + j];
                    for (int kk = 0; kk < i; kk++) sum -= A(i, kk) * b[kk * ldb + j];
                    b[i * ldb + j] = diag == Diag.Unit ? sum : sum / A(i, i);
                }
        }
        else
        {
            // Back substitution: row m-1..0
            for (int i = m - 1; i >= 0; i--)
                for (int j = 0; j < n; j++)
                {
                    double sum = b[i * ldb + j];
                    for (int kk = i + 1; kk < m; kk++) sum -= A(i, kk) * b[kk * ldb + j];
                    b[i * ldb + j] = diag == Diag.Unit ? sum : sum / A(i, i);
                }
        }
    }

    [Fact]
    public void Trsm_FP64_LeftLowerNoTransNonUnit_SingleRhs_MatchesReference()
    {
        const int m = 5, n = 1;
        var rng = new Random(42);
        double[] a = new double[m * m];
        double[] b = new double[m * n];
        // Lower-triangular A with a strong diagonal so it is well-conditioned.
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j <= i; j++) a[i * m + j] = rng.NextDouble() * 2 - 1;
            a[i * m + i] += m; // dominant diagonal
        }
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] expected = (double[])b.Clone();
        ReferenceTrsmLeft(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 1.0, a, m, expected, n);

        double[] actual = (double[])b.Clone();
        BlasManagedLib.Trsm<double>(
            Side.Left, Uplo.Lower, transA: false, Diag.NonUnit,
            m, n, 1.0, a, m, actual, n);

        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(expected[i], actual[i], 10); // 10 decimal places
    }

    private static void ReferenceTrsmRight(
        Uplo uplo, bool transA, Diag diag, int m, int n,
        double alpha, double[] a, int lda, double[] b, int ldb)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) b[i * ldb + j] *= alpha;
        double A(int r, int c) => transA ? a[c * lda + r] : a[r * lda + c];
        bool lower = (uplo == Uplo.Lower) ^ transA;
        if (!lower)
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++)
                {
                    double sum = b[i * ldb + j];
                    for (int kk = 0; kk < j; kk++) sum -= b[i * ldb + kk] * A(kk, j);
                    b[i * ldb + j] = diag == Diag.Unit ? sum : sum / A(j, j);
                }
        else
            for (int j = n - 1; j >= 0; j--)
                for (int i = 0; i < m; i++)
                {
                    double sum = b[i * ldb + j];
                    for (int kk = j + 1; kk < n; kk++) sum -= b[i * ldb + kk] * A(kk, j);
                    b[i * ldb + j] = diag == Diag.Unit ? sum : sum / A(j, j);
                }
    }

    public static System.Collections.Generic.IEnumerable<object[]> FullMatrix()
    {
        foreach (var side in new[] { Side.Left, Side.Right })
        foreach (var uplo in new[] { Uplo.Upper, Uplo.Lower })
        foreach (var trans in new[] { false, true })
        foreach (var diag in new[] { Diag.NonUnit, Diag.Unit })
            yield return new object[] { side, uplo, trans, diag };
    }

    [Theory]
    [MemberData(nameof(FullMatrix))]
    public void Trsm_FP64_FullCoverage_MultiRhs_MatchesReference(
        Side side, Uplo uplo, bool trans, Diag diag)
    {
        const int m = 7, n = 4;            // multi-RHS
        int triDim = side == Side.Left ? m : n;
        var rng = new Random(123);
        double[] a = new double[triDim * triDim];
        for (int i = 0; i < triDim; i++)
        {
            for (int j = 0; j < triDim; j++)
                if ((uplo == Uplo.Upper && j >= i) || (uplo == Uplo.Lower && j <= i))
                    a[i * triDim + j] = rng.NextDouble() * 2 - 1;
            a[i * triDim + i] += triDim; // dominant diagonal
        }
        double[] b = new double[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] expected = (double[])b.Clone();
        if (side == Side.Left)
            ReferenceTrsmLeft(side, uplo, trans, diag, m, n, 1.0, a, triDim, expected, n);
        else
            ReferenceTrsmRight(uplo, trans, diag, m, n, 1.0, a, triDim, expected, n);

        double[] actual = (double[])b.Clone();
        BlasManagedLib.Trsm<double>(side, uplo, trans, diag, m, n, 1.0, a, triDim, actual, n);

        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(expected[i], actual[i], 9);
    }

    [Fact]
    public void Trsm_FP32_LeftLower_MatchesReference()
    {
        const int m = 6, n = 3;
        var rng = new Random(7);
        float[] a = new float[m * m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j <= i; j++) a[i * m + j] = (float)(rng.NextDouble() * 2 - 1);
            a[i * m + i] += m;
        }
        float[] b = new float[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // FP64 reference for accuracy.
        double[] a64 = Array.ConvertAll(a, x => (double)x);
        double[] e64 = new double[b.Length];
        for (int i = 0; i < b.Length; i++) e64[i] = b[i];
        ReferenceTrsmLeft(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 1.0, a64, m, e64, n);

        float[] actual = (float[])b.Clone();
        BlasManagedLib.Trsm<float>(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 1f, a, m, actual, n);

        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(e64[i], actual[i], 3); // FP32: 3 decimal places
    }

    [Fact]
    public void Trsm_Alpha_ScalesRightHandSide()
    {
        const int m = 4, n = 2;
        var rng = new Random(99);
        double[] a = new double[m * m];
        for (int i = 0; i < m; i++) { for (int j = 0; j <= i; j++) a[i*m+j] = rng.NextDouble(); a[i*m+i] += m; }
        double[] b = new double[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble();

        double[] expected = (double[])b.Clone();
        ReferenceTrsmLeft(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 2.5, a, m, expected, n);
        double[] actual = (double[])b.Clone();
        BlasManagedLib.Trsm<double>(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 2.5, a, m, actual, n);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 9);
    }
}
