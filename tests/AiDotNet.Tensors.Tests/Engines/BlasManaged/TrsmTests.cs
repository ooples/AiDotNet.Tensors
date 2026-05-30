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
}
