using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class SymmTests
{
    // Reference SYMM. side=Left: C = alpha*A*B + beta*C (A is m×m symmetric).
    // side=Right: C = alpha*B*A + beta*C (A is n×n symmetric). A stored in uplo triangle.
    private static double[] ReferenceSymm(
        Side side, Uplo uplo, int m, int n, double alpha,
        double[] a, int lda, double[] b, int ldb, double beta, double[] c, int ldc)
    {
        int s = side == Side.Left ? m : n;
        double Asym(int i, int j)
        {
            bool stored = uplo == Uplo.Lower ? j <= i : j >= i;
            return stored ? a[i * lda + j] : a[j * lda + i];
        }
        var outC = (double[])c.Clone();
        for (int i = 0; i < m; i++)
            for (int l = 0; l < n; l++)
            {
                double acc = 0;
                if (side == Side.Left)
                    for (int p = 0; p < s; p++) acc += Asym(i, p) * b[p * ldb + l];
                else
                    for (int p = 0; p < s; p++) acc += b[i * ldb + p] * Asym(p, l);
                outC[i * ldc + l] = alpha * acc + beta * c[i * ldc + l];
            }
        return outC;
    }

    [Fact]
    public void Symm_FP64_LeftLower_MatchesReference()
    {
        const int m = 5, n = 3;
        var rng = new Random(42);
        double[] a = new double[m * m];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        double[] b = new double[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        double[] c = new double[m * n];
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 2 - 1;

        double[] expected = ReferenceSymm(Side.Left, Uplo.Lower, m, n, 1.0, a, m, b, n, 0.0, c, n);
        double[] actual = (double[])c.Clone();
        BlasManagedLib.Symm<double>(Side.Left, Uplo.Lower, m, n, 1.0, a, m, b, n, 0.0, actual, n);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 10);
    }

    public static System.Collections.Generic.IEnumerable<object[]> Matrix()
    {
        foreach (var side in new[] { Side.Left, Side.Right })
        foreach (var uplo in new[] { Uplo.Upper, Uplo.Lower })
            yield return new object[] { side, uplo };
    }

    [Theory]
    [MemberData(nameof(Matrix))]
    public void Symm_FP64_Coverage_AlphaBeta_MatchesReference(Side side, Uplo uplo)
    {
        const int m = 6, n = 4;
        int s = side == Side.Left ? m : n;
        var rng = new Random(321);
        double[] a = new double[s * s];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        double[] b = new double[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        double[] c = new double[m * n];
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 2 - 1;

        double alpha = 1.7, beta = -0.5;
        double[] expected = ReferenceSymm(side, uplo, m, n, alpha, a, s, b, n, beta, c, n);
        double[] actual = (double[])c.Clone();
        BlasManagedLib.Symm<double>(side, uplo, m, n, alpha, a, s, b, n, beta, actual, n);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 9);
    }

    [Fact]
    public void Symm_FP32_LeftLower_MatchesReference()
    {
        const int m = 5, n = 4;
        var rng = new Random(11);
        float[] a = new float[m * m];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] b = new float[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] c = new float[m * n];
        for (int i = 0; i < c.Length; i++) c[i] = (float)(rng.NextDouble() * 2 - 1);

        double[] a64 = Array.ConvertAll(a, x => (double)x);
        double[] b64 = Array.ConvertAll(b, x => (double)x);
        double[] c64 = Array.ConvertAll(c, x => (double)x);
        double[] expected = ReferenceSymm(Side.Left, Uplo.Lower, m, n, 1.0, a64, m, b64, n, 0.0, c64, n);

        float[] actual = (float[])c.Clone();
        BlasManagedLib.Symm<float>(Side.Left, Uplo.Lower, m, n, 1f, a, m, b, n, 0f, actual, n);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 3);
    }
}
