using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// TryGemmEx is C := op(A)·op(B) (beta = 0) and passes the BetaZero hint to the managed GEMM, which then skips the
/// output pre-clear on write-first routes. Every route must therefore fully overwrite C: the output is pre-filled
/// with NaN (any element left unwritten, or accumulated into, stays NaN or wrong) and compared with a double
/// reference. Shapes cover the GotoGemm per-tile route with M/N tails, the tiny-shape bypass, GEMV, and transposes.
/// </summary>
public class GemmOverwritesDirtyOutputTests
{
    [Theory]
    [InlineData(300, 517, 260, false, false)]   // GotoGemm per-tile, ragged M/N tails
    [InlineData(1024, 512, 2048, false, false)] // GotoGemm, deep-ish K
    [InlineData(5, 7, 3, false, false)]         // tiny-shape bypass
    [InlineData(1, 300, 200, false, false)]     // GEMV row
    [InlineData(200, 1, 300, false, false)]     // GEMV column
    [InlineData(150, 130, 90, true, false)]     // transposed A (strategy path)
    [InlineData(150, 130, 90, false, true)]     // transposed B (strategy path)
    public void Float_OverwritesNaNPrefilledOutput(int m, int n, int k, bool transA, bool transB)
    {
        var rng = new Random(m * 31 + n * 7 + k);
        var a = new float[m * k]; var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);
        var c = new float[m * n];
        Array.Fill(c, float.NaN);

        int lda = transA ? m : k, ldb = transB ? k : n;
        Assert.True(BlasProvider.TryGemmEx(m, n, k, a, 0, lda, transA, b, 0, ldb, transB, c, 0, n));

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int p = 0; p < k; p++)
                    s += (double)(transA ? a[p * m + i] : a[i * k + p]) * (transB ? b[j * k + p] : b[p * n + j]);
                float got = c[i * n + j];
                Assert.False(float.IsNaN(got), $"C[{i},{j}] never written");
                Assert.True(Math.Abs(got - s) <= 1e-3 * Math.Max(1, Math.Abs(s)), $"C[{i},{j}] = {got}, expected {s}");
            }
    }

    [Theory]
    [InlineData(300, 517, 260)]
    [InlineData(1, 300, 200)]
    public void Double_OverwritesNaNPrefilledOutput(int m, int n, int k)
    {
        var rng = new Random(m + n + k);
        var a = new double[m * k]; var b = new double[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() - 0.5;
        var c = new double[m * n];
        Array.Fill(c, double.NaN);
        Assert.True(BlasProvider.TryGemmEx(m, n, k, a, 0, k, false, b, 0, n, false, c, 0, n));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int p = 0; p < k; p++) s += a[i * k + p] * b[p * n + j];
                Assert.True(Math.Abs(c[i * n + j] - s) <= 1e-9 * Math.Max(1, Math.Abs(s)), $"C[{i},{j}] = {c[i * n + j]}, expected {s}");
            }
    }
}
