#if NET5_0_OR_GREATER
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// The blocked route of <see cref="Avx2Streaming.RunFp32"/> (four or more rows over a row-major B): C += A * B against
/// an independent reference, on shapes that reach every tail. 5 x 25 x 257 gives a 16-column tile, an 8-column tile, a
/// one-column recursive tail, two K slices and a one-row remainder; ldb and ldc are padded, and the padding must stay
/// untouched. Transposed A takes the same route.
/// </summary>
public sealed class Avx2StreamingBlockedTests
{
    [SkippableTheory]
    [InlineData(5, 25, 257, false)]
    [InlineData(5, 25, 257, true)]
    [InlineData(4, 16, 256, false)]   // exactly one row block, one column tile, one K slice
    [InlineData(9, 33, 600, false)]   // three K slices, a 16 + 16 + 1 column split, a one-row remainder
    public void BlockedRoute_AccumulatesTheReferenceProduct(int m, int n, int k, bool transA)
    {
        Skip.IfNot(Avx2Streaming.IsSupported, "AVX2 + FMA are not available.");
        int lda = transA ? m + 3 : k + 3, ldb = n + 5, ldc = n + 7;
        var a = new float[transA ? k * lda : m * lda];
        var b = new float[k * ldb];
        var c = new float[m * ldc];
        for (int i = 0; i < a.Length; i++) a[i] = ((i * 29) % 13 - 6) * 0.0625f;
        for (int i = 0; i < b.Length; i++) b[i] = ((i * 17) % 11 - 5) * 0.125f;
        for (int i = 0; i < c.Length; i++) c[i] = ((i * 7) % 5 - 2) * 0.5f;
        var c0 = (float[])c.Clone();

        Avx2Streaming.RunFp32(a, lda, transA, b, ldb, false, c, ldc, m, n, k);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double expected = c0[i * ldc + j];
                for (int p = 0; p < k; p++)
                {
                    float av = transA ? a[p * lda + i] : a[i * lda + p];
                    expected += (double)av * b[p * ldb + j];
                }
                Assert.True(System.Math.Abs(expected - c[i * ldc + j]) <= 1e-3 * (1 + System.Math.Abs(expected)),
                    $"C[{i},{j}] = {c[i * ldc + j]}, expected {expected}");
            }
            for (int j = n; j < ldc; j++) Assert.Equal(c0[i * ldc + j], c[i * ldc + j]);
        }
    }
}
#endif
