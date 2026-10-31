using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// A float product narrower than one 8-wide vector block over a deep K (an im2col convolution on a 2 x 2 map at batch
/// 1) is computed through a transposed copy of B and the dot-product kernel. It must equal C = A * B for strided B and
/// C (ldb, ldc wider than n), leave C's padding columns alone, and the products that bypass it (one column, a short K,
/// eight columns) must still be right.
/// </summary>
public sealed class BlasManagedNarrowNTests
{
    [Theory]
    [InlineData(64, 4, 4608, 4, 4)]
    [InlineData(37, 2, 1100, 5, 3)]      // odd rows, strided B and C
    [InlineData(16, 7, 1024, 9, 11)]     // the route's edges: 16 rows, 7 columns, K 1024
    [InlineData(40, 1, 2048, 1, 1)]      // one column: not the route
    [InlineData(40, 4, 512, 4, 4)]       // short K: not the route
    [InlineData(40, 8, 2048, 8, 8)]      // eight columns: not the route
    public void NarrowProduct_MatchesTheReference(int m, int n, int k, int ldb, int ldc)
    {
        var a = new float[m * k];
        var b = new float[k * ldb];
        var c = new float[m * ldc];
        for (int i = 0; i < a.Length; i++) a[i] = ((i * 31) % 17 - 8) * 0.03125f;
        for (int i = 0; i < b.Length; i++) b[i] = ((i * 13) % 11 - 5) * 0.0625f;
        for (int i = 0; i < c.Length; i++) c[i] = -7f;

        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, ldb, false, c, ldc, m, n, k);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double expected = 0;
                for (int p = 0; p < k; p++) expected += (double)a[i * k + p] * b[p * ldb + j];
                Assert.True(System.Math.Abs(expected - c[i * ldc + j]) <= 1e-3 * (1 + System.Math.Abs(expected)),
                    $"C[{i},{j}] = {c[i * ldc + j]}, expected {expected}");
            }
            for (int j = n; j < ldc; j++) Assert.Equal(-7f, c[i * ldc + j]);
        }
    }
}
