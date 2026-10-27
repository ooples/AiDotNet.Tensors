using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Gemm = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public sealed class DenseTransATailTests
{
    [Fact]
    public void ParallelPackedGemm_DoesNotRetainUnusedArenaBuffers()
    {
        const int m = 516, n = 512, k = 512;
        var a = new float[k * m];
        var b = new float[k * n];
        var c = new float[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = 1;
        for (int i = 0; i < b.Length; i++) b[i] = 1;
        using var arena = AiDotNet.Tensors.Helpers.TensorArena.Create();
        int tracked = arena.TrackedArrayCount;
        var options = new BlasOptions<float>
        {
            PackingMode = PackingMode.ForcePackBoth, NumThreads = 2,
            Mc = 96, Nc = 512, Kc = 128,
        };
        for (int repeat = 0; repeat < 2; repeat++)
        {
            Gemm.Gemm<float>(a, m, true, b, n, false, c, n, m, n, k, in options);
            Assert.Equal(tracked, arena.TrackedArrayCount);
            Assert.All(c, value => Assert.Equal((float)k, value));
        }
    }

    [Theory]
    [InlineData(512, 914, 512)]
    [InlineData(513, 515, 513)]
    [InlineData(512, 2048, 3072)]
    public void TransposedA_WithTails_PreservesReferencePaddingAndThreadDeterminism(int m, int n, int k)
    {
        int lda = m + 3, ldb = n + 5, ldc = n + 7;
        var random = new Random(1564);
        var a = new float[k * lda];
        var b = new float[k * ldb];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(random.NextDouble() - .5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(random.NextDouble() - .5);
        float[]? first = null;
        foreach (int threads in new[] { 1, 3, 8 })
        {
            var c = new float[m * ldc];
            for (int i = 0; i < c.Length; i++) c[i] = float.NaN;
            var options = new BlasOptions<float> { NumThreads = threads };
            Gemm.Gemm<float>(a, lda, true, b, ldb, false, c, ldc, m, n, k, in options);
            for (int row = 0; row < m; row++)
            {
                for (int col = n; col < ldc; col++) Assert.True(float.IsNaN(c[row * ldc + col]));
                for (int col = 0; col < n; col++)
                {
                    Assert.False(float.IsNaN(c[row * ldc + col]));
                    if (first is not null) Assert.Equal(first[row * ldc + col], c[row * ldc + col]);
                }
            }
            // Check interior, both tails and the corner against an independent double accumulation.
            foreach (int row in new[] { 0, m / 2, m - 3, m - 1 })
            foreach (int col in new[] { 0, n / 2, n - 3, n - 1 })
            {
                double expected = 0;
                for (int p = 0; p < k; p++) expected += (double)a[p * lda + row] * b[p * ldb + col];
                Assert.InRange(Math.Abs(c[row * ldc + col] - expected), 0, 1e-3);
            }
            first ??= c;
        }
    }
}
