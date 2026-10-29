using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// A small product against a prepacked B splits its rows finer when the column sub-blocks alone are too few for the
/// threads, and runs in one dispatch in which every tile packs its own rows of A and walks all K panels. These are the
/// batch-1 ViT projections plus a two-panel, ragged case. Each must equal the direct sum to float summation's bound and
/// repeat bit for bit.
/// </summary>
public sealed class CachedGemmSmallProductTests
{
    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return d;
    }

    [Theory]
    [InlineData(65, 768, 192)]
    [InlineData(65, 192, 192)]
    [InlineData(65, 192, 768)]
    [InlineData(65, 192, 576)]
    [InlineData(37, 1000, 100)]   // two K panels, rows and columns off the tile sizes
    public void EqualsTheDirectSum_AndRepeats(int m, int k, int n)
    {
        var a = Rand(1, m * k);
        var b = Rand(2, k * n);
        var c = new float[m * n];
        SimdGemm.SgemmWithCachedB(a, b, c, m, k, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0, magnitude = 0;
                for (int p = 0; p < k; p++)
                {
                    double term = (double)a[i * k + p] * b[p * n + j];
                    sum += term; magnitude += Math.Abs(term);
                }
                Assert.True(Math.Abs(sum - c[i * n + j]) <= k * Math.Pow(2, -24) * magnitude,
                    $"[{i},{j}]: {c[i * n + j]} vs {sum}");
            }
        for (int r = 0; r < 3; r++)
        {
            var again = new float[m * n];
            SimdGemm.SgemmWithCachedB(a, b, again, m, k, n);
            Assert.Equal(c, again);
        }
    }
}
