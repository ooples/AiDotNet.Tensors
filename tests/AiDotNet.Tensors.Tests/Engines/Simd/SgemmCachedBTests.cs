#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

public class SgemmCachedBTests
{
    [Fact]
    public void CachedB_NarrowN_MatchesNaiveReference()
    {
        const int m = 13;
        const int k = 17;
        const int n = 10;

        var rng = new Random(0x299300);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var actual = new float[m * n];
        SimdGemm.SgemmWithCachedB(a, b, actual, m, k, n);

        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                float sum = 0f;
                for (int p = 0; p < k; p++)
                    sum += a[row * k + p] * b[p * n + col];
                expected[row * n + col] = sum;
            }
        }

        for (int i = 0; i < actual.Length; i++)
            Assert.True(MathF.Abs(actual[i] - expected[i]) < 1e-4f,
                $"Mismatch at {i}: expected {expected[i]}, actual {actual[i]}");
    }
}
#endif
