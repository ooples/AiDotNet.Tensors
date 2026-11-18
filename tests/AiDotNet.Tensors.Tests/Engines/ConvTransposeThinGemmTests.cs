using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Perf-Serial")]
public class ConvTransposeThinGemmTests
{
    [Theory]
    [InlineData(192)]
    [InlineData(194)]
    [InlineData(384)]
    [InlineData(512)]
    public void Thin_product_matches_double_reference_and_is_independent_of_thread_count(int k)
    {
        const int m = 1024, n = 32;
        var random = new Random(724 + k);
        var a = new float[k * m];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(random.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(random.NextDouble() - 0.5);
        var aBefore = (float[])a.Clone();
        var bBefore = (float[])b.Clone();
        var single = new float[m * n + 17];
        var parallel = new float[single.Length];
        Array.Fill(single, 777f);
        Array.Fill(parallel, 777f);
        int prior = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            if (!ConvTransposeThinGemm.TryMultiply(a, b, single, m, n, k))
            {
                Assert.All(single, value => Assert.Equal(777f, value));
                return;
            }
            CpuParallelSettings.MaxDegreeOfParallelism = 8;
            Assert.True(ConvTransposeThinGemm.TryMultiply(a, b, parallel, m, n, k));
            Assert.Equal(single, parallel);
            for (int row = 0; row < m; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    double expected = 0;
                    for (int p = 0; p < k; p++) expected += (double)a[p * m + row] * b[p * n + col];
                    Assert.InRange(Math.Abs(single[row * n + col] - expected), 0, 0.00002);
                }
            }
            for (int i = m * n; i < single.Length; i++) Assert.Equal(777f, single[i]);
            Assert.Equal(aBefore, a);
            Assert.Equal(bBefore, b);
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }
    }

    [Theory]
    [InlineData(1024, 64, 512)]
    [InlineData(1023, 32, 512)]
    [InlineData(1024, 32, 513)]
    [InlineData(1024, 32, 128)]
    public void Unsupported_shapes_leave_the_destination_untouched(int m, int n, int k)
    {
        var output = new[] { 999f };
        Assert.False(ConvTransposeThinGemm.TryMultiply(Array.Empty<float>(), Array.Empty<float>(), output, m, n, k));
        Assert.Equal(999f, output[0]);
    }
}
