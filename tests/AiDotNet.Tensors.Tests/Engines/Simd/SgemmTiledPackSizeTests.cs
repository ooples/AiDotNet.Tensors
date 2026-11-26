using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

#if NET5_0_OR_GREATER   // SgemmTiled, and its packing, is compiled only on .NET 5+
namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// SgemmTiled rents its packed panels per call. They were sized by the Mc/Kc/Nc ceilings (B: 512 x 4096 floats,
/// 8 MB) whatever the shape, so a compiled training step's small weight-gradient GEMMs (transposed A) allocated
/// 8 MB on each worker thread that ran one.
/// </summary>
public sealed class SgemmTiledPackSizeTests
{
    [Fact]
    public void SmallProblem_PacksOnlyItsOwnPanels()
    {
        // A [64 x 32] by [32 x 16] product: one micro-tile-rounded panel of each operand.
        var (packedA, packedB) = SimdGemm.TiledPackSizes(128, 64, 32, 16);

        Assert.Equal(66 * 32, packedA);   // 64 rows round up to 11 micro-tiles of 6
        Assert.Equal(32 * 16, packedB);
    }

    [Fact]
    public void LargeProblem_IsBoundedByThePanelCeilings()
    {
        var (packedA, packedB) = SimdGemm.TiledPackSizes(128, 5000, 5000, 5000);

        Assert.Equal(132 * 512, packedA);
        Assert.Equal(512 * 4096, packedB);
    }

    [Theory]
    [InlineData(7, 5, 19)]
    [InlineData(64, 32, 16)]
    [InlineData(130, 600, 33)]
    [InlineData(200, 70, 4100)]
    public void TransposedProducts_MatchTheReference(int m, int k, int n)
    {
        var random = new Random(m * 31 + k * 7 + n);
        var a = new float[k * m];   // stored [k, m], read transposed
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(random.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(random.NextDouble() - 0.5);
        var c = new float[m * n];

        SimdGemm.SgemmAddInternal(a, m, true, b, n, false, c, m, k, n, allowParallel: false);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double expected = 0;
                for (int p = 0; p < k; p++) expected += (double)a[p * m + i] * b[p * n + j];
                Assert.True(Math.Abs(expected - c[i * n + j]) <= 1e-4 * Math.Max(1.0, Math.Abs(expected)),
                    $"c[{i},{j}] = {c[i * n + j]}, expected {expected}");
            }
        }
    }
}
#endif
