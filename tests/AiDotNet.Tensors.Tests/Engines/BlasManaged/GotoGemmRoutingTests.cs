using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public sealed class GotoGemmRoutingTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(32)]
    [InlineData(47)]
    public void IsPreferredForThreadBudget_RejectsSmallAndMidSizedMachines(int threadBudget)
    {
        Assert.False(GotoGemmFp32.IsPreferredForThreadBudget(threadBudget));
    }

    [Theory]
    [InlineData(48)]
    [InlineData(64)]
    [InlineData(128)]
    public void IsPreferredForThreadBudget_AllowsManyCoreMachines(int threadBudget)
    {
        Assert.True(GotoGemmFp32.IsPreferredForThreadBudget(threadBudget));
    }

    // #653: below the 48-thread gate, PackBoth splits along M only, at no fewer than 64 rows per
    // block. A transformer-sized M therefore cannot fill the budget, and those shapes route to the
    // 2D-tiled kernel instead. Large-M GEMMs keep PackBoth.
    [Theory]
    [InlineData(256, 16)]
    [InlineData(256, 32)]
    [InlineData(512, 16)]
    [InlineData(64, 2)]
    public void PackBothUnderOccupies_WhenItsMBlocksCannotFillTheBudget(int m, int threadBudget)
    {
        Assert.True(GotoGemmFp32.PackBothUnderOccupies(m, threadBudget));
    }

    [Theory]
    [InlineData(1024, 16)]
    [InlineData(4096, 32)]
    [InlineData(256, 4)]
    [InlineData(256, 1)]
    public void PackBothUnderOccupies_IsFalseWhenPackBothCanFillTheBudget(int m, int threadBudget)
    {
        Assert.False(GotoGemmFp32.PackBothUnderOccupies(m, threadBudget));
    }

    [Theory]
    [InlineData(256, 768, 768)]
    [InlineData(256, 768, 3072)]
    [InlineData(250, 768, 520)]
    public void TransformerShapes_OnTheNewRoute_MatchADoubleReference(int m, int k, int n)
    {
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 16;
            var rng = RandomHelper.CreateSeededRandom(653);
            var a = new Tensor<float>(new[] { m, k });
            var b = new Tensor<float>(new[] { k, n });
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);

            var c = new CpuEngine().BatchMatMul(a, b);

            Assert.Equal(new[] { m, n }, c.Shape.ToArray());
            double worst = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double expected = 0;
                    for (int p = 0; p < k; p++) expected += (double)a[i, p] * b[p, j];
                    worst = Math.Max(worst, Math.Abs(expected - c[i, j]));
                }

            // K=768 products of values in [-0.5, 0.5] sum to O(5); float accumulation error is ~1e-5,
            // and a mis-tiled or skipped block is off by O(0.1) or more.
            Assert.True(worst < 1e-4, $"max |C - C_ref| = {worst:E3}");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }

    // The routed per-tile kernel writes C first (overwrite kernel on the first K-panel, tail strips
    // zeroed per tile), so BlasManaged no longer clears C before it. A NaN-filled destination proves
    // every element is written: any cell left to a "+=" over stale memory stays NaN.
    [Theory]
    [InlineData(256, 768, 768)]
    [InlineData(250, 768, 520)]
    [InlineData(256, 3072, 768)]
    public void RoutedPath_OverwritesAGarbageDestination(int m, int k, int n)
    {
        int before = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 16;
            var rng = RandomHelper.CreateSeededRandom(9);
            var a = new float[m * k];
            var b = new float[k * n];
            var c = new float[m * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < c.Length; i++) c[i] = float.NaN;

            AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
                a, k, false, b, n, false, c, n, m, n, k,
                new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune, BetaZero = true });

            double worst = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double expected = 0;
                    for (int p = 0; p < k; p++) expected += (double)a[i * k + p] * b[p * n + j];
                    float actual = c[i * n + j];
                    Assert.False(float.IsNaN(actual), $"C[{i},{j}] was never written");
                    worst = Math.Max(worst, Math.Abs(expected - actual));
                }
            Assert.True(worst < 1e-3, $"max |C - C_ref| = {worst:E3}");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = before;
        }
    }
}
