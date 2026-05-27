using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-C (#371): small-total-work shapes route to the pack-free Streaming path
/// (eliminating the pack-A rent+pack overhead that dominates compute at small
/// scale), while larger shapes keep the packing path. Verifies both the routing
/// decision and that the pack-free path is numerically identical to PackBoth.
/// </summary>
public class SmallShapeStreamingDispatchTests
{
    [Theory]
    // Small total work (< 1M) AND not already caught by the k<32 / M·N<1024
    // rules → pack-free Streaming.
    [InlineData(64, 64, 64)]      // Tiny_64sq = 262k
    [InlineData(96, 96, 96)]      // 884k
    [InlineData(80, 80, 100)]     // 640k, k>=32, M·N>=1024
    public void SmallTotalWork_RoutesToStreaming(int m, int n, int k)
    {
        Assert.Equal(PackingMode.ForceStreaming,
            Dispatcher.SelectStrategy<double>(m, n, k, default));
    }

    [Theory]
    // Large total work keeps the packing path (pack + cache blocking pays off).
    [InlineData(512, 512, 64, PackingMode.ForcePackAOnly)]   // WideFat = 16.7M, k<128
    [InlineData(256, 256, 256, PackingMode.ForcePackBoth)]   // 16.8M, k>=128
    [InlineData(128, 128, 128, PackingMode.ForcePackBoth)]   // 2.1M (> 1M), k>=128
    [InlineData(512, 2048, 512, PackingMode.ForcePackBoth)]  // FFN up
    public void LargeWork_KeepsPackingPath(int m, int n, int k, PackingMode expected)
    {
        Assert.Equal(expected, Dispatcher.SelectStrategy<double>(m, n, k, default));
    }

    [Fact]
    public void ExplicitPackingMode_IsHonored_OverSmallShapeStreaming()
    {
        // A caller forcing PackBoth must win even for a small shape that would
        // otherwise route pack-free.
        var opts = new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth };
        Assert.Equal(PackingMode.ForcePackBoth,
            Dispatcher.SelectStrategy<double>(64, 64, 64, opts));
    }

    [Theory]
    // The issue's correctness acceptance: pack-free output must equal PackBoth.
    [InlineData(64, 64, 64)]            // Tiny_64sq → now pack-free by default
    [InlineData(512, 512, 64)]          // WideFat → still PackAOnly by default
    public void DefaultDispatch_MatchesForcePackBoth(int m, int n, int k)
    {
        var rng = new Random(17);
        var a = new double[m * k];
        var b = new double[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        var cDefault = new double[m * n];
        var cPackBoth = new double[m * n];

        BlasManagedLib.Gemm<double>(a, k, false, b, n, false, cDefault, n, m, n, k);
        BlasManagedLib.Gemm<double>(a, k, false, b, n, false, cPackBoth, n, m, n, k,
            new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth });

        for (int i = 0; i < cDefault.Length; i++)
            Assert.True(Math.Abs(cDefault[i] - cPackBoth[i]) < 1e-9,
                $"Mismatch at {i}: default={cDefault[i]:G6}, packBoth={cPackBoth[i]:G6}");
    }

    [Fact]
    public void SmallShape_PackFree_FloatMatchesPackBoth()
    {
        const int m = 64, n = 64, k = 64;
        var rng = new Random(5);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var cDefault = new float[m * n];
        var cPackBoth = new float[m * n];
        BlasManagedLib.Gemm<float>(a, k, false, b, n, false, cDefault, n, m, n, k);
        BlasManagedLib.Gemm<float>(a, k, false, b, n, false, cPackBoth, n, m, n, k,
            new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth });

        for (int i = 0; i < cDefault.Length; i++)
            Assert.True(MathF.Abs(cDefault[i] - cPackBoth[i]) < 1e-3f,
                $"Mismatch at {i}: default={cDefault[i]:G6}, packBoth={cPackBoth[i]:G6}");
    }
}
