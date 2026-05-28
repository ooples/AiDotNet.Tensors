using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-C (#371) + #375 hybrid: small-total-work shapes route pack-free on all
/// hardware (the pack-A overhead dominates at small scale everywhere). Larger
/// shapes' strategy is now HARDWARE-AWARE — what wins on a 32-thread Ryzen
/// (blocking) differs from a 16-thread box (streaming), so those routes are
/// asserted against the per-hardware StrategyDefaultTable with explicit keys,
/// not as a single universal answer. Numerical equality to PackBoth is preserved.
/// </summary>
// DefaultDispatch/SmallShape correctness compares strategy outputs, sensitive to the
// global reduction-order state; serialize so a concurrent mutator can't flip it (#375 de-flake).
[Collection("BlasManaged-Stats-Serial")]
public class SmallShapeStreamingDispatchTests
{
    [Theory]
    // Small total work (< 1M): pack-free Streaming on ALL hardware (work-based,
    // not hardware-dependent — the pack round-trip never pays at this scale).
    [InlineData(64, 64, 64)]      // Tiny_64sq = 262k
    [InlineData(96, 96, 96)]      // 884k
    [InlineData(80, 80, 100)]     // 640k, k>=32, M·N>=1024
    public void SmallTotalWork_RoutesToStreaming(int m, int n, int k)
    {
        Assert.Equal(PackingMode.ForceStreaming,
            Dispatcher.SelectStrategy<double>(m, n, k, default));
    }

    [Theory]
    // Large total work, hardware-AGNOSTIC: high-K / huge-work shapes route to
    // PackBoth on every key (pack-B amortises across many ic-blocks everywhere).
    [InlineData(256, 256, 256)]   // 16.8M, k≥256
    [InlineData(512, 2048, 512)]  // 536M, FFN up
    public void LargeHighKWork_RoutesToPackBoth_AllHardware(int m, int n, int k)
    {
        Assert.Equal(PackingMode.ForcePackBoth,
            Dispatcher.SelectStrategy<double>(m, n, k, default));
    }

    [Theory]
    // HARDWARE-DEPENDENT routing — the #375 collision shapes. The table is consulted only
    // for the TRANSPOSED shapes Sub-S declines, so values are the transposed optima:
    //   - 512×512×64 (ThinK): PackBoth on both core bands (transposed Streaming is 2.5×
    //     slower — G12-caught; the non-transposed Streaming win never reaches this table).
    //   - 128³ (true cube): cpu16 → Streaming (transposed Streaming wins), cpu32 → PackBoth.
    [InlineData("avx2", "amd", 2, 512, 512, 64, PackingMode.ForcePackAOnly)]   // cpu32 ThinK → blocking
    [InlineData("avx2", "amd", 1, 512, 512, 64, PackingMode.ForcePackBoth)]    // cpu16 ThinK → blocking (transposed)
    [InlineData("avx2", "amd", 2, 128, 128, 128, PackingMode.ForcePackBoth)]   // cpu32 cube → blocking
    [InlineData("avx2", "amd", 1, 128, 128, 128, PackingMode.ForceStreaming)]  // cpu16 cube → streaming
    public void LargeLowKWork_RoutingIsHardwareAware(string simd, string vendor, int bucket,
        int m, int n, int k, PackingMode expected)
    {
        var key = new HardwareFingerprint.HwKey(simd, vendor, bucket);
        Assert.Equal(expected, StrategyDefaultTable.Route(key, m, n, k));
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
