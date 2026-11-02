using System;
using System.IO;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// A DisableAutotune request runs the strategy the default selection picks, exactly as asking for that strategy by name
/// does. Resolved late, a product whose default is PackAOnly kept the 6-row PackBoth tile, tail-split on it and ran the
/// aligned rows through PackAOnly on the scalar 4 x 4 tile: the CNN classifier's 3136 x 64 x 128 A^T weight gradient
/// (the linear layers ask for DisableAutotune) took 6-8 ms on AVX-512 runners whose default table picks PackAOnly.
/// The learned-strategy cache stands in for such a table here, so the test does not depend on the host's.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public sealed class BlasManagedDisableAutotuneTests
{
    private static float[] Data(int length, int seed)
    {
        var rng = new Random(seed);
        var d = new float[length];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return d;
    }

    private static float[] Run(float[] a, float[] b, int m, int n, int k, PackingMode mode)
    {
        var c = new float[m * n];
        for (int i = 0; i < c.Length; i++) c[i] = float.NaN;   // BlasManaged overwrites C
        BlasManagedLib.Gemm<float>(a, m, true, b, n, false, c, n, m, n, k, new BlasOptions<float> { PackingMode = mode });
        return c;
    }

    [Theory]
    [InlineData(PackingMode.ForcePackAOnly, 3136, 128)]
    [InlineData(PackingMode.ForcePackBoth, 3136, 128)]
    [InlineData(PackingMode.ForceStreaming, 3136, 128)]
    [InlineData(PackingMode.ForcePackBoth, 3137, 129)]
    public void DisableAutotune_RunsTheDefaultStrategy_AsNamingItDoes(PackingMode defaultStrategy, int m, int n)
    {
        const int k = 64;   // A^T [k, m]: the CNN classifier's weight gradient
        bool? threadDeterministic = BlasProvider.GetThreadLocalDeterministicMode();
        // The learned cache is a fast-mode feature (deterministic mode routes by the hardware table alone).
        BlasProvider.SetThreadLocalDeterministicMode(false);
        const string cacheVariable = "AIDOTNET_AUTOTUNE_CACHE_PATH";
        string? previousCache = Environment.GetEnvironmentVariable(cacheVariable);
        string testCache = Path.Combine(Path.GetTempPath(), "aidotnet-review-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable(cacheVariable, testCache);
        BlasManagedAutotune.ClearStrategyMemo();
        var shape = BlasManagedAutotune.EncodeShape<float>(m, n, k, true, false, 0, 0, false, false);
        try
        {
            BlasManagedAutotune.StoreStrategy(shape, defaultStrategy, ParallelismAxis.M, 64, 128, 64, 0, BlasKernelVersion.Current);
            // Give every possible aligned interior a conflicting choice. The outer resolved plan must survive recursion.
            if (m == 3137)
                foreach (int alignedM in new[] { 3136, 3132 })
                    foreach (int alignedN in new[] { 128, 120 })
                        BlasManagedAutotune.StoreStrategy(
                            BlasManagedAutotune.EncodeShape<float>(alignedM, alignedN, k, true, false, 0, 0, false, false),
                            PackingMode.ForceStreaming, ParallelismAxis.M, 64, 128, 64, 0, BlasKernelVersion.Current);
            var auto = default(BlasOptions<float>);
            Assert.Equal(defaultStrategy, Dispatcher.SelectStrategy<float>(m, n, k, true, false, in auto));

            var a = Data(k * m, 1);
            var b = Data(k * n, 2);
            var named = Run(a, b, m, n, k, defaultStrategy);
            var disabled = Run(a, b, m, n, k, PackingMode.DisableAutotune);

            for (int i = 0; i < named.Length; i++)
                Assert.True(Unsafe.As<float, int>(ref named[i]) == Unsafe.As<float, int>(ref disabled[i]),
                    $"C[{i / n},{i % n}]: DisableAutotune {disabled[i]:R}, {defaultStrategy} {named[i]:R}");
            for (int row = 0; row < m; row += 97)
                for (int col = 0; col < n; col += 13)
                {
                    double s = 0;
                    for (int p = 0; p < k; p++) s += (double)a[p * m + row] * b[p * n + col];
                    Assert.True(Math.Abs(s - disabled[row * n + col]) <= 1e-4 * (1 + Math.Abs(s)),
                        $"C[{row},{col}] = {disabled[row * n + col]}, expected {s}");
                }
        }
        finally
        {
            BlasManagedAutotune.ClearStrategyMemo();
            Environment.SetEnvironmentVariable(cacheVariable, previousCache);
            BlasProvider.SetThreadLocalDeterministicMode(threadDeterministic);
            if (Directory.Exists(testCache)) Directory.Delete(testCache, recursive: true);
        }
    }

    [Fact]
    public void ForceStreaming_MatchesTheStreamingKernelBitForBit()
    {
        const int m = 3136, n = 128, k = 64;
        var a = Data(k * m, 31);
        var b = Data(k * n, 32);
        var expected = new float[m * n];
        var options = new BlasOptions<float> { PackingMode = PackingMode.ForceStreaming };
        StreamingStrategy.Run<float>(a, m, true, b, n, false, expected, n, m, n, k, in options);
        var actual = Run(a, b, m, n, k, PackingMode.ForceStreaming);
        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(Unsafe.As<float, int>(ref expected[i]), Unsafe.As<float, int>(ref actual[i]));
        // Observe the real routing gate: these kernels can produce identical bits on some hosts.
        // Auto must enter for this shape, while an explicit strategy must decline the same shortcut.
        var auto = default(BlasOptions<float>);
        Assert.True(BlasManagedLib.TryRouteToSimdGemm<float>(a, m, true, b, n, false, expected, n, m, n, k, in auto));
        Assert.False(BlasManagedLib.TryRouteToSimdGemm<float>(a, m, true, b, n, false, expected, n, m, n, k, in options));
    }

    [Theory]
    [InlineData(PackingMode.DisableAutotune, 2f, false)]
    [InlineData(PackingMode.DisableAutotune, 0f, true)]
    [InlineData(PackingMode.Auto, 2f, false)]
    [InlineData(PackingMode.Auto, 0f, true)]
    public void Gemm_PreservesOutputScale(PackingMode mode, float scale, bool explicitScale)
    {
        const int m = 3136, n = 128, k = 64;
        var a = Data(m * k, 11);
        var b = Data(k * n, 12);
        var baseline = Run(a, b, m, n, k, mode);
        var scaled = new float[m * n];
        BlasManagedLib.Gemm<float>(a, m, true, b, n, false, scaled, n, m, n, k,
            new BlasOptions<float>
            {
                PackingMode = mode,
                Epilogue = new Epilogue<float> { OutputScale = scale, HasOutputScale = explicitScale }
            });
        for (int i = 0; i < scaled.Length; i++)
            Assert.Equal(baseline[i] * scale, scaled[i]);
    }

    [Fact]
    public void DisableAutotune_DeterministicOutput_IsBitExactAcrossThreadCounts()
    {
        const int m = 64, n = 128, k = 3136;
        var a = Data(m * k, 21);
        var b = Data(k * n, 22);
        var automatic = new float[m * n];
        var single = new float[m * n];
        BlasManagedLib.Gemm<float>(a, k, false, b, n, false, automatic, n, m, n, k,
            new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune, Mode = BlasMode.Deterministic });
        BlasManagedLib.Gemm<float>(a, k, false, b, n, false, single, n, m, n, k,
            new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune, Mode = BlasMode.Deterministic, NumThreads = -1 });
        for (int i = 0; i < automatic.Length; i++)
            Assert.Equal(Unsafe.As<float, int>(ref single[i]), Unsafe.As<float, int>(ref automatic[i]));
    }
}
