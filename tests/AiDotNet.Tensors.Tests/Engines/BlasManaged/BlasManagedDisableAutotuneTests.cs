using System;
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
    [InlineData(PackingMode.ForcePackAOnly)]
    [InlineData(PackingMode.ForcePackBoth)]
    [InlineData(PackingMode.ForceStreaming)]
    public void DisableAutotune_RunsTheDefaultStrategy_AsNamingItDoes(PackingMode defaultStrategy)
    {
        const int m = 3136, n = 128, k = 64;   // A^T [k, m]: the CNN classifier's weight gradient
        bool? threadDeterministic = BlasProvider.GetThreadLocalDeterministicMode();
        bool deterministic = BlasProvider.IsDeterministicMode;
        // The learned cache is a fast-mode feature (deterministic mode routes by the hardware table alone).
        if (threadDeterministic is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        BlasProvider.SetDeterministicMode(false);
        var shape = BlasManagedAutotune.EncodeShape<float>(m, n, k, true, false, 0, 0, false, false);
        try
        {
            BlasManagedAutotune.StoreStrategy(shape, defaultStrategy, ParallelismAxis.M, 64, 128, 64, 0, BlasKernelVersion.Current);
            var auto = default(BlasOptions<float>);
            Assert.Equal(defaultStrategy, Dispatcher.SelectStrategy<float>(m, n, k, true, false, in auto));

            var a = Data(k * m, 1);
            var b = Data(k * n, 2);
            var named = Run(a, b, m, n, k, defaultStrategy);
            var disabled = Run(a, b, m, n, k, PackingMode.DisableAutotune);

            for (int i = 0; i < named.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(named[i]) == BitConverter.SingleToInt32Bits(disabled[i]),
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
            // A stale-version store drops the memo entry, and the persisted one now reads as a miss.
            BlasManagedAutotune.StoreStrategy(shape, defaultStrategy, ParallelismAxis.M, 64, 128, 64, 0, "stale-test-entry");
            BlasProvider.SetDeterministicMode(deterministic);
            BlasProvider.SetThreadLocalDeterministicMode(threadDeterministic);
        }
    }
}
