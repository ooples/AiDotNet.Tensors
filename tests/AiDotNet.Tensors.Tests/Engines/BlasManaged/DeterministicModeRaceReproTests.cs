// Copyright (c) AiDotNet. All rights reserved.
// Reproduction + fix verification for the flaky BlasManaged bit-match / cached-packed-buffer
// tests (PrePackedB_Output_BitMatches_LivePack, ScalarKernelTests.Gemm_WithoutMarkDirty_*,
// Gemm_WithPackedAHandle_*). Root cause: BlasProvider.IsDeterministicMode is process-global
// and is read on the CALLING thread inside BlasManaged.Gemm (tile pick, autotune, strategy
// gate). A concurrent test flipping the global flag mid-call (xUnit runs other collections in
// parallel; DisableParallelization proved unreliable on CI) changes the GEMM path/tiling, which
// (a) drifts a bit-exact comparison and (b) can reject a pre-pack handle and re-pack from a
// mutated source array. Pinning THREAD-LOCAL deterministic mode (which IsDeterministicMode
// resolves before the global) makes the calling-thread reads immune to the flip.

using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class DeterministicModeRaceReproTests
{
    /// <summary>
    /// With a background thread hammering the process-global deterministic flag, the
    /// cached-packed-buffer contract must still hold when the test pins its THREAD-LOCAL
    /// mode: the handle stays current (no re-pack from the mutated source) and the result
    /// matches the ORIGINAL weight. This is the robust fix the production tests adopt.
    /// </summary>
    [Fact]
    public void CachedPackedBuffer_HoldsUnderConcurrentGlobalModeFlips_WhenThreadLocalPinned()
    {
        int m = 8, n = 8, k = 8;
        var rng = new Random(42);
        var aOriginal = new double[m * k];
        for (int i = 0; i < aOriginal.Length; i++) aOriginal[i] = rng.NextDouble() * 2 - 1;
        var b = new double[k * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        var expected = Naive(aOriginal, b, m, n, k);

        bool globalBefore = BlasProvider.IsDeterministicMode;
        bool? tlBefore = BlasProvider.GetThreadLocalDeterministicMode();

        using var cts = new CancellationTokenSource();
        var flipper = Task.Run(() =>
        {
            // Simulate a concurrent test in another collection flipping the GLOBAL flag.
            bool d = false;
            while (!cts.IsCancellationRequested)
            {
                BlasProvider.SetDeterministicMode(d);
                d = !d;
                Thread.SpinWait(50);
            }
        });

        try
        {
            // The fix: pin THREAD-LOCAL mode on the calling thread. All of Gemm's mode reads
            // (tile pick, autotune, strategy gate) resolve thread-local-first, so the flipper
            // can't disturb them.
            BlasProvider.SetThreadLocalDeterministicMode(true);

            for (int iter = 0; iter < 200; iter++)
            {
                var a = (double[])aOriginal.Clone();
                var handle = BlasManagedLib.PrePackA<double>(a, lda: k, transA: false, m, k);
                var options = new BlasOptions<double> { PackedA = handle, PackingMode = PackingMode.ForcePackBoth };

                // Mutate the source AFTER pre-pack, WITHOUT MarkDirty — the cached buffer must win.
                for (int i = 0; i < a.Length; i++) a[i] += 100.0;

                var result = new double[m * n];
                BlasManagedLib.Gemm<double>(a, k, false, b, n, false, result, n, m, n, k, options);

                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(expected[i] - result[i]) < 1e-9,
                        $"iter {iter} idx {i}: cached-buffer contract broke under concurrent global flip " +
                        $"(expected {expected[i]} from original weight, got {result[i]})");
                handle.Dispose();
            }
        }
        finally
        {
            cts.Cancel();
            flipper.Wait(TimeSpan.FromSeconds(5));
            BlasProvider.SetThreadLocalDeterministicMode(tlBefore);
            BlasProvider.SetDeterministicMode(globalBefore);
        }
    }

    private static double[] Naive(double[] a, double[] b, int m, int n, int k)
    {
        var e = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
                e[i * n + j] = acc;
            }
        return e;
    }
}
