// Copyright (c) AiDotNet. All rights reserved.
// CpuParallelSettings.ParallelForOrSerial routes its parallel path through the persistent
// worker pool (PersistentParallelExecutor) instead of raw Parallel.For when UseCooperativePool
// is on (the default). These pin that the routing is a pure scheduling change — bit-identical
// results to Parallel.For (the Action overload's iterations write disjoint outputs, so order is
// irrelevant) — including under many concurrent callers.
//
// Serialized (it toggles the process-wide CpuParallelSettings.UseCooperativePool) and restores
// the prior value in a finally so it can't leak into other tests.

using System;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Pool-Serial")]
public class ParallelForCooperativeRoutingTests
{
    private static double Kernel(double x) => Math.Sqrt(Math.Abs(x)) + x * x - 0.5 * x;

    [Fact]
    public void Cooperative_MatchesParallelFor_BitIdentical()
    {
        const int n = 50_000; // well above the serial grain → the parallel path runs
        var src = new double[n];
        var rng = new Random(7);
        for (int i = 0; i < n; i++) src[i] = rng.NextDouble() * 2 - 1;

        bool prev = CpuParallelSettings.UseCooperativePool;
        try
        {
            var reference = new double[n];
            CpuParallelSettings.UseCooperativePool = false;
            CpuParallelSettings.ParallelForOrSerial(0, n, n, i => reference[i] = Kernel(src[i]));

            var coop = new double[n];
            CpuParallelSettings.UseCooperativePool = true;
            CpuParallelSettings.ParallelForOrSerial(0, n, n, i => coop[i] = Kernel(src[i]));

            for (int i = 0; i < n; i++)
                Assert.Equal(reference[i], coop[i]); // exact — disjoint writes, no reduction
        }
        finally { CpuParallelSettings.UseCooperativePool = prev; }
    }

    [Fact]
    public void Cooperative_CorrectUnderManyConcurrentCallers()
    {
        const int n = 30_000;
        var src = new double[n];
        var rng = new Random(11);
        for (int i = 0; i < n; i++) src[i] = rng.NextDouble();
        var expected = new double[n];
        for (int i = 0; i < n; i++) expected[i] = Kernel(src[i]);

        bool prev = CpuParallelSettings.UseCooperativePool;
        try
        {
            CpuParallelSettings.UseCooperativePool = true;
            int mismatches = 0;
            var threads = new Thread[8];
            for (int t = 0; t < threads.Length; t++)
            {
                threads[t] = new Thread(() =>
                {
                    for (int rep = 0; rep < 100; rep++)
                    {
                        var outp = new double[n];
                        CpuParallelSettings.ParallelForOrSerial(0, n, n, i => outp[i] = Kernel(src[i]));
                        for (int i = 0; i < n; i++)
                            if (outp[i] != expected[i]) { Interlocked.Increment(ref mismatches); break; }
                    }
                });
                threads[t].Start();
            }
            foreach (var th in threads) th.Join();
            Assert.Equal(0, mismatches);
        }
        finally { CpuParallelSettings.UseCooperativePool = prev; }
    }

    [Fact]
    public void Cooperative_RethrowsBodyException()
    {
        bool prev = CpuParallelSettings.UseCooperativePool;
        try
        {
            CpuParallelSettings.UseCooperativePool = true;
            var ex = Assert.ThrowsAny<Exception>(() =>
                CpuParallelSettings.ParallelForOrSerial(0, 40_000, 40_000, i =>
                {
                    if (i == 12_345) throw new InvalidOperationException("boom-12345");
                }));
            Assert.Contains("boom-12345", ex.Message);
        }
        finally { CpuParallelSettings.UseCooperativePool = prev; }
    }
}
