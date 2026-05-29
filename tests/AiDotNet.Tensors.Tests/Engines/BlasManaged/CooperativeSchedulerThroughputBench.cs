// Copyright (c) AiDotNet. All rights reserved.
// Phase 2 benchmark: does the cooperative scheduler beat the legacy
// StreamingWorkerPool under CONCURRENT GEMM dispatch? The legacy pool runs a
// contended second caller fully serial (#492); the cooperative scheduler lets
// concurrent dispatches interleave their chunks on a shared queue. This measures
// wall-clock for N threads each running many small-K (StreamingStrategy) GEMMs,
// legacy vs cooperative, and reports the ratio.
//
// Category=Performance so it's excluded from normal/CI runs (timing-sensitive). The
// only hard assertion is a generous anti-regression / anti-deadlock guard; the
// decision to flip CooperativeGemmScheduler.Enabled on is made from the reported
// numbers, not a brittle perf threshold.

using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged.Pool;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
[Trait("Category", "Performance")]
public class CooperativeSchedulerThroughputBench
{
    private readonly ITestOutputHelper _out;
    public CooperativeSchedulerThroughputBench(ITestOutputHelper output) => _out = output;

    [Theory]
    // K<32 → StreamingStrategy → StreamingWorkerPool (the path wired to the
    // cooperative scheduler). Small vs large M·N to see where each pool wins:
    // small GEMMs are dominated by per-chunk coordination overhead; large GEMMs
    // expose the legacy serial-fallback's wasted parallelism under concurrency.
    [InlineData(256, 256, 24, 200, 0)]    // small, many callers: ~1.5M work/GEMM
    [InlineData(1024, 1024, 24, 40, 0)]   // large, many callers: ~25M work/GEMM
    [InlineData(1024, 1024, 24, 60, 2)]   // large, FEW callers — legacy serializes the 2nd to 1 core
    [InlineData(1024, 1024, 24, 50, 4)]   // large, few callers
    public void ConcurrentGemm_Cooperative_Vs_Legacy_Throughput(int m, int n, int k, int itersPerThread, int threadsOverride)
    {
        int threads = threadsOverride > 0 ? threadsOverride : Math.Max(4, Environment.ProcessorCount);

        bool beforeDet = BlasProvider.IsDeterministicMode;
        bool beforeCoop = CooperativeGemmScheduler.Enabled;
        try
        {
            BlasProvider.SetDeterministicMode(true); // both paths → managed

            var rng = new Random(123);
            var a = new float[m * k];
            var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            // Warm up both paths (JIT, pool spin-up, autotune-free deterministic route).
            CooperativeGemmScheduler.Enabled = false; RunConcurrent(a, b, m, n, k, threads, 5);
            CooperativeGemmScheduler.Enabled = true; RunConcurrent(a, b, m, n, k, threads, 5);

            CooperativeGemmScheduler.Enabled = false;
            double legacyMs = RunConcurrent(a, b, m, n, k, threads, itersPerThread);

            CooperativeGemmScheduler.Enabled = true;
            double coopMs = RunConcurrent(a, b, m, n, k, threads, itersPerThread);

            double speedup = legacyMs / coopMs;
            _out.WriteLine($"Concurrent GEMM {m}x{n}x{k}, {threads} threads x {itersPerThread} iters:");
            _out.WriteLine($"  legacy  StreamingWorkerPool : {legacyMs,8:F1} ms");
            _out.WriteLine($"  cooperative scheduler       : {coopMs,8:F1} ms");
            _out.WriteLine($"  speedup (legacy/coop)       : {speedup,8:F2}x");

            // Anti-regression / anti-deadlock guard only: cooperative must not be
            // catastrophically slower. The go/no-go on flipping Enabled is read from
            // the numbers above, not asserted here.
            Assert.True(coopMs < legacyMs * 5.0,
                $"Cooperative scheduler is >5x slower than legacy ({coopMs:F1} vs {legacyMs:F1} ms) — investigate before enabling.");
        }
        finally
        {
            CooperativeGemmScheduler.Enabled = beforeCoop;
            BlasProvider.SetDeterministicMode(beforeDet);
        }
    }

    private static double RunConcurrent(float[] a, float[] b, int m, int n, int k, int threads, int iters)
    {
        using var cts = new CancellationTokenSource();
        using var gate = new Barrier(threads + 1);
        var workers = new Task[threads];
        for (int t = 0; t < threads; t++)
        {
            workers[t] = Task.Factory.StartNew(() =>
            {
                var c = new float[m * n];
                gate.SignalAndWait();
                // Observe cancellation between GEMMs so a scheduler deadlock can't hang
                // the benchmark indefinitely (which would skip the caller's finally).
                for (int it = 0; it < iters && !cts.IsCancellationRequested; it++)
                    BlasManagedLib.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k);
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }
        gate.SignalAndWait();
        var sw = Stopwatch.StartNew();
        // Bounded wait: on a deadlock, cancel + quiesce and surface it rather than hang.
        bool done = Task.WaitAll(workers, TimeSpan.FromSeconds(120));
        sw.Stop();
        if (!done)
        {
            cts.Cancel();
            Task.WaitAll(workers, TimeSpan.FromSeconds(30));
            throw new Xunit.Sdk.XunitException(
                $"RunConcurrent({m}x{n}x{k}, {threads} threads) did not finish in 120s — possible scheduler deadlock.");
        }
        return sw.Elapsed.TotalMilliseconds;
    }
}
