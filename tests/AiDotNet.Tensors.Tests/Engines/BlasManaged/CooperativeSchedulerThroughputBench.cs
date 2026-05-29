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

    // Repetitions per shape. Odd so the median is a real sample. Each rep uses a
    // DISTINCT seed (fresh operands) so the verdict is robust to a single lucky/
    // unlucky operand layout — not just to per-run timing jitter. A single A/B
    // sample on this scheduler swings 0.7x↔1.3x run-to-run; the median over many
    // seeds is what we trust to decide retirement of the legacy pool.
    private const int Reps = 9;

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

            var legacyMsAll = new double[Reps];
            var coopMsAll = new double[Reps];
            var ratios = new double[Reps];

            for (int rep = 0; rep < Reps; rep++)
            {
                // Fresh operands per rep with a distinct seed.
                var rng = new Random(1000 + rep * 7919);
                var a = new float[m * k];
                var b = new float[k * n];
                for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
                for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

                // Warm both paths on THIS rep's data (JIT, pool spin-up, deterministic route).
                CooperativeGemmScheduler.Enabled = false; RunConcurrent(a, b, m, n, k, threads, 5);
                CooperativeGemmScheduler.Enabled = true; RunConcurrent(a, b, m, n, k, threads, 5);

                double legacyMs, coopMs;
                // Alternate measurement order per rep so neither path systematically
                // benefits from a warmer cache / thermal ramp by always running second.
                if ((rep & 1) == 0)
                {
                    CooperativeGemmScheduler.Enabled = false; legacyMs = RunConcurrent(a, b, m, n, k, threads, itersPerThread);
                    CooperativeGemmScheduler.Enabled = true; coopMs = RunConcurrent(a, b, m, n, k, threads, itersPerThread);
                }
                else
                {
                    CooperativeGemmScheduler.Enabled = true; coopMs = RunConcurrent(a, b, m, n, k, threads, itersPerThread);
                    CooperativeGemmScheduler.Enabled = false; legacyMs = RunConcurrent(a, b, m, n, k, threads, itersPerThread);
                }

                legacyMsAll[rep] = legacyMs;
                coopMsAll[rep] = coopMs;
                ratios[rep] = legacyMs / coopMs;
            }

            double medLegacy = Median(legacyMsAll);
            double medCoop = Median(coopMsAll);
            double medRatio = Median(ratios);
            double minRatio = Min(ratios);
            double maxRatio = Max(ratios);

            _out.WriteLine($"Concurrent GEMM {m}x{n}x{k}, {threads} threads x {itersPerThread} iters, {Reps} seeds:");
            _out.WriteLine($"  legacy  StreamingWorkerPool (median) : {medLegacy,8:F1} ms");
            _out.WriteLine($"  cooperative scheduler       (median) : {medCoop,8:F1} ms");
            _out.WriteLine($"  speedup legacy/coop  median={medRatio:F2}x  min={minRatio:F2}x  max={maxRatio:F2}x");
            _out.WriteLine($"    >1.0 = cooperative faster; per-seed ratios = [{string.Join(", ", System.Linq.Enumerable.Select(ratios, r => r.ToString("F2")))}]");

            // Anti-regression / anti-deadlock guard only, on the MEDIAN so single-seed
            // noise can't fail it: cooperative must not be catastrophically slower. The
            // go/no-go on flipping Enabled is read from the reported medians, not asserted.
            Assert.True(medCoop < medLegacy * 5.0,
                $"Cooperative scheduler median is >5x slower than legacy ({medCoop:F1} vs {medLegacy:F1} ms) — investigate before enabling.");
        }
        finally
        {
            CooperativeGemmScheduler.Enabled = beforeCoop;
            BlasProvider.SetDeterministicMode(beforeDet);
        }
    }

    private static double Median(double[] xs)
    {
        var copy = (double[])xs.Clone();
        Array.Sort(copy);
        int mid = copy.Length / 2;
        return (copy.Length & 1) == 1 ? copy[mid] : (copy[mid - 1] + copy[mid]) / 2.0;
    }

    private static double Min(double[] xs)
    {
        double m = xs[0];
        for (int i = 1; i < xs.Length; i++) if (xs[i] < m) m = xs[i];
        return m;
    }

    private static double Max(double[] xs)
    {
        double m = xs[0];
        for (int i = 1; i < xs.Length; i++) if (xs[i] > m) m = xs[i];
        return m;
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
