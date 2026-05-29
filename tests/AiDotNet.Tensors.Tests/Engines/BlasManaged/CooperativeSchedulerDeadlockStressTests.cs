// Copyright (c) AiDotNet. All rights reserved.
// Isolation harness for the intermittent hang observed when the cooperative GEMM
// scheduler is ENABLED under concurrent dispatch. Unlike the throughput bench, this
// does NOT toggle Enabled or keep the legacy pool's workers alive — it drives REAL
// concurrent GEMMs purely through the cooperative scheduler, in many short batches,
// so a single run either reproduces the hang fast (a batch exceeds its watchdog) or
// builds confidence that it is gone. Category=Performance so it is excluded from the
// normal/CI run (it is a deliberately heavy, timing-sensitive reproducer).

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
public class CooperativeSchedulerDeadlockStressTests
{
    private readonly ITestOutputHelper _out;
    public CooperativeSchedulerDeadlockStressTests(ITestOutputHelper output) => _out = output;

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void ConcurrentGemm_ThroughCooperativeScheduler_NeverHangs(int threads)
    {
        const int batches = 60;
        const int itersPerThreadPerBatch = 20;
        const int m = 1024, n = 1024, k = 24;
        var watchdog = TimeSpan.FromSeconds(20);

        bool beforeDet = BlasProvider.IsDeterministicMode;
        bool beforeCoop = CooperativeGemmScheduler.Enabled;
        try
        {
            BlasProvider.SetDeterministicMode(true);
            CooperativeGemmScheduler.Enabled = true;

            var rng = new Random(20260529);
            var a = new float[m * k];
            var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            for (int batch = 0; batch < batches; batch++)
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
                        for (int it = 0; it < itersPerThreadPerBatch && !cts.IsCancellationRequested; it++)
                            BlasManagedLib.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k);
                    }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
                }
                gate.SignalAndWait();
                var sw = Stopwatch.StartNew();
                bool done = Task.WaitAll(workers, watchdog);
                sw.Stop();
                if (!done)
                {
                    cts.Cancel();
                    Task.WaitAll(workers, TimeSpan.FromSeconds(30));
                    throw new Xunit.Sdk.XunitException(
                        $"HANG reproduced: batch {batch}/{batches} with {threads} concurrent GEMMs " +
                        $"did not finish within {watchdog.TotalSeconds:F0}s through the cooperative scheduler.");
                }
            }
            _out.WriteLine($"{threads} threads: {batches} batches x {itersPerThreadPerBatch} iters each completed, no hang.");
        }
        finally
        {
            CooperativeGemmScheduler.Enabled = beforeCoop;
            BlasProvider.SetDeterministicMode(beforeDet);
        }
    }

    // Decisive isolation of the LEGACY pool: Enabled stays false throughout, so the
    // cooperative scheduler's worker threads are NEVER created — only the legacy
    // StreamingWorkerPool's (cores-1) workers exist. If the legacy pool hangs HERE, it is
    // a real production bug (legacy is the default). If it only hangs when the extra
    // coop threads are also alive (the toggling test below), the hang is dual-pool
    // oversubscription, not a legacy defect.
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void ConcurrentGemm_LegacyPoolOnly_NeverHangs(int threads)
    {
        const int batches = 80;
        const int itersPerThreadPerBatch = 30;
        const int m = 1024, n = 1024, k = 24;
        var watchdog = TimeSpan.FromSeconds(25);

        bool beforeDet = BlasProvider.IsDeterministicMode;
        bool beforeCoop = CooperativeGemmScheduler.Enabled;
        try
        {
            BlasProvider.SetDeterministicMode(true);
            CooperativeGemmScheduler.Enabled = false; // legacy path only; coop workers never spawn

            var rng = new Random(135792468);
            var a = new float[m * k];
            var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            for (int batch = 0; batch < batches; batch++)
            {
                if (!RunBatch(a, b, m, n, k, threads, itersPerThreadPerBatch, watchdog))
                    throw new Xunit.Sdk.XunitException(
                        $"HANG reproduced in LEGACY pool alone: batch {batch}/{batches}, threads={threads} " +
                        $"did not finish within {watchdog.TotalSeconds:F0}s (coop scheduler never enabled).");
            }
            _out.WriteLine($"{threads} threads legacy-only: {batches} batches completed, no hang.");
        }
        finally
        {
            CooperativeGemmScheduler.Enabled = beforeCoop;
            BlasProvider.SetDeterministicMode(beforeDet);
        }
    }

    // Reproduces the BENCH condition: toggle Enabled each batch so BOTH the legacy
    // StreamingWorkerPool AND the cooperative scheduler keep their (cores-1) worker
    // threads alive simultaneously (~62 background threads on a 32-logical machine),
    // both spin-waiting. If the "intermittent deadlock" is really spin-livelock from
    // that dual-pool oversubscription (not a scheduler defect), this is where it shows.
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void ConcurrentGemm_TogglingBothPools_BenchConditionRepro(int threads)
    {
        const int batches = 80;
        const int itersPerThreadPerBatch = 30;
        const int m = 1024, n = 1024, k = 24;
        var watchdog = TimeSpan.FromSeconds(25);

        bool beforeDet = BlasProvider.IsDeterministicMode;
        bool beforeCoop = CooperativeGemmScheduler.Enabled;
        try
        {
            BlasProvider.SetDeterministicMode(true);

            var rng = new Random(987654321);
            var a = new float[m * k];
            var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            // Warm both pools so both sets of worker threads are alive at once.
            CooperativeGemmScheduler.Enabled = false; RunBatch(a, b, m, n, k, threads, 3, TimeSpan.FromSeconds(25));
            CooperativeGemmScheduler.Enabled = true; RunBatch(a, b, m, n, k, threads, 3, TimeSpan.FromSeconds(25));

            for (int batch = 0; batch < batches; batch++)
            {
                CooperativeGemmScheduler.Enabled = (batch & 1) == 0;
                int b0 = batch;
                if (!RunBatch(a, b, m, n, k, threads, itersPerThreadPerBatch, watchdog))
                    throw new Xunit.Sdk.XunitException(
                        $"HANG reproduced under dual-pool toggling: batch {b0}/{batches}, threads={threads}, " +
                        $"Enabled={CooperativeGemmScheduler.Enabled} did not finish within {watchdog.TotalSeconds:F0}s.");
            }
            _out.WriteLine($"{threads} threads toggling: {batches} batches completed, no hang.");
        }
        finally
        {
            CooperativeGemmScheduler.Enabled = beforeCoop;
            BlasProvider.SetDeterministicMode(beforeDet);
        }
    }

    private static bool RunBatch(float[] a, float[] b, int m, int n, int k, int threads, int iters, TimeSpan watchdog)
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
                for (int it = 0; it < iters && !cts.IsCancellationRequested; it++)
                    BlasManagedLib.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k);
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }
        gate.SignalAndWait();
        bool done = Task.WaitAll(workers, watchdog);
        if (!done)
        {
            cts.Cancel();
            Task.WaitAll(workers, TimeSpan.FromSeconds(30));
        }
        return done;
    }
}
