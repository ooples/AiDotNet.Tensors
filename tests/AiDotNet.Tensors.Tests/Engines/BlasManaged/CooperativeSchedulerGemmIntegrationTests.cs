// Copyright (c) AiDotNet. All rights reserved.
// End-to-end proof that routing the real GEMM dispatch through
// CooperativeGemmScheduler (the Phase 2 migration seam in StreamingWorkerPool)
// (a) produces correct results and (b) preserves the deterministic-parallel
// bit-exactness contract. Toggles the process-wide CooperativeGemmScheduler.Enabled
// flag, so it runs in the serial collection.

using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged.Pool;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class CooperativeSchedulerGemmIntegrationTests
{
    [Fact]
    public void Gemm_ThroughCooperativeScheduler_BitIdenticalAcrossThreadCounts()
    {
        // Streaming-strategy shape so dispatch flows through StreamingWorkerPool →
        // (when enabled) CooperativeGemmScheduler. Deterministic mode → managed,
        // M/N-axis split → bit-exact regardless of how the scheduler interleaves.
        const int m = 256, n = 256, k = 24;
        bool beforeDet = BlasProvider.IsDeterministicMode;
        bool beforeCoop = CooperativeGemmScheduler.Enabled;
        try
        {
            BlasProvider.SetDeterministicMode(true);

            var rng = new Random(7);
            var a = new float[m * k];
            var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            // Reference: legacy pool.
            CooperativeGemmScheduler.Enabled = false;
            var cRef = new float[m * n];
            BlasManagedLib.Gemm<float>(a, k, false, b, n, false, cRef, n, m, n, k,
                new BlasOptions<float> { NumThreads = Math.Max(4, Environment.ProcessorCount) });

            // Cooperative scheduler must produce bit-identical output.
            CooperativeGemmScheduler.Enabled = true;
            var cCoop = new float[m * n];
            BlasManagedLib.Gemm<float>(a, k, false, b, n, false, cCoop, n, m, n, k,
                new BlasOptions<float> { NumThreads = Math.Max(4, Environment.ProcessorCount) });

            for (int i = 0; i < cRef.Length; i++)
                Assert.Equal(cRef[i], cCoop[i]);
        }
        finally
        {
            CooperativeGemmScheduler.Enabled = beforeCoop;
            BlasProvider.SetDeterministicMode(beforeDet);
        }
    }

    [Fact]
    public void ConcurrentGemms_ThroughCooperativeScheduler_StayCorrect()
    {
        // Many threads run the same GEMM concurrently while the cooperative scheduler
        // is enabled; every result must match the single-thread reference (deterministic
        // mode → bit-exact) with no deadlock. This is the concurrent-inference scenario
        // the scheduler exists for: the legacy pool would serialize these; here they
        // interleave on the shared queue.
        const int m = 128, n = 128, k = 24;
        bool beforeDet = BlasProvider.IsDeterministicMode;
        bool beforeCoop = CooperativeGemmScheduler.Enabled;
        try
        {
            BlasProvider.SetDeterministicMode(true);
            CooperativeGemmScheduler.Enabled = true;

            var rng = new Random(9);
            var a = new float[m * k];
            var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            var cRef = new float[m * n];
            BlasManagedLib.Gemm<float>(a, k, false, b, n, false, cRef, n, m, n, k,
                new BlasOptions<float> { NumThreads = -1 }); // serial reference

            int threads = Math.Max(8, Environment.ProcessorCount * 2);
            const int iters = 30;
            var drifts = new ConcurrentBag<string>();
            using var gate = new Barrier(threads + 1);
            var workers = new Task[threads];
            for (int t = 0; t < threads; t++)
            {
                workers[t] = Task.Factory.StartNew(() =>
                {
                    gate.SignalAndWait();
                    for (int it = 0; it < iters; it++)
                    {
                        var c = new float[m * n];
                        BlasManagedLib.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k);
                        for (int i = 0; i < cRef.Length; i++)
                            if (c[i] != cRef[i]) { drifts.Add($"idx {i}: {c[i]} vs {cRef[i]}"); break; }
                    }
                }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
            }
            gate.SignalAndWait();
            bool done = Task.WaitAll(workers, TimeSpan.FromSeconds(120));
            Assert.True(done, "Concurrent GEMMs via cooperative scheduler did not finish in 120s — possible deadlock.");
            Assert.True(drifts.IsEmpty,
                $"{drifts.Count} concurrent GEMMs drifted from the serial reference. " +
                $"First: {(drifts.IsEmpty ? "" : System.Linq.Enumerable.First(drifts))}");
        }
        finally
        {
            CooperativeGemmScheduler.Enabled = beforeCoop;
            BlasProvider.SetDeterministicMode(beforeDet);
        }
    }
}
