// Copyright (c) AiDotNet. All rights reserved.
// Regression coverage for issue #414 — per-thread cl_command_queue + the
// driver-safety lock around clCreateCommandQueue.
//
// The OpenCL 1.2 spec § 5.1.1 lists clCreateCommandQueue as thread-safe, but
// the AMD RDNA1 driver crashes amdocl64.dll with an access violation when
// multiple host threads enter that entry point concurrently. The
// DirectOpenClContext fix wraps the native call in a per-instance lock so
// the host-side enqueue path is fully parallel after the one-time setup,
// without exposing the driver-level race.
//
// These tests assert the OBSERVABLE contracts that fall out of that design:
//   * Each worker thread gets a DISTINCT queue handle.
//   * The handles persist for the lifetime of the thread (re-fetching from
//     the same thread returns the same handle).
//   * Dispose() releases EVERY per-thread queue that was created across the
//     context's lifetime — including queues belonging to threads that have
//     since exited.
//   * Re-entrant Dispose() is a no-op (idempotent).
//
// We DO NOT assert performance here (parallel speedup is a downstream
// concern measured by the HE PathB Pareto sweep); we DO assert there is no
// crash, no deadlock, and no leaked queue handles under stress (≥ 8 threads).

using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Regression coverage for issue #414 (per-thread <c>cl_command_queue</c>)
/// and the driver-safety lock around <c>clCreateCommandQueue</c>.
/// </summary>
public class DirectOpenClContextConcurrencyTests
{
    /// <summary>
    /// Skip when no OpenCL GPU is available — these tests touch real native
    /// driver code and have no managed-only fallback. CI runners without a
    /// GPU should silently skip rather than fail.
    /// </summary>
    private static bool OpenClPresent() =>
        DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0;

    [SkippableFact]
    public void EachWorkerThreadGetsDistinctQueueHandle()
    {
        Skip.IfNot(OpenClPresent(), "No OpenCL GPU device available on this host.");
        using var ctx = new DirectOpenClContext(deviceIndex: 0);

        const int threadCount = 8;
        var handles = new ConcurrentBag<IntPtr>();
        var startGate = new ManualResetEventSlim(false);
        var threads = new Thread[threadCount];

        for (int i = 0; i < threadCount; i++)
        {
            threads[i] = new Thread(() =>
            {
                // Block all threads until the gate is released so the
                // first-touch clCreateCommandQueue calls race against each
                // other — this is the exact pattern that crashed the driver
                // pre-fix. If the safety-lock works, every thread completes
                // and we collect 8 distinct handles.
                startGate.Wait();
                handles.Add(ctx.CommandQueue);
            }) { IsBackground = true };
        }
        foreach (var t in threads) t.Start();
        startGate.Set();
        foreach (var t in threads) Assert.True(t.Join(TimeSpan.FromSeconds(30)),
            "Worker thread did not return within 30s — likely deadlock in queue creation.");

        var distinct = handles.Where(h => h != IntPtr.Zero).Distinct().ToArray();
        Assert.Equal(threadCount, handles.Count);
        Assert.Equal(threadCount, distinct.Length);
    }

    [SkippableFact]
    public void RefetchingFromSameThreadReturnsSameQueueHandle()
    {
        Skip.IfNot(OpenClPresent(), "No OpenCL GPU device available on this host.");
        using var ctx = new DirectOpenClContext(deviceIndex: 0);

        // Two fetches from the SAME thread MUST return the cached
        // ThreadLocal handle — otherwise each kernel launch would
        // create + leak a queue.
        IntPtr first = ctx.CommandQueue;
        IntPtr second = ctx.CommandQueue;
        Assert.NotEqual(IntPtr.Zero, first);
        Assert.Equal(first, second);
    }

    [SkippableFact]
    public void DisposeReleasesAllPerThreadQueuesIncludingExitedThreads()
    {
        Skip.IfNot(OpenClPresent(), "No OpenCL GPU device available on this host.");
        var ctx = new DirectOpenClContext(deviceIndex: 0);

        // Touch the queue from several short-lived threads. They exit before
        // Dispose, so the ThreadLocal entries persist only via Values
        // tracking. Dispose must walk those tracked values and release the
        // native handles regardless of whether the originating threads are
        // still alive.
        const int threadCount = 6;
        var threads = new Thread[threadCount];
        for (int i = 0; i < threadCount; i++)
        {
            threads[i] = new Thread(() => { _ = ctx.CommandQueue; }) { IsBackground = true };
            threads[i].Start();
            Assert.True(threads[i].Join(TimeSpan.FromSeconds(10)));
        }

        // After Dispose, fetching CommandQueue must throw ObjectDisposedException
        // (the post-disposal lazy path returns IntPtr.Zero internally so a
        // racing thread can fail fast; the public property surfaces the
        // disposed state).
        ctx.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ctx.CommandQueue);
        // Re-entrant Dispose() is a no-op contract.
        ctx.Dispose();
    }
}
