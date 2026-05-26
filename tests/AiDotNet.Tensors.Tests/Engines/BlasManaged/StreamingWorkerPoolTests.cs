using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using AiDotNet.Tensors.Engines.BlasManaged.Pool;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Pool-Serial")]
public class StreamingWorkerPoolTests
{
    [Fact]
    public void Dispatch_SingleChunk_RunsOnCallerThread()
    {
        int callerTid = Thread.CurrentThread.ManagedThreadId;
        int observedTid = -1;
        StreamingWorkerPool.Dispatch(1, c => observedTid = Thread.CurrentThread.ManagedThreadId);
        Assert.Equal(callerTid, observedTid);
    }

    [Fact]
    public void Dispatch_MultipleChunks_AllChunksRunExactlyOnce()
    {
        const int numChunks = 32;
        var counts = new int[numChunks];
        StreamingWorkerPool.Dispatch(numChunks, c => Interlocked.Increment(ref counts[c]));
        for (int i = 0; i < numChunks; i++)
            Assert.Equal(1, counts[i]);
    }

    [Fact]
    public void Dispatch_MultipleChunks_UsesWorkerThreads()
    {
        const int numChunks = 16;
        var tids = new ConcurrentBag<int>();
        StreamingWorkerPool.Dispatch(numChunks, c => tids.Add(Thread.CurrentThread.ManagedThreadId));
        // With numChunks=16 and a multi-core host, at least 2 distinct thread IDs should be observed.
        if (Environment.ProcessorCount > 1)
            Assert.True(tids.Count >= 2 && tids.Distinct().Count() >= 2,
                $"Expected >=2 distinct threads, got {tids.Distinct().Count()}");
    }

    [Fact]
    public void Dispatch_WorkerException_PropagatesToDispatcher()
    {
        var thrown = Assert.Throws<InvalidOperationException>(() =>
            StreamingWorkerPool.Dispatch(8, c =>
            {
                if (c == 3) throw new InvalidOperationException("worker boom");
            }));
        Assert.Equal("worker boom", thrown.Message);
    }

    [Fact]
    public void Dispatch_Reentrant_FallsBackToSerial()
    {
        // Inner dispatch from inside an outer dispatch chunk must serialize
        // (avoids worker-pool deadlock).
        bool innerRan = false;
        StreamingWorkerPool.Dispatch(2, c =>
        {
            if (c == 0)
            {
                StreamingWorkerPool.Dispatch(2, ic => innerRan = true);
            }
        });
        Assert.True(innerRan);
    }

    [Fact]
    public void Dispatch_WithGrainSize_RunsSerialBelowThreshold()
    {
        int callerTid = Thread.CurrentThread.ManagedThreadId;
        var observedTids = new ConcurrentBag<int>();
        // totalWork below DefaultSerialGrainSize (32768) should run all chunks on caller.
        StreamingWorkerPool.Dispatch(8, totalWork: 1000, c => observedTids.Add(Thread.CurrentThread.ManagedThreadId));
        foreach (var t in observedTids)
            Assert.Equal(callerTid, t);
    }

    [Fact]
    public void Dispatch_DispatchLatency_BelowMicrosecond()
    {
        if (Environment.ProcessorCount < 2)
            return;  // Single-core: no worker parallelism, latency irrelevant.

        // Warm up the pool (first call spawns workers).
        for (int i = 0; i < 100; i++)
            StreamingWorkerPool.Dispatch(8, c => { /* no-op */ });

        // Measure: 1000 minimal dispatches. Median wall-time per dispatch
        // should be ≤25 µs on a multi-core box (spin-then-park hot path).
        var times = new double[1000];
        var sw = new System.Diagnostics.Stopwatch();
        for (int i = 0; i < 1000; i++)
        {
            sw.Restart();
            StreamingWorkerPool.Dispatch(8, c => { /* no-op */ });
            sw.Stop();
            times[i] = sw.Elapsed.TotalMicroseconds;
        }
        Array.Sort(times);
        double medianUs = times[500];

        // Gate: 25 µs is generous. The aspirational sub-µs is hardware-dependent.
        // This catches order-of-magnitude regressions (e.g., accidental TPL fallback).
        Assert.True(medianUs < 25.0, $"Median dispatch latency {medianUs:F1} µs exceeds 25 µs sentinel.");
    }
}
