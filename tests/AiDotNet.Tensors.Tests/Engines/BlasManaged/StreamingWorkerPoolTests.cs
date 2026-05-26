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
}
