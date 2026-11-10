using System;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// The collection these cache tests share. AutoTensorCache's byte budget is process-wide and Clear empties every
// thread's pools, which other classes call (TensorConcatenateTests); a Clear between a Return and its Assert.Same
// failed the test intermittently, so the collection runs with no other test in parallel.
[CollectionDefinition("DeferredMaterializerRegistrySerial", DisableParallelization = true)]
public sealed class AutoTensorCacheSerialCollection
{
}

// AutoTensorCache is process-wide; keep these off the parallel runner so another test's rentals cannot interleave.
[Collection("DeferredMaterializerRegistrySerial")]
public class AutoTensorCacheBudgetTests
{
    /// <summary>
    /// The per-shape copy limits bound each shape, not the total: before the aggregate cap, every distinct shape a
    /// model produced kept its copies for the life of the thread.
    /// </summary>
    [Fact]
    public void Return_StopsPoolingAtTheByteBudget()
    {
        long prior = AutoTensorCache.MaxPooledBytes;
        try
        {
            AutoTensorCache.Clear();
            AutoTensorCache.MaxPooledBytes = 3 * 1000 * sizeof(float);
            var shapes = new[] { new[] { 1000, 1 }, new[] { 1, 1000 }, new[] { 10, 100 }, new[] { 100, 10 } };
            var returned = new Tensor<float>[shapes.Length];
            for (int i = 0; i < shapes.Length; i++)
            {
                returned[i] = new Tensor<float>(shapes[i]);
                AutoTensorCache.Return(returned[i]);
            }

            // The first three fit the budget and come back; the fourth was refused.
            for (int i = 0; i < 3; i++) Assert.Same(returned[i], AutoTensorCache.RentOrAllocate<float>(shapes[i]));
            Assert.NotSame(returned[3], AutoTensorCache.RentOrAllocate<float>(shapes[3]));

            // Renting gave the budget back.
            AutoTensorCache.Return(returned[3]);
            Assert.Same(returned[3], AutoTensorCache.RentOrAllocate<float>(shapes[3]));
        }
        finally
        {
            AutoTensorCache.MaxPooledBytes = prior;
            AutoTensorCache.Clear();
        }
    }

    /// <summary>
    /// The budget is shared by every thread: the pools are per thread, and a parallel kernel's workers each filled
    /// their own, so a per-thread budget multiplied by the worker count.
    /// </summary>
    [Fact]
    public void Budget_IsSharedAcrossThreads()
    {
        long prior = AutoTensorCache.MaxPooledBytes;
        try
        {
            AutoTensorCache.Clear();
            AutoTensorCache.MaxPooledBytes = 1000 * sizeof(float);
            var worker = new Thread(() => AutoTensorCache.Return(new Tensor<float>(new[] { 25, 40 })));
            worker.Start();
            worker.Join();

            var mine = new Tensor<float>(new[] { 40, 25 });
            AutoTensorCache.Return(mine);
            Assert.NotSame(mine, AutoTensorCache.RentOrAllocate<float>(new[] { 40, 25 }));
        }
        finally
        {
            AutoTensorCache.MaxPooledBytes = prior;
            AutoTensorCache.Clear();
        }
    }

    /// <summary>
    /// Clear empties every thread's pools but runs on one thread: this thread's shape counts must not keep it refusing
    /// buffers as if its emptied pools were still full.
    /// </summary>
    [Fact]
    public void Clear_FromAnotherThread_ResetsThisThreadsShapeCounts()
    {
        long prior = AutoTensorCache.MaxPooledBytes;
        try
        {
            AutoTensorCache.Clear();
            AutoTensorCache.MaxPooledBytes = 1000 * sizeof(float);
            var shape = new[] { 20, 50 };
            AutoTensorCache.Return(new Tensor<float>(shape)); // fills this thread's budget

            var clearer = new Thread(AutoTensorCache.Clear);
            clearer.Start();
            clearer.Join();

            var after = new Tensor<float>(shape);
            AutoTensorCache.Return(after);
            Assert.Same(after, AutoTensorCache.RentOrAllocate<float>(shape));
        }
        finally
        {
            AutoTensorCache.MaxPooledBytes = prior;
            AutoTensorCache.Clear();
        }
    }

    /// <summary>
    /// Clear races the pools' other users, which take no lock. It used to zero the byte total and then empty the
    /// pools, so a return counted in between was dropped still counted, and a rent subtracted after the zeroing took
    /// the total below zero; either way the total no longer matched what the pools held.
    /// </summary>
    [Fact]
    public void Clear_DuringConcurrentReturnsAndRents_KeepsTheByteTotalExact()
    {
        long prior = AutoTensorCache.MaxPooledBytes;
        try
        {
            AutoTensorCache.Clear();
            AutoTensorCache.MaxPooledBytes = AutoTensorCache.DefaultMaxPooledBytes;
            var shape = new[] { 8, 8 };
            int stop = 0;
            var workers = new Thread[4];
            for (int w = 0; w < workers.Length; w++)
            {
                workers[w] = new Thread(() =>
                {
                    // Several tensors pooled at a time, so a Clear finds pools to empty while returns and rents run.
                    while (Volatile.Read(ref stop) == 0)
                    {
                        for (int i = 0; i < 8; i++) AutoTensorCache.Return(new Tensor<float>(shape));
                        for (int i = 0; i < 8; i++) AutoTensorCache.RentOrAllocate<float>(shape);
                    }
                });
                workers[w].Start();
            }

            for (int i = 0; i < 20000; i++) AutoTensorCache.Clear();
            Volatile.Write(ref stop, 1);
            foreach (var worker in workers) worker.Join();

            // Nothing runs now, so the running total must be what the pools hold; a final Clear would hide drift.
            Assert.Equal(AutoTensorCache.CountPooledBytes(), AutoTensorCache.PooledBytes);
            AutoTensorCache.Clear();
            Assert.Equal(0, AutoTensorCache.PooledBytes);
        }
        finally
        {
            AutoTensorCache.MaxPooledBytes = prior;
            AutoTensorCache.Clear();
        }
    }

    /// <summary>
    /// A thread that exits leaves its pooled tensors unreachable. Their bytes must leave the shared budget when its
    /// pools are reaped (on the next thread's first use, or a Clear), or the budget shrinks by them for good.
    /// </summary>
    [Fact]
    public void ExitedThreadsPooledBytes_LeaveTheBudget()
    {
        long prior = AutoTensorCache.MaxPooledBytes;
        try
        {
            AutoTensorCache.Clear();
            AutoTensorCache.MaxPooledBytes = AutoTensorCache.DefaultMaxPooledBytes;
            RunOnExitingThread(() => AutoTensorCache.Return(new Tensor<float>(new[] { 25, 40 })));
            Assert.Equal(1000L * sizeof(float), AutoTensorCache.PooledBytes);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            // A new thread's first use reaps the exited thread's pools.
            RunOnExitingThread(() => AutoTensorCache.RentOrAllocate<float>(new[] { 3, 3 }));
            Assert.Equal(AutoTensorCache.CountPooledBytes(), AutoTensorCache.PooledBytes);
            Assert.Equal(0, AutoTensorCache.PooledBytes);
        }
        finally
        {
            AutoTensorCache.MaxPooledBytes = prior;
            AutoTensorCache.Clear();
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void RunOnExitingThread(ThreadStart work)
    {
        var thread = new Thread(work);
        thread.Start();
        thread.Join();
    }
}
