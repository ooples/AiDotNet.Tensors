// Copyright (c) AiDotNet. All rights reserved.
// Issue #313 regression — PersistentParallelExecutor must not dispatch
// to its worker pool when the total elementwise work is below the
// configured grain size. Below threshold, every call must run on the
// caller's thread, with no semaphore wakeup / no completion-event
// wait. The original ViT-Base profile showed ~90% wall time in those
// signal/wait paths because BatchNorm and similar small-tensor
// kernels were dispatching even when the per-channel work was a few
// hundred elements.

using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class PersistentExecutorGrainSizeTests
{
    /// <summary>
    /// Below the grain size, every chunk must execute on the calling
    /// thread — no wake to the worker pool. Verified by capturing
    /// each chunk's <c>Thread.CurrentThread.ManagedThreadId</c>.
    /// </summary>
    [Fact]
    public void Execute_BelowGrainSize_RunsAllChunksOnCallerThread()
    {
        int callerId = Thread.CurrentThread.ManagedThreadId;
        int numChunks = 8;
        long totalWork = PersistentParallelExecutor.DefaultSerialGrainSize / 2;
        var observed = new int[numChunks];

        PersistentParallelExecutor.Instance.Execute(numChunks, totalWork, c =>
        {
            observed[c] = Thread.CurrentThread.ManagedThreadId;
        });

        for (int c = 0; c < numChunks; c++)
            Assert.True(observed[c] == callerId,
                $"Chunk {c} ran on thread {observed[c]}, expected caller thread {callerId} "
                + $"because totalWork ({totalWork}) < DefaultSerialGrainSize "
                + $"({PersistentParallelExecutor.DefaultSerialGrainSize}).");
    }

    /// <summary>
    /// Above the grain size, the parallel path must still actually
    /// parallelize. We assert that AT LEAST one chunk ran on a thread
    /// other than the caller — proving the worker pool was engaged.
    /// </summary>
    [Fact]
    public void Execute_AboveGrainSize_DispatchesToWorkerPool()
    {
        // Skip on single-CPU hosts where there are no workers.
        if (Environment.ProcessorCount <= 1) return;

        int callerId = Thread.CurrentThread.ManagedThreadId;
        int numChunks = Math.Max(2, Environment.ProcessorCount);
        long totalWork = (long)PersistentParallelExecutor.DefaultSerialGrainSize * 4;
        var observed = new int[numChunks];

        PersistentParallelExecutor.Instance.Execute(numChunks, totalWork, c =>
        {
            // Spin a little so chunks don't all serialize through chunk 0
            // before the workers have a chance to wake.
            int x = 0;
            for (int i = 0; i < 5_000; i++) x ^= i;
            observed[c] = Thread.CurrentThread.ManagedThreadId ^ (x & 0);
        });

        bool sawOtherThread = false;
        for (int c = 0; c < numChunks; c++)
            if (observed[c] != callerId) { sawOtherThread = true; break; }
        Assert.True(sawOtherThread,
            $"All {numChunks} chunks ran on caller thread {callerId} despite "
            + $"totalWork ({totalWork}) >> DefaultSerialGrainSize "
            + $"({PersistentParallelExecutor.DefaultSerialGrainSize}). The worker pool "
            + "should have been engaged.");
    }

    /// <summary>
    /// The 2-arg <c>Execute(numChunks, action)</c> overload (existing
    /// API) must continue to dispatch to the worker pool unconditionally
    /// — callers that haven't migrated to grain-size still get the old
    /// behavior. Backward compatibility canary.
    /// </summary>
    [Fact]
    public void Execute_NoGrainSizeOverload_PreservesExistingDispatchBehavior()
    {
        // Skip on single-CPU hosts where there are no workers.
        if (Environment.ProcessorCount <= 1) return;

        int callerId = Thread.CurrentThread.ManagedThreadId;
        int numChunks = Math.Max(2, Environment.ProcessorCount);
        var observed = new int[numChunks];

        PersistentParallelExecutor.Instance.Execute(numChunks, c =>
        {
            int x = 0;
            for (int i = 0; i < 5_000; i++) x ^= i;
            observed[c] = Thread.CurrentThread.ManagedThreadId ^ (x & 0);
        });

        bool sawOtherThread = false;
        for (int c = 0; c < numChunks; c++)
            if (observed[c] != callerId) { sawOtherThread = true; break; }
        Assert.True(sawOtherThread,
            "2-arg Execute should continue to dispatch to the worker pool — "
            + "backward-compatibility for callers that haven't been migrated.");
    }

    /// <summary>
    /// The action body must still observe each chunk index exactly once
    /// in both modes — serial-fallback must not skip or duplicate.
    /// </summary>
    [Theory]
    [InlineData(true)]   // small work → serial fallback
    [InlineData(false)]  // large work → parallel dispatch
    public void Execute_VisitsEachChunkExactlyOnce(bool small)
    {
        int numChunks = 16;
        long totalWork = small
            ? PersistentParallelExecutor.DefaultSerialGrainSize / 4
            : (long)PersistentParallelExecutor.DefaultSerialGrainSize * 4;
        int[] visits = new int[numChunks];

        PersistentParallelExecutor.Instance.Execute(numChunks, totalWork, c =>
        {
            Interlocked.Increment(ref visits[c]);
        });

        for (int c = 0; c < numChunks; c++)
            Assert.True(visits[c] == 1, $"Chunk {c} visited {visits[c]} times (expected exactly 1).");
    }

    /// <summary>
    /// Exception parity between serial-fallback and parallel modes.
    /// The parallel <see cref="PersistentParallelExecutor.Execute(int, Action{int})"/>
    /// path already runs every chunk before re-throwing the first
    /// captured exception (see WorkerLoop's catch block + Execute's
    /// re-throw at the end of the lock); the serial-fallback path
    /// must do the same so callers don't observe a behavior change
    /// when work happens to cross the grain-size threshold.
    /// </summary>
    [Theory]
    [InlineData(true)]   // small work → serial fallback
    [InlineData(false)]  // large work → parallel dispatch
    public void Execute_RethrowsFirstException_AfterAllChunksRan(bool small)
    {
        int numChunks = 8;
        long totalWork = small
            ? PersistentParallelExecutor.DefaultSerialGrainSize / 4
            : (long)PersistentParallelExecutor.DefaultSerialGrainSize * 4;
        int[] visits = new int[numChunks];

        // Throw on chunk 2; every chunk index must still be visited.
        var thrown = Assert.ThrowsAny<Exception>(() =>
        {
            PersistentParallelExecutor.Instance.Execute(numChunks, totalWork, c =>
            {
                Interlocked.Increment(ref visits[c]);
                if (c == 2) throw new InvalidOperationException("planted-exception-from-chunk-2");
            });
        });

        Assert.Contains("planted-exception-from-chunk-2", thrown.Message);
        for (int c = 0; c < numChunks; c++)
            Assert.True(visits[c] == 1,
                $"Chunk {c} visited {visits[c]} times after a sibling chunk threw — "
                + "expected each chunk to run exactly once regardless of mode.");
    }

    /// <summary>
    /// Integration check on the actual BatchNorm hot path (issue #313's
    /// principal target). With a tiny [1, 32, 7, 7] shape — the kind of
    /// late-stage EfficientNet/MobileNet layer where channels=32 channels
    /// gets dispatched but per-channel work is just 49 elements — the
    /// call must complete without engaging worker threads.
    ///
    /// We don't directly assert "no thread dispatch" here (the kernel
    /// is called transitively through CpuEngine.BatchNorm); instead we
    /// assert a much weaker invariant: the call returns the correct
    /// result. The throughput improvement is the real outcome and is
    /// covered by the per-thread-id checks above.
    /// </summary>
    [Fact]
    public void BatchNorm_TinySpatial_StillProducesCorrectOutput()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 1, 32, 7, 7 });
        var gamma = new Tensor<float>(new[] { 32 });
        var beta = new Tensor<float>(new[] { 32 });
        var rng = new Random(11);
        var xs = x.AsWritableSpan();
        for (int i = 0; i < xs.Length; i++) xs[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var gs = gamma.AsWritableSpan();
        var bs = beta.AsWritableSpan();
        for (int c = 0; c < 32; c++) { gs[c] = 1f; bs[c] = 0f; }

        var output = engine.BatchNorm(x, gamma, beta, 1e-5, out _, out _);
        Assert.Equal(new[] { 1, 32, 7, 7 }, output._shape);

        // Channel-wise mean ≈ 0 / variance ≈ 1 invariant — same as the
        // #310 regression test, but at the size where the #313 serial-
        // fallback path actually fires (per-channel = 49 elements ≪
        // grain size, so dispatch is skipped).
        var outSpan = output.AsSpan();
        for (int c = 0; c < 32; c++)
        {
            double sum = 0, sumSq = 0;
            for (int s = 0; s < 49; s++) { double v = outSpan[c * 49 + s]; sum += v; sumSq += v * v; }
            double mean = sum / 49;
            double var = sumSq / 49 - mean * mean;
            Assert.InRange(mean, -1e-3, 1e-3);
            Assert.InRange(var, 1.0 - 1e-1, 1.0 + 1e-1);
        }
    }
}
