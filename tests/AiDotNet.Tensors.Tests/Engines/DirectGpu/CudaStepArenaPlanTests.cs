using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="CudaStepArena"/>'s plan on host bookkeeping alone: the arena hands out device pointers into a block it
/// never touches, so a fake block address drives its recording, packing and replay without a device. Pool buffers are
/// host wrappers of a fake pointer, never freed.
/// </summary>
public sealed class CudaStepArenaPlanTests
{
    private static readonly IntPtr FakeBlock = new(0x10_0000);

    private static CudaStepArena NewArena() => new(IntPtr.Zero, 0, _ => FakeBlock, _ => { });

    private static long s_fakePointer = 0x1000;

    private static CudaBackend.CudaGpuBuffer FakePoolBuffer(int size)
    {
        var buffer = new CudaBackend.CudaGpuBuffer(IntPtr.Zero, new IntPtr(Interlocked.Add(ref s_fakePointer, 0x1000)), size);
        GC.SuppressFinalize(buffer);   // a fake pointer: nothing to free
        return buffer;
    }

    // One recorded allocation as the backend makes it: the arena declines, the pool serves, the arena tracks it.
    private static CudaBackend.CudaGpuBuffer RecordedAllocation(CudaStepArena arena, int size)
    {
        Assert.Null(arena.TryRent(size));
        var buffer = FakePoolBuffer(size);
        arena.Track(buffer);
        return buffer;
    }

    // The warm-up step, then a recorded step of `count` same-size temporaries each returned before the next.
    private static void WarmUpAndRecordSequentialTemporaries(CudaStepArena arena, int count, int size)
    {
        TrainingStep.Begin();
        Assert.Null(arena.TryRent(size));   // warm-up: one-time state, not recorded
        TrainingStep.End();
        TrainingStep.Begin();
        for (int k = 0; k < count; k++) arena.OnReturned(RecordedAllocation(arena, size));
        TrainingStep.End();
    }

    [Fact]
    public void Same_size_temporaries_in_sequence_plan_and_replay_without_pairwise_conflicts()
    {
        // 10,000 temporaries, one live at a time, all at offset 0: a per-slot list of every slot sharing memory held
        // about 10^8 entries (400 MB) and every replayed allocation scanned 10^4 of them.
        const int count = 10_000, size = 64;
        using var arena = NewArena();
        WarmUpAndRecordSequentialTemporaries(arena, count, size);

        var clock = Stopwatch.StartNew();
        TrainingStep.Begin();
        for (int k = 0; k < count; k++)
        {
            var slot = arena.TryRent(size);
            Assert.NotNull(slot);
            slot?.Dispose();
        }
        TrainingStep.End();
        clock.Stop();
        Assert.Equal(count, arena.SlotHits);
        Assert.Equal(0, arena.PoolFallbacks);
        Assert.Equal(256, arena.PlannedBytes);   // one aligned slot, reused by every temporary
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(20), $"planning and replaying took {clock.Elapsed}");
    }

    [Fact]
    public void A_slot_that_outlives_its_recorded_lifetime_is_never_aliased()
    {
        using var arena = NewArena();
        WarmUpAndRecordSequentialTemporaries(arena, 3, 64);

        TrainingStep.Begin();
        var kept = arena.TryRent(64);          // slot 0, kept past its recorded return
        Assert.NotNull(kept);
        Assert.Null(arena.TryRent(64));        // slot 1 shares slot 0's memory: the pool serves it
        kept?.Dispose();
        var third = arena.TryRent(64);         // slot 2: slot 0 is dead now
        Assert.NotNull(third);
        if (third is not null) Assert.Equal(FakeBlock, third.Handle);
        third?.Dispose();
        TrainingStep.End();
        Assert.Equal(2, arena.SlotHits);
        Assert.Equal(1, arena.PoolFallbacks);
    }

    [Fact]
    public void Another_threads_allocation_does_not_drop_the_owners_pending_record()
    {
        // Another thread's TryRent cleared the pending record between the owner's TryRent and Track, so the owner's buffer
        // was never recorded: the plan had no slot for it and every replay of it went to the pool.
        using var arena = NewArena();
        TrainingStep.Begin();
        Assert.Null(arena.TryRent(64));   // claims the arena for this thread; warm-up
        TrainingStep.End();

        TrainingStep.Begin();
        Assert.Null(arena.TryRent(64));   // recording: allocation 0 is pending
        bool otherWasServed = false;
        var other = new Thread(() =>
        {
            TrainingStep.Begin();
            otherWasServed = arena.TryRent(64) is not null;   // not the arena's thread: declined, touching nothing
            arena.Track(FakePoolBuffer(64));
            TrainingStep.End();
        });
        other.Start();
        other.Join();
        Assert.False(otherWasServed);
        var owned = FakePoolBuffer(64);
        arena.Track(owned);
        arena.OnReturned(owned);
        TrainingStep.End();

        TrainingStep.Begin();
        var slot = arena.TryRent(64);
        Assert.NotNull(slot);
        slot?.Dispose();
        TrainingStep.End();
        Assert.Equal(1, arena.SlotHits);
    }

    [Fact]
    public void Only_one_thread_claims_an_unowned_arena()
    {
        using var arena = NewArena();
        const int threads = 8;
        using var start = new ManualResetEventSlim(false);
        var workers = new List<Thread>();
        for (int t = 0; t < threads; t++)
        {
            var worker = new Thread(() =>
            {
                TrainingStep.Begin();
                start.Wait();
                arena.TryRent(64);
                arena.TryRent(64);   // a second call is the thread's second allocation of its warm-up step
                TrainingStep.End();
                TrainingStep.Begin();
                // Only the owner records: its recording step's allocation is pending, and its Track records it.
                arena.TryRent(64);
                var buffer = FakePoolBuffer(64);
                arena.Track(buffer);
                arena.OnReturned(buffer);
                TrainingStep.End();
            });
            workers.Add(worker);
            worker.Start();
        }
        start.Set();
        foreach (var worker in workers) worker.Join();

        // Exactly one thread's recording reached the arena: the owner's. A second claimant would have recorded too.
        var recordedField = typeof(CudaStepArena).GetField("_sizes", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
            ?? throw new InvalidOperationException("_sizes field not found.");
        var sizes = recordedField.GetValue(arena) as List<int> ?? throw new InvalidOperationException("_sizes is not a list.");
        Assert.Single(sizes);
    }
}
