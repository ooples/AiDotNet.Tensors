using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
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
    public void A_plan_with_no_room_for_its_block_falls_back_to_the_pool_and_says_so()
    {
        using var arena = new CudaStepArena(IntPtr.Zero, 0,
            _ => throw new InvalidOperationException("no device memory for the plan"), _ => { });
        WarmUpAndRecordSequentialTemporaries(arena, 2, 64);
        TrainingStep.Begin();
        Assert.Null(arena.TryRent(64));   // builds the plan, cannot place it: the pool serves
        Assert.Null(arena.TryRent(64));
        TrainingStep.End();
        Assert.Equal(0, arena.SlotHits);
        Assert.Equal(0, arena.PlannedBytes);
        Assert.Contains(GpuLaunchProbe.Fallbacks, entry => entry.Contains("CudaStepArena.BuildPlan: InvalidOperationException"));
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

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference RecordAndDrop(CudaStepArena arena, int size)
        => new(RecordedAllocation(arena, size));

    [Fact]
    public void A_recorded_buffer_dropped_undisposed_is_not_kept_alive_by_the_arena()
    {
        // The record was keyed by the buffer, so a buffer dropped without being disposed stayed reachable from the arena:
        // its finalizer could not run, and its memory stayed out until the next step or the backend's dispose.
        using var arena = NewArena();
        TrainingStep.Begin();
        Assert.Null(arena.TryRent(64));   // warm-up
        TrainingStep.End();
        TrainingStep.Begin();
        var dropped = RecordAndDrop(arena, 64);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        Assert.False(dropped.IsAlive, "the arena's record kept a dropped buffer alive");
        TrainingStep.End();
        GC.KeepAlive(arena);
    }

    [Fact]
    public void A_replay_that_races_the_arenas_dispose_never_hands_out_freed_memory()
    {
        // Dispose ran between the replay's read of the block pointer and its live count: it saw nothing live, freed the
        // block, and the replay handed out a slot over the freed memory.
        int freed = 0;
        var arena = new CudaStepArena(IntPtr.Zero, 0, _ => FakeBlock, ptr =>
        {
            Assert.Equal(FakeBlock, ptr);
            Interlocked.Increment(ref freed);
        });
        WarmUpAndRecordSequentialTemporaries(arena, 1, 64);
        TrainingStep.Begin();
        arena.BeforeSlotGoesLiveForTests = arena.Dispose;
        var slot = arena.TryRent(64);
        TrainingStep.End();
        Assert.Null(slot);
        Assert.Equal(1, freed);   // freed once, by the dispose, and not again by the undone slot
        Assert.Equal(0, arena.SlotHits);
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
        // Both threads read the arena as unowned before either claims it (the barrier holds each between the read and the
        // claim), the interleaving where a read-then-write claim let both own it. Only one claim may win.
        using var arena = NewArena();
        const int threads = 2;
        using var bothRead = new Barrier(threads);
        bool barrierTimedOut = false;
        arena.BeforeClaimForTests = () =>
        {
            if (!bothRead.SignalAndWait(TimeSpan.FromSeconds(30))) barrierTimedOut = true;
        };
        var workers = new List<Thread>();
        for (int t = 0; t < threads; t++)
        {
            var worker = new Thread(() =>
            {
                TrainingStep.Begin();
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
        foreach (var worker in workers) worker.Join();
        Assert.False(barrierTimedOut, "a thread never reached the claim: the interleaving was not forced");

        // Exactly one thread's recording reached the arena: the owner's. A second claimant would have recorded too.
        var recordedField = typeof(CudaStepArena).GetField("_sizes", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
            ?? throw new InvalidOperationException("_sizes field not found.");
        var sizes = recordedField.GetValue(arena) as List<int> ?? throw new InvalidOperationException("_sizes is not a list.");
        Assert.Single(sizes);
    }
}
