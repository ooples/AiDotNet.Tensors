// Copyright (c) AiDotNet. All rights reserved.
// Issue #478 — cross-arena persistent buffer pool. A caller that creates +
// disposes a fresh TensorArena PER training step (e.g.
// NeuralNetworkBase.TrainWithTape) must NOT re-allocate the large transient
// working set every step. Dispose returns large backing arrays to a
// thread-static persistent pool; the next arena rents them back. These tests
// pin that reuse contract (no per-step GC churn) and its bounds.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// Reuse the arena collection so these serialize with the other arena tests —
// TensorArena.Current and the persistent pool are both [ThreadStatic] and xUnit
// reuses worker threads across collections.
[Collection(nameof(TensorArenaPinnedTests))]
public class TensorArenaPersistentPoolTests
{
    // 2M float elements = 8 MB — larger than ArrayPool<T>.Shared's 2^20-element
    // max bucket, i.e. exactly the paper-scale gradient-buffer size that
    // ArrayPool refuses to pool and that #478 is about.
    private const int LargeElems = 2 * 1024 * 1024;

    /// <summary>
    /// The core #478 guarantee (TFM-agnostic): creating + disposing a fresh
    /// arena every "step" and renting a large buffer each time reuses backing
    /// arrays from the cross-arena persistent pool instead of allocating fresh.
    /// Proven via the reuse-hit counter (recycled buffers are zeroed to preserve
    /// the de-facto zero-init contract, so a content sentinel can't distinguish
    /// reuse from fresh — the counter can).
    /// </summary>
    [Fact]
    public void LargeBuffer_ReusedAcrossArenaLifetimes_RegistersReuseHits()
    {
        TensorArena.ClearPersistentPool();
        int[] shape = { LargeElems };

        // Cycle 1: fresh allocation, returned to the pool on dispose. No reuse yet.
        using (var arena = TensorArena.Create())
        {
            var t = TensorAllocator.RentUninitialized<float>(shape);
            t.AsWritableSpan()[0] = 1f;
        }
        Assert.Equal(0, TensorArena.PersistentReuseHits);

        // Cycles 2..N: each must pull the backing array from the persistent pool.
        for (int i = 0; i < 5; i++)
        {
            using var arena = TensorArena.Create();
            var t = TensorAllocator.RentUninitialized<float>(shape);
            t.AsWritableSpan()[0] = i;
        }
        Assert.Equal(5, TensorArena.PersistentReuseHits);
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Same guarantee, measured directly: steady-state per-step allocation must
    /// be bookkeeping-only, NOT the multi-MB backing array. Uses
    /// <c>GC.GetTotalAllocatedBytes</c> (net5+ only).
    /// </summary>
    [Fact]
    public void LargeBuffer_ReusedAcrossArenaLifetimes_NoPerStepChurn()
    {
        TensorArena.ClearPersistentPool();
        int[] shape = { LargeElems };

        // Warmup: a few full create→rent→dispose cycles to reach steady state.
        for (int i = 0; i < 4; i++)
        {
            using var arena = TensorArena.Create();
            var t = TensorAllocator.RentUninitialized<float>(shape);
            t.AsWritableSpan()[0] = i; // touch so nothing is elided
        }

        long before = GC.GetTotalAllocatedBytes(precise: true);
        using (var arena = TensorArena.Create())
        {
            var t = TensorAllocator.RentUninitialized<float>(shape);
            t.AsWritableSpan()[0] = 42f;
        }
        long delta = GC.GetTotalAllocatedBytes(precise: true) - before;

        long bufferBytes = (long)LargeElems * sizeof(float); // 8 MB
        Assert.True(delta < bufferBytes / 4,
            $"Per-cycle allocation was {delta} bytes; the {bufferBytes}-byte buffer is " +
            "being re-allocated every cycle — the cross-arena persistent pool is not reusing it.");
    }
#endif

    /// <summary>
    /// Reuse must not corrupt: a buffer rented via the CLEAR tier
    /// (<see cref="TensorAllocator.Rent{T}"/>) is zero-initialized even when it
    /// is a recycled buffer that previously held non-zero data.
    /// </summary>
    [Fact]
    public void RecycledClearTierBuffer_IsZeroInitialized()
    {
        TensorArena.ClearPersistentPool();
        int[] shape = { LargeElems };

        // Cycle 1: stamp non-zero data into a clear-tier buffer, then dispose
        // (returns the dirty buffer to the persistent pool).
        using (var arena = TensorArena.Create())
        {
            var t = TensorAllocator.Rent<float>(shape);
            var span = t.AsWritableSpan();
            for (int i = 0; i < 16; i++) span[i] = 123.5f;
        }

        // Cycle 2: rent the same size from the clear tier — it should hand back
        // the recycled buffer, but zeroed.
        using (var arena = TensorArena.Create())
        {
            var t = TensorAllocator.Rent<float>(shape);
            var span = t.AsSpan();
            for (int i = 0; i < 16; i++)
                Assert.Equal(0f, span[i]);
        }
    }

    /// <summary>
    /// Regression for the #1624 training-scale OOM: a single deep-model step rents
    /// the SAME large size MANY times (every op produces a like-shaped intermediate
    /// that the persistent gradient tape keeps live for the whole backward pass, so
    /// the arena cannot reuse them within the step). The cross-arena pool must
    /// retain ALL of them so the next step reuses every buffer — the old fixed
    /// per-(type,size) cap of 4 retained only a sliver and re-allocated the rest
    /// each step, which under a constrained heap (DOTNET_GCHeapHardLimit) churns
    /// faster than gen2 collects and OOMs even though the LIVE set is flat.
    /// </summary>
    [Fact]
    public void HighFanOutLargeSize_PoolsEveryBufferAcrossLifetimes()
    {
        TensorArena.ClearPersistentPool();
        int[] shape = { LargeElems };
        const int fanOut = 16; // far more than the old per-size cap of 4

        // Lifetime 1: rent the same large size fanOut times. No Reset between
        // rents, so each rent allocates a distinct backing array (matching a
        // step whose tape keeps every intermediate live). Fresh — no reuse yet.
        using (var arena = TensorArena.Create())
        {
            for (int i = 0; i < fanOut; i++)
            {
                var t = TensorAllocator.RentUninitialized<float>(shape);
                t.AsWritableSpan()[0] = i;
            }
        }
        Assert.Equal(0, TensorArena.PersistentReuseHits);

        // Lifetime 2: the same fan-out must reuse EVERY buffer from the pool, not
        // just the first 4. One reuse hit per rent == fully allocation-free step.
        using (var arena = TensorArena.Create())
        {
            for (int i = 0; i < fanOut; i++)
            {
                var t = TensorAllocator.RentUninitialized<float>(shape);
                t.AsWritableSpan()[0] = i;
            }
        }
        Assert.Equal(fanOut, TensorArena.PersistentReuseHits);
    }

    /// <summary>
    /// The byte budget MUST bound the pool: a model with an effectively unbounded
    /// set of distinct large shapes can't grow the persistent pool past
    /// MaxPersistTotalBytes — surplus buffers are dropped for GC on return. This
    /// is the mechanism that prevents the fix itself from becoming an unbounded
    /// leak. Exercised via the per-thread test override of the budget (production
    /// budget is gigabytes); we set it to exactly <c>budgetBuffers</c> large
    /// buffers and assert only that many are retained — the next lifetime reuses
    /// exactly <c>budgetBuffers</c> and re-allocates the rest.
    /// </summary>
    [Fact]
    public void ByteBudget_BoundsPool_DropsSurplusBuffers()
    {
        TensorArena.ClearPersistentPool();
        int[] shape = { LargeElems };
        const int budgetBuffers = 4;
        const int fanOut = 16; // 4x the budget — the surplus 12 must be dropped
        long bufferBytes = (long)LargeElems * sizeof(float); // 8 MB

        long savedOverride = TensorArena.MaxPersistTotalBytesOverrideForTests;
        TensorArena.MaxPersistTotalBytesOverrideForTests = budgetBuffers * bufferBytes;
        try
        {
            // Lifetime 1: return fanOut buffers; only budgetBuffers fit the budget,
            // the rest are dropped for GC (no reuse recorded yet).
            using (var arena = TensorArena.Create())
            {
                for (int i = 0; i < fanOut; i++)
                {
                    var t = TensorAllocator.RentUninitialized<float>(shape);
                    t.AsWritableSpan()[0] = i;
                }
            }
            Assert.Equal(0, TensorArena.PersistentReuseHits);

            // Lifetime 2: only the budgetBuffers that were retained can be reused;
            // the remaining (fanOut - budgetBuffers) rents allocate fresh. So the
            // reuse-hit count is capped at the budget, proving the pool is bounded.
            using (var arena = TensorArena.Create())
            {
                for (int i = 0; i < fanOut; i++)
                {
                    var t = TensorAllocator.RentUninitialized<float>(shape);
                    t.AsWritableSpan()[0] = i;
                }
            }
            Assert.Equal(budgetBuffers, TensorArena.PersistentReuseHits);
        }
        finally
        {
            TensorArena.MaxPersistTotalBytesOverrideForTests = savedOverride;
            TensorArena.ClearPersistentPool();
        }
    }

    /// <summary>
    /// Small buffers (below the persist threshold) are intentionally NOT pooled
    /// across lifetimes — they GC cheaply and pooling them would bloat the
    /// dictionary. This guards the threshold so the pool stays focused on the
    /// large buffers that actually drive the churn.
    /// </summary>
    [Fact]
    public void SmallBuffers_NotRetainedAcrossLifetimes()
    {
        TensorArena.ClearPersistentPool();
        int[] shape = { 64 }; // 256 bytes — far below PersistThresholdElems

        for (int i = 0; i < 4; i++)
        {
            using var arena = TensorArena.Create();
            var t = TensorAllocator.RentUninitialized<float>(shape);
            t.AsWritableSpan()[0] = i;
        }

        // A small buffer is re-allocated each cycle (no persistent retention),
        // so per-cycle allocation includes the array. We only assert this does
        // not THROW / mis-pool — correctness, not a perf bound (small allocs
        // are fine). Renting again must still succeed and be writable.
        using var arena2 = TensorArena.Create();
        var t2 = TensorAllocator.RentUninitialized<float>(shape);
        t2.AsWritableSpan()[0] = 7f;
        Assert.Equal(7f, t2.AsSpan()[0]);
    }
}
