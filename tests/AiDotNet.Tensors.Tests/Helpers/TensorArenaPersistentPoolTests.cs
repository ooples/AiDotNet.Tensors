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

    /// <summary>
    /// Deep-model churn guard: a training step for an N-layer network rents MANY
    /// same-size large buffers (one hidden-state / gradient per layer), not "O(1)
    /// per size". The cross-arena pool must retain and reuse ALL of them across a
    /// fresh-arena-per-step trainer — with the old per-size cap of 4 an 18-layer
    /// model dropped N-4 buffers on Dispose and re-allocated them every step
    /// (~2 GB/step Double[] churn measured on VALL-E-X-clone via PerfView). Renting
    /// 16 same-size large buffers per cycle must yield 16 reuse hits next cycle.
    /// </summary>
    [Fact]
    public void ManySameSizeBuffers_PerStep_AllReusedAcrossLifetimes()
    {
        TensorArena.ClearPersistentPool();
        const int n = 16;                       // > old cap (4); models a deep stack
        int[] shape = { 128 * 1024 };           // 512 KB float — above PersistThresholdElems

        // Cycle 1: cold — allocate n fresh, all returned to the pool on dispose.
        using (var arena = TensorArena.Create())
        {
            for (int i = 0; i < n; i++)
            {
                var t = TensorAllocator.RentUninitialized<float>(shape);
                t.AsWritableSpan()[0] = i;
            }
        }
        Assert.Equal(0, TensorArena.PersistentReuseHits);

        // Cycle 2: every one of the n same-size buffers must come from the pool.
        using (var arena = TensorArena.Create())
        {
            for (int i = 0; i < n; i++)
            {
                var t = TensorAllocator.RentUninitialized<float>(shape);
                t.AsWritableSpan()[0] = i;
            }
        }
        Assert.Equal(n, TensorArena.PersistentReuseHits);
    }

    /// <summary>
    /// The retention cap is a bound, not just a floor: renting MORE than
    /// MaxPersistPerSize (64) same-size buffers per cycle must retain AT MOST the
    /// cap across lifetimes — the excess is dropped on Dispose, so a later cycle
    /// sees exactly `cap` reuse hits (not all 65). Pins MaxPersistPerSize as the
    /// sole guard against unbounded per-size pool growth (CodeRabbit #767).
    /// </summary>
    [Fact]
    public void SameSizeBuffers_BeyondCap_RetentionIsCapped()
    {
        TensorArena.ClearPersistentPool();
        const int cap = 64;                     // MaxPersistPerSize
        const int n = cap + 1;                  // one past the cap
        int[] shape = { 128 * 1024 };           // above PersistThresholdElems

        using (var arena = TensorArena.Create())
            for (int i = 0; i < n; i++) TensorAllocator.RentUninitialized<float>(shape).AsWritableSpan()[0] = i;
        Assert.Equal(0, TensorArena.PersistentReuseHits);

        using (var arena = TensorArena.Create())
            for (int i = 0; i < n; i++) TensorAllocator.RentUninitialized<float>(shape).AsWritableSpan()[0] = i;

        // Only `cap` of the n=65 buffers were retained; the 65th was dropped on
        // Dispose and re-allocated fresh in cycle 2 — so exactly `cap` reuse hits.
        Assert.Equal(cap, TensorArena.PersistentReuseHits);
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
    /// The RentPersistentZeroed / ReturnPersistentBuffer API (used for long-lived
    /// fused-optimizer Adam m/v moment state that must outlive the transient
    /// per-step arena) hands back a ZEROED buffer, pools it across return/rent, and
    /// re-zeroes stale data on reuse. This is the transparent replacement for the
    /// per-reconfigure `new double[len]` that PerfView flagged in
    /// ConfigureOptimizerDouble (~2 GB churn on the VALL-E-X-clone warmup).
    /// </summary>
    [Fact]
    public void RentPersistentZeroed_ZeroesPoolsAndRezeroesOnReuse()
    {
        TensorArena.ClearPersistentPool();
        int len = 128 * 1024; // above PersistThresholdElems

        var a = TensorArena.RentPersistentZeroed<double>(len);
        Assert.Equal(len, a.Length);
        for (int i = 0; i < len; i++) Assert.Equal(0.0, a[i]); // rented zeroed
        a[0] = 42.0; a[len - 1] = 7.0;                          // dirty it
        TensorArena.ReturnPersistentBuffer(a);
        Assert.Equal(0, TensorArena.PersistentReuseHits);

        var b = TensorArena.RentPersistentZeroed<double>(len);
        Assert.Equal(1, TensorArena.PersistentReuseHits);       // reused, not re-allocated
        Assert.Same(a, b);                                      // same backing array
        for (int i = 0; i < len; i++) Assert.Equal(0.0, b[i]);  // stale 42/7 re-zeroed
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
