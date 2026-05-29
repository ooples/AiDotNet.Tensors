// Copyright (c) AiDotNet. All rights reserved.
// Issue #478 — Part 2 ("safety net for non-arena callers"). The arena-active
// per-step churn is covered by TensorArenaPersistentPoolTests; this file pins
// the complementary contract the issue called out for callers running WITHOUT
// an active TensorArena (inference, ad-hoc ops, any path that doesn't wrap a
// TensorArena scope):
//
//   A large (> ArrayPool<T>.Shared's 2^20-element max bucket) buffer rented via
//   TensorAllocator and returned via TensorPool.Return must be REUSED on the
//   next rent of the same size, not GC-allocated fresh every time.
//
// #478's worry was that TensorAllocator's large-buffer path bottomed out at
// ArrayPool<T>.Shared, which refuses to retain > 2^20-element arrays (its
// largest bucket) — so Rent returned a fresh array and Return dropped it,
// churning ~1 GB/step at paper scale. On net5+ TensorAllocator's Rent/Return
// already route through the element-cap-free ThreadLocalTensorCache BEFORE
// ArrayPool, so the reuse holds — but nothing pinned it. These tests are that
// guard: if the large-buffer path ever regresses to ArrayPool.Shared (or grows
// an element cap), the reuse / no-churn assertions fail.
//
// net471 NOTE: the non-arena pooling tier (ThreadLocalTensorCache + ArrayPool)
// is gated behind NET5_0_OR_GREATER; on net471 non-arena RentUninitialized
// returns a plain `new Tensor<T>` every time (no reuse). Paper-scale TRAINING
// on net471 is still covered because it runs inside a TensorArena, whose
// cross-arena persistent pool (TensorArenaPersistentPoolTests) is NOT TFM-gated.
// The net471 test below pins that documented "fresh, correct, no-pool" behavior.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class NonArenaLargeBufferReuseTests
{
    // 2^21 float elements = 8 MB — strictly larger than ArrayPool<T>.Shared's
    // 2^20-element (4 MB float) max bucket, i.e. exactly the paper-scale
    // gradient-buffer size ArrayPool refuses to pool and that #478 is about.
    private const int LargeElems = 2 * 1024 * 1024;

    private static void AssertNoActiveArena()
    {
        // The whole point of "Part 2" is the no-arena path; if a prior test on
        // this reused xUnit worker thread leaked an active arena, the buffer
        // would flow through the arena tier instead and this guard would be
        // testing the wrong thing.
        Assert.Null(TensorArena.Current);
    }

    /// <summary>
    /// Faithful guard for the issue's NAMED hot path: the gradient backward free
    /// path returns intermediates via <c>AutoTensorCache.Return</c>
    /// (GradientsScope), NOT <c>TensorPool.Return</c>. AutoTensorCache keeps its
    /// own per-shape pool of whole <see cref="Tensor{T}"/> objects (TFM-agnostic —
    /// not gated behind the net5 ThreadLocalTensorCache tier), so a large buffer
    /// rented + returned through it must come back by IDENTITY on the next
    /// same-shape rent, with no active arena. If this regresses (e.g. the
    /// per-shape cap drops to 0, the RAM-budget element cap falls below this size,
    /// or Return stops pooling) the &gt; 2^20-element backward buffer churns the GC
    /// every step — the exact #478 symptom — and this fails.
    /// </summary>
    [Fact]
    public void LargeBuffer_NoArena_AutoTensorCache_ReusesByIdentity()
    {
        AssertNoActiveArena();
        AutoTensorCache.Clear();
        int[] shape = { LargeElems };

        var first = AutoTensorCache.RentOrAllocate<float>(shape);
        // A freshly-rented tensor has no GradFn, so AutoTensorCache.Return will
        // pool it (the live-GradFn guard that protects tape-pinned tensors from
        // reuse doesn't apply here).
        Assert.Null(first.GradFn);
        float[] firstBacking = first.GetDataArray();
        AutoTensorCache.Return(first);

        var second = AutoTensorCache.RentOrAllocate<float>(shape);
        float[] secondBacking = second.GetDataArray();

        Assert.Same(firstBacking, secondBacking);

        AutoTensorCache.Clear();
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Core Part-2 guarantee, measured directly: with NO arena active, renting a
    /// &gt; 2^20-element buffer and returning it must make the next same-size rent
    /// reuse the backing array — steady-state per-cycle allocation is bookkeeping
    /// only, not the multi-MB array. Uses <c>GC.GetTotalAllocatedBytes</c> (net5+).
    /// </summary>
    [Fact]
    public void LargeBuffer_NoArena_RentReturnRent_DoesNotChurn()
    {
        AssertNoActiveArena();
        ThreadLocalTensorCache<float>.Clear();
        int[] shape = { LargeElems };

        // Warmup: a few rent→return cycles to populate the thread-local cache and
        // reach steady state (first rent allocates fresh; Return retains it).
        for (int i = 0; i < 4; i++)
        {
            var t = TensorAllocator.RentUninitialized<float>(shape);
            t.AsWritableSpan()[0] = i; // touch so nothing is elided
            TensorPool.Return(t);
        }

        long before = GC.GetTotalAllocatedBytes(precise: true);
        var rented = TensorAllocator.RentUninitialized<float>(shape);
        rented.AsWritableSpan()[0] = 42f;
        TensorPool.Return(rented);
        long delta = GC.GetTotalAllocatedBytes(precise: true) - before;

        long bufferBytes = (long)LargeElems * sizeof(float); // 8 MB
        Assert.True(delta < bufferBytes / 4,
            $"Per-cycle allocation was {delta} bytes; the {bufferBytes}-byte buffer is " +
            "being re-allocated every cycle — the non-arena large-buffer path is bottoming " +
            "out at ArrayPool.Shared (which drops > 2^20-element arrays) instead of reusing " +
            "via the cap-free thread-local cache.");

        // Cleanup so we don't leak the 8 MB buffer into other tests on this thread.
        ThreadLocalTensorCache<float>.Clear();
    }

    /// <summary>
    /// Reuse proof by identity: the SAME backing array instance comes back on a
    /// subsequent same-size rent after a return — the cap-free thread-local cache
    /// is genuinely recycling the &gt; 2^20-element buffer, not handing out a new
    /// one. (Identity, not just byte-equality, so a fresh-zeroed array can't pass
    /// by accident.)
    /// </summary>
    [Fact]
    public void LargeBuffer_NoArena_SameBackingArrayReturnedAfterReturn()
    {
        AssertNoActiveArena();
        ThreadLocalTensorCache<float>.Clear();
        int[] shape = { LargeElems };

        var first = TensorAllocator.RentUninitialized<float>(shape);
        float[] firstBacking = first.GetDataArray();
        TensorPool.Return(first);

        var second = TensorAllocator.RentUninitialized<float>(shape);
        float[] secondBacking = second.GetDataArray();
        TensorPool.Return(second);

        Assert.Same(firstBacking, secondBacking);

        ThreadLocalTensorCache<float>.Clear();
    }

    /// <summary>
    /// The clear tier (<see cref="TensorAllocator.Rent{T}"/>) must hand back a
    /// zeroed buffer even when reusing a recycled &gt; 2^20-element array that
    /// previously held non-zero data — reuse must never leak the prior renter's
    /// bytes (the SIMD-overhang corruption class from #311, here on the non-arena
    /// large-buffer path).
    /// </summary>
    [Fact]
    public void LargeBuffer_NoArena_ClearTierReuse_IsZeroInitialized()
    {
        AssertNoActiveArena();
        ThreadLocalTensorCache<float>.Clear();
        int[] shape = { LargeElems };

        var dirty = TensorAllocator.Rent<float>(shape);
        var dspan = dirty.AsWritableSpan();
        for (int i = 0; i < 32; i++) dspan[i] = 777.25f;
        TensorPool.Return(dirty);

        var recycled = TensorAllocator.Rent<float>(shape);
        var rspan = recycled.AsSpan();
        for (int i = 0; i < 32; i++)
            Assert.Equal(0f, rspan[i]);
        TensorPool.Return(recycled);

        ThreadLocalTensorCache<float>.Clear();
    }
#else
    /// <summary>
    /// net471 has no non-arena pooling tier (the ThreadLocalTensorCache + ArrayPool
    /// block is gated behind NET5_0_OR_GREATER). This pins the documented behavior
    /// there: a non-arena large-buffer rent returns a correct, writable, exact-size
    /// tensor every time (no reuse — that's fine, training on net471 pools via the
    /// TensorArena cross-arena persistent pool instead, which is not TFM-gated).
    /// </summary>
    [Fact]
    public void LargeBuffer_NoArena_Net471_ReturnsCorrectFreshTensor()
    {
        AssertNoActiveArena();
        int[] shape = { LargeElems };

        var t = TensorAllocator.RentUninitialized<float>(shape);
        Assert.Equal(LargeElems, t.Length);
        var span = t.AsWritableSpan();
        span[0] = 5f;
        span[LargeElems - 1] = 9f;
        Assert.Equal(5f, t.AsSpan()[0]);
        Assert.Equal(9f, t.AsSpan()[LargeElems - 1]);
    }
#endif
}
