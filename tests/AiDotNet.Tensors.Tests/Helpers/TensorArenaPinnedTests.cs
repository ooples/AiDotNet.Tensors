// Copyright (c) AiDotNet. All rights reserved.
// PR #333 — two-tier TensorArena (Pinned + Scratch). Pinned allocations
// (model weights, optimizer state, BatchNorm running stats) must survive
// Reset() and NEVER get re-issued as scratch on the next iteration.
// Without that guarantee, a Reset between training steps would let a
// later Rent return the same backing array as the weight tensor and
// downstream writes would corrupt the weight in place.

using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class TensorArenaPinnedTests
{
    /// <summary>
    /// Pinned allocations must persist across Reset() — the backing array
    /// stays valid and its contents survive. Without this, every Train
    /// iteration would zero out the layer weights.
    /// </summary>
    [Fact]
    public void Pinned_SurvivesReset_ContentsPreserved()
    {
        using var arena = TensorArena.Create();

        var pinned = TensorAllocator.RentPinned<float>(new[] { 4 });
        var span = pinned.AsWritableSpan();
        span[0] = 1f; span[1] = 2f; span[2] = 3f; span[3] = 4f;

        arena.Reset();

        var spanAfter = pinned.AsSpan();
        Assert.Equal(1f, spanAfter[0]);
        Assert.Equal(2f, spanAfter[1]);
        Assert.Equal(3f, spanAfter[2]);
        Assert.Equal(4f, spanAfter[3]);
    }

    /// <summary>
    /// Reset() must NOT hand the pinned array back as scratch on the next
    /// iteration. Verified by writing to the pinned tensor, calling Reset,
    /// then renting a scratch tensor of the same size and writing to it —
    /// the original pinned tensor's contents must be unchanged.
    /// </summary>
    [Fact]
    public void Pinned_NotReissuedAsScratch_AfterReset()
    {
        using var arena = TensorArena.Create();

        // Allocate a pinned tensor and stamp a signature into it.
        var weight = TensorAllocator.RentPinned<float>(new[] { 8 });
        var weightSpan = weight.AsWritableSpan();
        for (int i = 0; i < 8; i++) weightSpan[i] = 100f + i;

        // Mid-iteration the caller does some scratch work — both before and
        // after Reset — and we check that the scratch allocator never picks
        // up the pinned tensor's underlying array.
        var scratchBefore = TensorAllocator.Rent<float>(new[] { 8 });
        var scratchBeforeSpan = scratchBefore.AsWritableSpan();
        for (int i = 0; i < 8; i++) scratchBeforeSpan[i] = 999f;

        arena.Reset();

        var scratchAfter = TensorAllocator.Rent<float>(new[] { 8 });
        var scratchAfterSpan = scratchAfter.AsWritableSpan();
        for (int i = 0; i < 8; i++) scratchAfterSpan[i] = -7f;

        // The pinned tensor's contents must still be the signature.
        var weightSpanAfter = weight.AsSpan();
        for (int i = 0; i < 8; i++)
            Assert.Equal(100f + i, weightSpanAfter[i]);
    }

    /// <summary>
    /// PinnedArrayCount is the diagnostic that confirms pinned tier exists
    /// and grows monotonically across allocations. Used by perf tests /
    /// memory probes that assert weights land in the right tier.
    /// </summary>
    [Fact]
    public void PinnedArrayCount_GrowsOnEachRent_AndDoesNotResetOnReset()
    {
        using var arena = TensorArena.Create();

        Assert.Equal(0, arena.PinnedArrayCount);

        _ = TensorAllocator.RentPinned<float>(new[] { 4 });
        Assert.Equal(1, arena.PinnedArrayCount);

        _ = TensorAllocator.RentPinned<float>(new[] { 16 });
        Assert.Equal(2, arena.PinnedArrayCount);

        arena.Reset();

        // Reset must not drop pinned references — that would let GC reclaim
        // weight backing arrays still held by the model.
        Assert.Equal(2, arena.PinnedArrayCount);

        _ = TensorAllocator.RentPinned<double>(new[] { 32 });
        Assert.Equal(3, arena.PinnedArrayCount);
    }

    /// <summary>
    /// When no arena is active, RentPinned must still return a usable
    /// tensor (graceful degradation). The contract is "this tensor
    /// survives across iterations" — without an arena, plain CLR allocation
    /// satisfies that trivially because the array isn't in any pool.
    /// </summary>
    [Fact]
    public void RentPinned_NoActiveArena_FallsBackToCleanAllocation()
    {
        // Outside any `using var arena = TensorArena.Create()` block.
        var pinned = TensorAllocator.RentPinned<double>(new[] { 5 });
        Assert.NotNull(pinned);
        Assert.Equal(5, pinned.Length);

        // Should be zero-initialized like a fresh new Tensor<T>(shape).
        var span = pinned.AsSpan();
        for (int i = 0; i < 5; i++)
            Assert.Equal(0.0, span[i]);
    }

    /// <summary>
    /// Edge case: zero-size shape must not throw and must return a usable
    /// tensor. The pinned allocator should short-circuit to a plain
    /// Tensor&lt;T&gt;(shape) without consuming the pinned tier.
    /// </summary>
    [Fact]
    public void RentPinned_ZeroSize_ReturnsUsableTensor()
    {
        using var arena = TensorArena.Create();
        int countBefore = arena.PinnedArrayCount;

        var pinned = TensorAllocator.RentPinned<float>(new[] { 0 });
        Assert.NotNull(pinned);
        Assert.Equal(0, pinned.Length);

        // Zero-size short-circuit must not pollute the pinned tier counter.
        Assert.Equal(countBefore, arena.PinnedArrayCount);
    }
}
