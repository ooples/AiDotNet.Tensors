// Copyright (c) AiDotNet. All rights reserved.
//
// Regression + invariant suite for the TensorArena "zero-alloc training"
// mechanism (Helpers/TensorArena.cs). These tests pin down the exact shape
// bug fixed in AiDotNet #1804: the tensor RING buckets pooled buffers by
// element COUNT, so a same-count / different-shape rent after Reset() must
// re-issue the pooled tensor under the REQUESTED shape (ArenaReshapeInPlace),
// including recomputing strides AND clearing the row-major strides cache.
// Without the fix a stale-shaped tensor flows downstream and either crashes
// shape checks or reads through the wrong strides.

using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Serializes the arena reshape/regression tests onto a single runner thread.
/// <see cref="TensorArena.Current"/> is [ThreadStatic] and xUnit reuses worker
/// threads across collections, so a leaked arena from a parallel test could be
/// observed here. Same rationale as <see cref="TensorArenaPinnedTests"/>.
/// </summary>
[CollectionDefinition(nameof(TensorArenaReshapeRegressionTests), DisableParallelization = true)]
public sealed class TensorArenaReshapeRegressionTestsCollection { }

[Collection(nameof(TensorArenaReshapeRegressionTests))]
public class TensorArenaReshapeRegressionTests
{
    /// <summary>
    /// THE bug (AiDotNet #1804). Rent a [3,256] tensor (count 768) from the
    /// ring, stamp linear values, Reset, then rent count 768 at shape [256,3].
    /// The ring re-issues the SAME backing buffer; it MUST be reshaped to the
    /// requested [256,3] with fresh row-major strides [3,1] (not the stale
    /// [3,256] strides [256,1]). Flat indexing must still see the original
    /// linear data, and 2D indexing must decode using the NEW strides.
    /// Without the fix this test fails: either the shape assert or the 2D read
    /// (row index up to 255 is out of range for a stale [3,256] shape).
    /// </summary>
    [Fact]
    public void ArenaReshapeInPlace_SameCountDifferentShape_ReshapesCorrectly()
    {
        using var arena = TensorArena.Create();

        var first = arena.TryRentTensor<float>(768, new[] { 3, 256 });
        Assert.NotNull(first);
        Assert.Equal(new[] { 3, 256 }, first!._shape);

        // Stamp known linear values: element i == i.
        for (int i = 0; i < 768; i++) first[i] = i;

        arena.Reset();

        // Same element count (768), DIFFERENT shape [256,3].
        var second = arena.TryRentTensor<float>(768, new[] { 256, 3 });
        Assert.NotNull(second);

        // Must be re-issued under the requested shape with correct metadata.
        Assert.Equal(new[] { 256, 3 }, second!._shape);
        Assert.Equal(768, second.Length);
        Assert.Equal(new[] { 3, 1 }, second.Strides.ToArray());
        // The cached row-major strides must have been invalidated by the reshape.
        Assert.Equal(new[] { 3, 1 }, second.RowMajorStrides);
        Assert.True(second.IsContiguous);

        // Same backing buffer -> linear data preserved (ring path never clears).
        for (int i = 0; i < 768; i++)
            Assert.Equal((float)i, second[i]);

        // 2D read must decode with the NEW [256,3] strides: [r,c] -> r*3 + c.
        for (int r = 0; r < 256; r++)
            for (int c = 0; c < 3; c++)
                Assert.Equal((float)(r * 3 + c), second[r, c]);
    }

    /// <summary>
    /// The reshape must be a metadata-only swap that also works back the other
    /// way (round-trip), so repeated permute-style reuse across steps is stable.
    /// </summary>
    [Fact]
    public void ArenaReshapeInPlace_RoundTripsAcrossResets()
    {
        using var arena = TensorArena.Create();

        var a = arena.TryRentTensor<double>(12, new[] { 3, 4 });
        Assert.Equal(new[] { 4, 1 }, a!.Strides.ToArray());

        arena.Reset();
        var b = arena.TryRentTensor<double>(12, new[] { 4, 3 });
        Assert.Equal(new[] { 4, 3 }, b!._shape);
        Assert.Equal(new[] { 3, 1 }, b.Strides.ToArray());

        arena.Reset();
        var c = arena.TryRentTensor<double>(12, new[] { 2, 6 });
        Assert.Equal(new[] { 2, 6 }, c!._shape);
        Assert.Equal(new[] { 6, 1 }, c.Strides.ToArray());
        Assert.Equal(12, c.Length);
    }

    /// <summary>
    /// Ring reuse hands back the SAME backing buffer without clearing (the
    /// documented "caller overwrites every element" contract). This verifies
    /// both halves: (a) the buffer is genuinely reused (still carries the prior
    /// sentinel immediately after re-rent, proving zero-alloc reuse), and
    /// (b) once the consumer writes every element the reads are exactly what it
    /// wrote (no stale leak into a full-overwrite consumer).
    /// </summary>
    [Fact]
    public void RingReuse_ReusesBuffer_CallerFullyOverwrites()
    {
        using var arena = TensorArena.Create();

        var t1 = arena.TryRentTensor<float>(64, new[] { 64 });
        Assert.NotNull(t1);
        for (int i = 0; i < 64; i++) t1![i] = 7f; // sentinel

        arena.Reset();

        var t2 = arena.TryRentTensor<float>(64, new[] { 64 });
        Assert.NotNull(t2);
        // (a) Same buffer reused: uninitialized ring path did NOT clear, so the
        // sentinel is still present. This is the zero-alloc reuse guarantee.
        Assert.Same(t1, t2);
        for (int i = 0; i < 64; i++) Assert.Equal(7f, t2![i]);

        // (b) Consumer that writes every element fully controls contents.
        for (int i = 0; i < 64; i++) t2![i] = i * 2f;
        for (int i = 0; i < 64; i++) Assert.Equal(i * 2f, t2![i]);
    }

    /// <summary>
    /// Pinned/heap allocations (optimizer moments, weights) must NOT be
    /// recycled by Reset(). A pinned tensor's contents survive a Reset +
    /// re-rent-of-same-size cycle even when the scratch consumer scribbles
    /// over freshly rented buffers.
    /// </summary>
    [Fact]
    public void Reset_PreservesPinnedAndHeap_MomentsSurvive()
    {
        using var arena = TensorArena.Create();

        // Simulate Adam moment state via the pinned tier.
        var moment = TensorAllocator.RentPinned<float>(new[] { 32 });
        var mspan = moment.AsWritableSpan();
        for (int i = 0; i < 32; i++) mspan[i] = 0.5f + i;

        // A plain heap tensor (new ctor) is not arena-managed at all.
        var heap = new Tensor<float>(new[] { 32 });
        for (int i = 0; i < 32; i++) heap[i] = 1000f + i;

        // Do scratch work, reset, do more scratch work over the SAME size.
        var scratch1 = arena.TryRentTensor<float>(32, new[] { 32 });
        for (int i = 0; i < 32; i++) scratch1![i] = -1f;

        arena.Reset();

        var scratch2 = arena.TryRentTensor<float>(32, new[] { 32 });
        for (int i = 0; i < 32; i++) scratch2![i] = -99f;

        // Pinned moment untouched.
        var mAfter = moment.AsSpan();
        for (int i = 0; i < 32; i++) Assert.Equal(0.5f + i, mAfter[i]);

        // Heap tensor untouched.
        for (int i = 0; i < 32; i++) Assert.Equal(1000f + i, heap[i]);
    }

    /// <summary>
    /// Nested arenas must not cross-contaminate. An inner arena's rented
    /// buffers are independent of the outer arena's; after the inner disposes
    /// (restoring the outer as Current), the outer's buffers are intact and a
    /// fresh outer rent does not collide with inner data.
    /// </summary>
    [Fact]
    public void NestedArena_NoCrossContamination()
    {
        using var outer = TensorArena.Create();
        Assert.Same(outer, TensorArena.Current);

        var outerBuf = outer.TryRentTensor<float>(48, new[] { 48 });
        for (int i = 0; i < 48; i++) outerBuf![i] = 11f;

        using (var inner = TensorArena.Create())
        {
            Assert.Same(inner, TensorArena.Current);
            var innerBuf = inner.TryRentTensor<float>(48, new[] { 48 });
            // Distinct backing object from the outer buffer.
            Assert.NotSame(outerBuf, innerBuf);
            for (int i = 0; i < 48; i++) innerBuf![i] = 22f;
            // Writing inner must not touch outer.
            for (int i = 0; i < 48; i++) Assert.Equal(11f, outerBuf![i]);
        }

        // Inner disposed -> outer restored as Current.
        Assert.Same(outer, TensorArena.Current);

        // Outer buffer still intact.
        for (int i = 0; i < 48; i++) Assert.Equal(11f, outerBuf![i]);

        // A further outer reset + rent reuses the OUTER buffer (same object),
        // never anything from the disposed inner arena.
        outer.Reset();
        var outerBuf2 = outer.TryRentTensor<float>(48, new[] { 48 });
        Assert.Same(outerBuf, outerBuf2);
    }
}
