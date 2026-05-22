// Copyright (c) AiDotNet. All rights reserved.
// Regression coverage for TensorAllocator's int→long dim-product fix.
//
// Before this fix `Rent` / `RentPinned` accumulated the dim product in
// `checked(int * int)`, so any shape whose element count exceeded
// Int32.MaxValue (~2.1 B) threw the generic `System.OverflowException`
// from inside the multiplication itself. That surfaced upstream on
// AiDotNet (#1408 SonarCloud run 26241806890) as opaque crashes inside
// TimeMachine / DQN / OWLViT / DGCNN / TabTransformer / TabDPT /
// SlimSAM / TriaffineNER — the error never named the requested shape,
// so callers couldn't tell whether the alloc was 2 B + 1 elements (the
// realistic case) or 2^62 elements (a real bug). Long arithmetic lets
// the allocator name the size and point at `WeightRegistry.AllocateStreaming`
// before throwing.

using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class TensorAllocatorOverflowTests
{
    [Fact]
    public void Rent_NullShape_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => TensorAllocator.Rent<float>(null!));
    }

    [Fact]
    public void RentPinned_NullShape_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => TensorAllocator.RentPinned<float>(null!));
    }

    [Fact]
    public void Rent_NegativeDim_ThrowsArgumentOutOfRange_NamingTheDimAndIndex()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(
            () => TensorAllocator.Rent<float>(new[] { 4, -1, 8 }));
        // Diagnostic must name the offending dim by index so lazy-layer
        // sentinel-propagation bugs are debuggable upstream.
        Assert.Contains("shape[1]", ex.Message);
        Assert.Contains("-1", ex.Message);
    }

    [Fact]
    public void RentPinned_NegativeDim_ThrowsArgumentOutOfRange()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(
            () => TensorAllocator.RentPinned<float>(new[] { 2, -3 }));
        Assert.Contains("shape[1]", ex.Message);
    }

    [Fact]
    public void Rent_ShapeProductExceedsInt32_ThrowsInvalidOperationWithLongCount()
    {
        // 65537 * 65537 = 4,295,098,369 > Int32.MaxValue (2,147,483,647).
        // Old code: `checked(int * int)` throws bare OverflowException with
        // no shape context. New code: InvalidOperationException naming the
        // long element count and pointing at the streaming pool.
        var ex = Assert.Throws<InvalidOperationException>(
            () => TensorAllocator.Rent<float>(new[] { 65537, 65537 }));
        Assert.Contains("4295098369", ex.Message);
        Assert.Contains("Array.MaxLength", ex.Message);
        Assert.Contains("WeightRegistry.AllocateStreaming", ex.Message);
    }

    [Fact]
    public void RentPinned_ShapeProductExceedsInt32_ThrowsInvalidOperationWithLongCount()
    {
        var ex = Assert.Throws<InvalidOperationException>(
            () => TensorAllocator.RentPinned<float>(new[] { 50000, 50000 }));
        // 50000 * 50000 = 2_500_000_000 > Int32.MaxValue.
        Assert.Contains("2500000000", ex.Message);
        Assert.Contains("WeightRegistry.AllocateStreaming", ex.Message);
    }

    [Fact]
    public void Rent_ZeroElementShape_ReturnsEmptyTensor_NotThrow()
    {
        // A zero dim short-circuits the product to 0 — pinning / arena
        // logic shouldn't allocate, and the tensor must come back with
        // the exact requested shape.
        using var t = TensorAllocator.Rent<float>(new[] { 4, 0, 8 });
        Assert.Equal(0, t.Length);
        Assert.Equal(3, t.Shape.Length);
        Assert.Equal(4, t.Shape[0]);
        Assert.Equal(0, t.Shape[1]);
        Assert.Equal(8, t.Shape[2]);
    }
}
