// Copyright (c) AiDotNet. All rights reserved.
// The arena's tensor ring was keyed by element COUNT only, then cast the cached wrapper to Tensor<T>.
// One arena serving both float and double operations of the same size (a float model whose pitch
// preprocessing runs a double FFT, a mixed-precision step) handed a Tensor<double> to a Tensor<float>
// request: InvalidCastException in TensorAllocator.RentUninitialized.

using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

[Collection(nameof(TensorArenaPinnedTests))]
public class TensorArenaElementTypeTests
{
    [Fact]
    public void SameSizeRentsOfDifferentElementTypes_EachGetTheirOwnType()
    {
        using var arena = TensorArena.Create();

        var asDouble = TensorAllocator.RentUninitialized<double>(new[] { 4, 4 });
        var asFloat = TensorAllocator.RentUninitialized<float>(new[] { 4, 4 });

        Assert.IsType<Tensor<double>>(asDouble);
        Assert.IsType<Tensor<float>>(asFloat);
    }

    [Fact]
    public void AfterReset_ARentOfTheOtherElementTypeDoesNotReceiveTheCachedWrapper()
    {
        using var arena = TensorArena.Create();

        // Step 1 leaves a double wrapper of 16 elements in the ring.
        TensorAllocator.RentUninitialized<double>(new[] { 16 });
        arena.Reset();

        // Step 2 asks for a float tensor of the same element count first.
        var asFloat = TensorAllocator.RentUninitialized<float>(new[] { 16 });
        var asDouble = TensorAllocator.RentUninitialized<double>(new[] { 16 });

        Assert.IsType<Tensor<float>>(asFloat);
        Assert.IsType<Tensor<double>>(asDouble);
        Assert.Equal(16, asFloat.Length);
        Assert.Equal(16, asDouble.Length);
    }

    [Fact]
    public void InterleavedElementTypes_ReuseTheirOwnWrappersAcrossResets()
    {
        using var arena = TensorArena.Create();

        var firstDouble = TensorAllocator.RentUninitialized<double>(new[] { 8 });
        var firstFloat = TensorAllocator.RentUninitialized<float>(new[] { 8 });
        arena.Reset();

        // Zero-allocation reuse still works per type: each type gets back its own cached wrapper.
        Assert.Same(firstFloat, TensorAllocator.RentUninitialized<float>(new[] { 8 }));
        Assert.Same(firstDouble, TensorAllocator.RentUninitialized<double>(new[] { 8 }));
    }
}
