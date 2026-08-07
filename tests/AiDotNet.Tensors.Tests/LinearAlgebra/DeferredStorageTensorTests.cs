// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Acceptance tests for deferred-storage tensors — <see cref="Tensor{T}.CreateDeferred"/> and the
/// <see cref="WeightLifetime.Deferred"/> branch of the materialization gate.
/// </summary>
/// <remarks>
/// The contract these pin down: a tensor's SHAPE and its STORAGE are separable. Shape metadata is
/// exact and free; the buffer (and the pending fill) appear on first value access, exactly once.
/// That is what lets a caller count a model's parameters, size a checkpoint or plan a memory
/// budget before allocating anything — and because the count comes from the same construction that
/// later allocates, the two cannot disagree.
/// </remarks>
public class DeferredStorageTensorTests
{
    [Fact]
    public void Shape_metadata_is_exact_while_storage_is_absent()
    {
        var t = Tensor<float>.CreateDeferred([4, 8, 2]);

        Assert.True(t.IsStorageDeferred);
        Assert.Equal(64, t.Length);
        Assert.Equal(3, t.Rank);
        Assert.Equal([4, 8, 2], t.Shape.ToArray());
        Assert.Equal(WeightLifetime.Deferred, t.Lifetime);

        // Reading metadata must not have allocated.
        Assert.True(t.IsStorageDeferred);
    }

    [Fact]
    public void First_value_read_materializes_and_runs_the_initializer()
    {
        int initializerRuns = 0;
        var t = Tensor<float>.CreateDeferred([3], w =>
        {
            initializerRuns++;
            for (int i = 0; i < w.Length; i++) w[i] = (i + 1) * 10f;
        });

        Assert.Equal(0, initializerRuns);
        Assert.True(t.IsStorageDeferred);

        Assert.Equal(20f, t[1]);

        Assert.False(t.IsStorageDeferred);
        Assert.Equal(1, initializerRuns);
        Assert.Equal([10f, 20f, 30f], Enumerable.Range(0, 3).Select(i => t[i]));
    }

    [Fact]
    public void Initializer_runs_exactly_once_across_many_reads()
    {
        int runs = 0;
        var t = Tensor<float>.CreateDeferred([5], _ => runs++);

        for (int i = 0; i < 5; i++) _ = t[i];
        _ = t[0];

        Assert.Equal(1, runs);
    }

    [Fact]
    public void Initializer_runs_exactly_once_under_concurrent_first_access()
    {
        int runs = 0;
        var t = Tensor<float>.CreateDeferred([256], w =>
        {
            System.Threading.Interlocked.Increment(ref runs);
            for (int i = 0; i < w.Length; i++) w[i] = 7f;
        });

        Parallel.For(0, 32, _ => { for (int i = 0; i < 256; i++) Assert.Equal(7f, t[i]); });

        Assert.Equal(1, runs);
        Assert.False(t.IsStorageDeferred);
    }

    [Fact]
    public void Deferred_without_initializer_materializes_zeroed()
    {
        var t = Tensor<float>.CreateDeferred([4]);

        Assert.Equal(0f, t[2]);
        Assert.False(t.IsStorageDeferred);
        Assert.All(Enumerable.Range(0, 4).Select(i => t[i]), v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Writing_a_value_materializes_before_the_write_lands()
    {
        var t = Tensor<float>.CreateDeferred([3], w => { for (int i = 0; i < w.Length; i++) w[i] = 1f; });

        t[1] = 99f;

        // The initializer's fill must not clobber a write that triggered it.
        Assert.Equal(1f, t[0]);
        Assert.Equal(99f, t[1]);
        Assert.Equal(1f, t[2]);
    }

    [Fact]
    public void Deferred_matches_an_eagerly_allocated_tensor_of_the_same_shape()
    {
        var eager = new Tensor<float>([6, 3]);
        var deferred = Tensor<float>.CreateDeferred([6, 3]);

        Assert.Equal(eager.Length, deferred.Length);
        Assert.Equal(eager.Rank, deferred.Rank);
        Assert.Equal(eager.Shape.ToArray(), deferred.Shape.ToArray());
    }

    [Fact]
    public void CloneDeepCopy_of_a_deferred_tensor_carries_the_initialized_values()
    {
        var t = Tensor<float>.CreateDeferred([4], w => { for (int i = 0; i < w.Length; i++) w[i] = i; });

        var clone = t.CloneDeepCopy();

        Assert.Equal(4, clone.Length);
        for (int i = 0; i < 4; i++) Assert.Equal((float)i, clone[i]);
    }

    [Fact]
    public void Transform_of_a_deferred_tensor_sees_initialized_values_not_an_empty_buffer()
    {
        var t = Tensor<float>.CreateDeferred([3], w => { for (int i = 0; i < w.Length; i++) w[i] = 2f; });

        var doubled = t.Transform(v => v * 2f);

        Assert.Equal(3, doubled.Length);
        for (int i = 0; i < 3; i++) Assert.Equal(4f, doubled[i]);
    }

    [Fact]
    public void A_never_touched_deferred_tensor_stays_unallocated()
    {
        // The whole point: 100 million logical elements, no allocation, exact count.
        var huge = Tensor<float>.CreateDeferred([10_000, 10_000]);

        Assert.Equal(100_000_000, huge.Length);
        Assert.True(huge.IsStorageDeferred);
    }
}
