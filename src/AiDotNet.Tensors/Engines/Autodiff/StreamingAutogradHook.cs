// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Issue #276 sub-feature 2: backward replay coordinates with the
/// streaming weight pool. Before the backward kernel for any op reads
/// from one of its input tensors, the autodiff dispatcher calls
/// <see cref="OnInputAccessed{T}"/> which routes the streaming weights
/// through <see cref="WeightRegistry.Materialize{T}"/>.
///
/// <para>This is the integration glue that closes the loop: weight is
/// tagged <see cref="WeightLifetime.Streaming"/> at construction →
/// registered with <see cref="WeightRegistry"/> (which drops the
/// tensor's backing storage and hands canonical bytes to the pool) →
/// pool may evict the bytes to disk under memory pressure → backward
/// replay rehydrates them before the kernel needs them.</para>
///
/// <para><b>Why this delegates to <see cref="WeightRegistry.Materialize{T}"/>
/// instead of doing the read itself:</b> <c>RegisterWeight</c> calls
/// <c>DropStorageForStreaming</c> which replaces the tensor's backing
/// <see cref="Vector{T}"/> with <see cref="Vector{T}.Empty"/>. After
/// that point <see cref="Tensor{T}.AsWritableSpan"/> returns an empty
/// span, so writing rehydrated bytes into the span (which an earlier
/// implementation of this hook did via its own <c>DeserializeFromBytes</c>
/// path) would either crash or silently no-op. <see cref="WeightRegistry.Materialize{T}"/>
/// goes through <see cref="TensorBase{T}.RestoreStorageFromBytes"/>,
/// which allocates a fresh <see cref="Vector{T}"/> of the right size
/// and replaces the tensor's storage — that's the only path that
/// correctly inflates a streaming weight back from the pool.</para>
/// </summary>
public static class StreamingAutogradHook
{
    /// <summary>Called by the backward dispatcher before any kernel reads
    /// <paramref name="input"/>. Promotes the entry's LRU position; if
    /// the entry has been evicted, rehydrates from the backing store and
    /// reinstates the tensor's backing storage.</summary>
    public static void OnInputAccessed<T>(Tensor<T> input)
    {
        if (input is null) return;
        if (input.Lifetime != WeightLifetime.Streaming) return;
        if (input.StreamingPoolHandle < 0) return;
        // Materialize handles all three branches:
        //   1. Already resident → bumps LRU (keeps hot weights warm).
        //   2. Evicted → snapshots bytes under registry lock, then
        //      drops the lock and calls RestoreStorageFromBytes
        //      which allocates a fresh Vector and atomically swaps
        //      the tensor's storage. This is the only path that
        //      correctly rebuilds a streaming weight whose backing
        //      Vector was replaced by Vector.Empty at register time.
        //   3. Not registered (handle < 0) → no-op (already filtered above).
        WeightRegistry.Materialize(input);
    }

    /// <summary>Called when a forward op records an entry; touches the
    /// LRU for any input that's a streaming weight so reading from it
    /// during forward also keeps it warm.</summary>
    public static void OnForwardRecorded<T>(Tensor<T>[]? inputs)
    {
        if (inputs is null) return;
        for (int i = 0; i < inputs.Length; i++) OnInputAccessed(inputs[i]);
    }
}
