// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Issue #276 sub-feature 2: backward replay coordinates with the
/// streaming weight pool. Before the backward kernel for any op reads
/// from one of its input tensors, the autodiff dispatcher calls
/// <see cref="OnInputAccessed{T}"/> which routes to
/// <see cref="StreamingTensorPool.MarkAccessed"/> /
/// <see cref="StreamingTensorPool.Rehydrate"/> for any input that's
/// registered with the pool.
///
/// <para>This is the integration glue that closes the loop: weight is
/// tagged <see cref="WeightLifetime.Streaming"/> at construction →
/// registered with <see cref="WeightRegistry"/> → its byte payload
/// can evict from the pool when memory pressure rises → backward
/// replay rehydrates it before the kernel needs it. Same pattern as
/// DeepSpeed ZeRO-Offload's prefetch coordination.</para>
/// </summary>
public static class StreamingAutogradHook
{
    /// <summary>Called by the backward dispatcher before any kernel reads
    /// <paramref name="input"/>. Promotes the entry's LRU position; if
    /// the entry has been evicted, rehydrates from the backing store.</summary>
    public static void OnInputAccessed<T>(Tensor<T> input)
    {
        if (input is null) return;
        if (input.Lifetime != WeightLifetime.Streaming) return;
        if (input.StreamingPoolHandle < 0) return;
        var pool = WeightRegistry.StreamingPool;
        // Mark accessed so the LRU keeps it resident; rehydrate if evicted.
        pool.MarkAccessed(input.StreamingPoolHandle);
        var bytes = pool.Rehydrate(input.StreamingPoolHandle);

        // Deserialize the rehydrated bytes back into the Tensor<T>'s live
        // storage. This is what closes the streaming-pool loop: when the
        // pool evicts a weight, the Tensor<T> buffer goes stale; on next
        // access we restore it from the backing-store snapshot.
        DeserializeFromBytes(input, bytes);
    }

    private static void DeserializeFromBytes<T>(Tensor<T> tensor, ReadOnlySpan<byte> src)
    {
        // Write into a fresh T[] of the right element type via Buffer.BlockCopy,
        // then copy that into the tensor's writable span using Tensor<T>'s
        // own typed accessor (which handles the underlying buffer copy).
        var srcArr = src.ToArray();
        var dstSpan = tensor.AsWritableSpan();
        if (typeof(T) == typeof(float))
        {
            var typed = new float[dstSpan.Length];
            Buffer.BlockCopy(srcArr, 0, typed, 0, Math.Min(srcArr.Length, typed.Length * sizeof(float)));
            for (int i = 0; i < dstSpan.Length; i++)
                dstSpan[i] = (T)(object)typed[i];
            return;
        }
        if (typeof(T) == typeof(double))
        {
            var typed = new double[dstSpan.Length];
            Buffer.BlockCopy(srcArr, 0, typed, 0, Math.Min(srcArr.Length, typed.Length * sizeof(double)));
            for (int i = 0; i < dstSpan.Length; i++)
                dstSpan[i] = (T)(object)typed[i];
            return;
        }
        if (typeof(T) == typeof(int))
        {
            var typed = new int[dstSpan.Length];
            Buffer.BlockCopy(srcArr, 0, typed, 0, Math.Min(srcArr.Length, typed.Length * sizeof(int)));
            for (int i = 0; i < dstSpan.Length; i++)
                dstSpan[i] = (T)(object)typed[i];
            return;
        }
        if (typeof(T) == typeof(long))
        {
            var typed = new long[dstSpan.Length];
            Buffer.BlockCopy(srcArr, 0, typed, 0, Math.Min(srcArr.Length, typed.Length * sizeof(long)));
            for (int i = 0; i < dstSpan.Length; i++)
                dstSpan[i] = (T)(object)typed[i];
            return;
        }
        if (typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16))
        {
            for (int i = 0; i < dstSpan.Length; i++)
            {
                ushort raw = (ushort)(srcArr[i * 2] | (srcArr[i * 2 + 1] << 8));
                dstSpan[i] = (T)(object)AiDotNet.Tensors.NumericOperations.BFloat16.FromRawBits(raw);
            }
            return;
        }
        // Unsupported T: the registry rejects unknown types at registration,
        // so this path is unreachable in practice. No-op for safety.
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
