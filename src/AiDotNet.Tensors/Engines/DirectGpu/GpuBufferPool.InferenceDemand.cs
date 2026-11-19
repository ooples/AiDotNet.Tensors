using System;
using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

internal sealed partial class GpuBufferPool<TBuffer> where TBuffer : class, IGpuBuffer, IPoolableGpuBuffer
{
    private long _inferenceRegimeReleasedBuffers, _inferenceRegimeReleasedCapacityBytes;
    internal long InferenceRegimeNativeReleaseRequestedBuffers => Interlocked.Read(ref _inferenceRegimeReleasedBuffers);
    internal long InferenceRegimeNativeReleaseRequestedCapacityBytes => Interlocked.Read(ref _inferenceRegimeReleasedCapacityBytes);

    internal bool TryRentForBackend(int size, IDirectGpuBackend? backend, out TBuffer? buffer)
        => TryRentForBackend(size, GpuBufferPoolAffinity.Global, backend, out buffer);

    internal bool TryRentForBackend(int size, GpuBufferPoolAffinity affinity, IDirectGpuBackend? backend,
        out TBuffer? buffer)
    {
        if (TryRent(size, affinity, out buffer)) return true;
        if (Volatile.Read(ref _disposed) != 0 || size <= 0 || size > _maxSize) return false;
        if (backend is not null && InferenceWorkspaceDemandScope.TryRentLargerIdleBuffer(
                backend, this, size, affinity, out buffer)) return true;
        if (backend is not null && InferenceWorkspaceDemandScope.TryReturnIncompatibleIdleCache(backend, this, size, affinity))
            return TryRent(size, affinity, out buffer); // A concurrent return may have supplied the new size class.
        return false;
    }

    // Only called after the ordinary size-class rent missed. Increasing bucket order chooses the smallest
    // reusable class without enumerating the pool or allocating a search list. Physical capacity never exceeds
    // four times the logical request, and the existing parked-element budget remains unchanged.
    internal bool TryRentLargerIdleBuffer(int size, GpuBufferPoolAffinity affinity, out TBuffer? buffer)
    {
        buffer = null;
        if (_exactSizes || size <= 0 || size > _maxSize || Volatile.Read(ref _disposed) != 0) return false;
        int requestedBucket = BucketKey(size);
        if (requestedBucket <= 0) return false; // Power-of-two rounding overflow: preserve the native fallback.
        long limit = Math.Min((long)size * 4, int.MaxValue);
        for (long bucket = (long)requestedBucket * 2; bucket <= limit; bucket *= 2)
            if (TryRentFromBucket(size, (int)bucket, affinity, out buffer)) return true;
        return false;
    }

    internal int ReturnIncompatibleIdleForLargerRequest(int requested, GpuBufferPoolAffinity affinity)
    {
        if (requested <= 0 || requested > _maxSize || Volatile.Read(ref _disposed) != 0) return 0;
        List<TBuffer>? detached = null;
        lock (_lifecycleGate)
        {
            if (Volatile.Read(ref _disposed) != 0) return 0;
            bool any = false;
            // Return is serialized by this gate. Rent may remove a cell concurrently; removed cells are never freed
            // here. A same/larger idle cell means this is not a proven transition to a larger working size class.
            foreach (var bucket in _buckets.Values)
                if (bucket.Buffers.TryGetValue(affinity, out var idle))
                    foreach (var candidate in idle)
                    {
                        any = true;
                        if (candidate.Capacity >= requested) return 0;
                    }
            if (!any) return 0;
            foreach (var bucket in _buckets.Values)
                if (bucket.Buffers.TryGetValue(affinity, out var idle))
                    while (idle.TryPop(out var candidate))
                    {
                        Interlocked.Decrement(ref bucket.Count);
                        Interlocked.Add(ref _pooledElements, -candidate.Capacity);
                        UnmarkMeasurementParked(candidate);
                        (detached ??= new List<TBuffer>()).Add(candidate);
                    }
        }
        // Native release can enter the driver's/context's own locks or callbacks. Never hold the pool gate across it.
        // A nested/closing inference scope cannot lend these detached private cells; every cell still gets released.
        if (detached is null) return 0;
        Exception? firstFailure = null;
        foreach (var buffer in detached)
        {
            long capacityBytes = checked((long)buffer.Capacity * sizeof(float));
            try
            {
                buffer.Release();
                Interlocked.Increment(ref _inferenceRegimeReleasedBuffers);
                Interlocked.Add(ref _inferenceRegimeReleasedCapacityBytes, capacityBytes);
            }
            catch (Exception error) { firstFailure ??= error; }
        }
        if (firstFailure is not null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(firstFailure).Throw();
        return detached.Count;
    }
}
