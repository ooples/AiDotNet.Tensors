using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.LinearAlgebra;

public static partial class WeightRegistry
{
    /// <summary>
    /// Configures backing storage and registers a quiescent model's weights as one transaction.
    /// Allocation, serialization, or I/O failure leaves the prior configuration and all supplied
    /// tensors unchanged. Duplicate references are registered once. The caller must prevent
    /// concurrent use or mutation of the supplied model while configuring it.
    /// </summary>
    /// <remarks>
    /// Streaming owners retain their data until <see cref="TryFinalizeDeferredDrop{T}"/> is called,
    /// allowing a framework to publish its model and layer configuration before reclaiming storage.
    /// On success this registry owns the allocator. On failure the supplied allocator remains
    /// caller-owned. Disposal of a retired, empty backend is post-commit diagnostic cleanup.
    /// </remarks>
    public static void ConfigureAndRegisterBatch<T>(
        GpuOffloadOptions options,
        IReadOnlyList<Tensor<T>> weights,
        IGpuOffloadAllocator? offloadAllocator = null)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("Weight streaming requires a little-endian host.");
        WeightLifetime lifetime = offloadAllocator is null ? WeightLifetime.Streaming : WeightLifetime.GpuOffload;
        List<Tensor<T>> unique = ValidateBatch(weights, lifetime, offloadAllocator, configuring: true);
        DrainInFlightPrefetches();
        StreamingTensorPool? candidatePool = lifetime == WeightLifetime.Streaming
            ? new StreamingTensorPool(options)
            : null;
        StreamingTensorPool? retiredPool = null;
        IGpuOffloadAllocator? retiredAllocator = null;
        try
        {
            lock (_lock)
            {
                PruneDeadOwnersUnlocked();
                PruneDeadOffloadOwnersUnlocked();
                if (_streamingPool is not null && _streamingPool.RegisteredEntryCount > 0)
                    throw new InvalidOperationException("WeightRegistry.Configure: existing streaming pool has registered entries. Unregister all weights first.");
                if (_streamingPool is not null && _streamingPool.ReservedBytes > 0)
                    throw new InvalidOperationException("WeightRegistry.Configure: existing streaming pool has outstanding reservations. Complete or abandon those allocations first.");
                if (_offloadRegistrations.Count > 0)
                    throw new InvalidOperationException("WeightRegistry.Configure: the active GPU offload allocator has live allocations. Unregister all offloaded weights first.");

                List<PendingRegistration<T>> pending =
                    StageBatchUnlocked(unique, lifetime, options, candidatePool, offloadAllocator);
                retiredPool = _streamingPool;
                retiredAllocator = _offloadAllocator;
                _options = options;
                _streamingPool = candidatePool;
                _offloadAllocator = offloadAllocator;
                PublishBatch(pending, lifetime);
            }
        }
        catch
        {
            DisposeRetiredBackend(candidatePool);
            throw;
        }

        DisposeRetiredBackend(retiredPool);
        if (!ReferenceEquals(retiredAllocator, offloadAllocator))
            DisposeRetiredBackend(retiredAllocator);
    }

    /// <summary>
    /// Atomically adds newly materialized weights to the current configuration. Existing compatible
    /// registrations are preserved. Work and rollback are proportional to the new weights, rather
    /// than copying the entire live registry. Streaming reservations are restored on failure.
    /// </summary>
    public static void RegisterBatch<T>(IReadOnlyList<Tensor<T>> weights, WeightLifetime lifetime)
    {
        lock (_lock)
        {
            List<Tensor<T>> unique = ValidateBatch(weights, lifetime, _offloadAllocator, configuring: false);
            if (unique.Count == 0) return;
            StreamingTensorPool? pool = lifetime == WeightLifetime.Streaming ? StreamingPoolUnlocked() : null;
            List<PendingRegistration<T>> pending = StageBatchUnlocked(unique, lifetime, _options, pool, _offloadAllocator);
            PublishBatch(pending, lifetime);
        }
    }

    private static List<Tensor<T>> ValidateBatch<T>(
        IReadOnlyList<Tensor<T>> weights, WeightLifetime lifetime,
        IGpuOffloadAllocator? allocator, bool configuring)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (lifetime is not WeightLifetime.Streaming and not WeightLifetime.GpuOffload)
            throw new ArgumentOutOfRangeException(nameof(lifetime));
        if (lifetime == WeightLifetime.GpuOffload && (allocator is null || !allocator.IsAvailable))
            throw new InvalidOperationException("The requested GPU offload allocator is unavailable.");
        var result = new List<Tensor<T>>(weights.Count);
        var seen = new HashSet<Tensor<T>>(WeightReferenceComparer<T>.Instance);
        for (int i = 0; i < weights.Count; i++)
        {
            Tensor<T> weight = weights[i] ?? throw new ArgumentException("A weight cannot be null.", nameof(weights));
            if (!seen.Add(weight) || weight.Length == 0) continue;
            bool registered = weight.StreamingPoolHandle >= 0 || weight.OffloadRegistryHandle >= 0
                || weight.OffloadHostPointer != IntPtr.Zero;
            if (registered)
            {
                if (!configuring && weight.Lifetime == lifetime) continue;
                throw new InvalidOperationException("A registered weight cannot change backing configuration.");
            }
            if (configuring && weight.StreamingReservedBytes != 0)
                throw new InvalidOperationException("A weight with a live streaming reservation cannot change pools.");
            if (lifetime == WeightLifetime.Streaming && (weight.IsView || !weight.IsContiguous || weight._storageOffset != 0))
                throw new NotSupportedException("Streaming requires owning contiguous weight tensors, not views.");
            if (lifetime == WeightLifetime.GpuOffload) CheckedStreamingByteCount<T>(weight.Length);
            result.Add(weight);
        }
        return result;
    }

    private readonly struct PendingRegistration<T>
    {
        internal PendingRegistration(Tensor<T> weight, long handle, byte encoding,
            GpuOffloadHandle offload, long reservedBytes)
        {
            Weight = weight;
            Handle = handle;
            Encoding = encoding;
            Offload = offload;
            ReservedBytes = reservedBytes;
        }
        internal Tensor<T> Weight { get; }
        internal long Handle { get; }
        internal byte Encoding { get; }
        internal GpuOffloadHandle Offload { get; }
        internal long ReservedBytes { get; }
    }

    private static List<PendingRegistration<T>> StageBatchUnlocked<T>(
        List<Tensor<T>> weights, WeightLifetime lifetime, GpuOffloadOptions options,
        StreamingTensorPool? pool, IGpuOffloadAllocator? allocator)
    {
        var pending = new List<PendingRegistration<T>>(weights.Count);
        try
        {
            foreach (Tensor<T> weight in weights)
            {
                if (lifetime == WeightLifetime.Streaming)
                {
                    if (pool is null) throw new InvalidOperationException("No streaming pool was staged.");
                    var (encoding, stochastic) = ResolveStoreEncoding(weight, options);
                    byte[] bytes;
                    if (encoding == StreamingEncoding.Lossless) bytes = SerializeLossless(weight);
                    else
                    {
                        int byteCount = encoding switch
                        {
                            StreamingEncoding.Bf16 => CheckedBf16ByteCount(weight.Length),
                            StreamingEncoding.Int8 => CheckedInt8ByteCount(weight.Length, weight.Int8QuantRows),
                            StreamingEncoding.Int4 => CheckedInt4ByteCount(weight.Length),
                            _ => CheckedStreamingByteCount<T>(weight.Length),
                        };
                        bytes = new byte[byteCount];
                        SerializeToBytes(weight, bytes, encoding, stochastic);
                    }
                    var owner = new WeakReference<IStreamingDroppable>(weight);
                    long handle = pool.RegisterReserved(bytes, weight.StreamingReservedBytes);
                    pending.Add(new PendingRegistration<T>(weight, handle, encoding, default, weight.StreamingReservedBytes));
                    _ownerByHandle.Add(handle, owner);
                }
                else
                {
                    if (allocator is null) throw new InvalidOperationException("No GPU allocator was staged.");
                    int byteCount = CheckedStreamingByteCount<T>(weight.Length);
                    var bytes = new byte[byteCount];
                    SerializeToBytes(weight, bytes);
                    GpuOffloadHandle allocation = allocator.Allocate(byteCount, OffloadScheme.Pinned);
                    long handle = _nextOffloadRegistrationHandle++;
                    pending.Add(new PendingRegistration<T>(weight, handle, StreamingEncoding.Native, allocation, 0));
                    if (allocation.HostPointer == IntPtr.Zero || allocation.Bytes < byteCount)
                        throw new InvalidOperationException("The GPU allocator returned an invalid host allocation.");
                    Marshal.Copy(bytes, 0, allocation.HostPointer, byteCount);
                    _offloadRegistrations.Add(handle, new OffloadRegistration(weight, allocator, allocation));
                }
            }
            return pending;
        }
        catch
        {
            foreach (PendingRegistration<T> item in pending)
            {
                if (lifetime == WeightLifetime.Streaming)
                {
                    _ownerByHandle.Remove(item.Handle);
                    pool?.UnregisterAndRestoreReservation(item.Handle, item.ReservedBytes);
                }
                else
                {
                    _offloadRegistrations.Remove(item.Handle);
                    try { allocator?.Free(item.Offload); }
                    catch (Exception cleanupError) { System.Diagnostics.Trace.TraceWarning("GPU batch rollback cleanup failed: {0}", cleanupError); }
                }
            }
            throw;
        }
    }

    private static void PublishBatch<T>(List<PendingRegistration<T>> pending, WeightLifetime lifetime)
    {
        foreach (PendingRegistration<T> item in pending)
        {
            Tensor<T> weight = item.Weight;
            weight.Lifetime = lifetime;
            if (lifetime == WeightLifetime.Streaming)
            {
                weight.StreamingStoreEncoding = item.Encoding;
                weight.StreamingReservedBytes = 0;
                weight.StreamingDropDeferred = true;
                weight.StreamingPoolHandle = item.Handle;
            }
            else
            {
                weight.OffloadHostPointer = item.Offload.HostPointer;
                weight.OffloadDevicePointer = item.Offload.DevicePointer;
                weight.OffloadOpaqueHandle = item.Offload.BackendOpaque;
                weight.OffloadByteCount = item.Offload.Bytes;
                weight.OffloadRegistryHandle = item.Handle;
            }
        }
    }

    private static void DisposeRetiredBackend(IDisposable? backend)
    {
        try { backend?.Dispose(); }
        catch (Exception cleanupError) { System.Diagnostics.Trace.TraceWarning("Retired weight backend cleanup failed: {0}", cleanupError); }
    }

    private sealed class WeightReferenceComparer<T> : IEqualityComparer<Tensor<T>>
    {
        internal static readonly WeightReferenceComparer<T> Instance = new();
        public bool Equals(Tensor<T>? x, Tensor<T>? y) => ReferenceEquals(x, y);
        public int GetHashCode(Tensor<T> value) => RuntimeHelpers.GetHashCode(value);
    }
}
