// Copyright (c) AiDotNet. All rights reserved.
// IGpuBuffer wrapper for Vulkan storage and staging buffer pairs.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Wraps a Vulkan device-local storage buffer and host-visible staging buffer
/// as an <see cref="IGpuBuffer"/> for use with <see cref="IDirectGpuBackend"/> operations.
/// </summary>
public sealed class VulkanGpuBuffer : IGpuBuffer, IPoolableGpuBuffer
{
    internal readonly VulkanBuffer Storage;
    internal readonly VulkanBuffer Staging;
    private int _disposed;
    private readonly VulkanGpuBuffer? _viewParent;

    // Creating and destroying a storage + staging pair costs about 0.7 ms for a small buffer and several
    // milliseconds for a large one, many times the ~80 microseconds a dispatch takes, so a released buffer is kept
    // for the next allocation of the same size (as the CUDA, HIP and OpenCL backends do). Only an exact size match is
    // reused: a Vulkan buffer's logical size is its allocation size everywhere a transfer or a kernel reads it.
    // Every dispatch is submitted and waited for, so a released buffer has no work in flight.
    // As many per size as OpenCL keeps per bucket: a loop that never disposes its results gets them all back at once,
    // when a collection finalizes them. The memory budget below is the real bound.
    private const int MaxPooledBuffersPerBucket = 4096;
    private const int PooledMemoryDivisor = 4;
    private const long MinimumReclaimCharge = 128 * 1024;

    /// <summary>Released buffers kept for reuse. AIDOTNET_VULKAN_BUFFER_POOL=0 disables pooling.</summary>
    internal static readonly GpuBufferPool<VulkanGpuBuffer> Pool =
        new GpuBufferPool<VulkanGpuBuffer>(MaxPooledBuffersPerBucket, int.MaxValue, exactSizes: true);

    internal static bool PoolingEnabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_VULKAN_BUFFER_POOL") != "0";

    private static int s_poolBudgetSet;
    private static long s_driverAllocations;

    /// <summary>Buffers created through the driver (pool misses) since the process started.</summary>
    internal static long DriverAllocations => System.Threading.Interlocked.Read(ref s_driverAllocations);

    /// <inheritdoc/>
    int IPoolableGpuBuffer.Capacity => Size;

    /// <inheritdoc/>
    /// <remarks>
    /// Nothing to do: a parked object only carries its allocation, and <see cref="Create"/> hands that allocation
    /// out under a new object (see there).
    /// </remarks>
    void IPoolableGpuBuffer.MarkRented(int size)
    {
    }

    /// <inheritdoc/>
    void IPoolableGpuBuffer.Release()
    {
        Storage.Dispose();
        Staging.Dispose();
    }

    // The pool may park at most a quarter of the device's memory (each parked buffer also holds a host-visible
    // staging buffer of the same size).
    private static void EnsurePoolBudget()
    {
        if (System.Threading.Interlocked.Exchange(ref s_poolBudgetSet, 1) != 0) return;
        long deviceBytes = VulkanDevice.Instance.TotalDeviceLocalMemoryBytes;
        if (deviceBytes > 0) Pool.MaxPooledElements = deviceBytes / PooledMemoryDivisor / sizeof(float);
    }

    /// <summary>Frees of Vulkan buffers collected without Dispose, run by the next allocation.</summary>
    internal static readonly DeviceFreeQueue PendingFrees = new();

    /// <inheritdoc/>
    public int Size { get; }

    /// <inheritdoc/>
    /// <remarks>
    /// Returns the actual byte size of the underlying Vulkan buffer.
    /// For standard float buffers, this is Size * 4.
    /// For FP16 buffers created via ConvertToFp16, the element count remains
    /// the same but each element is 2 bytes instead of 4; callers should
    /// account for this at a higher level.
    /// </remarks>
    public long SizeInBytes => (long)Size * sizeof(float);

    /// <inheritdoc/>
    public IntPtr Handle => Storage.Handle;

    internal VulkanGpuBuffer(VulkanBuffer storage, VulkanBuffer staging, int size)
    {
        Storage = storage ?? throw new ArgumentNullException(nameof(storage));
        Staging = staging ?? throw new ArgumentNullException(nameof(staging));
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Element count must be positive.");
        Size = size;
    }

    /// <summary>A view wrapper: holds <paramref name="parent"/> so its finalizer cannot free memory the view still uses.</summary>
    internal VulkanGpuBuffer(VulkanBuffer storage, VulkanBuffer staging, int size, VulkanGpuBuffer parent)
        : this(storage, staging, size)
    {
        _viewParent = parent;
    }

    /// <summary>
    /// Creates a new VulkanGpuBuffer with the given element count.
    /// </summary>
    internal static VulkanGpuBuffer Create(int size)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Element count must be positive.");
        // Run the frees queued by collected buffers; on out-of-memory collect unreachable buffers, run their frees and
        // retry once (see DeviceMemoryReclaim).
        if (!PendingFrees.IsEmpty) PendingFrees.Drain();
        // Counted like the OpenCL and HIP allocators count theirs, pool hits included: a result nobody disposed comes
        // back only when the GC finalizes it, and a GPU loop allocates too little managed memory to collect often.
        // A small buffer is charged MinimumReclaimCharge: a driver allocation costs Vulkan about 0.7 ms however small it
        // is, so small results have to come back from the collector by count; charged by their bytes alone, sixteen
        // thousand 16 KB results would be created before the first collection returned any.
        DeviceMemoryReclaim.OnDeviceAllocation(Math.Max((long)size * sizeof(float), MinimumReclaimCharge));
        // A rented allocation goes out under a NEW object, never the parked one. A parked object that sat in the pool
        // through a collection has been promoted, and a result dropped while holding a promoted object is not
        // finalized by the young collections that return results to the pool: it came back only at the next older
        // collection, and the pool ran empty in between (measured: every other 2,000-call stretch of an engine loop
        // paid a driver allocation per call). A new object is young, so the next young collection returns it.
        if (PoolingEnabled && Pool.TryRent(size, out var pooled) &&
            pooled is not null)
            return new VulkanGpuBuffer(pooled.Storage, pooled.Staging, size);
        if (PoolingEnabled && DeviceMemoryReclaim.CollectOnPoolMiss())
        {
            PendingFrees.Drain();
            if (Pool.TryRent(size, out pooled) && pooled is not null)
                return new VulkanGpuBuffer(pooled.Storage, pooled.Staging, size);
        }
        System.Threading.Interlocked.Increment(ref s_driverAllocations);
        // Out of memory: the parked buffers are the first thing to give back.
        return DeviceMemoryReclaim.AllocateWithRetry(() => CreateCore(size), ex => ex is GpuOutOfMemoryException,
            () => { PendingFrees.Drain(); Pool.DrainAll(); });
    }

    private static VulkanGpuBuffer CreateCore(int size)
    {
        VulkanBuffer? storage = null;
        VulkanBuffer? staging = null;
        try
        {
            VulkanBuffer.LastAllocationResult = VulkanNativeBindings.VK_SUCCESS;
            storage = VulkanBuffer.CreateStorageBuffer(size);
            staging = storage is null ? null : VulkanBuffer.CreateStagingBuffer(size);
            if (storage is null || staging is null)
            {
                int vk = VulkanBuffer.LastAllocationResult;
                if (vk == VulkanNativeBindings.VK_ERROR_OUT_OF_DEVICE_MEMORY || vk == VulkanNativeBindings.VK_ERROR_OUT_OF_HOST_MEMORY)
                    throw new GpuOutOfMemoryException($"Failed to allocate Vulkan GPU buffer: {vk}", vk);
                throw new InvalidOperationException("Failed to allocate Vulkan GPU buffer.");
            }

            return new VulkanGpuBuffer(storage, staging, size);
        }
        catch
        {
            storage?.Dispose();
            staging?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates a new VulkanGpuBuffer and uploads initial data.
    /// </summary>
    /// <remarks>
    /// The upload is performed synchronously via a staging buffer copy.
    /// For large data sets, consider pre-allocating and uploading in batches.
    /// If upload fails or the staging buffer is not writable, the buffer is
    /// disposed to prevent resource leaks and an exception is thrown.
    /// </remarks>
    internal static VulkanGpuBuffer Create(float[] data, VulkanBufferTransfer transfer)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.Length == 0) throw new ArgumentException("Data array must not be empty.", nameof(data));
        if (transfer is null) throw new ArgumentNullException(nameof(transfer));

        var buffer = Create(data.Length);
        try
        {
            // WriteData returns silently if staging is not host-visible or disposed.
            // Since we just created the buffer, these conditions indicate a serious
            // allocation failure, so we verify the staging buffer is valid first.
            if (buffer.Staging.Handle == IntPtr.Zero)
                throw new InvalidOperationException("Staging buffer handle is invalid after allocation.");

            buffer.Staging.WriteData(data);
            transfer.CopyToDevice(buffer.Staging, buffer.Storage);
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
        return buffer;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
        if (PoolingEnabled)
        {
            EnsurePoolBudget();
            Pool.Return(this); // releases it when the pool is full
            return;
        }
        Storage.Dispose();
        Staging.Dispose();
    }

    // An undisposed buffer (a result nothing references any more) frees its device memory through the queue.
    ~VulkanGpuBuffer()
    {
        if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0) return;
        var storage = Storage;
        var staging = Staging;
        int size = Size;
        // This object is gone; its allocation goes back to the pool under a new wrapper, on the next allocation's
        // thread.
        PendingFrees.Enqueue(() =>
        {
            if (!PoolingEnabled) { storage.Dispose(); staging.Dispose(); return; }
            var parked = new VulkanGpuBuffer(storage, staging, size);
            System.Threading.Volatile.Write(ref parked._disposed, 1);
            GC.SuppressFinalize(parked);
            EnsurePoolBudget();
            Pool.Return(parked);
        });
    }
}
