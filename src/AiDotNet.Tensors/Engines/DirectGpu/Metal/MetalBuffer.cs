// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Metal GPU buffer implementation for Apple Silicon.

using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Metal GPU buffer implementation of IGpuBuffer.
/// </summary>
/// <remarks>
/// <para><b>Unified Memory Architecture:</b></para>
/// <para>
/// On Apple Silicon, Metal buffers use unified memory that is shared between
/// CPU and GPU. This means:
/// </para>
/// <list type="bullet">
/// <item>No explicit copy operations needed between CPU and GPU</item>
/// <item>The Contents property returns a direct pointer to the shared memory</item>
/// <item>Synchronization is automatic for StorageModeShared buffers</item>
/// </list>
/// <para><b>Storage Modes:</b></para>
/// <list type="bullet">
/// <item><b>Shared</b>: CPU and GPU can both access. Best for Apple Silicon.</item>
/// <item><b>Private</b>: GPU-only access. Best for intermediate results.</item>
/// <item><b>Managed</b>: Metal manages synchronization (macOS only).</item>
/// </list>
/// </remarks>
public sealed class MetalGpuBuffer : IGpuBuffer, IPoolableGpuBuffer
{
    private const long MinimumReclaimCharge = 128 * 1024;

    /// <summary>AIDOTNET_METAL_BUFFER_POOL=0 disables Metal buffer pooling.</summary>
    internal static bool PoolingEnabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_METAL_BUFFER_POOL") != "0";

    private static long s_driverAllocations;

    /// <summary>Buffers allocated from the Metal driver (pool misses) since the process started.</summary>
    internal static long DriverAllocations => System.Threading.Interlocked.Read(ref s_driverAllocations);

    // MetalDevice reports a failed allocation by throwing rather than returning zero. The first attempt's failure becomes
    // the zero that starts the reclaim-and-retry (ReclaimForRetry drains the pool and collects unreachable buffers); the
    // retry calls the device directly, so a second failure reaches the caller with the device's own exception.
    private static IntPtr FirstAttempt(Func<IntPtr> allocate)
    {
        try { return allocate(); }
        catch (InvalidOperationException ex) when (ex is not ObjectDisposedException) { return IntPtr.Zero; }
    }

    // The count is process-wide, so it is kept by a static method rather than written from each constructor.
    private static void CountDriverAllocation() => System.Threading.Interlocked.Increment(ref s_driverAllocations);

    private readonly MetalDevice? _device;
    private IntPtr _buffer;
    private bool _disposed;
    private readonly bool _pooled;
    private readonly object _lock = new();
    private readonly long _byteOffset;
    // A view keeps its parent reachable, so the parent's finalizer cannot release memory the view still addresses.
    private readonly MetalGpuBuffer? _viewParent;
    // The buffer that owns the MTLBuffer: this one, or the root of a view chain.
    private MetalGpuBuffer Owner => _viewParent is null ? this : _viewParent.Owner;
    // Live views of this owner's MTLBuffer, and device copies queued to read it and not yet completed. While either is
    // non-zero the allocation is still addressed after the owner is disposed, so it must not go back to the pool, where
    // the next rent would clear or overwrite it. Released instead, it stays alive through the view's own retain or the
    // command buffer's retained reference until that last use ends.
    private int _liveViews;
    private int _deviceReadsInFlight;

    private bool IsSharedWithDevice =>
        System.Threading.Volatile.Read(ref _liveViews) != 0 || System.Threading.Volatile.Read(ref _deviceReadsInFlight) != 0;

    /// <summary>Marks a queued device copy that reads this buffer; pair with <see cref="EndDeviceRead"/>.</summary>
    internal void BeginDeviceRead() => System.Threading.Interlocked.Increment(ref Owner._deviceReadsInFlight);

    /// <summary>Marks a device copy from <see cref="BeginDeviceRead"/> as completed.</summary>
    internal void EndDeviceRead() => System.Threading.Interlocked.Decrement(ref Owner._deviceReadsInFlight);
    private DeviceMemoryHold _memory; // what GpuMemoryStats counts for this memory (nothing for a view)
    private long _reclaimCharge;
    private long _reclaimEpoch = -1;

    /// <inheritdoc/>
    int IPoolableGpuBuffer.Capacity => Size;

    /// <inheritdoc/>
    void IPoolableGpuBuffer.MarkRented(int size)
    {
        // The pool is exact-size (MetalDevice.BufferPool), so a rent is always for this buffer's whole capacity and nothing
        // about the wrapper changes; anything else would hand out a wrong-sized buffer.
        if (size != Size)
            throw new InvalidOperationException(
                $"The exact-size Metal pool rented a {Size}-element buffer for a {size}-element request.");
    }

    /// <inheritdoc/>
    void IPoolableGpuBuffer.Release()
    {
        _memory.OnFreed();
        if (_buffer != IntPtr.Zero)
        {
            Release(_buffer);
            _buffer = IntPtr.Zero;
        }
    }
    /// <summary>
    /// Gets the native Metal buffer handle.
    /// </summary>
    public IntPtr Handle => _buffer;

    /// <summary>Byte offset of this buffer's first element in <see cref="Handle"/>: zero except for a view.</summary>
    public long ByteOffset => _byteOffset;

    /// <summary>True when this buffer is a view into another buffer's memory (see <see cref="IGpuBufferViews"/>).</summary>
    internal bool IsView => _viewParent is not null;

    /// <summary>
    /// Gets the number of float elements in the buffer.
    /// </summary>
    public int Size { get; }

    /// <summary>
    /// Gets the total size in bytes.
    /// </summary>
    public long SizeInBytes => (long)Size * sizeof(float);

    /// <summary>
    /// Gets whether the buffer is disposed.
    /// </summary>
    public bool IsDisposed => _disposed;

    /// <summary>Gets the Metal device that allocated this buffer, when known.</summary>
    internal MetalDevice? OwningDevice => _device;

    /// <summary>
    /// Gets the storage mode of the buffer.
    /// </summary>
    public MTLResourceOptions StorageMode { get; }

    /// <summary>
    /// Gets a pointer to the buffer contents (for shared/managed buffers).
    /// </summary>
    /// <remarks>
    /// On Apple Silicon with shared storage mode, this provides direct access
    /// to the unified memory, allowing zero-copy data access from CPU.
    /// </remarks>
    public IntPtr Contents
    {
        get
        {
            ThrowIfDisposed();
            if (_buffer == IntPtr.Zero)
            {
                return IntPtr.Zero;
            }
            IntPtr contents = SendMessage(_buffer, Selectors.Contents);
            return contents == IntPtr.Zero ? IntPtr.Zero : IntPtr.Add(contents, checked((int)_byteOffset));
        }
    }

    /// <summary>
    /// Creates a new Metal buffer with the specified size.
    /// </summary>
    /// <param name="device">The Metal device.</param>
    /// <param name="elementCount">Number of float elements.</param>
    /// <param name="options">Resource options for the buffer.</param>
    internal MetalGpuBuffer(MetalDevice device, int elementCount, MTLResourceOptions options = MTLResourceOptions.StorageModeShared)
    {
        if (device is null || !device.IsAvailable)
        {
            throw new ArgumentException("Invalid Metal device", nameof(device));
        }

        if (elementCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Element count must be positive");
        }

        Size = elementCount;
        _device = device;
        StorageMode = options;

        var sizeInBytes = (ulong)elementCount * sizeof(float);
        if (!PendingFrees.IsEmpty) PendingFrees.Drain();
        _buffer = FirstAttempt(() => device.CreateBuffer(sizeInBytes, options));
        if (_buffer == IntPtr.Zero)
        {
            // Out of memory: free what unreachable and pooled buffers hold, then retry once (see DeviceMemoryReclaim).
            ReclaimForRetry(device);
            _buffer = device.CreateBuffer(sizeInBytes, options);
        }

        if (_buffer == IntPtr.Zero)
        {
            throw new OutOfMemoryException($"Failed to allocate Metal buffer of {sizeInBytes} bytes");
        }
        CountDriverAllocation();
        _pooled = options == MTLResourceOptions.StorageModeShared;
        _memory.OnAllocated(device.MemoryAccount, (long)sizeInBytes);
    }

    private MetalGpuBuffer(MetalDevice device, IntPtr buffer, int elementCount, MTLResourceOptions options)
    {
        _device = device;
        _buffer = buffer;
        Size = elementCount;
        StorageMode = options;
        _pooled = options == MTLResourceOptions.StorageModeShared;
    }

    /// <summary>
    /// Rents a pooled StorageModeShared Metal buffer when available, or allocates a new one.
    /// </summary>
    internal static MetalGpuBuffer Create(MetalDevice device, int elementCount, bool zeroInitialize = true)
    {
        if (device is null || !device.IsAvailable)
            throw new ArgumentException("Invalid Metal device", nameof(device));
        if (elementCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Element count must be positive");

        if (!PendingFrees.IsEmpty) PendingFrees.Drain();
        long charge = Math.Max((long)elementCount * sizeof(float), MinimumReclaimCharge);
        long reclaimEpoch = DeviceMemoryReclaim.OnDeviceAllocation(charge);

        MetalGpuBuffer buffer;
        if (PoolingEnabled && device.BufferPool.TryRent(elementCount, out var parked) && parked is not null)
        {
            buffer = FromParked(device, parked, elementCount);
            if (zeroInitialize) buffer.Clear();
        }
        else if (PoolingEnabled && DeviceMemoryReclaim.CollectOnPoolMiss() &&
                 PendingFreesDrainAndTryRent(device, elementCount, out parked) && parked is not null)
        {
            buffer = FromParked(device, parked, elementCount);
            if (zeroInitialize) buffer.Clear();
        }
        else
        {
            // MTLDevice newBufferWithLength: guarantees zero-filled memory on fresh driver allocations.
            buffer = new MetalGpuBuffer(device, elementCount, MTLResourceOptions.StorageModeShared);
        }

        buffer._reclaimCharge = charge;
        buffer._reclaimEpoch = reclaimEpoch;
        return buffer;
    }

    /// <summary>
    /// Rents a pooled StorageModeShared Metal buffer and copies <paramref name="data"/> into unified memory,
    /// or allocates a new buffer initialized with <paramref name="data"/>.
    /// </summary>
    internal static MetalGpuBuffer Create(MetalDevice device, float[] data)
    {
        if (device is null || !device.IsAvailable)
            throw new ArgumentException("Invalid Metal device", nameof(device));
        if (data is null || data.Length == 0)
            throw new ArgumentException("Data cannot be null or empty", nameof(data));

        if (!PendingFrees.IsEmpty) PendingFrees.Drain();
        long charge = Math.Max((long)data.Length * sizeof(float), MinimumReclaimCharge);
        long reclaimEpoch = DeviceMemoryReclaim.OnDeviceAllocation(charge);

        MetalGpuBuffer buffer;
        if (PoolingEnabled && device.BufferPool.TryRent(data.Length, out var parked) && parked is not null)
        {
            buffer = FromParked(device, parked, data.Length);
            buffer.CopyFrom(data);
        }
        else if (PoolingEnabled && DeviceMemoryReclaim.CollectOnPoolMiss() &&
                 PendingFreesDrainAndTryRent(device, data.Length, out parked) && parked is not null)
        {
            buffer = FromParked(device, parked, data.Length);
            buffer.CopyFrom(data);
        }
        else
        {
            buffer = new MetalGpuBuffer(device, data, MTLResourceOptions.StorageModeShared);
        }

        buffer._reclaimCharge = charge;
        buffer._reclaimEpoch = reclaimEpoch;
        return buffer;
    }

    private static bool PendingFreesDrainAndTryRent(MetalDevice device, int elementCount, out MetalGpuBuffer? parked)
    {
        PendingFrees.Drain();
        return device.BufferPool.TryRent(elementCount, out parked);
    }

    private static MetalGpuBuffer FromParked(MetalDevice device, MetalGpuBuffer parked, int size)
    {
        var rented = new MetalGpuBuffer(device, parked._buffer, size, parked.StorageMode);
        parked._buffer = IntPtr.Zero;
        rented._memory = parked._memory.TransferToLease();
        rented._memory.OnRented();
        return rented;
    }

    /// <summary>
    /// Creates a new Metal buffer with initial data.
    /// </summary>
    /// <param name="device">The Metal device.</param>
    /// <param name="data">Initial data to copy to the buffer.</param>
    /// <param name="options">Resource options for the buffer.</param>
    internal MetalGpuBuffer(MetalDevice device, float[] data, MTLResourceOptions options = MTLResourceOptions.StorageModeShared)
    {
        if (device is null || !device.IsAvailable)
        {
            throw new ArgumentException("Invalid Metal device", nameof(device));
        }

        if (data is null || data.Length == 0)
        {
            throw new ArgumentException("Data cannot be null or empty", nameof(data));
        }

        Size = data.Length;
        _device = device;
        StorageMode = options;

        var sizeInBytes = (ulong)data.Length * sizeof(float);
        if (!PendingFrees.IsEmpty) PendingFrees.Drain();
        var handle = GCHandle.Alloc(data, GCHandleType.Pinned);

        try
        {
            IntPtr source = handle.AddrOfPinnedObject();
            _buffer = FirstAttempt(() => device.CreateBufferWithData(source, sizeInBytes, options));
            if (_buffer == IntPtr.Zero)
            {
                // Out of memory: free what unreachable and pooled buffers hold, then retry once (see DeviceMemoryReclaim).
                ReclaimForRetry(device);
                _buffer = device.CreateBufferWithData(handle.AddrOfPinnedObject(), sizeInBytes, options);
            }
        }
        finally
        {
            handle.Free();
        }

        if (_buffer == IntPtr.Zero)
        {
            throw new OutOfMemoryException($"Failed to allocate Metal buffer of {sizeInBytes} bytes with data");
        }
        CountDriverAllocation();
        _pooled = options == MTLResourceOptions.StorageModeShared;
        _memory.OnAllocated(device.MemoryAccount, (long)sizeInBytes);
    }

    /// <summary>
    /// A view of <paramref name="elementCount"/> floats starting <paramref name="elementOffset"/> floats into
    /// <paramref name="parent"/>: the same MTLBuffer bound at a byte offset. The view retains the MTLBuffer, so its
    /// own release is balanced and the memory stays valid while either object is alive.
    /// </summary>
    internal MetalGpuBuffer(MetalGpuBuffer parent, int elementOffset, int elementCount)
    {
        if (parent is null) throw new ArgumentNullException(nameof(parent));
        if (elementOffset < 0 || elementCount <= 0 || (long)elementOffset + elementCount > parent.Size)
            throw new ArgumentOutOfRangeException(nameof(elementOffset));
        if (parent.Handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(MetalGpuBuffer));
        _buffer = parent.Handle;
        _device = parent._device;
        _viewParent = parent;
        _byteOffset = parent._byteOffset + (long)elementOffset * sizeof(float);
        Size = elementCount;
        StorageMode = parent.StorageMode;
        Retain(_buffer);
        System.Threading.Interlocked.Increment(ref Owner._liveViews);
    }
    /// <summary>
    /// Creates a wrapper around an existing Metal buffer handle.
    /// </summary>
    /// <param name="handle">The native buffer handle.</param>
    /// <param name="elementCount">Number of float elements.</param>
    /// <param name="options">Resource options of the buffer.</param>
    internal MetalGpuBuffer(IntPtr handle, int elementCount, MTLResourceOptions options = MTLResourceOptions.StorageModeShared)
    {
        if (handle == IntPtr.Zero)
        {
            throw new ArgumentException("Invalid buffer handle", nameof(handle));
        }

        if (elementCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Element count must be positive");
        }

        _buffer = handle;
        Size = elementCount;
        StorageMode = options;

        // Retain the buffer since we're taking ownership
        Retain(_buffer);
    }

    /// <summary>
    /// Copies data from the GPU buffer to a CPU array.
    /// </summary>
    /// <param name="destination">Destination array.</param>
    /// <param name="offset">Offset in the buffer (in elements).</param>
    /// <param name="count">Number of elements to copy.</param>
    public void CopyTo(float[] destination, int offset = 0, int count = -1)
    {
        ThrowIfDisposed();

        if (destination is null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (count < 0)
        {
            count = Size - offset;
        }

        if (offset < 0 || offset >= Size)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        if (count <= 0 || offset + count > Size || count > destination.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        var contents = Contents;
        if (contents == IntPtr.Zero)
        {
            throw new InvalidOperationException("Buffer contents are not accessible (possibly private storage mode)");
        }

        // Copy from unified memory to the destination array
        var sourcePtr = IntPtr.Add(contents, offset * sizeof(float));
        Marshal.Copy(sourcePtr, destination, 0, count);
    }

    /// <summary>
    /// Copies data from a CPU array to the GPU buffer.
    /// </summary>
    /// <param name="source">Source array.</param>
    /// <param name="offset">Offset in the buffer (in elements).</param>
    /// <param name="count">Number of elements to copy.</param>
    public void CopyFrom(float[] source, int offset = 0, int count = -1)
    {
        ThrowIfDisposed();

        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (count < 0)
        {
            count = Math.Min(source.Length, Size - offset);
        }

        if (offset < 0 || offset >= Size)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        if (count <= 0 || offset + count > Size || count > source.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        var contents = Contents;
        if (contents == IntPtr.Zero)
        {
            throw new InvalidOperationException("Buffer contents are not accessible (possibly private storage mode)");
        }

        // Copy from source array to unified memory
        var destPtr = IntPtr.Add(contents, offset * sizeof(float));
        Marshal.Copy(source, 0, destPtr, count);

        // For managed storage mode, notify Metal that the range was modified
        if ((StorageMode & MTLResourceOptions.StorageModeManaged) == MTLResourceOptions.StorageModeManaged)
        {
            NotifyModified(offset, count);
        }
    }

    /// <summary>
    /// Copies raw bytes from the GPU buffer to a CPU array.
    /// </summary>
    public void CopyBytesTo(byte[] destination, int offsetInBytes = 0, int count = -1)
    {
        ThrowIfDisposed();

        if (destination is null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (count < 0)
        {
            count = Math.Min(destination.Length, checked((int)(SizeInBytes - offsetInBytes)));
        }

        if (offsetInBytes < 0 || offsetInBytes >= SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(offsetInBytes));
        }

        // (long) cast: offsetInBytes + count are both int, so a large offset+count would overflow to a
        // negative int and slip past this bound before reaching Marshal.Copy. Evaluate in long.
        if (count < 0 || (long)offsetInBytes + count > SizeInBytes || count > destination.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        if (count == 0)
        {
            return;
        }

        var contents = Contents;
        if (contents == IntPtr.Zero)
        {
            throw new InvalidOperationException("Buffer contents are not accessible");
        }

        Marshal.Copy(IntPtr.Add(contents, offsetInBytes), destination, 0, count);
    }

    /// <summary>
    /// Copies raw bytes from a CPU array to the GPU buffer.
    /// </summary>
    public void CopyBytesFrom(byte[] source, int offsetInBytes = 0, int count = -1)
    {
        ThrowIfDisposed();

        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (count < 0)
        {
            count = Math.Min(source.Length, checked((int)(SizeInBytes - offsetInBytes)));
        }

        if (offsetInBytes < 0 || offsetInBytes >= SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(offsetInBytes));
        }

        // (long) cast: see CopyBytesTo — int offsetInBytes + count can overflow past this bound.
        if (count < 0 || (long)offsetInBytes + count > SizeInBytes || count > source.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        if (count == 0)
        {
            return;
        }

        var contents = Contents;
        if (contents == IntPtr.Zero)
        {
            throw new InvalidOperationException("Buffer contents are not accessible");
        }

        Marshal.Copy(source, 0, IntPtr.Add(contents, offsetInBytes), count);

        if ((StorageMode & MTLResourceOptions.StorageModeManaged) == MTLResourceOptions.StorageModeManaged)
        {
            int offsetElements = offsetInBytes / sizeof(float);
            int countElements = (count + sizeof(float) - 1) / sizeof(float);
            NotifyModified(offsetElements, countElements);
        }
    }

    /// <summary>
    /// Copies data from a pointer to the GPU buffer.
    /// </summary>
    /// <param name="source">Source pointer.</param>
    /// <param name="sizeInBytes">Number of bytes to copy.</param>
    /// <param name="offsetInBytes">Offset in the buffer (in bytes).</param>
    public unsafe void CopyFromPointer(IntPtr source, int sizeInBytes, int offsetInBytes = 0)
    {
        ThrowIfDisposed();

        if (source == IntPtr.Zero)
        {
            throw new ArgumentException("Source pointer cannot be null", nameof(source));
        }

        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        if (offsetInBytes < 0 || offsetInBytes + sizeInBytes > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(offsetInBytes));
        }

        var contents = Contents;
        if (contents == IntPtr.Zero)
        {
            throw new InvalidOperationException("Buffer contents are not accessible");
        }

        var destPtr = IntPtr.Add(contents, offsetInBytes);
        Buffer.MemoryCopy(source.ToPointer(), destPtr.ToPointer(), SizeInBytes - offsetInBytes, sizeInBytes);

        // For managed storage mode, notify Metal that the range was modified
        if ((StorageMode & MTLResourceOptions.StorageModeManaged) == MTLResourceOptions.StorageModeManaged)
        {
            int offsetElements = offsetInBytes / sizeof(float);
            int countElements = (sizeInBytes + sizeof(float) - 1) / sizeof(float);
            NotifyModified(offsetElements, countElements);
        }
    }

    /// <summary>
    /// Gets a span view of the buffer contents.
    /// </summary>
    /// <returns>A span representing the buffer data.</returns>
    public unsafe Span<float> AsSpan()
    {
        ThrowIfDisposed();

        var contents = Contents;
        if (contents == IntPtr.Zero)
        {
            throw new InvalidOperationException("Buffer contents are not accessible");
        }

        return new Span<float>(contents.ToPointer(), Size);
    }

    /// <summary>
    /// Gets a read-only span view of the buffer contents.
    /// </summary>
    /// <returns>A read-only span representing the buffer data.</returns>
    public unsafe ReadOnlySpan<float> AsReadOnlySpan()
    {
        ThrowIfDisposed();

        var contents = Contents;
        if (contents == IntPtr.Zero)
        {
            throw new InvalidOperationException("Buffer contents are not accessible");
        }

        return new ReadOnlySpan<float>(contents.ToPointer(), Size);
    }

    /// <summary>
    /// Fills the buffer with a constant value.
    /// </summary>
    /// <param name="value">Value to fill with.</param>
    public void Fill(float value)
    {
        ThrowIfDisposed();

        var contents = Contents;
        if (contents == IntPtr.Zero)
        {
            throw new InvalidOperationException("Buffer contents are not accessible");
        }

        // Use the span to fill efficiently
        var span = AsSpan();
        span.Fill(value);

        // Notify modification for managed storage
        if ((StorageMode & MTLResourceOptions.StorageModeManaged) == MTLResourceOptions.StorageModeManaged)
        {
            NotifyModified(0, Size);
        }
    }

    /// <summary>
    /// Clears the buffer to zero.
    /// </summary>
    public void Clear()
    {
        Fill(0f);
    }

    /// <summary>
    /// Notifies Metal that a range of the buffer was modified (for managed storage mode).
    /// </summary>
    /// <param name="offset">Offset in elements.</param>
    /// <param name="count">Number of elements modified.</param>
    public void NotifyModified(int offset, int count)
    {
        if (_buffer == IntPtr.Zero)
        {
            return;
        }

        // Call [MTLBuffer didModifyRange:NSMakeRange(offset, count)]
        var range = new MetalNativeBindings.NSRange(
            (ulong)(_byteOffset + (long)offset * sizeof(float)),
            (ulong)(count * sizeof(float)));
        MetalNativeBindings.SendMessageNSRange(
            _buffer, MetalNativeBindings.Selectors.DidModifyRange, range);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MetalGpuBuffer));
        }
    }

    /// <summary>
    /// Disposes the Metal buffer.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        IntPtr bufferToPark = IntPtr.Zero;
        DeviceMemoryHold memoryLease = default;
        MetalDevice? poolDevice = null;

        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            if (_viewParent is not null) System.Threading.Interlocked.Decrement(ref Owner._liveViews);
            long charge = System.Threading.Interlocked.Exchange(ref _reclaimCharge, 0);
            if (charge > 0) DeviceMemoryReclaim.OnDeviceRelease(charge, _reclaimEpoch);

            if (_pooled && PoolingEnabled && _viewParent is null && _buffer != IntPtr.Zero && !IsSharedWithDevice &&
                _device is { IsDisposed: false } dev)
            {
                _memory.OnReturned();
                memoryLease = _memory.TransferToLease();
                bufferToPark = _buffer;
                _buffer = IntPtr.Zero;
                poolDevice = dev;
            }
            else
            {
                _memory.OnFreed();
                if (_buffer != IntPtr.Zero)
                {
                    Release(_buffer);
                    _buffer = IntPtr.Zero;
                }
            }
        }

        GC.SuppressFinalize(this);

        if (poolDevice is not null && bufferToPark != IntPtr.Zero)
        {
            var parked = new MetalGpuBuffer(poolDevice, bufferToPark, Size, StorageMode)
            {
                _disposed = true,
                _memory = memoryLease
            };
            GC.SuppressFinalize(parked);
            poolDevice.BufferPool.Return(parked);
        }
    }

    /// <summary>Frees of Metal buffers collected without Dispose, run by the next allocation.</summary>
    internal static readonly DeviceFreeQueue PendingFrees = new();

    private static void ReclaimForRetry(MetalDevice? device)
    {
        DeviceMemoryReclaim.CollectUnreachable();
        PendingFrees.Drain();
        device?.BufferPool.DrainAll();
    }

    // An undisposed buffer (a result nothing references any more) returns to the pool or releases through the queue.
    ~MetalGpuBuffer()
    {
        IntPtr buffer;
        DeviceMemoryHold memoryLease = default;
        MetalDevice? poolDevice = null;
        int size = Size;
        MTLResourceOptions storageMode = StorageMode;

        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            if (_viewParent is not null) System.Threading.Interlocked.Decrement(ref Owner._liveViews);
            long charge = System.Threading.Interlocked.Exchange(ref _reclaimCharge, 0);
            DeviceMemoryReclaim.OnDeviceFinalized(charge, _reclaimEpoch);
            buffer = _buffer;
            _buffer = IntPtr.Zero;

            if (_pooled && PoolingEnabled && _viewParent is null && buffer != IntPtr.Zero && !IsSharedWithDevice &&
                _device is { IsDisposed: false } dev)
            {
                memoryLease = _memory.TransferToLease();
                poolDevice = dev;
            }
            else
            {
                _memory.OnFreed();
            }
        }

        if (buffer == IntPtr.Zero) return;
        if (poolDevice is not null)
        {
            PendingFrees.Enqueue(() =>
            {
                if (poolDevice.IsDisposed || !PoolingEnabled)
                {
                    memoryLease.OnFreed();
                    Release(buffer);
                    return;
                }
                memoryLease.OnReturned();
                var parked = new MetalGpuBuffer(poolDevice, buffer, size, storageMode)
                {
                    _disposed = true,
                    _memory = memoryLease
                };
                GC.SuppressFinalize(parked);
                poolDevice.BufferPool.Return(parked);
            });
        }
        else
        {
            PendingFrees.Enqueue(() => Release(buffer));
        }
    }

    public override string ToString()
    {
        return $"MetalGpuBuffer[{Size} floats, {SizeInBytes / 1024.0:F1} KB, {StorageMode}]";
    }
}
