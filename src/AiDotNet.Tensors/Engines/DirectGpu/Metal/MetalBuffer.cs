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
public sealed class MetalGpuBuffer : IGpuBuffer
{
    private readonly MetalDevice? _device;
    private IntPtr _buffer;
    private bool _disposed;
    private readonly object _lock = new();
    private readonly long _byteOffset;
    // A view keeps its parent reachable, so the parent's finalizer cannot release memory the view still addresses.
    private readonly MetalGpuBuffer? _viewParent;
    private DeviceMemoryHold _memory; // what GpuMemoryStats counts for this memory (nothing for a view)

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
        _buffer = device.CreateBuffer(sizeInBytes, options);
        if (_buffer == IntPtr.Zero)
        {
            // Out of memory: free what unreachable buffers hold, then retry once (see DeviceMemoryReclaim).
            ReclaimForRetry();
            _buffer = device.CreateBuffer(sizeInBytes, options);
        }

        if (_buffer == IntPtr.Zero)
        {
            throw new OutOfMemoryException($"Failed to allocate Metal buffer of {sizeInBytes} bytes");
        }
        _memory.OnAllocated(device.MemoryAccount, (long)sizeInBytes);
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
            _buffer = device.CreateBufferWithData(handle.AddrOfPinnedObject(), sizeInBytes, options);
            if (_buffer == IntPtr.Zero)
            {
                // Out of memory: free what unreachable buffers hold, then retry once (see DeviceMemoryReclaim).
                ReclaimForRetry();
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

        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _memory.OnFreed();

            if (_buffer != IntPtr.Zero)
            {
                Release(_buffer);
                _buffer = IntPtr.Zero;
            }
        }
        GC.SuppressFinalize(this);
    }

    /// <summary>Frees of Metal buffers collected without Dispose, run by the next allocation.</summary>
    internal static readonly DeviceFreeQueue PendingFrees = new();

    private static void ReclaimForRetry()
    {
        DeviceMemoryReclaim.CollectUnreachable();
        PendingFrees.Drain();
    }

    // An undisposed buffer (a result nothing references any more) releases its MTLBuffer through the queue.
    ~MetalGpuBuffer()
    {
        IntPtr buffer;
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            _memory.OnFreed();
            buffer = _buffer;
            _buffer = IntPtr.Zero;
        }
        if (buffer != IntPtr.Zero) PendingFrees.Enqueue(() => Release(buffer));
    }

    public override string ToString()
    {
        return $"MetalGpuBuffer[{Size} floats, {SizeInBytes / 1024.0:F1} KB, {StorageMode}]";
    }
}
