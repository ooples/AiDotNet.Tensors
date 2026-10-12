// Copyright (c) AiDotNet. All rights reserved.
// WebGPU buffer implementation for browser GPU compute.
// Only available in .NET 7+ with Blazor WebAssembly.

#if NET7_0_OR_GREATER
using System;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// WebGPU buffer implementation for GPU compute operations in the browser.
/// </summary>
/// <remarks>
/// <para><b>WebGPU Buffer Model:</b></para>
/// <para>
/// WebGPU buffers are explicitly managed GPU memory with defined usage:
/// </para>
/// <list type="bullet">
/// <item><b>STORAGE</b>: For compute shader read/write</item>
/// <item><b>UNIFORM</b>: For constant data (parameters)</item>
/// <item><b>COPY_SRC/DST</b>: For data transfer operations</item>
/// <item><b>MAP_READ/WRITE</b>: For CPU-accessible staging buffers</item>
/// </list>
/// <para><b>Data Transfer Pattern:</b></para>
/// <para>
/// WebGPU uses explicit staging for CPU-GPU transfers:
/// 1. Create STORAGE buffer for compute operations
/// 2. Write data via queue.writeBuffer()
/// 3. Read data by copying to MAP_READ buffer, then mapping
/// </para>
/// </remarks>
public sealed class WebGpuBuffer : IGpuBuffer, IDisposable
{
    private readonly int _bufferId;
    private readonly int _sizeBytes;
    private readonly int _elementCount;
    private readonly WebGpuBufferUsage _usage;
    private int _disposedFlag;
    private readonly int _offsetBytes;
    // A view keeps its parent reachable, so the parent's finalizer cannot destroy memory the view still addresses.
    private readonly WebGpuBuffer? _viewParent;
    private bool _disposed => System.Threading.Volatile.Read(ref _disposedFlag) != 0;

    /// <summary>Frees of WebGPU buffers collected without Dispose, run by the next allocation.</summary>
    internal static readonly DeviceFreeQueue PendingFrees = new();

    /// <summary>
    /// Gets the buffer handle ID.
    /// </summary>
    public int BufferId => _bufferId;

    /// <summary>Byte offset of this buffer's first element in <see cref="BufferId"/>: zero except for a view.</summary>
    public int OffsetBytes => _offsetBytes;

    /// <summary>True when this buffer is a view into another buffer's memory (see <see cref="IGpuBufferViews"/>).</summary>
    internal bool IsView => _viewParent is not null;

    /// <summary>
    /// Gets the buffer size in bytes.
    /// </summary>
    public int SizeBytes => _sizeBytes;

    /// <summary>
    /// Gets the number of float elements in the buffer.
    /// </summary>
    public int ElementCount => _elementCount;

    /// <summary>
    /// Gets the buffer usage flags.
    /// </summary>
    public WebGpuBufferUsage Usage => _usage;

    /// <summary>
    /// Gets whether this buffer is valid for use.
    /// </summary>
    public bool IsValid => _bufferId >= 0 && !_disposed && (_viewParent is null || _viewParent.IsValid);

    /// <summary>
    /// Gets the number of elements in the buffer (IGpuBuffer implementation).
    /// </summary>
    public int Size => _elementCount;

    /// <summary>
    /// Gets the total size in bytes (IGpuBuffer implementation).
    /// </summary>
    public long SizeInBytes => _sizeBytes;

    /// <summary>
    /// Gets the native handle. For WebGPU, this returns IntPtr.Zero as handles are JS-side.
    /// The actual buffer ID is accessible via BufferId property.
    /// </summary>
    public IntPtr Handle => IntPtr.Zero;

    /// <summary>
    /// Creates a new WebGPU buffer.
    /// </summary>
    /// <param name="elementCount">Number of float elements.</param>
    /// <param name="usage">Buffer usage flags.</param>
    public WebGpuBuffer(int elementCount, WebGpuBufferUsage usage = WebGpuBufferUsage.Storage | WebGpuBufferUsage.CopySrc | WebGpuBufferUsage.CopyDst)
    {
        if (elementCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Element count must be positive");
        }

        _elementCount = elementCount;
        _sizeBytes = elementCount * sizeof(float);
        _usage = usage;

        // Ensure buffer size is aligned to 4 bytes (required by WebGPU)
        var alignedSize = (_sizeBytes + 3) & ~3;
        if (!PendingFrees.IsEmpty) PendingFrees.Drain();
        int bufferId = WebGpuNativeBindings.CreateBuffer(alignedSize, (int)usage);
        if (bufferId < 0)
        {
            // The binding does not say why creation failed; free what unreachable buffers hold and retry once
            // (see DeviceMemoryReclaim) before reporting the failure.
            DeviceMemoryReclaim.CollectUnreachable();
            PendingFrees.Drain();
            bufferId = WebGpuNativeBindings.CreateBuffer(alignedSize, (int)usage);
        }
        _bufferId = bufferId;

        if (_bufferId < 0)
        {
            throw new InvalidOperationException("Failed to create WebGPU buffer");
        }
    }

    /// <summary>
    /// A view of <paramref name="elementCount"/> floats starting <paramref name="elementOffset"/> floats into
    /// <paramref name="parent"/>: the same device buffer, bound and transferred at a byte offset.
    /// </summary>
    internal WebGpuBuffer(WebGpuBuffer parent, int elementOffset, int elementCount)
    {
        if (parent is null) throw new ArgumentNullException(nameof(parent));
        if (elementOffset < 0 || elementCount <= 0 || (long)elementOffset + elementCount > parent._elementCount)
            throw new ArgumentOutOfRangeException(nameof(elementOffset));
        parent.ThrowIfDisposed();
        _bufferId = parent._bufferId;
        _viewParent = parent;
        _offsetBytes = checked(parent._offsetBytes + elementOffset * sizeof(float));
        _elementCount = elementCount;
        _sizeBytes = elementCount * sizeof(float);
        _usage = parent._usage;
    }
    /// <summary>
    /// Creates a WebGPU buffer with initial data.
    /// </summary>
    /// <param name="data">Initial data to upload.</param>
    /// <param name="usage">Buffer usage flags.</param>
    public WebGpuBuffer(float[] data, WebGpuBufferUsage usage = WebGpuBufferUsage.Storage | WebGpuBufferUsage.CopySrc | WebGpuBufferUsage.CopyDst)
        : this(data.Length, usage)
    {
        CopyFrom(data);
    }

    /// <summary>
    /// Copies data from a float array to the GPU buffer.
    /// </summary>
    /// <param name="data">Source data array.</param>
    public void CopyFrom(float[] data)
    {
        ThrowIfDisposed();

        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (data.Length > _elementCount)
        {
            throw new ArgumentException($"Data length ({data.Length}) exceeds buffer capacity ({_elementCount})", nameof(data));
        }

        WebGpuNativeBindings.WriteBuffer(_bufferId, data, _offsetBytes);
    }

    /// <summary>
    /// Copies data from the GPU buffer to a float array.
    /// </summary>
    /// <param name="data">Destination data array.</param>
    public void CopyTo(float[] data)
    {
        // Synchronous wrapper for async operation
        CopyToAsync(data).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Copies raw bytes from a CPU array to the GPU buffer.
    /// </summary>
    public void CopyBytesFrom(byte[] data)
    {
        ThrowIfDisposed();

        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (data.Length > _sizeBytes)
        {
            throw new ArgumentException($"Data length ({data.Length}) exceeds buffer capacity ({_sizeBytes})", nameof(data));
        }

        WebGpuNativeBindings.WriteByteBuffer(_bufferId, data, _offsetBytes);
    }

    /// <summary>
    /// Copies raw bytes from the GPU buffer to a CPU array.
    /// </summary>
    public void CopyBytesTo(byte[] data)
    {
        CopyBytesToAsync(data).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Asynchronously copies raw bytes from the GPU buffer to a CPU array.
    /// </summary>
    public async Task CopyBytesToAsync(byte[] data)
    {
        ThrowIfDisposed();

        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (data.Length > _sizeBytes)
        {
            throw new ArgumentException($"Data length ({data.Length}) exceeds buffer capacity ({_sizeBytes})", nameof(data));
        }

        var result = await WebGpuNativeBindings.ReadByteBufferAsync(_bufferId, data.Length, _offsetBytes);
        Array.Copy(result, data, Math.Min(result.Length, data.Length));
    }

    /// <summary>
    /// Asynchronously copies data from the GPU buffer to a float array.
    /// </summary>
    /// <param name="data">Destination data array.</param>
    public async Task CopyToAsync(float[] data)
    {
        ThrowIfDisposed();

        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (data.Length > _elementCount)
        {
            throw new ArgumentException($"Data length ({data.Length}) exceeds buffer capacity ({_elementCount})", nameof(data));
        }

        var result = await WebGpuNativeBindings.ReadBufferAsync(_bufferId, data.Length * sizeof(float), _offsetBytes);
        Array.Copy(result, data, Math.Min(result.Length, data.Length));
    }

    /// <summary>
    /// Downloads all data from the buffer.
    /// </summary>
    /// <returns>Array containing all buffer data.</returns>
    public float[] Download()
    {
        return DownloadAsync().GetAwaiter().GetResult();
    }

    /// <summary>
    /// Asynchronously downloads all data from the buffer.
    /// </summary>
    /// <returns>Array containing all buffer data.</returns>
    public async Task<float[]> DownloadAsync()
    {
        ThrowIfDisposed();
        return await WebGpuNativeBindings.ReadBufferAsync(_bufferId, _sizeBytes, _offsetBytes);
    }

    /// <summary>
    /// Copies data from another WebGPU buffer.
    /// </summary>
    /// <param name="source">Source buffer.</param>
    /// <param name="srcOffset">Offset in source buffer (in floats).</param>
    /// <param name="dstOffset">Offset in this buffer (in floats).</param>
    /// <param name="count">Number of floats to copy.</param>
    public void CopyFromBuffer(WebGpuBuffer source, int srcOffset, int dstOffset, int count)
    {
        ThrowIfDisposed();

        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        source.ThrowIfDisposed();

        WebGpuNativeBindings.CopyBufferToBuffer(
            source.BufferId,
            source.OffsetBytes + srcOffset * sizeof(float),
            _bufferId,
            _offsetBytes + dstOffset * sizeof(float),
            count * sizeof(float));
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(WebGpuBuffer));
        }
    }

    /// <summary>
    /// Disposes the buffer and releases GPU resources.
    /// </summary>
    public void Dispose()
    {
        if (System.Threading.Interlocked.Exchange(ref _disposedFlag, 1) != 0)
        {
            return;
        }

        // A view owns no device memory: its parent destroys the buffer.
        if (_bufferId >= 0 && _viewParent is null)
        {
            WebGpuNativeBindings.DestroyBuffer(_bufferId);
        }
        GC.SuppressFinalize(this);
    }

    // An undisposed buffer (a result nothing references any more) destroys its device buffer through the queue.
    ~WebGpuBuffer()
    {
        if (System.Threading.Interlocked.Exchange(ref _disposedFlag, 1) != 0 || _bufferId < 0 || _viewParent is not null) return;
        int bufferId = _bufferId;
        PendingFrees.Enqueue(() => WebGpuNativeBindings.DestroyBuffer(bufferId));
    }

    public override string ToString()
    {
        return $"WebGpuBuffer[ID={_bufferId}, Elements={_elementCount}, Usage={_usage}]";
    }
}
#endif
