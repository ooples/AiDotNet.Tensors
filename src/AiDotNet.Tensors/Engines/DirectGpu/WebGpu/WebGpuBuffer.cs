// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
public sealed class WebGpuBuffer : IGpuBuffer, IPoolableGpuBuffer, IDisposable
{
    internal const WebGpuBufferUsage DefaultStorageUsage =
        WebGpuBufferUsage.Storage | WebGpuBufferUsage.CopySrc | WebGpuBufferUsage.CopyDst;

    private const int MaxPooledBuffersPerBucket = 4096;
    private const long MinimumReclaimCharge = 128 * 1024;

    /// <summary>Released default-usage storage buffers kept for reuse. AIDOTNET_WEBGPU_BUFFER_POOL=0 disables pooling.</summary>
    internal static readonly GpuBufferPool<WebGpuBuffer> Pool =
        new(MaxPooledBuffersPerBucket, int.MaxValue, exactSizes: true);

    internal static bool PoolingEnabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_WEBGPU_BUFFER_POOL") != "0";

    private static long s_driverAllocations;

    /// <summary>
    /// Default-usage storage buffers allocated from WebGPU (pool misses) since the process started. Only those are pooled,
    /// so a uniform or staging buffer an op creates for itself is not a pool miss and is not counted.
    /// </summary>
    internal static long DriverAllocations => System.Threading.Interlocked.Read(ref s_driverAllocations);

    // The count is process-wide, so it is kept by a static method rather than written from each constructor.
    private static void CountDriverAllocation() => System.Threading.Interlocked.Increment(ref s_driverAllocations);

    private int _bufferId;
    private readonly int _sizeBytes;
    private readonly int _elementCount;
    private readonly WebGpuBufferUsage _usage;
    private readonly bool _pooled;
    private int _disposedFlag;
    private readonly int _offsetBytes;
    // A view keeps its parent reachable, so the parent's finalizer cannot destroy memory the view still addresses.
    private readonly WebGpuBuffer? _viewParent;
    private AiDotNet.Tensors.Engines.DevicePrimitives.DeviceMemoryHold _memory; // what GpuMemoryStats counts (nothing for a view)
    private long _reclaimCharge;
    private long _reclaimEpoch = -1;

    /// <inheritdoc/>
    int IPoolableGpuBuffer.Capacity => _elementCount;

    /// <inheritdoc/>
    void IPoolableGpuBuffer.MarkRented(int size)
    {
        // The pool is exact-size (a WebGPU buffer has no logical size below its allocation), so a rent is always for this
        // buffer's whole capacity and nothing about the wrapper changes; anything else would hand out a wrong-sized buffer.
        if (size != _elementCount)
            throw new InvalidOperationException(
                $"The exact-size WebGPU pool rented a {_elementCount}-element buffer for a {size}-element request.");
    }

    /// <inheritdoc/>
    void IPoolableGpuBuffer.Release()
    {
        _memory.OnFreed();
        int id = System.Threading.Interlocked.Exchange(ref _bufferId, -1);
        if (id >= 0)
        {
            WebGpuNativeBindings.DestroyBuffer(id);
        }
    }

    // The parked-storage budget never drops below 16M floats (64 MB): a device reporting a small maximum buffer size
    // would otherwise cap the pool below a single large activation, and every allocation of that size would miss.
    private const long MinPooledElements = 16L * 1024 * 1024;

    internal static void ConfigurePoolBudget(long maxBufferBytes)
    {
        if (maxBufferBytes > 0)
        {
            // Bound total parked storage to MaxBufferSize (e.g. 256 MB default -> 64M float elements).
            Pool.MaxPooledElements = Math.Max(maxBufferBytes / sizeof(float), MinPooledElements);
        }
    }

    internal static void ReleaseAndDrain()
    {
        PendingFrees.Drain();
        Pool.DrainAll();
    }

    /// <summary>The device's memory counters. Every WebGPU backend shares one device (WebGpuDevice.Instance).</summary>
    internal static readonly AiDotNet.Tensors.Engines.DevicePrimitives.DeviceMemoryAccount Account =
        AiDotNet.Tensors.Engines.DevicePrimitives.DeviceMemoryAccount.For("webgpu", 0);
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
            // The binding does not say why creation failed; free what unreachable and pooled buffers hold and retry once
            // (see DeviceMemoryReclaim) before reporting the failure.
            DeviceMemoryReclaim.CollectUnreachable();
            PendingFrees.Drain();
            Pool.DrainAll();
            bufferId = WebGpuNativeBindings.CreateBuffer(alignedSize, (int)usage);
        }
        _bufferId = bufferId;

        if (_bufferId < 0)
        {
            throw new InvalidOperationException("Failed to create WebGPU buffer");
        }
        if (usage == DefaultStorageUsage) CountDriverAllocation();
        _pooled = usage == DefaultStorageUsage && WebGpuDevice.Instance.IsInitialized;
        _memory.OnAllocated(Account, alignedSize);
    }

    private WebGpuBuffer(int bufferId, int elementCount, WebGpuBufferUsage usage)
    {
        _bufferId = bufferId;
        _elementCount = elementCount;
        _sizeBytes = elementCount * sizeof(float);
        _usage = usage;
        _pooled = usage == DefaultStorageUsage;
    }

    internal static WebGpuBuffer Rent(int elementCount, out bool fromPool)
    {
        if (elementCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(elementCount), "Element count must be positive");
        }

        if (!PendingFrees.IsEmpty) PendingFrees.Drain();
        long charge = Math.Max((long)elementCount * sizeof(float), MinimumReclaimCharge);
        long reclaimEpoch = DeviceMemoryReclaim.OnDeviceAllocation(charge);

        WebGpuBuffer buffer;
        if (PoolingEnabled && Pool.TryRent(elementCount, out var parked) && parked is not null)
        {
            buffer = FromParked(parked, elementCount);
            fromPool = true;
        }
        else if (PoolingEnabled && DeviceMemoryReclaim.CollectOnPoolMiss() &&
                 PendingFreesDrainAndTryRent(elementCount, out parked) && parked is not null)
        {
            buffer = FromParked(parked, elementCount);
            fromPool = true;
        }
        else
        {
            buffer = new WebGpuBuffer(elementCount, DefaultStorageUsage);
            fromPool = false;
        }

        buffer._reclaimCharge = charge;
        buffer._reclaimEpoch = reclaimEpoch;
        return buffer;
    }

    private static bool PendingFreesDrainAndTryRent(int elementCount, out WebGpuBuffer? parked)
    {
        PendingFrees.Drain();
        return Pool.TryRent(elementCount, out parked);
    }

    private static WebGpuBuffer FromParked(WebGpuBuffer parked, int elementCount)
    {
        int id = System.Threading.Interlocked.Exchange(ref parked._bufferId, -1);
        var rented = new WebGpuBuffer(id, elementCount, parked._usage);
        rented._memory = parked._memory.TransferToLease();
        rented._memory.OnRented();
        return rented;
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
    /// Disposes the buffer and returns default storage buffers to <see cref="Pool"/> (or destroys non-pooled buffers).
    /// </summary>
    public void Dispose()
    {
        if (System.Threading.Interlocked.Exchange(ref _disposedFlag, 1) != 0)
        {
            return;
        }

        long charge = System.Threading.Interlocked.Exchange(ref _reclaimCharge, 0);
        if (charge > 0) DeviceMemoryReclaim.OnDeviceRelease(charge, _reclaimEpoch);

        int id = System.Threading.Interlocked.Exchange(ref _bufferId, -1);
        GC.SuppressFinalize(this);

        // A view owns no device memory: its parent destroys or recycles the buffer.
        if (_viewParent is not null || id < 0)
        {
            _memory.OnFreed();
            return;
        }

        if (_pooled && PoolingEnabled && WebGpuDevice.Instance.IsInitialized)
        {
            _memory.OnReturned();
            var parked = new WebGpuBuffer(id, _elementCount, _usage)
            {
                _disposedFlag = 1,
                _memory = _memory.TransferToLease()
            };
            GC.SuppressFinalize(parked);
            Pool.Return(parked);
        }
        else
        {
            _memory.OnFreed();
            WebGpuNativeBindings.DestroyBuffer(id);
        }
    }

    // An undisposed buffer (a result nothing references any more) returns to Pool or destroys its device buffer through the queue.
    ~WebGpuBuffer()
    {
        if (System.Threading.Interlocked.Exchange(ref _disposedFlag, 1) != 0) return;
        long charge = System.Threading.Interlocked.Exchange(ref _reclaimCharge, 0);
        DeviceMemoryReclaim.OnDeviceFinalized(charge, _reclaimEpoch);

        int bufferId = System.Threading.Interlocked.Exchange(ref _bufferId, -1);
        if (bufferId < 0 || _viewParent is not null)
        {
            _memory.OnFreed();
            return;
        }

        if (_pooled && PoolingEnabled && WebGpuDevice.Instance.IsInitialized)
        {
            var memoryLease = _memory.TransferToLease();
            int elementCount = _elementCount;
            WebGpuBufferUsage usage = _usage;
            PendingFrees.Enqueue(() =>
            {
                if (!PoolingEnabled || !WebGpuDevice.Instance.IsInitialized)
                {
                    memoryLease.OnFreed();
                    WebGpuNativeBindings.DestroyBuffer(bufferId);
                    return;
                }
                memoryLease.OnReturned();
                var parked = new WebGpuBuffer(bufferId, elementCount, usage)
                {
                    _disposedFlag = 1,
                    _memory = memoryLease
                };
                GC.SuppressFinalize(parked);
                Pool.Return(parked);
            });
        }
        else
        {
            _memory.OnFreed();
            PendingFrees.Enqueue(() => WebGpuNativeBindings.DestroyBuffer(bufferId));
        }
    }

    public override string ToString()
    {
        return $"WebGpuBuffer[ID={_bufferId}, Elements={_elementCount}, Usage={_usage}]";
    }
}
#endif
