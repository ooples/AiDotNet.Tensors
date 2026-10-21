using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A small device-to-host copy that need not block the host when it is issued: on a backend with an asynchronous
/// copy it is queued behind the work already submitted and the host waits only when it reads the value.
/// </summary>
/// <remarks>
/// A training step that downloads a scalar (its loss, or whether its gradients are finite) synchronously drains the
/// device every step, so the host can never prepare step N+1 while the device runs step N. Reading the same scalar
/// through a readback costs that wait only when, and if, something reads it.
/// </remarks>
internal interface IDeviceReadback : IDisposable
{
    /// <summary>Copies the first <paramref name="byteCount"/> bytes of <paramref name="source"/> as they will be once
    /// all work already submitted to the backend has run. Replaces any earlier copy.</summary>
    void Enqueue(IGpuBuffer source, int byteCount);

    /// <summary>True when a copy was queued and has not yet been read by <see cref="Read"/>.</summary>
    bool HasUnreadValue { get; }

    /// <summary>Waits for the queued copy and returns its bytes. The span is valid until the next
    /// <see cref="Enqueue"/> or <see cref="IDisposable.Dispose"/>.</summary>
    ReadOnlySpan<byte> Read();
}

/// <summary>Creates <see cref="IDeviceReadback"/> slots.</summary>
internal interface IDeviceReadbackFactory
{
    IDeviceReadback CreateReadback(int byteCapacity);
}

/// <summary>
/// The readback for a backend without an asynchronous device-to-host copy: <see cref="Enqueue"/> fixes the value on the
/// device and <see cref="Read"/> downloads it, so a step that never reads pays no wait at all - the point of the
/// interface. The download is the host's only wait, and it waits for work the device has usually long finished by then.
/// <para>The source is a float buffer (a training step's loss). Its prefix is copied on the device into this
/// readback's own float buffer, which is downloaded: a backend's byte download accepts only its byte buffers (OpenCL
/// threw "Buffer must be a DirectOpenClGpuByteBuffer", failing every compiled GPU training step off CUDA), and its
/// float download reads a whole buffer, where the loss sits in a larger pooled one. The copy must happen at the
/// enqueue, not the read: the source is a pooled buffer the steps after it reuse, so only the copy pins which values
/// this readback returns.</para>
/// </summary>
internal sealed class SynchronousDeviceReadback : IDeviceReadback
{
    private readonly IDirectGpuBackend _backend;
    private readonly int _capacity;
    private byte[] _bytes = Array.Empty<byte>();
    private IGpuBuffer? _staging;
    private int _queuedBytes;
    private bool _downloaded;
    private bool _queued;
    private bool _disposed;

    public SynchronousDeviceReadback(IDirectGpuBackend backend, int byteCapacity)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (byteCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(byteCapacity));
        _backend = backend;
        _capacity = byteCapacity;
    }

    public bool HasUnreadValue { get; private set; }

    public void Enqueue(IGpuBuffer source, int byteCount)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(SynchronousDeviceReadback));
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (byteCount <= 0 || byteCount > _capacity) throw new ArgumentOutOfRangeException(nameof(byteCount));
        int floats = (byteCount + sizeof(float) - 1) / sizeof(float);
        if (source.Size < floats)
            throw new ArgumentOutOfRangeException(nameof(byteCount), $"The buffer holds {source.Size} floats, fewer than {byteCount} bytes.");
        // A backend's float download reads a whole buffer, and the source is usually a larger pooled one: copy the
        // requested prefix into this readback's own buffer on the device and download only that. The copy is queued
        // behind the work already submitted and does not wait; the download is left to Read.
        var staging = _staging ??= _backend.AllocateBuffer((_capacity + sizeof(float) - 1) / sizeof(float));
        _backend.Copy(source, 0, staging, 0, floats);
        _queuedBytes = byteCount;
        _downloaded = false;
        _queued = true;
        HasUnreadValue = true;
    }

    public ReadOnlySpan<byte> Read()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(SynchronousDeviceReadback));
        if (!_queued) throw new InvalidOperationException("Nothing was queued for readback.");
        if (!_downloaded)
        {
            // The one wait, taken where the value is wanted rather than where it was queued. The staging copy already
            // decided which values these are, so waiting later cannot change them.
            var staging = _staging ?? throw new InvalidOperationException("The queued readback has no staging buffer.");
            // The download is the only part that needs the backend alive. An owner resolves a pending value before it
            // tears the backend down (CompiledTrainingPlan materializes the loss host array first); past that there is
            // nothing to download, and the InvalidOperationException is what those owners already handle.
            if (!_backend.IsAvailable)
                throw new InvalidOperationException("The backend is gone, so the queued readback cannot be downloaded.");
            var values = _backend.DownloadBuffer(staging);
            if (_bytes.Length != _queuedBytes) _bytes = new byte[_queuedBytes];
            Buffer.BlockCopy(values, 0, _bytes, 0, _queuedBytes);
            _downloaded = true;
        }

        HasUnreadValue = false;
        return _bytes;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // A backend that is already gone took its buffers with it.
        if (_backend.IsAvailable) _staging?.Dispose();
        _staging = null;
    }
}
