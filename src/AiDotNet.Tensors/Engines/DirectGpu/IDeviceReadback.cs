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
/// The readback for a backend without an asynchronous device-to-host copy: <see cref="Enqueue"/> downloads at once
/// (the download waits for the submitted work, so the value is the same), and <see cref="Read"/> returns it.
/// </summary>
internal sealed class SynchronousDeviceReadback : IDeviceReadback
{
    private readonly IDirectGpuBackend _backend;
    private readonly int _capacity;
    private byte[] _bytes = Array.Empty<byte>();
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
        _bytes = _backend.DownloadByteBuffer(source, byteCount);
        _queued = true;
        HasUnreadValue = true;
    }

    public ReadOnlySpan<byte> Read()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(SynchronousDeviceReadback));
        if (!_queued) throw new InvalidOperationException("Nothing was queued for readback.");
        HasUnreadValue = false;
        return _bytes;
    }

    public void Dispose() => _disposed = true;
}
