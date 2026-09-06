using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Adapts an asynchronous GPU backend's native timing events to the kernel-tuning device-clock contract.
/// The caller retains ownership of the backend and stream.
/// </summary>
public sealed class GpuKernelTuningDeviceClock : IKernelTuningDeviceClock
{
    private readonly IAsyncGpuBackend _backend;
    private readonly IGpuStream _stream;

    /// <summary>Uses the backend's default compute stream.</summary>
    public GpuKernelTuningDeviceClock(IAsyncGpuBackend backend)
        : this(backend, (backend ?? throw new ArgumentNullException(nameof(backend))).DefaultStream)
    {
    }

    /// <summary>Uses an explicitly selected stream that also receives the tuned operation.</summary>
    public GpuKernelTuningDeviceClock(IAsyncGpuBackend backend, IGpuStream stream)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        if (!backend.SupportsEvents)
            throw new NotSupportedException("The GPU backend does not support device timing events.");
    }

    /// <inheritdoc />
    public IKernelTuningDeviceTimestamp Record()
    {
        IGpuEvent gpuEvent = _backend.CreateEvent(enableTiming: true);
        try
        {
            _backend.RecordEvent(gpuEvent, _stream);
            return new Timestamp(this, gpuEvent);
        }
        catch
        {
            gpuEvent.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public TimeSpan GetElapsed(
        IKernelTuningDeviceTimestamp start,
        IKernelTuningDeviceTimestamp end)
    {
        Timestamp startTimestamp = ValidateTimestamp(start, nameof(start));
        Timestamp endTimestamp = ValidateTimestamp(end, nameof(end));
        float milliseconds = _backend.GetEventElapsedTime(
            startTimestamp.Event,
            endTimestamp.Event);
        if (float.IsNaN(milliseconds) || float.IsInfinity(milliseconds))
            throw new InvalidOperationException("The GPU event clock returned a non-finite duration.");
        return TimeSpan.FromMilliseconds(milliseconds);
    }

    private Timestamp ValidateTimestamp(IKernelTuningDeviceTimestamp timestamp, string parameterName)
    {
        if (timestamp is not Timestamp result || !ReferenceEquals(result.Owner, this))
        {
            throw new ArgumentException(
                "The timestamp was not created by this GPU device clock.",
                parameterName);
        }
        return result;
    }

    private sealed class Timestamp : IKernelTuningDeviceTimestamp
    {
        internal Timestamp(GpuKernelTuningDeviceClock owner, IGpuEvent gpuEvent)
        {
            Owner = owner;
            Event = gpuEvent;
        }

        internal GpuKernelTuningDeviceClock Owner { get; }
        internal IGpuEvent Event { get; }

        public ValueTask WaitAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Event.Synchronize();
            cancellationToken.ThrowIfCancellationRequested();
            return default;
        }

        public void Dispose() => Event.Dispose();
    }
}
