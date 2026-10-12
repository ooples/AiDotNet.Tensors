// Copyright (c) AiDotNet. All rights reserved.
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IAsyncGpuBackend"/> for backends whose work runs on one in-order queue (Metal, Vulkan, WebGPU; #1142).
/// Every stream is that queue, so work on any stream is already ordered after all earlier work: waiting on an event
/// needs no device action, and an event is complete once the queue has drained past it. Recording an event and
/// querying completion therefore drain the queue, which is what makes their answers true rather than optimistic.
/// Transfers and kernels run through the backend's ordinary entry points on that queue.
/// </summary>
internal static class InOrderAsyncGpu
{
    public static IGpuStream CreateStream(IDirectGpuBackend backend, GpuStreamType type, int priority)
        => new InOrderGpuStream(backend, type, priority, isDefault: false);

    public static IGpuStream CreateDefaultStream(IDirectGpuBackend backend)
        => new InOrderGpuStream(backend, GpuStreamType.Default, 0, isDefault: true);

    public static void RecordEvent(IDirectGpuBackend backend, IGpuEvent gpuEvent, IGpuStream stream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        if (gpuEvent is not InOrderGpuEvent inOrder)
            throw new ArgumentException("The event was not created by this backend.", nameof(gpuEvent));
        inOrder.Record(backend);
    }

    public static void StreamWaitEvent(IGpuStream stream, IGpuEvent gpuEvent)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        if (gpuEvent is null) throw new ArgumentNullException(nameof(gpuEvent));
    }

    public static GpuSyncPoint CreateSyncPoint(IGpuStream stream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        return new InOrderSyncPoint(stream, stream.RecordEvent());
    }

    public static void UploadBufferAsync(IDirectGpuBackend backend, float[] data, IGpuBuffer buffer, IGpuStream stream)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        if (data.Length > buffer.Size)
            throw new ArgumentException($"Host data ({data.Length}) exceeds buffer capacity ({buffer.Size}).", nameof(data));
        if (data.Length == 0) return;
        using var staged = backend.AllocateBuffer(data);
        backend.Copy(staged, buffer, data.Length);
    }

    public static void DownloadBufferAsync(IDirectGpuBackend backend, IGpuBuffer buffer, float[] destination, IGpuStream stream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        backend.DownloadBuffer(buffer, destination);
    }

    public static IGpuBuffer AllocateBufferAsync(IDirectGpuBackend backend, float[] data, IGpuStream stream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        return backend.AllocateBuffer(data);
    }

    public static void CopyBufferAsync(IDirectGpuBackend backend, IGpuBuffer source, IGpuBuffer destination, int size, IGpuStream stream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        backend.Copy(source, destination, size);
    }

    public static void GemmAsync(IDirectGpuBackend backend, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta, IGpuStream stream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        backend.Gemm(A, B, C, M, N, K, alpha, beta);
    }

    /// <summary>
    /// GEMM + bias, then the activation, with the activations OpenCL's fused GEMM supports (none, ReLU, sigmoid,
    /// tanh, and the row-wise softmax); any other activation is refused the same way.
    /// </summary>
    public static void FusedGemmBiasActivationAsync(IDirectGpuBackend backend, IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias,
        IGpuBuffer output, int M, int N, int K, FusedActivationType activation, IGpuStream stream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        if (activation is not (FusedActivationType.None or FusedActivationType.ReLU or FusedActivationType.Sigmoid
            or FusedActivationType.Tanh or FusedActivationType.Softmax))
            throw new NotSupportedException($"Activation type {activation} not supported for fused GEMM");
        int size = checked(M * N);
        using var gemmBias = backend.GemmBias(A, B, bias, M, N, K);
        switch (activation)
        {
            case FusedActivationType.None: backend.Copy(gemmBias, output, size); break;
            case FusedActivationType.ReLU: backend.Relu(gemmBias, output, size); break;
            case FusedActivationType.Sigmoid: backend.Sigmoid(gemmBias, output, size); break;
            case FusedActivationType.Tanh: backend.Tanh(gemmBias, output, size); break;
            default: backend.Softmax(gemmBias, output, M, N); break;
        }
    }

    public static float GetEventElapsedTime(IGpuEvent start, IGpuEvent end)
    {
        if (start is null) throw new ArgumentNullException(nameof(start));
        if (end is null) throw new ArgumentNullException(nameof(end));
        return end.GetElapsedTime(start);
    }

    private sealed class InOrderGpuStream : IGpuStream
    {
        private readonly IDirectGpuBackend _backend;

        public InOrderGpuStream(IDirectGpuBackend backend, GpuStreamType type, int priority, bool isDefault)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            StreamType = type;
            Priority = priority;
            IsDefault = isDefault;
        }

        public IntPtr Handle => IntPtr.Zero;
        public GpuStreamType StreamType { get; }
        public bool IsDefault { get; }
        public int Priority { get; }
        public void Synchronize() => _backend.Synchronize();

        public IGpuEvent RecordEvent()
        {
            var gpuEvent = new InOrderGpuEvent();
            gpuEvent.Record(_backend);
            return gpuEvent;
        }

        public void WaitEvent(IGpuEvent gpuEvent)
        {
            if (gpuEvent is null) throw new ArgumentNullException(nameof(gpuEvent));
        }

        public bool Query()
        {
            _backend.Synchronize();
            return true;
        }

        public void Dispose()
        {
        }
    }

    /// <summary>Holds the host time at which the queue had drained past the point the event was recorded.</summary>
    internal sealed class InOrderGpuEvent : IGpuEvent
    {
        private long _timestamp;

        public IntPtr Handle => IntPtr.Zero;
        public bool IsRecorded { get; private set; }
        public bool IsComplete => IsRecorded;

        public void Record(IDirectGpuBackend backend)
        {
            backend.Synchronize();
            _timestamp = Stopwatch.GetTimestamp();
            IsRecorded = true;
        }

        public void Synchronize()
        {
            if (!IsRecorded) throw new InvalidOperationException("The event has not been recorded.");
        }

        public bool Query() => IsRecorded;

        public float GetElapsedTime(IGpuEvent startEvent)
        {
            if (startEvent is not InOrderGpuEvent start || !start.IsRecorded || !IsRecorded)
                throw new InvalidOperationException("Both events must be recorded on this backend.");
            return (float)((_timestamp - start._timestamp) * 1000.0 / Stopwatch.Frequency);
        }

        public void Dispose()
        {
        }
    }

    private sealed class InOrderSyncPoint : GpuSyncPoint
    {
        public InOrderSyncPoint(IGpuStream stream, IGpuEvent gpuEvent)
        {
            Stream = stream;
            Event = gpuEvent;
        }

        public override bool IsComplete => true;
        public override IGpuStream? Stream { get; }
        public override IGpuEvent? Event { get; }
        public override void Wait() { }
        public override bool Poll() => true;
    }
}
