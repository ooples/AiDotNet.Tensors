// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// The HIP side of DLPack exchange (<see cref="Interop.DLPack"/>): ROCm device memory (kDLROCM) shared by pointer with
/// another library, the stream ordering an export owes its consumer, and the buffer an imported pointer is wrapped in.
/// </summary>
/// <remarks>
/// An imported pointer becomes a <see cref="HipGpuBuffer"/> in its non-owning mode rather than a buffer type of its
/// own: every HIP kernel path takes its operands as <see cref="HipGpuBuffer"/>, so any other type would be refused by
/// the first kernel the imported tensor reached.
/// </remarks>
public sealed partial class HipBackend : IDLPackDeviceBackend
{
    // hipEventDisableTiming: an event only used for ordering needs no timestamp.
    private const uint HipEventDisableTiming = 0x2;

    int IDLPackDeviceBackend.DLPackDeviceType => 10;   // kDLROCM
    int IDLPackDeviceBackend.DeviceOrdinal => _hipDeviceIndex;
    IntPtr IDLPackDeviceBackend.WorkStreamHandle => _stream;
    void IDLPackDeviceBackend.OrderStreamAfterPendingWork(IntPtr consumerStream) => OrderStreamAfterPendingWork(consumerStream);
    IGpuBuffer IDLPackDeviceBackend.WrapExternalDeviceMemory(IntPtr devicePointer, int elementCount, int elementByteSize, Action release)
        => WrapExternalDeviceMemory(devicePointer, elementCount, elementByteSize, release);
    void IDLPackDeviceBackend.ReleaseAfterPendingWorkDeferred(Action release) => ReleaseAfterPendingWorkDeferred(release);
    IDisposable IDLPackDeviceBackend.HoldForExport(IGpuBuffer buffer, IntPtr consumerStream) => HoldForExport(buffer, consumerStream);

    /// <summary>
    /// Keeps <paramref name="buffer"/>'s device memory for a DLPack export: until the hold is disposed (the consumer's
    /// deleter), a release or pool return of the buffer is parked rather than run. Disposing the hold first makes this
    /// backend's stream wait for the work queued on <paramref name="consumerStream"/> (the consumer may still have
    /// kernels using the memory when it calls the deleter), then runs the parked calls.
    /// </summary>
    /// <param name="consumerStream">The stream the consumer works on, or zero when it synchronizes itself (NoSync).</param>
    internal IDisposable HoldForExport(IGpuBuffer buffer, IntPtr consumerStream)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        // Imported memory belongs to another library and is never freed here; re-exporting it needs only the ordering.
        var pin = buffer is HipGpuBuffer hip ? hip.HoldForExport() : null;
        return new ExportHold(this, pin, consumerStream);
    }

    private sealed class ExportHold : IDisposable
    {
        private readonly HipBackend _backend;
        private readonly IntPtr _consumerStream;
        private IDisposable? _pin;
        private int _disposed;

        public ExportHold(HipBackend backend, IDisposable? pin, IntPtr consumerStream)
        {
            _backend = backend;
            _pin = pin;
            _consumerStream = consumerStream;
        }

        public void Dispose()
        {
            if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0) return;
            try
            {
                if (_consumerStream != IntPtr.Zero && !DeviceMemoryReclaim.ProcessExiting)
                    _backend.WaitOnStream(_consumerStream);
            }
            finally
            {
                // Released even when the ordering failed (the consumer passed a stream it has since destroyed): the
                // memory would otherwise stay parked for the life of the process.
                _pin?.Dispose();
                _pin = null;
            }
        }
    }

    /// <summary>
    /// Makes <paramref name="consumerStream"/> wait for every operation queued so far on this backend's stream, without
    /// blocking the host: the producer's half of the DLPack stream contract.
    /// </summary>
    internal void OrderStreamAfterPendingWork(IntPtr consumerStream) => OrderStreams(waiter: consumerStream, producer: _stream);

    /// <summary>Makes this backend's stream wait for the work queued so far on <paramref name="consumerStream"/>.</summary>
    private void WaitOnStream(IntPtr consumerStream)
    {
        if (_disposed) return;   // torn down: nothing of ours can still run on the memory
        OrderStreams(waiter: _stream, producer: consumerStream);
    }

    // waiter waits for everything queued on producer so far: an event recorded on one, waited on by the other.
    private void OrderStreams(IntPtr waiter, IntPtr producer)
    {
        if (waiter == producer) return;   // same stream: already ordered
        ((IThreadContextBackend)this).EnsureContextCurrent();
        IntPtr ready = IntPtr.Zero;
        HipNativeBindings.CheckError(
            HipNativeBindings.hipEventCreateWithFlags(ref ready, HipEventDisableTiming), "hipEventCreateWithFlags(DLPack)");
        try
        {
            HipNativeBindings.CheckError(HipNativeBindings.hipEventRecord(ready, producer), "hipEventRecord(DLPack)");
            HipNativeBindings.CheckError(HipNativeBindings.hipStreamWaitEvent(waiter, ready, 0), "hipStreamWaitEvent(DLPack)");
        }
        finally
        {
            // Destroying a recorded event is legal: the wait binds to the record, not to the event object.
            HipNativeBindings.hipEventDestroy(ready);
        }
    }

    /// <summary>
    /// Wraps device memory another library owns as a buffer this backend's kernels can read and write. The buffer never
    /// frees the memory: releasing it first waits for this backend's queued work, then runs <paramref name="release"/>.
    /// </summary>
    internal IGpuBuffer WrapExternalDeviceMemory(IntPtr devicePointer, int elementCount, int elementByteSize, Action release)
    {
        if (devicePointer == IntPtr.Zero) throw new ArgumentException("Device pointer is null.", nameof(devicePointer));
        if (elementCount <= 0) throw new ArgumentOutOfRangeException(nameof(elementCount));
        if (elementByteSize != sizeof(float))
            throw new ArgumentOutOfRangeException(nameof(elementByteSize), "HIP buffers hold float32 elements.");
        if (release is null) throw new ArgumentNullException(nameof(release));
        return HipGpuBuffer.CreateExternal(devicePointer, elementCount, this, release);
    }

    // Releases of imported memory DEFERRED from finalizers (an imported tensor that was never disposed). The finalizer
    // thread never calls the driver: it queues the release, and the next allocation on any HIP backend waits for the
    // owning backend's stream and then runs it (DrainPendingExternalReleases).
    internal static readonly ConcurrentQueue<(HipBackend Backend, Action Release)> PendingExternalReleases = new();

    /// <summary>
    /// Queues <paramref name="release"/> to run once the work queued so far on this backend's stream has finished.
    /// Safe on the finalizer thread: it only enqueues.
    /// </summary>
    internal void ReleaseAfterPendingWorkDeferred(Action release)
    {
        if (release is null) throw new ArgumentNullException(nameof(release));
        PendingExternalReleases.Enqueue((this, release));
    }

    /// <summary>Runs the queued releases of imported memory, each after its owning backend's stream finished.</summary>
    internal static void DrainPendingExternalReleases()
    {
        while (PendingExternalReleases.TryDequeue(out var entry))
        {
            // At process exit the producer may already be gone; its memory goes with the process.
            if (DeviceMemoryReclaim.ProcessExiting) continue;
            try
            {
                entry.Backend.SynchronizeWorkStream();
            }
            catch (Exception ex)
            {
                // The wait failed (the backend was torn down meanwhile, or the device faulted): none of its kernels still
                // runs on the memory, so it is released below rather than leaked.
                System.Diagnostics.Debug.WriteLine($"HIP stream wait before a deferred DLPack release failed: {ex}");
            }
            try { entry.Release(); }
            catch (Exception ex)
            {
                // The producer's deleter failing must not fail the unrelated operation that drained it.
                System.Diagnostics.Debug.WriteLine($"A deferred DLPack deleter failed: {ex}");
            }
        }
    }

    /// <summary>Waits for this backend's stream: the release step of an imported buffer.</summary>
    internal void SynchronizeWorkStream()
    {
        if (_disposed) return;   // torn down: nothing can still be queued
        ((IThreadContextBackend)this).EnsureContextCurrent();
        HipNativeBindings.CheckError(HipNativeBindings.hipStreamSynchronize(_stream), "hipStreamSynchronize(DLPack release)");
    }
}
