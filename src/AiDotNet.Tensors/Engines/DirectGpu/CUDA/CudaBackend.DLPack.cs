using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The CUDA side of DLPack exchange (<see cref="Interop.DLPack"/>): the device ordinal a DLPack <c>DLDevice</c>
/// names, the stream ordering an export owes its consumer, and the buffer an imported device pointer is wrapped in.
/// </summary>
public sealed partial class CudaBackend : IDLPackDeviceBackend
{
    int IDLPackDeviceBackend.DLPackDeviceType => 2;   // kDLCUDA
    int IDLPackDeviceBackend.DeviceOrdinal => DeviceOrdinal;
    IntPtr IDLPackDeviceBackend.WorkStreamHandle => WorkStreamHandle;
    void IDLPackDeviceBackend.OrderStreamAfterPendingWork(IntPtr consumerStream) => OrderStreamAfterPendingWork(consumerStream);
    IGpuBuffer IDLPackDeviceBackend.WrapExternalDeviceMemory(IntPtr devicePointer, int elementCount, int elementByteSize, Action release)
        => WrapExternalDeviceMemory(devicePointer, elementCount, elementByteSize, release);
    void IDLPackDeviceBackend.ReleaseAfterPendingWorkDeferred(Action release) => ReleaseAfterPendingWorkDeferred(release);
    IDisposable IDLPackDeviceBackend.HoldForExport(IGpuBuffer buffer, IntPtr consumerStream) => HoldForExport(buffer, consumerStream);

    /// <summary>
    /// Keeps <paramref name="buffer"/>'s device memory for a DLPack export: until the hold is disposed (the consumer's
    /// deleter), no release of the buffer -- a TensorScope ending, an activation-cache eviction, a pool return, its
    /// finalizer -- frees it or hands it to another allocation; each is parked instead. Disposing the hold first makes
    /// this backend's stream wait for the work queued on <paramref name="consumerStream"/> (the consumer may still have
    /// kernels using the memory when it calls the deleter), then runs the parked releases, which are stream-ordered.
    /// </summary>
    /// <param name="consumerStream">The stream the consumer works on, or zero when it synchronizes itself (NoSync).</param>
    internal IDisposable HoldForExport(IGpuBuffer buffer, IntPtr consumerStream)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        // The same mechanism keeps a captured graph's memory: a pinned buffer parks every release until retirement.
        var pins = new CaptureMemoryPool();
        if (buffer is CudaGpuBuffer cuda) cuda.PinTo(pins);
        // Imported memory (ExternalDeviceBuffer) belongs to another library and is never freed here; re-exporting it
        // needs only the stream ordering.
        return new ExportHold(this, pins, consumerStream);
    }

    /// <summary>Makes this backend's stream wait for the work queued so far on <paramref name="consumerStream"/>.</summary>
    private void OrderAfterStream(IntPtr consumerStream)
    {
        if (_disposed && _cudaContext == IntPtr.Zero) return;   // torn down: nothing of ours can run on the memory
        using var _ = PushContext();
        if (consumerStream == _stream) return;
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuEventCreate(out IntPtr done, CudaNativeBindings.CU_EVENT_DISABLE_TIMING),
            "cuEventCreate(DLPack release)");
        try
        {
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventRecord(done, consumerStream), "cuEventRecord(DLPack release)");
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamWaitEvent(_stream, done, 0), "cuStreamWaitEvent(DLPack release)");
        }
        finally
        {
            CudaNativeBindings.cuEventDestroy(done);
        }
    }

    private sealed class ExportHold : IDisposable
    {
        private readonly CudaBackend _backend;
        private readonly IntPtr _consumerStream;
        private CaptureMemoryPool? _pins;

        public ExportHold(CudaBackend backend, CaptureMemoryPool pins, IntPtr consumerStream)
        {
            _backend = backend;
            _pins = pins;
            _consumerStream = consumerStream;
        }

        public void Dispose()
        {
            var pins = Interlocked.Exchange(ref _pins, null);
            if (pins is null) return;
            try
            {
                if (_consumerStream != IntPtr.Zero && !IsRuntimeTearingDown && !ProcessExiting)
                    _backend.OrderAfterStream(_consumerStream);
            }
            finally
            {
                // Released even when the ordering failed (the consumer passed a stream it has since destroyed): the
                // memory would otherwise leak for the life of the process.
                pins.Retire();
            }
        }
    }

    /// <summary>The CUDA device ordinal this backend was created for (<c>DLDevice.device_id</c>).</summary>
    internal int DeviceOrdinal { get; }

    /// <summary>
    /// Makes <paramref name="consumerStream"/> wait for every operation queued so far on this backend's stream, without
    /// blocking the host: the producer's half of the DLPack stream contract. The consumer may be another library's
    /// stream (torch's current stream, say); it lives in the same primary context.
    /// </summary>
    internal void OrderStreamAfterPendingWork(IntPtr consumerStream)
    {
        using var _ = PushContext();
        IntPtr producer = _stream;
        if (consumerStream == producer) return;   // same stream: already ordered
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuEventCreate(out IntPtr ready, CudaNativeBindings.CU_EVENT_DISABLE_TIMING),
            "cuEventCreate(DLPack export)");
        try
        {
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventRecord(ready, producer), "cuEventRecord(DLPack export)");
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuStreamWaitEvent(consumerStream, ready, 0), "cuStreamWaitEvent(DLPack export)");
        }
        finally
        {
            // Destroying a recorded event is legal: the wait binds to the record, not to the event object.
            CudaNativeBindings.cuEventDestroy(ready);
        }
    }

    /// <summary>The handle of the stream this backend orders its work on: what an importer passes a producer as
    /// <c>__dlpack__(stream=...)</c>.</summary>
    internal IntPtr WorkStreamHandle
    {
        get
        {
            using var _ = PushContext();
            return _stream;
        }
    }

    /// <summary>
    /// Wraps device memory another library owns as a buffer this backend's kernels can read and write. The buffer
    /// never frees the memory: disposing it first waits for this backend's queued work (a kernel may still be using the
    /// memory), then runs <paramref name="release"/> once.
    /// </summary>
    internal IGpuBuffer WrapExternalDeviceMemory(IntPtr devicePointer, int elementCount, int elementByteSize, Action release)
    {
        if (devicePointer == IntPtr.Zero) throw new ArgumentException("Device pointer is null.", nameof(devicePointer));
        if (elementCount <= 0) throw new ArgumentOutOfRangeException(nameof(elementCount));
        if (release is null) throw new ArgumentNullException(nameof(release));
        return new ExternalDeviceBuffer(this, devicePointer, elementCount, (long)elementCount * elementByteSize, release);
    }

    // Releases of imported memory DEFERRED from finalizers (an imported tensor that was never disposed). Like
    // PendingFinalizerFrees, the finalizer thread never calls the driver: it queues the release, and the next operation
    // on any CUDA backend waits for the owning backend's stream and then runs it (DrainPendingExternalReleases).
    internal static readonly ConcurrentQueue<(CudaBackend Backend, Action Release)> PendingExternalReleases = new();

    /// <summary>
    /// Queues <paramref name="release"/> to run once the work queued so far on this backend's stream has finished.
    /// Safe on the finalizer thread: it only enqueues.
    /// </summary>
    internal void ReleaseAfterPendingWorkDeferred(Action release)
    {
        if (release is null) throw new ArgumentNullException(nameof(release));
        PendingExternalReleases.Enqueue((this, release));
    }

    /// <summary>
    /// Runs the queued releases of imported memory: waits for each owning backend's stream, then runs the release.
    /// Called from <see cref="EnsureContextCurrent"/> on an operation thread that is not capturing.
    /// </summary>
    internal static void DrainPendingExternalReleases()
    {
        if (IsCapturingOnThisThread) return;
        List<(CudaBackend Backend, Action Release)>? deferred = null;
        while (PendingExternalReleases.TryDequeue(out var entry))
        {
            // At process or runtime teardown the producer may already be gone; its memory goes with the process.
            if (IsRuntimeTearingDown || ProcessExiting) continue;
            var backend = entry.Backend;
            bool tornDown = backend._disposed && backend._cudaContext == IntPtr.Zero;
            if (!tornDown)
            {
                bool capturing = false;
                try
                {
                    using var _ = backend.PushContext();
                    IntPtr stream = backend._stream;
                    // Another thread capturing this stream: a synchronize now is illegal and would invalidate its
                    // graph, so the release waits for a later operation.
                    capturing = CudaNativeBindings.cuStreamIsCapturing(stream, out int captureStatus) == CudaResult.Success
                        && captureStatus != 0;
                    if (!capturing)
                        CuBlasNative.CheckCudaResult(
                            CudaNativeBindings.cuStreamSynchronize(stream), "cuStreamSynchronize(DLPack deferred release)");
                }
                catch
                {
                    // The stream wait failed: the backend was torn down meanwhile, or the context faulted. Either way
                    // none of its kernels still runs on the memory, so it is released below rather than leaked.
                }
                if (capturing)
                {
                    (deferred ??= new List<(CudaBackend, Action)>()).Add(entry);
                    continue;
                }
            }
            try { entry.Release(); }
            catch { /* the producer's deleter failing must not fail the unrelated operation that drained it */ }
        }
        if (deferred is not null)
            foreach (var entry in deferred) PendingExternalReleases.Enqueue(entry);
    }

    /// <summary>Waits for this backend's stream; the release step of <see cref="ExternalDeviceBuffer"/>.</summary>
    private void SynchronizeWorkStream()
    {
        if (_disposed && _cudaContext == IntPtr.Zero) return;   // torn down: nothing can still be queued
        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_stream), "cuStreamSynchronize(DLPack release)");
    }

    private sealed class ExternalDeviceBuffer : IGpuBuffer
    {
        private readonly CudaBackend _backend;
        private Action? _release;
        private IntPtr _handle;

        public ExternalDeviceBuffer(CudaBackend backend, IntPtr handle, int size, long sizeInBytes, Action release)
        {
            _backend = backend;
            _handle = handle;
            Size = size;
            SizeInBytes = sizeInBytes;
            _release = release;
        }

        public int Size { get; }
        public long SizeInBytes { get; }
        public IntPtr Handle => Volatile.Read(ref _handle);

        /// <summary>The backend the memory was imported for.</summary>
        internal CudaBackend Owner => _backend;

        public void Dispose()
        {
            var release = Interlocked.Exchange(ref _release, null);
            if (release is null) return;
            try
            {
                _backend.SynchronizeWorkStream();
            }
            finally
            {
                Volatile.Write(ref _handle, IntPtr.Zero);
                release();
            }
        }
    }
}
