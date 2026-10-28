using System;
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
