using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Page-locked memory plus an event: the copy is queued behind the submitted work and the host waits only on Read.
public sealed partial class CudaBackend : IDeviceReadbackFactory
{
    IDeviceReadback IDeviceReadbackFactory.CreateReadback(int byteCapacity) => new CudaDeviceReadback(this, byteCapacity);

    private sealed class CudaDeviceReadback : IDeviceReadback
    {
        private readonly CudaBackend _backend;
        private readonly int _capacity;
        private IntPtr _pinned;
        private CudaEvent? _event;
        private int _bytes;
        private bool _unread;

        public CudaDeviceReadback(CudaBackend backend, int byteCapacity)
        {
            if (byteCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(byteCapacity));
            _backend = backend;
            _capacity = byteCapacity;
            using var _ = backend.PushContext();
            CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAllocHost(out _pinned, (ulong)byteCapacity), "cuMemAllocHost(readback)");
        }

        public bool HasUnreadValue => _unread;

        public void Enqueue(IGpuBuffer source, int byteCount)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));
            if (byteCount <= 0 || byteCount > _capacity) throw new ArgumentOutOfRangeException(nameof(byteCount));
            if (_pinned == IntPtr.Zero) throw new ObjectDisposedException(nameof(CudaDeviceReadback));
            if (source.Handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(source), "GPU buffer was released before its readback.");
            if (_backend._defaultStream is not { } stream)
                throw new InvalidOperationException("The CUDA backend has no compute stream.");
            using var _ = _backend.PushContext();
            // The previous copy may still be in flight into the same pinned bytes; queuing the next one on the same
            // stream orders it after that copy, so the bytes are never written twice at once.
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuMemcpyDtoHAsync(_pinned, source.Handle, (ulong)byteCount, _backend._stream),
                "cuMemcpyDtoHAsync(readback)");
            if (_event is null) _event = new CudaEvent(_backend, stream, enableTiming: false);
            else _event.Record(stream);
            _bytes = byteCount;
            _unread = true;
        }

        public unsafe ReadOnlySpan<byte> Read()
        {
            if (_pinned == IntPtr.Zero) throw new ObjectDisposedException(nameof(CudaDeviceReadback));
            if (_event is null) throw new InvalidOperationException("Nothing was queued for readback.");
            using (_backend.PushContext())
                _event.Synchronize();
            _unread = false;
            return new ReadOnlySpan<byte>((void*)_pinned, _bytes);
        }

        public void Dispose()
        {
            if (_pinned == IntPtr.Zero) return;
            if (!_backend.IsAvailable)
            {
                // The backend (and its context) went first: its queue is gone, so no copy can still be writing, and a
                // driver call now would touch a dead context. The page-locked bytes go back with the context.
                _event = null;
                _pinned = IntPtr.Zero;
                return;
            }
            using (_backend.PushContext())
            {
                // The copy may still be writing the pinned bytes; they are freed only once it is done.
                if (_event is { } evt && _unread) evt.Synchronize();
                _event?.Dispose();
                _event = null;
                CuBlasNative.cuMemFreeHost(_pinned);
            }
            _pinned = IntPtr.Zero;
        }
    }
}
