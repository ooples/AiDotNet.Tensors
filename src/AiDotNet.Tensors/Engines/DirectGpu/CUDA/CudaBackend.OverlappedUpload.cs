using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Host-to-device refreshes overlapped with the running step (see IOverlappedUploadBackend). Measured on the parity
// Transformer: the per-step input upload (256 KB) queued on the compute stream behind the previous step left the device
// idle ~135 us around a 41 us copy, between every two graph replays.
public sealed partial class CudaBackend : IOverlappedUploadBackend
{
    // Two slots per destination, used alternately, so the host fills one while the other may still be in flight.
    private sealed class UploadSlot
    {
        public IntPtr Pinned;
        public IGpuBuffer? Staging;
        public IntPtr Uploaded;   // recorded on the copy stream after the host-to-device copy
        public IntPtr Consumed;   // recorded on the compute stream after the device-to-device copy out of Staging
        public bool UploadedRecorded;
        public bool ConsumedRecorded;
    }

    private sealed class UploadTarget
    {
        public readonly UploadSlot[] Slots = { new UploadSlot(), new UploadSlot() };
        public int Next;
        public int Floats;
    }

    private readonly object _overlappedUploadLock = new object();
    private readonly Dictionary<IntPtr, UploadTarget> _uploadTargets = new Dictionary<IntPtr, UploadTarget>();
    private IntPtr _uploadStream;

    bool IOverlappedUploadBackend.TryUploadOverlapped(float[] data, IGpuBuffer destination)
    {
        if (data is null || destination is null || !IsAvailable || _backendStreamCaptureActive || IsStreamCapturing())
            return false;
        int floats = data.Length;
        if (floats <= 0 || destination.Handle == IntPtr.Zero || destination.Size < floats) return false;
        ulong bytes = (ulong)floats * sizeof(float);
        using var _ = PushContext();
        lock (_overlappedUploadLock)
        {
            if (_uploadStream == IntPtr.Zero)
                CuBlasNative.CheckCudaResult(
                    CudaNativeBindings.cuStreamCreate(out _uploadStream, CudaNativeBindings.CU_STREAM_NON_BLOCKING),
                    "cuStreamCreate(upload)");
            if (!_uploadTargets.TryGetValue(destination.Handle, out var target) || target.Floats != floats)
            {
                if (target is not null) ReleaseUploadTarget(target);
                target = new UploadTarget { Floats = floats };
                _uploadTargets[destination.Handle] = target;
            }
            var slot = target.Slots[target.Next];
            target.Next ^= 1;
            if (slot.Pinned == IntPtr.Zero)
            {
                CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAllocHost(out slot.Pinned, bytes), "cuMemAllocHost(upload)");
                slot.Staging = AllocateBuffer(floats);
                CuBlasNative.CheckCudaResult(
                    CudaNativeBindings.cuEventCreate(out slot.Uploaded, CudaNativeBindings.CU_EVENT_DISABLE_TIMING), "cuEventCreate(uploaded)");
                CuBlasNative.CheckCudaResult(
                    CudaNativeBindings.cuEventCreate(out slot.Consumed, CudaNativeBindings.CU_EVENT_DISABLE_TIMING), "cuEventCreate(consumed)");
                // The staging buffer is a stream-ordered allocation (and zero-fill) on the compute stream: the copy
                // stream may write it only after that. Without this the first upload into a slot raced the fill and
                // lost (the parity Transformer's first replay trained on partly zeroed labels).
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventRecord(slot.Consumed, _stream), "cuEventRecord(allocated)");
                slot.ConsumedRecorded = true;
            }
            var staging = slot.Staging ?? throw new InvalidOperationException("Upload slot has no staging buffer.");

            // The slot's previous host-to-device copy may still be reading its pinned bytes.
            if (slot.UploadedRecorded && CudaNativeBindings.cuEventQuery(slot.Uploaded) != CudaResult.Success)
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_uploadStream), "cuStreamSynchronize(upload)");
            unsafe
            {
                fixed (float* src = data)
                    Buffer.MemoryCopy(src, (void*)slot.Pinned, (long)bytes, (long)bytes);
            }

            // Copy stream: once the compute stream has copied the slot's previous contents out, refill it.
            if (slot.ConsumedRecorded)
                CuBlasNative.CheckCudaResult(
                    CudaNativeBindings.cuStreamWaitEvent(_uploadStream, slot.Consumed, 0), "cuStreamWaitEvent(consumed)");
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuMemcpyHtoDAsync(staging.Handle, slot.Pinned, bytes, _uploadStream), "cuMemcpyHtoDAsync(upload)");
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventRecord(slot.Uploaded, _uploadStream), "cuEventRecord(uploaded)");
            slot.UploadedRecorded = true;

            // Compute stream: everything queued after this sees the new contents.
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuStreamWaitEvent(_stream, slot.Uploaded, 0), "cuStreamWaitEvent(uploaded)");
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuMemcpyDtoDAsync(destination.Handle, staging.Handle, bytes, _stream), "cuMemcpyDtoDAsync(upload)");
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventRecord(slot.Consumed, _stream), "cuEventRecord(consumed)");
            slot.ConsumedRecorded = true;
            return true;
        }
    }

    private void ReleaseUploadTarget(UploadTarget target)
    {
        // Both streams must be past the slots before their memory goes.
        if (_uploadStream != IntPtr.Zero) CudaNativeBindings.cuStreamSynchronize(_uploadStream);
        CudaNativeBindings.cuStreamSynchronize(_stream);
        foreach (var slot in target.Slots)
        {
            if (slot.Pinned != IntPtr.Zero) CuBlasNative.cuMemFreeHost(slot.Pinned);
            slot.Staging?.Dispose();
            if (slot.Uploaded != IntPtr.Zero) CudaNativeBindings.cuEventDestroy(slot.Uploaded);
            if (slot.Consumed != IntPtr.Zero) CudaNativeBindings.cuEventDestroy(slot.Consumed);
            slot.Pinned = IntPtr.Zero; slot.Staging = null; slot.Uploaded = IntPtr.Zero; slot.Consumed = IntPtr.Zero;
            slot.UploadedRecorded = false; slot.ConsumedRecorded = false;
        }
    }

    // Called from Dispose while the context is still alive.
    private void DisposeOverlappedUploads()
    {
        lock (_overlappedUploadLock)
        {
            if (_uploadTargets.Count == 0 && _uploadStream == IntPtr.Zero) return;
            using var _ = PushContext();
            foreach (var target in _uploadTargets.Values)
            {
                try { ReleaseUploadTarget(target); } catch { }
            }
            _uploadTargets.Clear();
            if (_uploadStream != IntPtr.Zero)
            {
                try { CudaNativeBindings.cuStreamDestroy(_uploadStream); } catch { }
                _uploadStream = IntPtr.Zero;
            }
        }
    }
}
