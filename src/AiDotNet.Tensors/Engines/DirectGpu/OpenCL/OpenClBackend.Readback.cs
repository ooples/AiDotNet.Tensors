// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // #1502: a non-blocking clEnqueueReadBuffer into pinned host memory plus its event, the OpenCL counterpart of
    // CudaBackend's page-locked readback. The copy is queued behind the work already submitted and the host waits only
    // on Read, so a training step that defers its loss or its finiteness verdict never drains the queue itself.
    public sealed partial class OpenClBackend
    {
        private sealed class OpenClDeviceReadback : IDeviceReadback
        {
            private readonly OpenClBackend _backend;
            private readonly int _capacity;
            private readonly byte[] _host;
            private GCHandle _pin;
            private IntPtr _event;
            private int _bytes;
            private bool _queued;
            private bool _unread;

            public OpenClDeviceReadback(OpenClBackend backend, int byteCapacity)
            {
                if (byteCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(byteCapacity));
                _backend = backend ?? throw new ArgumentNullException(nameof(backend));
                _capacity = byteCapacity;
                _host = new byte[byteCapacity];
                _pin = GCHandle.Alloc(_host, GCHandleType.Pinned);
            }

            public bool HasUnreadValue => _unread;

            public void Enqueue(IGpuBuffer source, int byteCount)
            {
                if (!_pin.IsAllocated) throw new ObjectDisposedException(nameof(OpenClDeviceReadback));
                if (source is null) throw new ArgumentNullException(nameof(source));
                if (byteCount <= 0 || byteCount > _capacity) throw new ArgumentOutOfRangeException(nameof(byteCount));
                if (source is not DirectOpenClGpuBuffer buffer)
                    throw new ArgumentException("The readback reads this backend's float buffers.", nameof(source));
                if ((long)buffer.Size * sizeof(float) < byteCount)
                    throw new ArgumentOutOfRangeException(nameof(byteCount),
                        $"The buffer holds {buffer.Size} floats, fewer than {byteCount} bytes.");
                if (_backend._context is null || _backend._defaultStream is not { } stream)
                    throw new InvalidOperationException("The OpenCL backend has no command queue.");

                var memories = DirectOpenClSubmission.GetDirectSubmissionMemories(buffer.Buffer);
                try
                {
                    int err;
                    IntPtr transferEvent;
                    lock (DirectOpenClSubmission.Gate)
                    {
                        using var waits = DirectOpenClSubmission.PrepareLocked(stream.Handle, memories);
                        err = OpenClNativeBindings.EnqueueReadBufferWithEvent(stream.Handle, buffer.Buffer.Handle, 0,
                            UIntPtr.Zero, (UIntPtr)byteCount, _pin.AddrOfPinnedObject(), waits.Count, waits.Pointer,
                            out transferEvent);
                        if (err == OpenClNativeBindings.CL_SUCCESS)
                        {
                            // The copy is queued: its event is kept before CommitLocked, which may flush and throw, so
                            // Dispose still waits for it before unpinning. The previous copy into the same bytes is
                            // ordered ahead of this one on the in-order queue; its event is released, not waited on.
                            ReleaseEvent();
                            _event = transferEvent;
                            try
                            {
                                DirectOpenClSubmission.CommitLocked(stream.Handle, memories);
                            }
                            catch
                            {
                                // The bytes are being overwritten, so there is no value to read until the next Enqueue.
                                _queued = false;
                                _unread = false;
                                throw;
                            }
                        }
                    }
                    if (err != OpenClNativeBindings.CL_SUCCESS)
                        throw new InvalidOperationException($"clEnqueueReadBuffer(readback) failed: {err}");
                    // A driver may hold enqueued commands until a flush: without one, Read would be the first push.
                    int flushError = OpenClNativeBindings.Flush(stream.Handle);
                    if (flushError != OpenClNativeBindings.CL_SUCCESS)
                    {
                        _queued = false;
                        _unread = false;
                        throw new InvalidOperationException($"clFlush(readback) failed: {flushError}");
                    }
                }
                finally
                {
                    DirectOpenClSubmission.ReleaseDirectSubmissionMemories(memories);
                }
                _bytes = byteCount;
                _queued = true;
                _unread = true;
            }

            public ReadOnlySpan<byte> Read()
            {
                if (!_pin.IsAllocated) throw new ObjectDisposedException(nameof(OpenClDeviceReadback));
                if (!_queued) throw new InvalidOperationException("Nothing was queued for readback.");
                if (_event != IntPtr.Zero)
                {
                    // A failed wait keeps the event, so a retry or Dispose still waits before trusting the bytes.
                    int err = OpenClNativeBindings.WaitForEvents(1, new[] { _event });
                    if (err != OpenClNativeBindings.CL_SUCCESS)
                        throw new InvalidOperationException($"clWaitForEvents(readback) failed: {err}");
                    ReleaseEvent();
                }
                _unread = false;
                return new ReadOnlySpan<byte>(_host, 0, _bytes);
            }

            private void ReleaseEvent()
            {
                if (_event == IntPtr.Zero) return;
                OpenClNativeBindings.ReleaseEvent(_event);
                _event = IntPtr.Zero;
            }

            public void Dispose()
            {
                if (!_pin.IsAllocated) return;
                // The copy may still be writing the pinned bytes; they are unpinned only once it is done. A backend
                // that is already gone took its queue with it, so nothing can still be writing.
                if (_event != IntPtr.Zero)
                {
                    if (_backend.IsAvailable) OpenClNativeBindings.WaitForEvents(1, new[] { _event });
                    ReleaseEvent();
                }
                _pin.Free();
            }
        }
    }
}
