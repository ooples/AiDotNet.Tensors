// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #1520: a blit into a landing buffer, committed behind the work already submitted and not waited for; the host waits
// for that command buffer only on Read. A training step that defers its loss or finiteness verdict never drains the
// queue itself.
public sealed partial class MetalBackend
{
    private sealed class MetalDeviceReadback : IDeviceReadback
    {
        private readonly MetalBackend _backend;
        private readonly int _capacity;
        private readonly byte[] _host;
        private MetalGpuBuffer? _landing;
        private IntPtr _commandBuffer;
        private int _bytes;
        private bool _queued;
        private bool _unread;

        public MetalDeviceReadback(MetalBackend backend, int byteCapacity)
        {
            if (byteCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(byteCapacity));
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _capacity = byteCapacity;
            _host = new byte[byteCapacity];
            _landing = (MetalGpuBuffer)backend.AllocateBuffer((byteCapacity + sizeof(float) - 1) / sizeof(float));
        }

        public bool HasUnreadValue => _unread;

        public void Enqueue(IGpuBuffer source, int byteCount)
        {
            var landing = _landing ?? throw new ObjectDisposedException(nameof(MetalDeviceReadback));
            if (source is null) throw new ArgumentNullException(nameof(source));
            if (byteCount <= 0 || byteCount > _capacity) throw new ArgumentOutOfRangeException(nameof(byteCount));
            if (source is not MetalGpuBuffer buffer)
                throw new ArgumentException("The readback reads this backend's float buffers.", nameof(source));
            if ((long)buffer.Size * sizeof(float) < byteCount)
                throw new ArgumentOutOfRangeException(nameof(byteCount),
                    $"The buffer holds {buffer.Size} floats, fewer than {byteCount} bytes.");
            // A copy still in flight writes the same landing bytes: let it finish before queueing the next.
            Complete();
            var queue = _backend._commandQueue;
            var commandBuffer = queue.CreateCommandBuffer();
            if (commandBuffer == IntPtr.Zero) throw new InvalidOperationException("Failed to create a Metal command buffer.");
            var blit = queue.CreateBlitEncoder(commandBuffer);
            if (blit == IntPtr.Zero)
            {
                queue.ReturnCommandBuffer(commandBuffer);
                throw new InvalidOperationException("Failed to create a Metal blit encoder.");
            }
            MetalCommandQueue.CopyBuffer(blit, buffer.Handle, (ulong)buffer.ByteOffset, landing.Handle,
                (ulong)landing.ByteOffset, (ulong)byteCount);
            MetalCommandQueue.EndEncoding(blit);
            MetalCommandQueue.Commit(commandBuffer);
            _commandBuffer = commandBuffer;
            _bytes = byteCount;
            _queued = true;
            _unread = true;
        }

        public ReadOnlySpan<byte> Read()
        {
            var landing = _landing ?? throw new ObjectDisposedException(nameof(MetalDeviceReadback));
            if (!_queued) throw new InvalidOperationException("Nothing was queued for readback.");
            Complete();
            landing.CopyBytesTo(_host, 0, _bytes);
            _unread = false;
            return new ReadOnlySpan<byte>(_host, 0, _bytes);
        }

        // Waits for the queued blit and hands its command buffer back; a failed one surfaces here.
        private void Complete()
        {
            var commandBuffer = _commandBuffer;
            if (commandBuffer == IntPtr.Zero) return;
            _commandBuffer = IntPtr.Zero;
            try
            {
                MetalCommandQueue.WaitUntilCompleted(commandBuffer);
                if (MetalCommandQueue.GetError(commandBuffer) is { } error)
                    throw new InvalidOperationException($"The Metal readback copy failed: {error}");
            }
            finally
            {
                _backend._commandQueue.ReturnCommandBuffer(commandBuffer);
            }
        }

        public void Dispose()
        {
            var landing = _landing;
            if (landing is null) return;
            _landing = null;
            // The blit may still be writing the landing buffer; it is released once the copy is done.
            if (_backend.IsAvailable) Complete();
            landing.Dispose();
        }
    }
}
