// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #1521: a device-to-host copy recorded behind the work already submitted and submitted without waiting; the host waits
// only for that submission, on Read. A training step that defers its loss or finiteness verdict never drains the
// queue itself.
public sealed unsafe partial class VulkanBackend
{
    private sealed class VulkanDeviceReadback : IDeviceReadback
    {
        private readonly VulkanBackend _backend;
        private readonly int _capacity;
        private readonly byte[] _host;
        private VulkanGpuBuffer? _landing;
        private long _serial;
        private int _bytes;
        private bool _queued;
        private bool _unread;

        public VulkanDeviceReadback(VulkanBackend backend, int byteCapacity)
        {
            if (byteCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(byteCapacity));
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _capacity = byteCapacity;
            _host = new byte[byteCapacity];
            _landing = (VulkanGpuBuffer)backend.AllocateBuffer((byteCapacity + sizeof(float) - 1) / sizeof(float));
        }

        public bool HasUnreadValue => _unread;

        public void Enqueue(IGpuBuffer source, int byteCount)
        {
            var landing = _landing ?? throw new ObjectDisposedException(nameof(VulkanDeviceReadback));
            if (source is null) throw new ArgumentNullException(nameof(source));
            if (byteCount <= 0 || byteCount > _capacity) throw new ArgumentOutOfRangeException(nameof(byteCount));
            var buffer = AsVulkan(source);
            if ((long)buffer.Size * sizeof(float) < byteCount)
                throw new ArgumentOutOfRangeException(nameof(byteCount),
                    $"The buffer holds {buffer.Size} floats, fewer than {byteCount} bytes.");
            // The previous copy into the same staging bytes is ordered ahead of this one on the queue.
            _serial = _backend._device.SubmitCopyToHost(buffer.Storage, landing.Staging, (ulong)byteCount);
            _bytes = byteCount;
            _queued = true;
            _unread = true;
        }

        public ReadOnlySpan<byte> Read()
        {
            var landing = _landing ?? throw new ObjectDisposedException(nameof(VulkanDeviceReadback));
            if (!_queued) throw new InvalidOperationException("Nothing was queued for readback.");
            _backend._device.WaitForSubmission(_serial);
            landing.Staging.ReadRawBytes(_host);
            _unread = false;
            return new ReadOnlySpan<byte>(_host, 0, _bytes);
        }

        public void Dispose()
        {
            var landing = _landing;
            if (landing is null) return;
            _landing = null;
            // The copy may still be writing the staging memory; the buffer is released once it has executed.
            if (_queued && _backend.IsAvailable) _backend._device.WaitForSubmission(_serial);
            landing.Dispose();
        }
    }
}
