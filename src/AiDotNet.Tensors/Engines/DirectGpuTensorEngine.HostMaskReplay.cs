using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// The device copy (FP32 0/1, the layout the masked-fill kernels read) of a host boolean mask, such as an attention
    /// layer's causal mask. Uploading the mask per call is a synchronous host-to-device copy, which aborts a CUDA graph
    /// capture, and a captured graph would read a buffer freed after the call. The copy is made once outside a capture,
    /// read through a stable pointer, and re-uploaded only when the host mask changes (before a replay, too).
    /// </summary>
    private sealed class HostMaskDeviceCopy
    {
        internal IGpuBuffer? Buffer;
        // The backend the buffer belongs to: a copy is never read through another backend.
        internal IDirectGpuBackend? Backend;
        internal int Capacity;
        internal object? Storage;
        internal int Version;
    }

    // Keyed weakly by the host mask: a copy (and, through its buffer's finalizer, its device memory) lives as long as
    // the mask - the plan node that captured it.
    private readonly System.Runtime.CompilerServices.ConditionalWeakTable<Tensor<bool>, HostMaskDeviceCopy> _hostMaskCopies = new();
    private readonly object _hostMaskLock = new();
    // Every copy made, weakly, so the engine's disposal frees the device memory of those still alive: the table above
    // cannot be enumerated on every target, and the copies otherwise outlived the engine until their masks were collected.
    private readonly List<WeakReference<HostMaskDeviceCopy>> _hostMaskCopyList = new();

    /// <summary>Frees every live host-mask device copy (engine disposal).</summary>
    private void ReleaseHostMaskDeviceCopies()
    {
        lock (_hostMaskLock)
        {
            foreach (var weak in _hostMaskCopyList)
            {
                if (!weak.TryGetTarget(out var copy)) continue;
                lock (copy)
                {
                    var buffer = copy.Buffer;
                    var owner = copy.Backend;
                    copy.Buffer = null;
                    copy.Backend = null;
                    if (buffer is not null) ReleaseHostMaskBuffer(owner, buffer);
                }
            }
            _hostMaskCopyList.Clear();
        }
    }

    /// <summary>Device copies of host masks this engine holds (tests).</summary>
    internal int LiveHostMaskDeviceCopiesForTests
    {
        get
        {
            lock (_hostMaskLock)
            {
                int live = 0;
                foreach (var weak in _hostMaskCopyList)
                    if (weak.TryGetTarget(out var copy) && copy.Buffer is not null) live++;
                return live;
            }
        }
    }

    /// <summary>
    /// True for a mask that lives on the host only (no device buffer of its own): the masks whose per-call upload
    /// <see cref="GetHostMaskDeviceCopy"/> replaces.
    /// </summary>
    private static bool IsHostOnlyMask(Tensor<bool> mask)
        => mask.IsContiguous && mask._storageOffset == 0 && mask._gpuBuffer is null
           && !Helpers.HostSync.IsPending(mask.DataVector);

    private IGpuBuffer GetHostMaskDeviceCopy(IDirectGpuBackend backend, Tensor<bool> mask)
    {
        HostMaskDeviceCopy copy;
        lock (_hostMaskLock)
        {
            if (_hostMaskCopies.TryGetValue(mask, out var known) && known is not null) copy = known;
            else
            {
                // A copy first made inside a capture would be an upload in the captured stream. Refuse, so the capture
                // fails and the step runs uncaptured; the eager steps before a capture make the copies it reads.
                if (backend is IGpuGraphCapture capture && capture.IsStreamCapturing())
                    throw new NotSupportedException(
                        "A host mask's device copy cannot be made inside a graph capture: run the step uncaptured first.");
                copy = new HostMaskDeviceCopy();
                _hostMaskCopies.Add(mask, copy);
                _hostMaskCopyList.RemoveAll(w => !w.TryGetTarget(out _));
                _hostMaskCopyList.Add(new WeakReference<HostMaskDeviceCopy>(copy));
            }
        }
        RefreshHostMaskDeviceCopy(backend, copy, mask);
        RecordHostMaskBinding(copy, mask);
        return copy.Buffer ?? throw new InvalidOperationException("A host mask's device copy was not allocated.");
    }

    private static void RefreshHostMaskDeviceCopy(IDirectGpuBackend backend, HostMaskDeviceCopy copy, Tensor<bool> mask)
    {
        int n = mask.Length;
        object storage = mask.StorageIdentity;
        int version = mask.GpuCacheVersion;
        lock (copy)
        {
            if (copy.Buffer is not null && ReferenceEquals(copy.Backend, backend) && copy.Capacity == n
                && ReferenceEquals(copy.Storage, storage) && copy.Version == version) return;
            if (backend is IGpuGraphCapture capture && capture.IsStreamCapturing())
                throw new NotSupportedException("A host mask changed inside a graph capture; its device copy cannot be refreshed there.");
            if (copy.Buffer is not null && !ReferenceEquals(copy.Backend, backend))
            {
                // Made on another backend: its buffer is that context's memory, possibly already closed, so it is
                // disposed directly (which checks for a live context itself). A fresh copy is made on this one.
                var stale = copy.Buffer;
                copy.Buffer = null;
                copy.Backend = null;
                stale.Dispose();
            }
            if (copy.Buffer is not null && copy.Capacity != n)
                throw new InvalidOperationException("A host mask's device copy cannot be resized.");
            var values = new float[Math.Max(n, 1)];
            using (var lease = mask.Lease())
            {
                var span = lease.Span;
                for (int i = 0; i < n; i++) values[i] = span[i] ? 1f : 0f;
            }
            if (copy.Buffer is null)
            {
                copy.Buffer = backend.AllocateBuffer(values);
                copy.Backend = backend;
                copy.Capacity = n;
            }
            else backend.UploadBufferInPlace(values, copy.Buffer);
            copy.Storage = storage;
            copy.Version = version;
        }
    }

    private void RecordHostMaskBinding(HostMaskDeviceCopy copy, Tensor<bool> mask)
    {
        var bindings = t_indexReplayCapture;
        if (bindings is null || !ReferenceEquals(bindings.Engine, this)) return;
        foreach (var binding in bindings.Masks)
            if (ReferenceEquals(binding.Copy, copy)) return;
        if (copy.Buffer is not { } buffer) throw new InvalidOperationException("A host mask's device copy was not allocated.");
        bindings.Masks.Add((copy, mask, buffer.Handle));
    }

    // Before a replay (outside the capture): a mask the caller rewrote since the capture is uploaded into the buffer the
    // graph baked.
    private static void RefreshMaskGraphBindings(IDirectGpuBackend backend, List<(HostMaskDeviceCopy Copy, Tensor<bool> Source, IntPtr Handle)> masks)
    {
        foreach (var (copy, source, handle) in masks)
        {
            if (copy.Buffer?.Handle != handle || handle == IntPtr.Zero)
                throw new InvalidOperationException("A captured host-mask device buffer was released or replaced.");
            RefreshHostMaskDeviceCopy(backend, copy, source);
        }
    }

    // A host-mask copy's buffer: freed after the work queued on its backend while that backend is live and this engine's
    // own; otherwise (another engine's backend, possibly closed, or this one closing) disposed directly, which checks for
    // a live context itself. The fields are cleared before this runs, so a failure cannot leave a freed buffer behind.
    private void ReleaseHostMaskBuffer(IDirectGpuBackend? owner, IGpuBuffer buffer)
    {
        if (owner is not null && !IsDisposed && _directGpu?.Backend is { } own
            && ReferenceEquals(CrossBackendTensors.Unwrap(own), CrossBackendTensors.Unwrap(owner)))
        {
            try { DisposeBufferAfterQueuedUse(owner, buffer); return; }
            catch (ObjectDisposedException) { /* closed meanwhile: dispose directly below */ }
        }
        buffer.Dispose();
    }
}
