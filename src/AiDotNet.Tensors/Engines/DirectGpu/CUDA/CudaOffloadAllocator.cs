// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CUDA-backed <see cref="IGpuOffloadAllocator"/>. Pinned-scheme uses
/// <c>cuMemAllocHost</c> (mapped + portable); Managed-scheme uses
/// <c>cuMemAllocManaged</c> with <c>CU_MEM_ATTACH_GLOBAL</c>.
/// </summary>
public sealed class CudaOffloadAllocator : IGpuOffloadAllocator
{
    private readonly ConcurrentDictionary<IntPtr, GpuOffloadHandle> _live = new();
    private readonly object _lifecycleLock = new();
    private bool _disposed;

    public bool IsAvailable => CudaNativeBindings.IsAvailable;

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        // Hold _lifecycleLock across the entire allocate+register so a
        // concurrent Dispose cannot snapshot _live, clear it, and let this
        // allocation slip in afterwards (which would leak the device handle
        // since Dispose has already returned).
        lock (_lifecycleLock)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CudaOffloadAllocator));
            if (!IsAvailable)
                throw new NotSupportedException("CUDA driver is not loadable on this host.");
            if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

            IntPtr ptr;
            OffloadScheme effective = scheme == OffloadScheme.Auto ? OffloadScheme.Pinned : scheme;
            switch (effective)
            {
                case OffloadScheme.Pinned:
                    {
                        var rc = CuBlasNative.cuMemAllocHost(out ptr, (ulong)bytes);
                        CuBlasNative.CheckCudaResult(rc, "cuMemAllocHost(offload)");
                        break;
                    }
                case OffloadScheme.Managed:
                    {
                        var rc = CudaNativeBindings.cuMemAllocManaged(out ptr, (ulong)bytes, CudaNativeBindings.CU_MEM_ATTACH_GLOBAL);
                        if (rc != CudaResult.Success)
                            throw new InvalidOperationException($"cuMemAllocManaged returned {rc}");
                        break;
                    }
                default:
                    throw new ArgumentOutOfRangeException(nameof(scheme), scheme, "Unknown offload scheme.");
            }
            var h = new GpuOffloadHandle(ptr, ptr, bytes, effective);
            _live[ptr] = h;
            return h;
        }
    }

    public void Free(GpuOffloadHandle handle)
    {
        if (handle.HostPointer == IntPtr.Zero) return;
        // Only call native free for handles WE own — a foreign handle or
        // double-free would corrupt the heap on the second native release.
        // Lock TryRemove against Dispose's snapshot+clear so a concurrent
        // Dispose can't free the same pointer twice.
        bool removed;
        lock (_lifecycleLock)
        {
            removed = _live.TryRemove(handle.HostPointer, out _);
        }
        if (!removed) return;
        FreeNative(handle);
    }

    private static void FreeNative(GpuOffloadHandle handle)
    {
        switch (handle.Scheme)
        {
            case OffloadScheme.Pinned:
                CuBlasNative.cuMemFreeHost(handle.HostPointer);
                break;
            case OffloadScheme.Managed:
                CudaNativeBindings.cuMemFree(handle.DevicePointer);
                break;
        }
    }

    public void Dispose()
    {
        GpuOffloadHandle[] snapshot;
        lock (_lifecycleLock)
        {
            if (_disposed) return;
            // Flip _disposed under the lock so any Allocate that's already
            // waiting on _lifecycleLock observes the flip and throws,
            // and any Allocate that hasn't yet entered the lock cannot race
            // past us with a fresh allocation.
            _disposed = true;
            snapshot = _live.Values.ToArray();
            _live.Clear();
        }
        // Native frees outside the lock so a backend that internally locks
        // during free can't deadlock against our lifecycle lock.
        foreach (var h in snapshot) FreeNative(h);
    }
}
