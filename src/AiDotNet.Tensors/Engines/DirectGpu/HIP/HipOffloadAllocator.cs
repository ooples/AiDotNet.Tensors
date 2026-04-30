// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// HIP/ROCm-backed <see cref="IGpuOffloadAllocator"/>. AMD-side mirror of
/// <see cref="CUDA.CudaOffloadAllocator"/>. Pinned-scheme uses
/// <c>hipHostMalloc(Portable | Mapped)</c>; Managed-scheme uses
/// <c>hipMallocManaged</c>.
/// </summary>
public sealed class HipOffloadAllocator : IGpuOffloadAllocator
{
    private readonly ConcurrentDictionary<IntPtr, GpuOffloadHandle> _live = new();
    private readonly object _lifecycleLock = new();
    private bool _disposed;

    public bool IsAvailable
    {
        get
        {
            try
            {
                int n = 0;
                return HipNativeBindings.hipGetDeviceCount(ref n) == HipError.Success && n > 0;
            }
            catch { return false; }
        }
    }

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        // Lock across allocate+register so a concurrent Dispose cannot
        // snapshot _live, clear it, and let this allocation slip in
        // afterwards (which would leak the HIP allocation).
        lock (_lifecycleLock)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HipOffloadAllocator));
            if (!IsAvailable)
                throw new NotSupportedException("HIP runtime is not loadable on this host.");
            if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

            IntPtr ptr = IntPtr.Zero;
            OffloadScheme effective = scheme == OffloadScheme.Auto ? OffloadScheme.Pinned : scheme;
            switch (effective)
            {
                case OffloadScheme.Pinned:
                    {
                        var rc = HipNativeBindings.hipHostMalloc(ref ptr, (UIntPtr)bytes,
                            HipNativeBindings.HIP_HOST_MALLOC_PORTABLE | HipNativeBindings.HIP_HOST_MALLOC_MAPPED);
                        if (rc != HipError.Success)
                            throw new InvalidOperationException($"hipHostMalloc returned {rc}");
                        break;
                    }
                case OffloadScheme.Managed:
                    {
                        var rc = HipNativeBindings.hipMallocManaged(ref ptr, (UIntPtr)bytes, HipNativeBindings.HIP_MEM_ATTACH_GLOBAL);
                        if (rc != HipError.Success)
                            throw new InvalidOperationException($"hipMallocManaged returned {rc}");
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
        // Native free only for handles we own — a foreign handle or
        // double-free would corrupt the HIP heap on second release.
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
            case OffloadScheme.Pinned: HipNativeBindings.hipHostFree(handle.HostPointer); break;
            case OffloadScheme.Managed: HipNativeBindings.hipFree(handle.DevicePointer); break;
        }
    }

    public void Dispose()
    {
        GpuOffloadHandle[] snapshot;
        lock (_lifecycleLock)
        {
            if (_disposed) return;
            // Flip _disposed under the lock so any racing Allocate observes
            // the flip (or blocks waiting on the lock and then sees it on
            // entry) before we snapshot — no allocations can sneak in.
            _disposed = true;
            snapshot = System.Linq.Enumerable.ToArray(_live.Values);
            _live.Clear();
        }
        // Native frees outside the lock so a HIP backend that internally
        // serializes during free can't deadlock against our lifecycle lock.
        foreach (var h in snapshot) FreeNative(h);
    }
}
