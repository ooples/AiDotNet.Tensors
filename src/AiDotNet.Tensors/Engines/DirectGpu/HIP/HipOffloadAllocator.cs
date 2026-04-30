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

    public void Free(GpuOffloadHandle handle)
    {
        if (_disposed) return;
        if (handle.HostPointer == IntPtr.Zero) return;
        _live.TryRemove(handle.HostPointer, out _);
        switch (handle.Scheme)
        {
            case OffloadScheme.Pinned: HipNativeBindings.hipHostFree(handle.HostPointer); break;
            case OffloadScheme.Managed: HipNativeBindings.hipFree(handle.DevicePointer); break;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var h in _live.Values) Free(h);
        _live.Clear();
    }
}
