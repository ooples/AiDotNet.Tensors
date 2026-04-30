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
    private bool _disposed;

    public bool IsAvailable => CudaNativeBindings.IsAvailable;

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
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

    public void Free(GpuOffloadHandle handle)
    {
        if (handle.HostPointer == IntPtr.Zero) return;
        // Only call native free for handles WE own. A foreign handle (or a
        // double-free) would corrupt the heap on the second native release.
        if (!_live.TryRemove(handle.HostPointer, out _)) return;
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
        if (_disposed) return;
        // Snapshot live entries BEFORE flipping _disposed so Free() still
        // performs the native release. The previous code marked _disposed
        // first which made Free() a no-op and leaked every outstanding
        // CUDA allocation.
        var snapshot = _live.Values.ToArray();
        foreach (var h in snapshot) Free(h);
        _live.Clear();
        _disposed = true;
    }
}
