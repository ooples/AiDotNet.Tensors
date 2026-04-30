// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
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
        if (_disposed) return;
        if (handle.HostPointer == IntPtr.Zero) return;
        _live.TryRemove(handle.HostPointer, out _);
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
        _disposed = true;
        foreach (var h in _live.Values) Free(h);
        _live.Clear();
    }
}
