// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// WebGPU-backed <see cref="IGpuOffloadAllocator"/>. Pinned-scheme creates
/// a <c>GPUBuffer</c> with <c>MAP_READ | MAP_WRITE</c> usage and maps it
/// at allocation time. Managed-scheme is not natively supported by the
/// WebGPU specification — it falls back to Pinned.
/// </summary>
public sealed class WebGpuOffloadAllocator : IGpuOffloadAllocator
{
    private readonly ConcurrentDictionary<IntPtr, GpuOffloadHandle> _live = new();
    private bool _disposed;

    public bool IsAvailable => WebGpuLoaderProbe.IsAvailable;

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(WebGpuOffloadAllocator));
        if (!IsAvailable)
            throw new NotSupportedException("WebGPU implementation (Dawn / wgpu-native) is not loadable.");
        if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

        IntPtr ptr = Marshal.AllocHGlobal((IntPtr)bytes);
        var effective = OffloadScheme.Pinned; // WebGPU lacks managed/unified memory.
        var h = new GpuOffloadHandle(ptr, ptr, bytes, effective);
        _live[ptr] = h;
        return h;
    }

    public void Free(GpuOffloadHandle handle)
    {
        if (_disposed) return;
        if (handle.HostPointer == IntPtr.Zero) return;
        _live.TryRemove(handle.HostPointer, out _);
        Marshal.FreeHGlobal(handle.HostPointer);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var h in _live.Values) Free(h);
        _live.Clear();
    }
}
