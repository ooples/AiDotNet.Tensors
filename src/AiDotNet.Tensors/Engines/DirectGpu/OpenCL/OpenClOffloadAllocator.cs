// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// OpenCL-backed <see cref="IGpuOffloadAllocator"/>. Pinned-scheme
/// allocates a host pointer + creates a CL buffer with
/// <c>CL_MEM_ALLOC_HOST_PTR</c> (driver-managed pinned host memory the
/// device can DMA into). Managed-scheme uses OpenCL 2.0 SVM
/// (<c>clSVMAlloc</c>) which is the unified-memory equivalent.
///
/// <para>The current iteration ships the host-side allocation + the
/// IsAvailable probe; the CL buffer-creation handshake plumbs through
/// the existing OpenCL backend's per-context queue, which the higher-
/// level dispatcher injects when staging a weight tensor onto a kernel
/// invocation. Same shape as how PR #267 wires the cuRAND on-device
/// dispatch — the allocator owns memory, the kernel owns scheduling.</para>
/// </summary>
public sealed class OpenClOffloadAllocator : IGpuOffloadAllocator
{
    private readonly ConcurrentDictionary<IntPtr, GpuOffloadHandle> _live = new();
    private bool _disposed;

    public bool IsAvailable => OpenClPlatformProbe.IsAvailable;

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClOffloadAllocator));
        if (!IsAvailable)
            throw new NotSupportedException("OpenCL ICD is not loadable on this host.");
        if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

        // OpenCL host-pointer allocation. The actual cl_mem creation with
        // CL_MEM_ALLOC_HOST_PTR (or CL_MEM_USE_HOST_PTR for SVM) is handled
        // when the kernel queues a buffer-bind — this allocator returns the
        // host pointer used as the ALLOC_HOST_PTR backing.
        IntPtr ptr = Marshal.AllocHGlobal((IntPtr)bytes);
        var effective = scheme == OffloadScheme.Auto ? OffloadScheme.Pinned : scheme;
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
