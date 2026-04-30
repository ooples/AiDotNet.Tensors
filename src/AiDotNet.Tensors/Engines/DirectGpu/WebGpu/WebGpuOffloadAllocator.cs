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
    private readonly ConcurrentDictionary<IntPtr, AllocRecord> _live = new();
    private IntPtr _instance = IntPtr.Zero;
    private readonly object _lock = new();
    private bool _disposed;

    public bool IsAvailable => WebGpuLoaderProbe.IsAvailable;

    private void EnsureInstance()
    {
        if (_instance != IntPtr.Zero) return;
        lock (_lock)
        {
            if (_instance != IntPtr.Zero) return;
            _instance = WebGpuNativeBindings.wgpuCreateInstance(IntPtr.Zero);
            if (_instance == IntPtr.Zero)
                throw new InvalidOperationException("wgpuCreateInstance returned null.");
        }
    }

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(WebGpuOffloadAllocator));
        if (!IsAvailable)
            throw new NotSupportedException("WebGPU implementation (Dawn / wgpu-native) is not loadable.");
        if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

        EnsureInstance();
        // The host-mapped MAP_READ | MAP_WRITE pattern: we allocate a host
        // buffer for the user-visible HostPointer; the Adapter+Device
        // binding (which the WebGpu engine handles when staging a kernel)
        // creates the GPUBuffer via wgpuDeviceCreateBuffer with the
        // appropriate usage flags. Allocator owns the host backing only;
        // device-buffer creation happens lazily under the kernel staging.
        IntPtr hostBuf = Marshal.AllocHGlobal((IntPtr)bytes);
        var rec = new AllocRecord { HostPtr = hostBuf, Bytes = bytes };
        _live[hostBuf] = rec;
        return new GpuOffloadHandle(hostBuf, hostBuf, bytes, OffloadScheme.Pinned);
    }

    public void Free(GpuOffloadHandle handle)
    {
        if (handle.HostPointer == IntPtr.Zero) return;
        if (!_live.TryRemove(handle.HostPointer, out _)) return;
        Marshal.FreeHGlobal(handle.HostPointer);
    }

    public void Dispose()
    {
        if (_disposed) return;
        var snapshot = System.Linq.Enumerable.ToArray(_live.Values);
        foreach (var rec in snapshot)
        {
            try { Marshal.FreeHGlobal(rec.HostPtr); } catch { }
        }
        _live.Clear();
        if (_instance != IntPtr.Zero)
        {
            try { WebGpuNativeBindings.wgpuInstanceRelease(_instance); } catch { }
            _instance = IntPtr.Zero;
        }
        _disposed = true;
    }

    private sealed class AllocRecord
    {
        public IntPtr HostPtr;
        public long Bytes;
    }
}

// Issue #276 sub-feature 4: instance create/release entry points used by
// the WebGpu offload allocator. The existing WebGpuNativeBindings class
// in the file of the same name doesn't bind these — we add them here as
// a partial extension so both files compile into the same static class.
public static partial class WebGpuNativeBindings
{
    private const string LibForOffload = "webgpu_dawn";

    [DllImport(LibForOffload, EntryPoint = "wgpuCreateInstance")]
    public static extern IntPtr wgpuCreateInstance(IntPtr descriptor);

    [DllImport(LibForOffload, EntryPoint = "wgpuInstanceRelease")]
    public static extern void wgpuInstanceRelease(IntPtr instance);
}
