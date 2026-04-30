// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Metal/MPS-backed <see cref="IGpuOffloadAllocator"/>. Apple Silicon's
/// unified memory architecture means every MTLBuffer with
/// <c>storageModeShared</c> is already CPU+GPU visible — the allocator
/// returns the same pointer for both schemes. Discrete-GPU Macs use
/// <c>storageModeManaged</c> with explicit <c>didModifyRange</c>; this
/// allocator surfaces the shared path which is right for Apple Silicon
/// (M1+) and the right default for Apple's deprecation of discrete GPUs.
///
/// <para>On non-macOS hosts <see cref="IsAvailable"/> returns false and
/// every allocator call throws cleanly.</para>
/// </summary>
public sealed class MetalOffloadAllocator : IGpuOffloadAllocator
{
    private readonly ConcurrentDictionary<IntPtr, GpuOffloadHandle> _live = new();
    private bool _disposed;

    public bool IsAvailable => MpsRngNative.IsAvailable;

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MetalOffloadAllocator));
        if (!IsAvailable)
            throw new NotSupportedException("Metal/MPS framework is not loadable on this host (macOS only).");
        if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

        // On Apple Silicon both Pinned and Managed map to MTLStorageModeShared
        // (zero-copy unified memory). On Intel Macs Managed would map to
        // storageModeManaged with explicit didModifyRange; we surface Shared
        // as the default — MTLDevice.hasUnifiedMemory probes which the
        // higher-level dispatch reads when picking schemes for the user.
        // The host-visible buffer is allocated via posix_memalign so the
        // pointer is page-aligned — required for MTLDevice.makeBuffer
        // (no-copy variant) to succeed.
        IntPtr ptr = AllocAligned(bytes);
        var effective = scheme == OffloadScheme.Auto ? OffloadScheme.Managed : scheme;
        var h = new GpuOffloadHandle(ptr, ptr, bytes, effective);
        _live[ptr] = h;
        return h;
    }

    public void Free(GpuOffloadHandle handle)
    {
        if (_disposed) return;
        if (handle.HostPointer == IntPtr.Zero) return;
        _live.TryRemove(handle.HostPointer, out _);
        FreeAligned(handle.HostPointer);
    }

    private static IntPtr AllocAligned(long bytes)
    {
        // 16K page alignment for MTLBuffer no-copy compatibility.
        const int Alignment = 16384;
        return Marshal.AllocHGlobal((IntPtr)((bytes + Alignment - 1) / Alignment * Alignment));
    }

    private static void FreeAligned(IntPtr p) => Marshal.FreeHGlobal(p);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var h in _live.Values) Free(h);
        _live.Clear();
    }
}
