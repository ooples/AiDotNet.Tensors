// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Vulkan-backed <see cref="IGpuOffloadAllocator"/>. Pinned-scheme uses
/// <c>HOST_VISIBLE | DEVICE_LOCAL</c> memory (the "smart access memory"
/// path supported by AMD ReBAR and NVIDIA RTX 3000+ on PCIe 4.0).
/// Managed-scheme uses <c>HOST_COHERENT | HOST_VISIBLE</c> + driver
/// migration on commit.
///
/// <para>Vulkan's vkAllocateMemory + vkMapMemory pair gives the same
/// host-pointer + device-mapping contract as CUDA's pinned host. This
/// allocator returns the host pointer; the Vulkan dispatcher creates
/// the VkBuffer + binds the memory when a kernel reads the weight.</para>
/// </summary>
public sealed class VulkanOffloadAllocator : IGpuOffloadAllocator
{
    private readonly ConcurrentDictionary<IntPtr, GpuOffloadHandle> _live = new();
    private bool _disposed;

    public bool IsAvailable => VulkanLoaderProbe.IsAvailable;

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VulkanOffloadAllocator));
        if (!IsAvailable)
            throw new NotSupportedException("Vulkan loader is not registered on this host.");
        if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

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
