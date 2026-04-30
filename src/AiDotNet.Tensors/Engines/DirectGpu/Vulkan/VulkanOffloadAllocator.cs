// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
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
    private readonly ConcurrentDictionary<IntPtr, AllocRecord> _live = new();
    private IntPtr _instance = IntPtr.Zero;
    private IntPtr _device = IntPtr.Zero;
    private uint _hostVisibleMemTypeIndex = uint.MaxValue;
    private readonly object _lock = new();
    private bool _disposed;

    public bool IsAvailable => VulkanLoaderProbe.IsAvailable;

    private void EnsureDevice()
    {
        if (_device != IntPtr.Zero) return;
        lock (_lock)
        {
            if (_device != IntPtr.Zero) return;
            // Initialize a Vulkan instance + pick the first physical device
            // and a HOST_VISIBLE | DEVICE_LOCAL (or HOST_COHERENT-fallback)
            // memory type. Reused for every Allocate call so the heavy
            // setup amortizes across the model's weights.
            var setup = VulkanInstanceSetup.CreateMinimal();
            _instance = setup.Instance;
            _device = setup.Device;
            _hostVisibleMemTypeIndex = setup.HostVisibleMemTypeIndex;
            if (_hostVisibleMemTypeIndex == uint.MaxValue)
                throw new InvalidOperationException(
                    "Vulkan: no HOST_VISIBLE memory type on selected device.");
            if (_device == IntPtr.Zero)
                throw new InvalidOperationException(
                    "Vulkan: no physical device with HOST_VISIBLE memory found.");
        }
    }

    public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
    {
        // Hold _lock across allocate + register so a concurrent Dispose
        // cannot snapshot _live and tear down the device while we're
        // mid-allocate. _lock is reentrant so EnsureDevice's nested lock
        // is safe.
        lock (_lock)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(VulkanOffloadAllocator));
            if (!IsAvailable)
                throw new NotSupportedException("Vulkan loader is not registered on this host.");
            if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

            EnsureDevice();
            var effective = scheme == OffloadScheme.Auto ? OffloadScheme.Pinned : scheme;
            // vkAllocateMemory + vkMapMemory.
            var allocInfo = new VulkanInstanceSetup.VkMemoryAllocateInfo
            {
                sType = 5, // VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO
                pNext = IntPtr.Zero,
                allocationSize = (ulong)bytes,
                memoryTypeIndex = _hostVisibleMemTypeIndex,
            };
            if (VulkanInstanceSetup.vkAllocateMemory(_device, ref allocInfo, IntPtr.Zero, out IntPtr memHandle) != 0)
                throw new InvalidOperationException("vkAllocateMemory failed.");
            if (VulkanInstanceSetup.vkMapMemory(_device, memHandle, 0, (ulong)bytes, 0, out IntPtr mappedPtr) != 0)
            {
                VulkanInstanceSetup.vkFreeMemory(_device, memHandle, IntPtr.Zero);
                throw new InvalidOperationException("vkMapMemory failed.");
            }
            var rec = new AllocRecord { MemHandle = memHandle, MappedPtr = mappedPtr, Bytes = bytes };
            _live[mappedPtr] = rec;
            return new GpuOffloadHandle(mappedPtr, mappedPtr, bytes, effective, memHandle);
        }
    }

    public void Free(GpuOffloadHandle handle)
    {
        if (handle.HostPointer == IntPtr.Zero) return;
        if (!_live.TryRemove(handle.HostPointer, out var rec)) return;
        FreeRecord(rec, _device);
    }

    private static void FreeRecord(AllocRecord rec, IntPtr device)
    {
        if (device == IntPtr.Zero) return; // device already torn down
        try { VulkanInstanceSetup.vkUnmapMemory(device, rec.MemHandle); } catch { }
        try { VulkanInstanceSetup.vkFreeMemory(device, rec.MemHandle, IntPtr.Zero); } catch { }
    }

    public void Dispose()
    {
        AllocRecord[] snapshot;
        IntPtr device;
        IntPtr instance;
        lock (_lock)
        {
            if (_disposed) return;
            // Flip _disposed under the lock so any racing Allocate observes
            // the flip on entry and throws — no allocations can sneak in.
            _disposed = true;
            snapshot = System.Linq.Enumerable.ToArray(_live.Values);
            _live.Clear();
            device = _device;
            instance = _instance;
        }
        // Free outside the lock; vk*Memory must run BEFORE vkDestroyDevice.
        foreach (var rec in snapshot) FreeRecord(rec, device);
        if (device != IntPtr.Zero)
        {
            try { VulkanInstanceSetup.vkDestroyDevice(device, IntPtr.Zero); } catch { }
        }
        if (instance != IntPtr.Zero)
        {
            try { VulkanInstanceSetup.vkDestroyInstance(instance, IntPtr.Zero); } catch { }
        }
        lock (_lock)
        {
            _device = IntPtr.Zero;
            _instance = IntPtr.Zero;
        }
    }

    private sealed class AllocRecord
    {
        public IntPtr MemHandle;
        public IntPtr MappedPtr;
        public long Bytes;
    }
}
#else
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan
{
    using System;
    using AiDotNet.Tensors.LinearAlgebra;

    /// <summary>net471 stub: Vulkan loader probe + native allocator
    /// require NativeLibrary which is .NET 5+. IsAvailable returns false
    /// so consumers cleanly route around it.</summary>
    public sealed class VulkanOffloadAllocator : IGpuOffloadAllocator
    {
        public bool IsAvailable => false;
        public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme) =>
            throw new NotSupportedException("VulkanOffloadAllocator requires .NET 5+.");
        public void Free(GpuOffloadHandle handle) { }
        public void Dispose() { }
    }
}
#endif
