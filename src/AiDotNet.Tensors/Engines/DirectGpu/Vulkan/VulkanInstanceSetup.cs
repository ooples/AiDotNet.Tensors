// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Minimal Vulkan setup for the offload allocator. Creates an instance,
/// picks the first physical device with a HOST_VISIBLE memory type, and
/// creates a logical device. The allocator reuses this across every
/// allocation; teardown happens at <see cref="VulkanOffloadAllocator.Dispose"/>.
/// </summary>
internal static class VulkanInstanceSetup
{
    private const string Lib = "vulkan-1";

    [DllImport(Lib)] internal static extern int vkCreateInstance(ref VkInstanceCreateInfo info, IntPtr alloc, out IntPtr instance);
    [DllImport(Lib)] internal static extern void vkDestroyInstance(IntPtr instance, IntPtr alloc);
    [DllImport(Lib)] internal static extern int vkEnumeratePhysicalDevices(IntPtr instance, ref uint count, IntPtr[]? devices);
    [DllImport(Lib)] internal static extern void vkGetPhysicalDeviceMemoryProperties(IntPtr device, out VkPhysicalDeviceMemoryProperties props);
    [DllImport(Lib)] internal static extern int vkCreateDevice(IntPtr physical, ref VkDeviceCreateInfo info, IntPtr alloc, out IntPtr device);
    [DllImport(Lib)] internal static extern void vkDestroyDevice(IntPtr device, IntPtr alloc);
    [DllImport(Lib)] internal static extern void vkGetPhysicalDeviceQueueFamilyProperties(IntPtr device, ref uint count, IntPtr properties);
    [DllImport(Lib)] internal static extern int vkAllocateMemory(IntPtr device, ref VkMemoryAllocateInfo info, IntPtr alloc, out IntPtr mem);
    [DllImport(Lib)] internal static extern void vkFreeMemory(IntPtr device, IntPtr mem, IntPtr alloc);
    [DllImport(Lib)] internal static extern int vkMapMemory(IntPtr device, IntPtr mem, ulong offset, ulong size, uint flags, out IntPtr data);
    [DllImport(Lib)] internal static extern void vkUnmapMemory(IntPtr device, IntPtr mem);

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkInstanceCreateInfo
    {
        public int sType;            // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
        public IntPtr pNext;
        public uint flags;
        public IntPtr pApplicationInfo;
        public uint enabledLayerCount;
        public IntPtr ppEnabledLayerNames;
        public uint enabledExtensionCount;
        public IntPtr ppEnabledExtensionNames;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDeviceCreateInfo
    {
        public int sType;            // VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
        public IntPtr pNext;
        public uint flags;
        public uint queueCreateInfoCount;
        public IntPtr pQueueCreateInfos;
        public uint enabledLayerCount;
        public IntPtr ppEnabledLayerNames;
        public uint enabledExtensionCount;
        public IntPtr ppEnabledExtensionNames;
        public IntPtr pEnabledFeatures;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkMemoryAllocateInfo
    {
        public int sType;            // VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
        public IntPtr pNext;
        public ulong allocationSize;
        public uint memoryTypeIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDeviceQueueCreateInfo
    {
        public int sType;            // VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
        public IntPtr pNext;
        public uint flags;
        public uint queueFamilyIndex;
        public uint queueCount;
        public IntPtr pQueuePriorities;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPhysicalDeviceMemoryProperties
    {
        public uint memoryTypeCount;
        // 32 memory types, each 8 bytes (heapIndex + propertyFlags).
        // C# can't express VK_MAX_MEMORY_TYPES inline; we declare an
        // array large enough.
        public unsafe fixed uint memoryTypeFlags[32 * 2]; // [propertyFlags, heapIndex] per type
        public uint memoryHeapCount;
        public unsafe fixed long memoryHeapData[16 * 3];  // padding for the heap array
    }

    public const uint VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001;
    public const uint VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002;
    public const uint VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004;

    public sealed class Setup
    {
        public IntPtr Instance;
        public IntPtr Device;

        /// <summary>Generic HOST_VISIBLE memory type. Always set; falls back
        /// to the first HOST_VISIBLE type when no preferred type matches.
        /// Kept for back-compat — new code should use the scheme-specific
        /// fields below.</summary>
        public uint HostVisibleMemTypeIndex;

        /// <summary>HOST_VISIBLE | HOST_COHERENT memory type for Managed
        /// scheme (no explicit vkFlushMappedMemoryRanges needed). Falls
        /// back to <see cref="HostVisibleMemTypeIndex"/> when no coherent
        /// type exists — caller is responsible for flushing in that case.</summary>
        public uint HostCoherentMemTypeIndex;

        /// <summary>HOST_VISIBLE | DEVICE_LOCAL memory type for Pinned
        /// scheme — the "smart access memory" / ReBAR path that lets the
        /// GPU read host-resident memory without a staging copy. Falls
        /// back to <see cref="HostVisibleMemTypeIndex"/> on hardware
        /// without ReBAR (most pre-2020 dGPUs).</summary>
        public uint HostVisibleDeviceLocalMemTypeIndex;
    }

    public static Setup CreateMinimal()
    {
        var setup = new Setup
        {
            HostVisibleMemTypeIndex = uint.MaxValue,
            HostCoherentMemTypeIndex = uint.MaxValue,
            HostVisibleDeviceLocalMemTypeIndex = uint.MaxValue,
        };
        var instInfo = new VkInstanceCreateInfo { sType = 1 };
        if (vkCreateInstance(ref instInfo, IntPtr.Zero, out IntPtr instance) != 0)
            throw new InvalidOperationException("vkCreateInstance failed.");
        setup.Instance = instance;

        uint pdCount = 0;
        if (vkEnumeratePhysicalDevices(instance, ref pdCount, null) != 0 || pdCount == 0)
        {
            // Tear the instance down before throwing so we don't leak the
            // VkInstance handle when no physical devices are present.
            vkDestroyInstance(instance, IntPtr.Zero);
            throw new InvalidOperationException("Vulkan: no physical devices.");
        }
        var pds = new IntPtr[pdCount];
        if (vkEnumeratePhysicalDevices(instance, ref pdCount, pds) != 0)
        {
            vkDestroyInstance(instance, IntPtr.Zero);
            throw new InvalidOperationException("vkEnumeratePhysicalDevices failed.");
        }

        // Pick first device with at least one HOST_VISIBLE type, scanning
        // for both Pinned (HOST_VISIBLE | DEVICE_LOCAL) and Managed
        // (HOST_VISIBLE | HOST_COHERENT) preferences. Falls back to plain
        // HOST_VISIBLE on the same device when a preferred type isn't
        // available, so callers always get a usable index.
        IntPtr chosenPd = IntPtr.Zero;
        unsafe
        {
            for (int p = 0; p < pds.Length; p++)
            {
                vkGetPhysicalDeviceMemoryProperties(pds[p], out var props);
                uint? hostVisible = null;
                uint? hostCoherent = null;
                uint? hostVisibleDeviceLocal = null;
                for (uint t = 0; t < props.memoryTypeCount && t < 32; t++)
                {
                    uint flags = props.memoryTypeFlags[t * 2];
                    bool hv = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
                    bool hc = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
                    bool dl = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
                    if (!hv) continue;
                    hostVisible ??= t;
                    if (hc && hostCoherent is null) hostCoherent = t;
                    if (dl && hostVisibleDeviceLocal is null) hostVisibleDeviceLocal = t;
                }
                if (hostVisible.HasValue)
                {
                    chosenPd = pds[p];
                    setup.HostVisibleMemTypeIndex = hostVisible.Value;
                    setup.HostCoherentMemTypeIndex = hostCoherent ?? hostVisible.Value;
                    setup.HostVisibleDeviceLocalMemTypeIndex = hostVisibleDeviceLocal ?? hostVisible.Value;
                    break;
                }
            }
        }
        if (chosenPd == IntPtr.Zero)
        {
            vkDestroyInstance(instance, IntPtr.Zero);
            throw new InvalidOperationException("Vulkan: no HOST_VISIBLE memory type.");
        }

        // VkDeviceCreateInfo MUST include at least one VkDeviceQueueCreateInfo
        // — Vulkan rejects a no-queue logical device. Use queue family 0
        // with one queue at priority 1.0; that's the universal compute-
        // capable family on every modern desktop GPU.
        float[] priorities = { 1.0f };
        IntPtr prioritiesHandle = Marshal.AllocHGlobal(sizeof(float));
        Marshal.Copy(priorities, 0, prioritiesHandle, 1);
        var queueInfo = new VkDeviceQueueCreateInfo
        {
            sType = 2,
            queueFamilyIndex = 0,
            queueCount = 1,
            pQueuePriorities = prioritiesHandle,
        };
        IntPtr queueInfoHandle = Marshal.AllocHGlobal(Marshal.SizeOf<VkDeviceQueueCreateInfo>());
        Marshal.StructureToPtr(queueInfo, queueInfoHandle, false);
        try
        {
            var devInfo = new VkDeviceCreateInfo
            {
                sType = 3,
                queueCreateInfoCount = 1,
                pQueueCreateInfos = queueInfoHandle,
            };
            if (vkCreateDevice(chosenPd, ref devInfo, IntPtr.Zero, out IntPtr device) != 0)
            {
                vkDestroyInstance(instance, IntPtr.Zero);
                throw new InvalidOperationException("vkCreateDevice failed.");
            }
            setup.Device = device;
        }
        finally
        {
            Marshal.FreeHGlobal(queueInfoHandle);
            Marshal.FreeHGlobal(prioritiesHandle);
        }
        return setup;
    }
}
#endif
