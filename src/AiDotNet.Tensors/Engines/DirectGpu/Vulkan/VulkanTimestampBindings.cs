// Copyright (c) AiDotNet. All rights reserved.
// Vulkan timestamp-query bindings: the device clock the tuned-kernel registry times candidates on.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>Query pool creation information.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkQueryPoolCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public int queryType;
    public uint queryCount;
    public uint pipelineStatistics;
}

/// <summary>P/Invoke bindings for the timestamp queries of the Vulkan API.</summary>
internal static unsafe class VulkanTimestampBindings
{
    private const string VulkanWindows = "vulkan-1.dll";
    private const string VulkanLinux = "libvulkan.so.1";

    internal const int VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = 11;
    internal const int VK_QUERY_TYPE_TIMESTAMP = 2;
    internal const uint VK_QUERY_RESULT_64_BIT = 0x00000001;
    internal const uint VK_QUERY_RESULT_WAIT_BIT = 0x00000002;
    internal const uint VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x00002000;

    [DllImport(VulkanWindows, EntryPoint = "vkCreateQueryPool")]
    private static extern int vkCreateQueryPool_Windows(IntPtr device, VkQueryPoolCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pQueryPool);

    [DllImport(VulkanLinux, EntryPoint = "vkCreateQueryPool")]
    private static extern int vkCreateQueryPool_Linux(IntPtr device, VkQueryPoolCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pQueryPool);

    internal static int vkCreateQueryPool(IntPtr device, VkQueryPoolCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pQueryPool) =>
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? vkCreateQueryPool_Windows(device, pCreateInfo, pAllocator, out pQueryPool)
            : vkCreateQueryPool_Linux(device, pCreateInfo, pAllocator, out pQueryPool);

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyQueryPool")]
    private static extern void vkDestroyQueryPool_Windows(IntPtr device, IntPtr queryPool, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyQueryPool")]
    private static extern void vkDestroyQueryPool_Linux(IntPtr device, IntPtr queryPool, IntPtr pAllocator);

    internal static void vkDestroyQueryPool(IntPtr device, IntPtr queryPool, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyQueryPool_Windows(device, queryPool, pAllocator);
        else
            vkDestroyQueryPool_Linux(device, queryPool, pAllocator);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkCmdResetQueryPool")]
    private static extern void vkCmdResetQueryPool_Windows(IntPtr commandBuffer, IntPtr queryPool, uint firstQuery, uint queryCount);

    [DllImport(VulkanLinux, EntryPoint = "vkCmdResetQueryPool")]
    private static extern void vkCmdResetQueryPool_Linux(IntPtr commandBuffer, IntPtr queryPool, uint firstQuery, uint queryCount);

    internal static void vkCmdResetQueryPool(IntPtr commandBuffer, IntPtr queryPool, uint firstQuery, uint queryCount)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkCmdResetQueryPool_Windows(commandBuffer, queryPool, firstQuery, queryCount);
        else
            vkCmdResetQueryPool_Linux(commandBuffer, queryPool, firstQuery, queryCount);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkCmdWriteTimestamp")]
    private static extern void vkCmdWriteTimestamp_Windows(IntPtr commandBuffer, uint pipelineStage, IntPtr queryPool, uint query);

    [DllImport(VulkanLinux, EntryPoint = "vkCmdWriteTimestamp")]
    private static extern void vkCmdWriteTimestamp_Linux(IntPtr commandBuffer, uint pipelineStage, IntPtr queryPool, uint query);

    internal static void vkCmdWriteTimestamp(IntPtr commandBuffer, uint pipelineStage, IntPtr queryPool, uint query)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkCmdWriteTimestamp_Windows(commandBuffer, pipelineStage, queryPool, query);
        else
            vkCmdWriteTimestamp_Linux(commandBuffer, pipelineStage, queryPool, query);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkGetQueryPoolResults")]
    private static extern int vkGetQueryPoolResults_Windows(IntPtr device, IntPtr queryPool, uint firstQuery, uint queryCount, UIntPtr dataSize, void* pData, ulong stride, uint flags);

    [DllImport(VulkanLinux, EntryPoint = "vkGetQueryPoolResults")]
    private static extern int vkGetQueryPoolResults_Linux(IntPtr device, IntPtr queryPool, uint firstQuery, uint queryCount, UIntPtr dataSize, void* pData, ulong stride, uint flags);

    internal static int vkGetQueryPoolResults(IntPtr device, IntPtr queryPool, uint firstQuery, uint queryCount, UIntPtr dataSize, void* pData, ulong stride, uint flags) =>
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? vkGetQueryPoolResults_Windows(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags)
            : vkGetQueryPoolResults_Linux(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
}
