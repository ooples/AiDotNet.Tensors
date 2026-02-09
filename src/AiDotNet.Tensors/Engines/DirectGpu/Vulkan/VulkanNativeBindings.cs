// Copyright (c) AiDotNet. All rights reserved.
// Vulkan native P/Invoke bindings for cross-platform GPU compute.
// Supports Windows (vulkan-1.dll), Linux (libvulkan.so.1), and macOS via MoltenVK.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Native P/Invoke bindings for the Vulkan API.
/// </summary>
/// <remarks>
/// <para><b>Cross-Platform Support:</b></para>
/// <para>
/// Vulkan is available on Windows, Linux, Android, and macOS (via MoltenVK).
/// This implementation dynamically loads the appropriate library for each platform.
/// </para>
/// <para><b>Compute Pipeline Architecture:</b></para>
/// <list type="bullet">
/// <item><b>VkInstance</b>: Application-level Vulkan context</item>
/// <item><b>VkPhysicalDevice</b>: Physical GPU representation</item>
/// <item><b>VkDevice</b>: Logical device for GPU operations</item>
/// <item><b>VkQueue</b>: Command submission queue</item>
/// <item><b>VkBuffer</b>: GPU memory buffer</item>
/// <item><b>VkShaderModule</b>: SPIR-V compiled shader</item>
/// <item><b>VkPipeline</b>: Configured compute pipeline</item>
/// <item><b>VkCommandBuffer</b>: Recorded GPU commands</item>
/// </list>
/// </remarks>
public static unsafe class VulkanNativeBindings
{
    #region Platform Detection and Library Loading

    private const string VulkanWindows = "vulkan-1.dll";
    private const string VulkanLinux = "libvulkan.so.1";
    private const string VulkanMacOS = "libvulkan.1.dylib"; // MoltenVK

    /// <summary>
    /// Gets the Vulkan library name for the current platform.
    /// </summary>
    public static string LibraryName
    {
        get
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return VulkanWindows;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return VulkanLinux;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return VulkanMacOS;
            return VulkanWindows;
        }
    }

    /// <summary>
    /// Checks if Vulkan is supported on the current platform.
    /// </summary>
    public static bool IsPlatformSupported
    {
        get
        {
            try
            {
#if NET5_0_OR_GREATER
                // .NET 5+ can use NativeLibrary
                var handle = System.Runtime.InteropServices.NativeLibrary.Load(LibraryName);
                if (handle != IntPtr.Zero)
                {
                    System.Runtime.InteropServices.NativeLibrary.Free(handle);
                    return true;
                }
#else
                // For .NET Framework, try to call a Vulkan function
                // If it doesn't throw, Vulkan is available
                uint count = 0;
                var result = vkEnumeratePhysicalDevices(IntPtr.Zero, ref count, null);
                // VK_ERROR_INITIALIZATION_FAILED is expected with null instance
                // but it means Vulkan is loaded
                return true;
#endif
            }
            catch
            {
                // Library not available
            }
            return false;
        }
    }

    #endregion

    #region Result Codes

    public const int VK_SUCCESS = 0;
    public const int VK_NOT_READY = 1;
    public const int VK_TIMEOUT = 2;
    public const int VK_EVENT_SET = 3;
    public const int VK_EVENT_RESET = 4;
    public const int VK_INCOMPLETE = 5;
    public const int VK_ERROR_OUT_OF_HOST_MEMORY = -1;
    public const int VK_ERROR_OUT_OF_DEVICE_MEMORY = -2;
    public const int VK_ERROR_INITIALIZATION_FAILED = -3;
    public const int VK_ERROR_DEVICE_LOST = -4;
    public const int VK_ERROR_MEMORY_MAP_FAILED = -5;
    public const int VK_ERROR_LAYER_NOT_PRESENT = -6;
    public const int VK_ERROR_EXTENSION_NOT_PRESENT = -7;
    public const int VK_ERROR_FEATURE_NOT_PRESENT = -8;
    public const int VK_ERROR_INCOMPATIBLE_DRIVER = -9;
    public const int VK_ERROR_TOO_MANY_OBJECTS = -10;
    public const int VK_ERROR_FORMAT_NOT_SUPPORTED = -11;

    #endregion

    #region Structure Types

    public const int VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
    public const int VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
    public const int VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
    public const int VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
    public const int VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
    public const int VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
    public const int VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE = 6;
    public const int VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
    public const int VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
    public const int VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
    public const int VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
    public const int VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
    public const int VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
    public const int VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
    public const int VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
    public const int VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
    public const int VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
    public const int VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
    public const int VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
    public const int VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 44;
    public const int VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER = 45;
    public const int VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2 = 1000059006;

    #endregion

    #region Buffer Usage Flags

    public const uint VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001;
    public const uint VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002;
    public const uint VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010;
    public const uint VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020;

    #endregion

    #region Memory Property Flags

    public const uint VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001;
    public const uint VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002;
    public const uint VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004;
    public const uint VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 0x00000008;

    #endregion

    #region Descriptor Types

    public const int VK_DESCRIPTOR_TYPE_SAMPLER = 0;
    public const int VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1;
    public const int VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE = 2;
    public const int VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 3;
    public const int VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER = 4;
    public const int VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER = 5;
    public const int VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6;
    public const int VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
    public const int VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC = 8;
    public const int VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC = 9;

    #endregion

    #region Shader Stage Flags

    public const uint VK_SHADER_STAGE_VERTEX_BIT = 0x00000001;
    public const uint VK_SHADER_STAGE_FRAGMENT_BIT = 0x00000010;
    public const uint VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020;
    public const uint VK_SHADER_STAGE_ALL = 0x7FFFFFFF;

    #endregion

    #region Pipeline Bind Points

    public const int VK_PIPELINE_BIND_POINT_GRAPHICS = 0;
    public const int VK_PIPELINE_BIND_POINT_COMPUTE = 1;

    #endregion

    #region Command Buffer Levels

    public const int VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
    public const int VK_COMMAND_BUFFER_LEVEL_SECONDARY = 1;

    #endregion

    #region Queue Flags

    public const uint VK_QUEUE_GRAPHICS_BIT = 0x00000001;
    public const uint VK_QUEUE_COMPUTE_BIT = 0x00000002;
    public const uint VK_QUEUE_TRANSFER_BIT = 0x00000004;

    #endregion

    #region Other Constants

    public const ulong VK_WHOLE_SIZE = ~0UL;
    public const uint VK_QUEUE_FAMILY_IGNORED = ~0U;
    public const int VK_API_VERSION_1_0 = (1 << 22) | (0 << 12) | 0;
    public const int VK_API_VERSION_1_1 = (1 << 22) | (1 << 12) | 0;
    public const int VK_API_VERSION_1_2 = (1 << 22) | (2 << 12) | 0;
    public const int VK_API_VERSION_1_3 = (1 << 22) | (3 << 12) | 0;

    #endregion

    #region Instance Functions

    [DllImport(VulkanWindows, EntryPoint = "vkCreateInstance")]
    private static extern int vkCreateInstance_Windows(VkInstanceCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pInstance);

    [DllImport(VulkanLinux, EntryPoint = "vkCreateInstance")]
    private static extern int vkCreateInstance_Linux(VkInstanceCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pInstance);

    public static int vkCreateInstance(VkInstanceCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pInstance)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkCreateInstance_Windows(pCreateInfo, pAllocator, out pInstance);
        return vkCreateInstance_Linux(pCreateInfo, pAllocator, out pInstance);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyInstance")]
    private static extern void vkDestroyInstance_Windows(IntPtr instance, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyInstance")]
    private static extern void vkDestroyInstance_Linux(IntPtr instance, IntPtr pAllocator);

    public static void vkDestroyInstance(IntPtr instance, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyInstance_Windows(instance, pAllocator);
        else
            vkDestroyInstance_Linux(instance, pAllocator);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkEnumeratePhysicalDevices")]
    private static extern int vkEnumeratePhysicalDevices_Windows(IntPtr instance, ref uint pPhysicalDeviceCount, IntPtr* pPhysicalDevices);

    [DllImport(VulkanLinux, EntryPoint = "vkEnumeratePhysicalDevices")]
    private static extern int vkEnumeratePhysicalDevices_Linux(IntPtr instance, ref uint pPhysicalDeviceCount, IntPtr* pPhysicalDevices);

    public static int vkEnumeratePhysicalDevices(IntPtr instance, ref uint pPhysicalDeviceCount, IntPtr* pPhysicalDevices)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkEnumeratePhysicalDevices_Windows(instance, ref pPhysicalDeviceCount, pPhysicalDevices);
        return vkEnumeratePhysicalDevices_Linux(instance, ref pPhysicalDeviceCount, pPhysicalDevices);
    }

    #endregion

    #region Physical Device Functions

    [DllImport(VulkanWindows, EntryPoint = "vkGetPhysicalDeviceProperties")]
    private static extern void vkGetPhysicalDeviceProperties_Windows(IntPtr physicalDevice, VkPhysicalDeviceProperties* pProperties);

    [DllImport(VulkanLinux, EntryPoint = "vkGetPhysicalDeviceProperties")]
    private static extern void vkGetPhysicalDeviceProperties_Linux(IntPtr physicalDevice, VkPhysicalDeviceProperties* pProperties);

    public static void vkGetPhysicalDeviceProperties(IntPtr physicalDevice, VkPhysicalDeviceProperties* pProperties)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkGetPhysicalDeviceProperties_Windows(physicalDevice, pProperties);
        else
            vkGetPhysicalDeviceProperties_Linux(physicalDevice, pProperties);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkGetPhysicalDeviceMemoryProperties")]
    private static extern void vkGetPhysicalDeviceMemoryProperties_Windows(IntPtr physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties);

    [DllImport(VulkanLinux, EntryPoint = "vkGetPhysicalDeviceMemoryProperties")]
    private static extern void vkGetPhysicalDeviceMemoryProperties_Linux(IntPtr physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties);

    public static void vkGetPhysicalDeviceMemoryProperties(IntPtr physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkGetPhysicalDeviceMemoryProperties_Windows(physicalDevice, pMemoryProperties);
        else
            vkGetPhysicalDeviceMemoryProperties_Linux(physicalDevice, pMemoryProperties);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkGetPhysicalDeviceQueueFamilyProperties")]
    private static extern void vkGetPhysicalDeviceQueueFamilyProperties_Windows(IntPtr physicalDevice, ref uint pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties);

    [DllImport(VulkanLinux, EntryPoint = "vkGetPhysicalDeviceQueueFamilyProperties")]
    private static extern void vkGetPhysicalDeviceQueueFamilyProperties_Linux(IntPtr physicalDevice, ref uint pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties);

    public static void vkGetPhysicalDeviceQueueFamilyProperties(IntPtr physicalDevice, ref uint pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkGetPhysicalDeviceQueueFamilyProperties_Windows(physicalDevice, ref pQueueFamilyPropertyCount, pQueueFamilyProperties);
        else
            vkGetPhysicalDeviceQueueFamilyProperties_Linux(physicalDevice, ref pQueueFamilyPropertyCount, pQueueFamilyProperties);
    }

    #endregion

    #region Device Functions

    [DllImport(VulkanWindows, EntryPoint = "vkCreateDevice")]
    private static extern int vkCreateDevice_Windows(IntPtr physicalDevice, VkDeviceCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pDevice);

    [DllImport(VulkanLinux, EntryPoint = "vkCreateDevice")]
    private static extern int vkCreateDevice_Linux(IntPtr physicalDevice, VkDeviceCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pDevice);

    public static int vkCreateDevice(IntPtr physicalDevice, VkDeviceCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pDevice)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkCreateDevice_Windows(physicalDevice, pCreateInfo, pAllocator, out pDevice);
        return vkCreateDevice_Linux(physicalDevice, pCreateInfo, pAllocator, out pDevice);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyDevice")]
    private static extern void vkDestroyDevice_Windows(IntPtr device, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyDevice")]
    private static extern void vkDestroyDevice_Linux(IntPtr device, IntPtr pAllocator);

    public static void vkDestroyDevice(IntPtr device, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyDevice_Windows(device, pAllocator);
        else
            vkDestroyDevice_Linux(device, pAllocator);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkGetDeviceQueue")]
    private static extern void vkGetDeviceQueue_Windows(IntPtr device, uint queueFamilyIndex, uint queueIndex, out IntPtr pQueue);

    [DllImport(VulkanLinux, EntryPoint = "vkGetDeviceQueue")]
    private static extern void vkGetDeviceQueue_Linux(IntPtr device, uint queueFamilyIndex, uint queueIndex, out IntPtr pQueue);

    public static void vkGetDeviceQueue(IntPtr device, uint queueFamilyIndex, uint queueIndex, out IntPtr pQueue)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkGetDeviceQueue_Windows(device, queueFamilyIndex, queueIndex, out pQueue);
        else
            vkGetDeviceQueue_Linux(device, queueFamilyIndex, queueIndex, out pQueue);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDeviceWaitIdle")]
    private static extern int vkDeviceWaitIdle_Windows(IntPtr device);

    [DllImport(VulkanLinux, EntryPoint = "vkDeviceWaitIdle")]
    private static extern int vkDeviceWaitIdle_Linux(IntPtr device);

    public static int vkDeviceWaitIdle(IntPtr device)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkDeviceWaitIdle_Windows(device);
        return vkDeviceWaitIdle_Linux(device);
    }

    #endregion

    #region Buffer Functions

    [DllImport(VulkanWindows, EntryPoint = "vkCreateBuffer")]
    private static extern int vkCreateBuffer_Windows(IntPtr device, VkBufferCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pBuffer);

    [DllImport(VulkanLinux, EntryPoint = "vkCreateBuffer")]
    private static extern int vkCreateBuffer_Linux(IntPtr device, VkBufferCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pBuffer);

    public static int vkCreateBuffer(IntPtr device, VkBufferCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pBuffer)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkCreateBuffer_Windows(device, pCreateInfo, pAllocator, out pBuffer);
        return vkCreateBuffer_Linux(device, pCreateInfo, pAllocator, out pBuffer);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyBuffer")]
    private static extern void vkDestroyBuffer_Windows(IntPtr device, IntPtr buffer, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyBuffer")]
    private static extern void vkDestroyBuffer_Linux(IntPtr device, IntPtr buffer, IntPtr pAllocator);

    public static void vkDestroyBuffer(IntPtr device, IntPtr buffer, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyBuffer_Windows(device, buffer, pAllocator);
        else
            vkDestroyBuffer_Linux(device, buffer, pAllocator);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkGetBufferMemoryRequirements")]
    private static extern void vkGetBufferMemoryRequirements_Windows(IntPtr device, IntPtr buffer, VkMemoryRequirements* pMemoryRequirements);

    [DllImport(VulkanLinux, EntryPoint = "vkGetBufferMemoryRequirements")]
    private static extern void vkGetBufferMemoryRequirements_Linux(IntPtr device, IntPtr buffer, VkMemoryRequirements* pMemoryRequirements);

    public static void vkGetBufferMemoryRequirements(IntPtr device, IntPtr buffer, VkMemoryRequirements* pMemoryRequirements)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkGetBufferMemoryRequirements_Windows(device, buffer, pMemoryRequirements);
        else
            vkGetBufferMemoryRequirements_Linux(device, buffer, pMemoryRequirements);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkBindBufferMemory")]
    private static extern int vkBindBufferMemory_Windows(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset);

    [DllImport(VulkanLinux, EntryPoint = "vkBindBufferMemory")]
    private static extern int vkBindBufferMemory_Linux(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset);

    public static int vkBindBufferMemory(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkBindBufferMemory_Windows(device, buffer, memory, memoryOffset);
        return vkBindBufferMemory_Linux(device, buffer, memory, memoryOffset);
    }

    #endregion

    #region Memory Functions

    [DllImport(VulkanWindows, EntryPoint = "vkAllocateMemory")]
    private static extern int vkAllocateMemory_Windows(IntPtr device, VkMemoryAllocateInfo* pAllocateInfo, IntPtr pAllocator, out IntPtr pMemory);

    [DllImport(VulkanLinux, EntryPoint = "vkAllocateMemory")]
    private static extern int vkAllocateMemory_Linux(IntPtr device, VkMemoryAllocateInfo* pAllocateInfo, IntPtr pAllocator, out IntPtr pMemory);

    public static int vkAllocateMemory(IntPtr device, VkMemoryAllocateInfo* pAllocateInfo, IntPtr pAllocator, out IntPtr pMemory)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkAllocateMemory_Windows(device, pAllocateInfo, pAllocator, out pMemory);
        return vkAllocateMemory_Linux(device, pAllocateInfo, pAllocator, out pMemory);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkFreeMemory")]
    private static extern void vkFreeMemory_Windows(IntPtr device, IntPtr memory, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkFreeMemory")]
    private static extern void vkFreeMemory_Linux(IntPtr device, IntPtr memory, IntPtr pAllocator);

    public static void vkFreeMemory(IntPtr device, IntPtr memory, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkFreeMemory_Windows(device, memory, pAllocator);
        else
            vkFreeMemory_Linux(device, memory, pAllocator);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkMapMemory")]
    private static extern int vkMapMemory_Windows(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, out IntPtr ppData);

    [DllImport(VulkanLinux, EntryPoint = "vkMapMemory")]
    private static extern int vkMapMemory_Linux(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, out IntPtr ppData);

    public static int vkMapMemory(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, out IntPtr ppData)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkMapMemory_Windows(device, memory, offset, size, flags, out ppData);
        return vkMapMemory_Linux(device, memory, offset, size, flags, out ppData);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkUnmapMemory")]
    private static extern void vkUnmapMemory_Windows(IntPtr device, IntPtr memory);

    [DllImport(VulkanLinux, EntryPoint = "vkUnmapMemory")]
    private static extern void vkUnmapMemory_Linux(IntPtr device, IntPtr memory);

    public static void vkUnmapMemory(IntPtr device, IntPtr memory)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkUnmapMemory_Windows(device, memory);
        else
            vkUnmapMemory_Linux(device, memory);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkFlushMappedMemoryRanges")]
    private static extern int vkFlushMappedMemoryRanges_Windows(IntPtr device, uint memoryRangeCount, VkMappedMemoryRange* pMemoryRanges);

    [DllImport(VulkanLinux, EntryPoint = "vkFlushMappedMemoryRanges")]
    private static extern int vkFlushMappedMemoryRanges_Linux(IntPtr device, uint memoryRangeCount, VkMappedMemoryRange* pMemoryRanges);

    public static int vkFlushMappedMemoryRanges(IntPtr device, uint memoryRangeCount, VkMappedMemoryRange* pMemoryRanges)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkFlushMappedMemoryRanges_Windows(device, memoryRangeCount, pMemoryRanges);
        return vkFlushMappedMemoryRanges_Linux(device, memoryRangeCount, pMemoryRanges);
    }

    #endregion

    #region Shader Module Functions

    [DllImport(VulkanWindows, EntryPoint = "vkCreateShaderModule")]
    private static extern int vkCreateShaderModule_Windows(IntPtr device, VkShaderModuleCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pShaderModule);

    [DllImport(VulkanLinux, EntryPoint = "vkCreateShaderModule")]
    private static extern int vkCreateShaderModule_Linux(IntPtr device, VkShaderModuleCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pShaderModule);

    public static int vkCreateShaderModule(IntPtr device, VkShaderModuleCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pShaderModule)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkCreateShaderModule_Windows(device, pCreateInfo, pAllocator, out pShaderModule);
        return vkCreateShaderModule_Linux(device, pCreateInfo, pAllocator, out pShaderModule);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyShaderModule")]
    private static extern void vkDestroyShaderModule_Windows(IntPtr device, IntPtr shaderModule, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyShaderModule")]
    private static extern void vkDestroyShaderModule_Linux(IntPtr device, IntPtr shaderModule, IntPtr pAllocator);

    public static void vkDestroyShaderModule(IntPtr device, IntPtr shaderModule, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyShaderModule_Windows(device, shaderModule, pAllocator);
        else
            vkDestroyShaderModule_Linux(device, shaderModule, pAllocator);
    }

    #endregion

    #region Descriptor Set Layout Functions

    [DllImport(VulkanWindows, EntryPoint = "vkCreateDescriptorSetLayout")]
    private static extern int vkCreateDescriptorSetLayout_Windows(IntPtr device, VkDescriptorSetLayoutCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pSetLayout);

    [DllImport(VulkanLinux, EntryPoint = "vkCreateDescriptorSetLayout")]
    private static extern int vkCreateDescriptorSetLayout_Linux(IntPtr device, VkDescriptorSetLayoutCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pSetLayout);

    public static int vkCreateDescriptorSetLayout(IntPtr device, VkDescriptorSetLayoutCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pSetLayout)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkCreateDescriptorSetLayout_Windows(device, pCreateInfo, pAllocator, out pSetLayout);
        return vkCreateDescriptorSetLayout_Linux(device, pCreateInfo, pAllocator, out pSetLayout);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyDescriptorSetLayout")]
    private static extern void vkDestroyDescriptorSetLayout_Windows(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyDescriptorSetLayout")]
    private static extern void vkDestroyDescriptorSetLayout_Linux(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);

    public static void vkDestroyDescriptorSetLayout(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyDescriptorSetLayout_Windows(device, descriptorSetLayout, pAllocator);
        else
            vkDestroyDescriptorSetLayout_Linux(device, descriptorSetLayout, pAllocator);
    }

    #endregion

    #region Descriptor Pool Functions

    [DllImport(VulkanWindows, EntryPoint = "vkCreateDescriptorPool")]
    private static extern int vkCreateDescriptorPool_Windows(IntPtr device, VkDescriptorPoolCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pDescriptorPool);

    [DllImport(VulkanLinux, EntryPoint = "vkCreateDescriptorPool")]
    private static extern int vkCreateDescriptorPool_Linux(IntPtr device, VkDescriptorPoolCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pDescriptorPool);

    public static int vkCreateDescriptorPool(IntPtr device, VkDescriptorPoolCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pDescriptorPool)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkCreateDescriptorPool_Windows(device, pCreateInfo, pAllocator, out pDescriptorPool);
        return vkCreateDescriptorPool_Linux(device, pCreateInfo, pAllocator, out pDescriptorPool);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyDescriptorPool")]
    private static extern void vkDestroyDescriptorPool_Windows(IntPtr device, IntPtr descriptorPool, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyDescriptorPool")]
    private static extern void vkDestroyDescriptorPool_Linux(IntPtr device, IntPtr descriptorPool, IntPtr pAllocator);

    public static void vkDestroyDescriptorPool(IntPtr device, IntPtr descriptorPool, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyDescriptorPool_Windows(device, descriptorPool, pAllocator);
        else
            vkDestroyDescriptorPool_Linux(device, descriptorPool, pAllocator);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkAllocateDescriptorSets")]
    private static extern int vkAllocateDescriptorSets_Windows(IntPtr device, VkDescriptorSetAllocateInfo* pAllocateInfo, IntPtr* pDescriptorSets);

    [DllImport(VulkanLinux, EntryPoint = "vkAllocateDescriptorSets")]
    private static extern int vkAllocateDescriptorSets_Linux(IntPtr device, VkDescriptorSetAllocateInfo* pAllocateInfo, IntPtr* pDescriptorSets);

    public static int vkAllocateDescriptorSets(IntPtr device, VkDescriptorSetAllocateInfo* pAllocateInfo, IntPtr* pDescriptorSets)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkAllocateDescriptorSets_Windows(device, pAllocateInfo, pDescriptorSets);
        return vkAllocateDescriptorSets_Linux(device, pAllocateInfo, pDescriptorSets);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkUpdateDescriptorSets")]
    private static extern void vkUpdateDescriptorSets_Windows(IntPtr device, uint descriptorWriteCount, VkWriteDescriptorSet* pDescriptorWrites, uint descriptorCopyCount, IntPtr pDescriptorCopies);

    [DllImport(VulkanLinux, EntryPoint = "vkUpdateDescriptorSets")]
    private static extern void vkUpdateDescriptorSets_Linux(IntPtr device, uint descriptorWriteCount, VkWriteDescriptorSet* pDescriptorWrites, uint descriptorCopyCount, IntPtr pDescriptorCopies);

    public static void vkUpdateDescriptorSets(IntPtr device, uint descriptorWriteCount, VkWriteDescriptorSet* pDescriptorWrites, uint descriptorCopyCount, IntPtr pDescriptorCopies)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkUpdateDescriptorSets_Windows(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
        else
            vkUpdateDescriptorSets_Linux(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
    }

    #endregion

    #region Pipeline Functions

    [DllImport(VulkanWindows, EntryPoint = "vkCreatePipelineLayout")]
    private static extern int vkCreatePipelineLayout_Windows(IntPtr device, VkPipelineLayoutCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pPipelineLayout);

    [DllImport(VulkanLinux, EntryPoint = "vkCreatePipelineLayout")]
    private static extern int vkCreatePipelineLayout_Linux(IntPtr device, VkPipelineLayoutCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pPipelineLayout);

    public static int vkCreatePipelineLayout(IntPtr device, VkPipelineLayoutCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pPipelineLayout)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkCreatePipelineLayout_Windows(device, pCreateInfo, pAllocator, out pPipelineLayout);
        return vkCreatePipelineLayout_Linux(device, pCreateInfo, pAllocator, out pPipelineLayout);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyPipelineLayout")]
    private static extern void vkDestroyPipelineLayout_Windows(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyPipelineLayout")]
    private static extern void vkDestroyPipelineLayout_Linux(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator);

    public static void vkDestroyPipelineLayout(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyPipelineLayout_Windows(device, pipelineLayout, pAllocator);
        else
            vkDestroyPipelineLayout_Linux(device, pipelineLayout, pAllocator);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkCreateComputePipelines")]
    private static extern int vkCreateComputePipelines_Windows(IntPtr device, IntPtr pipelineCache, uint createInfoCount, VkComputePipelineCreateInfo* pCreateInfos, IntPtr pAllocator, IntPtr* pPipelines);

    [DllImport(VulkanLinux, EntryPoint = "vkCreateComputePipelines")]
    private static extern int vkCreateComputePipelines_Linux(IntPtr device, IntPtr pipelineCache, uint createInfoCount, VkComputePipelineCreateInfo* pCreateInfos, IntPtr pAllocator, IntPtr* pPipelines);

    public static int vkCreateComputePipelines(IntPtr device, IntPtr pipelineCache, uint createInfoCount, VkComputePipelineCreateInfo* pCreateInfos, IntPtr pAllocator, IntPtr* pPipelines)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkCreateComputePipelines_Windows(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
        return vkCreateComputePipelines_Linux(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyPipeline")]
    private static extern void vkDestroyPipeline_Windows(IntPtr device, IntPtr pipeline, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyPipeline")]
    private static extern void vkDestroyPipeline_Linux(IntPtr device, IntPtr pipeline, IntPtr pAllocator);

    public static void vkDestroyPipeline(IntPtr device, IntPtr pipeline, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyPipeline_Windows(device, pipeline, pAllocator);
        else
            vkDestroyPipeline_Linux(device, pipeline, pAllocator);
    }

    #endregion

    #region Command Pool Functions

    [DllImport(VulkanWindows, EntryPoint = "vkCreateCommandPool")]
    private static extern int vkCreateCommandPool_Windows(IntPtr device, VkCommandPoolCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pCommandPool);

    [DllImport(VulkanLinux, EntryPoint = "vkCreateCommandPool")]
    private static extern int vkCreateCommandPool_Linux(IntPtr device, VkCommandPoolCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pCommandPool);

    public static int vkCreateCommandPool(IntPtr device, VkCommandPoolCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pCommandPool)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkCreateCommandPool_Windows(device, pCreateInfo, pAllocator, out pCommandPool);
        return vkCreateCommandPool_Linux(device, pCreateInfo, pAllocator, out pCommandPool);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyCommandPool")]
    private static extern void vkDestroyCommandPool_Windows(IntPtr device, IntPtr commandPool, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyCommandPool")]
    private static extern void vkDestroyCommandPool_Linux(IntPtr device, IntPtr commandPool, IntPtr pAllocator);

    public static void vkDestroyCommandPool(IntPtr device, IntPtr commandPool, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyCommandPool_Windows(device, commandPool, pAllocator);
        else
            vkDestroyCommandPool_Linux(device, commandPool, pAllocator);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkResetCommandPool")]
    private static extern int vkResetCommandPool_Windows(IntPtr device, IntPtr commandPool, uint flags);

    [DllImport(VulkanLinux, EntryPoint = "vkResetCommandPool")]
    private static extern int vkResetCommandPool_Linux(IntPtr device, IntPtr commandPool, uint flags);

    public static int vkResetCommandPool(IntPtr device, IntPtr commandPool, uint flags)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkResetCommandPool_Windows(device, commandPool, flags);
        return vkResetCommandPool_Linux(device, commandPool, flags);
    }

    #endregion

    #region Command Buffer Functions

    [DllImport(VulkanWindows, EntryPoint = "vkAllocateCommandBuffers")]
    private static extern int vkAllocateCommandBuffers_Windows(IntPtr device, VkCommandBufferAllocateInfo* pAllocateInfo, IntPtr* pCommandBuffers);

    [DllImport(VulkanLinux, EntryPoint = "vkAllocateCommandBuffers")]
    private static extern int vkAllocateCommandBuffers_Linux(IntPtr device, VkCommandBufferAllocateInfo* pAllocateInfo, IntPtr* pCommandBuffers);

    public static int vkAllocateCommandBuffers(IntPtr device, VkCommandBufferAllocateInfo* pAllocateInfo, IntPtr* pCommandBuffers)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkAllocateCommandBuffers_Windows(device, pAllocateInfo, pCommandBuffers);
        return vkAllocateCommandBuffers_Linux(device, pAllocateInfo, pCommandBuffers);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkFreeCommandBuffers")]
    private static extern void vkFreeCommandBuffers_Windows(IntPtr device, IntPtr commandPool, uint commandBufferCount, IntPtr* pCommandBuffers);

    [DllImport(VulkanLinux, EntryPoint = "vkFreeCommandBuffers")]
    private static extern void vkFreeCommandBuffers_Linux(IntPtr device, IntPtr commandPool, uint commandBufferCount, IntPtr* pCommandBuffers);

    public static void vkFreeCommandBuffers(IntPtr device, IntPtr commandPool, uint commandBufferCount, IntPtr* pCommandBuffers)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkFreeCommandBuffers_Windows(device, commandPool, commandBufferCount, pCommandBuffers);
        else
            vkFreeCommandBuffers_Linux(device, commandPool, commandBufferCount, pCommandBuffers);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkBeginCommandBuffer")]
    private static extern int vkBeginCommandBuffer_Windows(IntPtr commandBuffer, VkCommandBufferBeginInfo* pBeginInfo);

    [DllImport(VulkanLinux, EntryPoint = "vkBeginCommandBuffer")]
    private static extern int vkBeginCommandBuffer_Linux(IntPtr commandBuffer, VkCommandBufferBeginInfo* pBeginInfo);

    public static int vkBeginCommandBuffer(IntPtr commandBuffer, VkCommandBufferBeginInfo* pBeginInfo)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkBeginCommandBuffer_Windows(commandBuffer, pBeginInfo);
        return vkBeginCommandBuffer_Linux(commandBuffer, pBeginInfo);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkEndCommandBuffer")]
    private static extern int vkEndCommandBuffer_Windows(IntPtr commandBuffer);

    [DllImport(VulkanLinux, EntryPoint = "vkEndCommandBuffer")]
    private static extern int vkEndCommandBuffer_Linux(IntPtr commandBuffer);

    public static int vkEndCommandBuffer(IntPtr commandBuffer)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkEndCommandBuffer_Windows(commandBuffer);
        return vkEndCommandBuffer_Linux(commandBuffer);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkResetCommandBuffer")]
    private static extern int vkResetCommandBuffer_Windows(IntPtr commandBuffer, uint flags);

    [DllImport(VulkanLinux, EntryPoint = "vkResetCommandBuffer")]
    private static extern int vkResetCommandBuffer_Linux(IntPtr commandBuffer, uint flags);

    public static int vkResetCommandBuffer(IntPtr commandBuffer, uint flags)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkResetCommandBuffer_Windows(commandBuffer, flags);
        return vkResetCommandBuffer_Linux(commandBuffer, flags);
    }

    #endregion

    #region Command Recording Functions

    [DllImport(VulkanWindows, EntryPoint = "vkCmdBindPipeline")]
    private static extern void vkCmdBindPipeline_Windows(IntPtr commandBuffer, int pipelineBindPoint, IntPtr pipeline);

    [DllImport(VulkanLinux, EntryPoint = "vkCmdBindPipeline")]
    private static extern void vkCmdBindPipeline_Linux(IntPtr commandBuffer, int pipelineBindPoint, IntPtr pipeline);

    public static void vkCmdBindPipeline(IntPtr commandBuffer, int pipelineBindPoint, IntPtr pipeline)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkCmdBindPipeline_Windows(commandBuffer, pipelineBindPoint, pipeline);
        else
            vkCmdBindPipeline_Linux(commandBuffer, pipelineBindPoint, pipeline);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkCmdBindDescriptorSets")]
    private static extern void vkCmdBindDescriptorSets_Windows(IntPtr commandBuffer, int pipelineBindPoint, IntPtr layout, uint firstSet, uint descriptorSetCount, IntPtr* pDescriptorSets, uint dynamicOffsetCount, uint* pDynamicOffsets);

    [DllImport(VulkanLinux, EntryPoint = "vkCmdBindDescriptorSets")]
    private static extern void vkCmdBindDescriptorSets_Linux(IntPtr commandBuffer, int pipelineBindPoint, IntPtr layout, uint firstSet, uint descriptorSetCount, IntPtr* pDescriptorSets, uint dynamicOffsetCount, uint* pDynamicOffsets);

    public static void vkCmdBindDescriptorSets(IntPtr commandBuffer, int pipelineBindPoint, IntPtr layout, uint firstSet, uint descriptorSetCount, IntPtr* pDescriptorSets, uint dynamicOffsetCount, uint* pDynamicOffsets)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkCmdBindDescriptorSets_Windows(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
        else
            vkCmdBindDescriptorSets_Linux(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkCmdPushConstants")]
    private static extern void vkCmdPushConstants_Windows(IntPtr commandBuffer, IntPtr layout, uint stageFlags, uint offset, uint size, void* pValues);

    [DllImport(VulkanLinux, EntryPoint = "vkCmdPushConstants")]
    private static extern void vkCmdPushConstants_Linux(IntPtr commandBuffer, IntPtr layout, uint stageFlags, uint offset, uint size, void* pValues);

    public static void vkCmdPushConstants(IntPtr commandBuffer, IntPtr layout, uint stageFlags, uint offset, uint size, void* pValues)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkCmdPushConstants_Windows(commandBuffer, layout, stageFlags, offset, size, pValues);
        else
            vkCmdPushConstants_Linux(commandBuffer, layout, stageFlags, offset, size, pValues);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkCmdDispatch")]
    private static extern void vkCmdDispatch_Windows(IntPtr commandBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);

    [DllImport(VulkanLinux, EntryPoint = "vkCmdDispatch")]
    private static extern void vkCmdDispatch_Linux(IntPtr commandBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);

    public static void vkCmdDispatch(IntPtr commandBuffer, uint groupCountX, uint groupCountY, uint groupCountZ)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkCmdDispatch_Windows(commandBuffer, groupCountX, groupCountY, groupCountZ);
        else
            vkCmdDispatch_Linux(commandBuffer, groupCountX, groupCountY, groupCountZ);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkCmdCopyBuffer")]
    private static extern void vkCmdCopyBuffer_Windows(IntPtr commandBuffer, IntPtr srcBuffer, IntPtr dstBuffer, uint regionCount, VkBufferCopy* pRegions);

    [DllImport(VulkanLinux, EntryPoint = "vkCmdCopyBuffer")]
    private static extern void vkCmdCopyBuffer_Linux(IntPtr commandBuffer, IntPtr srcBuffer, IntPtr dstBuffer, uint regionCount, VkBufferCopy* pRegions);

    public static void vkCmdCopyBuffer(IntPtr commandBuffer, IntPtr srcBuffer, IntPtr dstBuffer, uint regionCount, VkBufferCopy* pRegions)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkCmdCopyBuffer_Windows(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
        else
            vkCmdCopyBuffer_Linux(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkCmdPipelineBarrier")]
    private static extern void vkCmdPipelineBarrier_Windows(IntPtr commandBuffer, uint srcStageMask, uint dstStageMask, uint dependencyFlags, uint memoryBarrierCount, IntPtr pMemoryBarriers, uint bufferMemoryBarrierCount, VkBufferMemoryBarrier* pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers);

    [DllImport(VulkanLinux, EntryPoint = "vkCmdPipelineBarrier")]
    private static extern void vkCmdPipelineBarrier_Linux(IntPtr commandBuffer, uint srcStageMask, uint dstStageMask, uint dependencyFlags, uint memoryBarrierCount, IntPtr pMemoryBarriers, uint bufferMemoryBarrierCount, VkBufferMemoryBarrier* pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers);

    public static void vkCmdPipelineBarrier(IntPtr commandBuffer, uint srcStageMask, uint dstStageMask, uint dependencyFlags, uint memoryBarrierCount, IntPtr pMemoryBarriers, uint bufferMemoryBarrierCount, VkBufferMemoryBarrier* pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkCmdPipelineBarrier_Windows(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
        else
            vkCmdPipelineBarrier_Linux(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
    }

    #endregion

    #region Queue Functions

    [DllImport(VulkanWindows, EntryPoint = "vkQueueSubmit")]
    private static extern int vkQueueSubmit_Windows(IntPtr queue, uint submitCount, VkSubmitInfo* pSubmits, IntPtr fence);

    [DllImport(VulkanLinux, EntryPoint = "vkQueueSubmit")]
    private static extern int vkQueueSubmit_Linux(IntPtr queue, uint submitCount, VkSubmitInfo* pSubmits, IntPtr fence);

    public static int vkQueueSubmit(IntPtr queue, uint submitCount, VkSubmitInfo* pSubmits, IntPtr fence)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkQueueSubmit_Windows(queue, submitCount, pSubmits, fence);
        return vkQueueSubmit_Linux(queue, submitCount, pSubmits, fence);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkQueueWaitIdle")]
    private static extern int vkQueueWaitIdle_Windows(IntPtr queue);

    [DllImport(VulkanLinux, EntryPoint = "vkQueueWaitIdle")]
    private static extern int vkQueueWaitIdle_Linux(IntPtr queue);

    public static int vkQueueWaitIdle(IntPtr queue)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkQueueWaitIdle_Windows(queue);
        return vkQueueWaitIdle_Linux(queue);
    }

    #endregion

    #region Fence Functions

    [DllImport(VulkanWindows, EntryPoint = "vkCreateFence")]
    private static extern int vkCreateFence_Windows(IntPtr device, VkFenceCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pFence);

    [DllImport(VulkanLinux, EntryPoint = "vkCreateFence")]
    private static extern int vkCreateFence_Linux(IntPtr device, VkFenceCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pFence);

    public static int vkCreateFence(IntPtr device, VkFenceCreateInfo* pCreateInfo, IntPtr pAllocator, out IntPtr pFence)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkCreateFence_Windows(device, pCreateInfo, pAllocator, out pFence);
        return vkCreateFence_Linux(device, pCreateInfo, pAllocator, out pFence);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkDestroyFence")]
    private static extern void vkDestroyFence_Windows(IntPtr device, IntPtr fence, IntPtr pAllocator);

    [DllImport(VulkanLinux, EntryPoint = "vkDestroyFence")]
    private static extern void vkDestroyFence_Linux(IntPtr device, IntPtr fence, IntPtr pAllocator);

    public static void vkDestroyFence(IntPtr device, IntPtr fence, IntPtr pAllocator)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            vkDestroyFence_Windows(device, fence, pAllocator);
        else
            vkDestroyFence_Linux(device, fence, pAllocator);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkWaitForFences")]
    private static extern int vkWaitForFences_Windows(IntPtr device, uint fenceCount, IntPtr* pFences, uint waitAll, ulong timeout);

    [DllImport(VulkanLinux, EntryPoint = "vkWaitForFences")]
    private static extern int vkWaitForFences_Linux(IntPtr device, uint fenceCount, IntPtr* pFences, uint waitAll, ulong timeout);

    public static int vkWaitForFences(IntPtr device, uint fenceCount, IntPtr* pFences, uint waitAll, ulong timeout)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkWaitForFences_Windows(device, fenceCount, pFences, waitAll, timeout);
        return vkWaitForFences_Linux(device, fenceCount, pFences, waitAll, timeout);
    }

    [DllImport(VulkanWindows, EntryPoint = "vkResetFences")]
    private static extern int vkResetFences_Windows(IntPtr device, uint fenceCount, IntPtr* pFences);

    [DllImport(VulkanLinux, EntryPoint = "vkResetFences")]
    private static extern int vkResetFences_Linux(IntPtr device, uint fenceCount, IntPtr* pFences);

    public static int vkResetFences(IntPtr device, uint fenceCount, IntPtr* pFences)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return vkResetFences_Windows(device, fenceCount, pFences);
        return vkResetFences_Linux(device, fenceCount, pFences);
    }

    #endregion
}
