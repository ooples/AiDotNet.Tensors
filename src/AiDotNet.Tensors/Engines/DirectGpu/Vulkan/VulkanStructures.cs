// Copyright (c) AiDotNet. All rights reserved.
// Vulkan structure definitions for GPU compute operations.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Application information for Vulkan instance creation.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkApplicationInfo
{
    public int sType;
    public void* pNext;
    public byte* pApplicationName;
    public uint applicationVersion;
    public byte* pEngineName;
    public uint engineVersion;
    public uint apiVersion;
}

/// <summary>
/// Instance creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkInstanceCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public VkApplicationInfo* pApplicationInfo;
    public uint enabledLayerCount;
    public byte** ppEnabledLayerNames;
    public uint enabledExtensionCount;
    public byte** ppEnabledExtensionNames;
}

/// <summary>
/// Physical device properties.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkPhysicalDeviceProperties
{
    public uint apiVersion;
    public uint driverVersion;
    public uint vendorID;
    public uint deviceID;
    public int deviceType;
    public fixed byte deviceName[256];
    public fixed byte pipelineCacheUUID[16];
    public VkPhysicalDeviceLimits limits;
    public VkPhysicalDeviceSparseProperties sparseProperties;
}

/// <summary>
/// Physical device limits.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceLimits
{
    public uint maxImageDimension1D;
    public uint maxImageDimension2D;
    public uint maxImageDimension3D;
    public uint maxImageDimensionCube;
    public uint maxImageArrayLayers;
    public uint maxTexelBufferElements;
    public uint maxUniformBufferRange;
    public uint maxStorageBufferRange;
    public uint maxPushConstantsSize;
    public uint maxMemoryAllocationCount;
    public uint maxSamplerAllocationCount;
    public ulong bufferImageGranularity;
    public ulong sparseAddressSpaceSize;
    public uint maxBoundDescriptorSets;
    public uint maxPerStageDescriptorSamplers;
    public uint maxPerStageDescriptorUniformBuffers;
    public uint maxPerStageDescriptorStorageBuffers;
    public uint maxPerStageDescriptorSampledImages;
    public uint maxPerStageDescriptorStorageImages;
    public uint maxPerStageDescriptorInputAttachments;
    public uint maxPerStageResources;
    public uint maxDescriptorSetSamplers;
    public uint maxDescriptorSetUniformBuffers;
    public uint maxDescriptorSetUniformBuffersDynamic;
    public uint maxDescriptorSetStorageBuffers;
    public uint maxDescriptorSetStorageBuffersDynamic;
    public uint maxDescriptorSetSampledImages;
    public uint maxDescriptorSetStorageImages;
    public uint maxDescriptorSetInputAttachments;
    public uint maxVertexInputAttributes;
    public uint maxVertexInputBindings;
    public uint maxVertexInputAttributeOffset;
    public uint maxVertexInputBindingStride;
    public uint maxVertexOutputComponents;
    public uint maxTessellationGenerationLevel;
    public uint maxTessellationPatchSize;
    public uint maxTessellationControlPerVertexInputComponents;
    public uint maxTessellationControlPerVertexOutputComponents;
    public uint maxTessellationControlPerPatchOutputComponents;
    public uint maxTessellationControlTotalOutputComponents;
    public uint maxTessellationEvaluationInputComponents;
    public uint maxTessellationEvaluationOutputComponents;
    public uint maxGeometryShaderInvocations;
    public uint maxGeometryInputComponents;
    public uint maxGeometryOutputComponents;
    public uint maxGeometryOutputVertices;
    public uint maxGeometryTotalOutputComponents;
    public uint maxFragmentInputComponents;
    public uint maxFragmentOutputAttachments;
    public uint maxFragmentDualSrcAttachments;
    public uint maxFragmentCombinedOutputResources;
    public uint maxComputeSharedMemorySize;
    public uint maxComputeWorkGroupCountX;
    public uint maxComputeWorkGroupCountY;
    public uint maxComputeWorkGroupCountZ;
    public uint maxComputeWorkGroupInvocations;
    public uint maxComputeWorkGroupSizeX;
    public uint maxComputeWorkGroupSizeY;
    public uint maxComputeWorkGroupSizeZ;
    public uint subPixelPrecisionBits;
    public uint subTexelPrecisionBits;
    public uint mipmapPrecisionBits;
    public uint maxDrawIndexedIndexValue;
    public uint maxDrawIndirectCount;
    public float maxSamplerLodBias;
    public float maxSamplerAnisotropy;
    public uint maxViewports;
    public uint maxViewportDimensionsX;
    public uint maxViewportDimensionsY;
    public float viewportBoundsRangeMin;
    public float viewportBoundsRangeMax;
    public uint viewportSubPixelBits;
    public nuint minMemoryMapAlignment;
    public ulong minTexelBufferOffsetAlignment;
    public ulong minUniformBufferOffsetAlignment;
    public ulong minStorageBufferOffsetAlignment;
    public int minTexelOffset;
    public uint maxTexelOffset;
    public int minTexelGatherOffset;
    public uint maxTexelGatherOffset;
    public float minInterpolationOffset;
    public float maxInterpolationOffset;
    public uint subPixelInterpolationOffsetBits;
    public uint maxFramebufferWidth;
    public uint maxFramebufferHeight;
    public uint maxFramebufferLayers;
    public uint framebufferColorSampleCounts;
    public uint framebufferDepthSampleCounts;
    public uint framebufferStencilSampleCounts;
    public uint framebufferNoAttachmentsSampleCounts;
    public uint maxColorAttachments;
    public uint sampledImageColorSampleCounts;
    public uint sampledImageIntegerSampleCounts;
    public uint sampledImageDepthSampleCounts;
    public uint sampledImageStencilSampleCounts;
    public uint storageImageSampleCounts;
    public uint maxSampleMaskWords;
    public uint timestampComputeAndGraphics;
    public float timestampPeriod;
    public uint maxClipDistances;
    public uint maxCullDistances;
    public uint maxCombinedClipAndCullDistances;
    public uint discreteQueuePriorities;
    public float pointSizeRangeMin;
    public float pointSizeRangeMax;
    public float lineWidthRangeMin;
    public float lineWidthRangeMax;
    public float pointSizeGranularity;
    public float lineWidthGranularity;
    public uint strictLines;
    public uint standardSampleLocations;
    public ulong optimalBufferCopyOffsetAlignment;
    public ulong optimalBufferCopyRowPitchAlignment;
    public ulong nonCoherentAtomSize;
}

/// <summary>
/// Physical device sparse properties.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkPhysicalDeviceSparseProperties
{
    public uint residencyStandard2DBlockShape;
    public uint residencyStandard2DMultisampleBlockShape;
    public uint residencyStandard3DBlockShape;
    public uint residencyAlignedMipSize;
    public uint residencyNonResidentStrict;
}

/// <summary>
/// Queue family properties.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkQueueFamilyProperties
{
    public uint queueFlags;
    public uint queueCount;
    public uint timestampValidBits;
    public VkExtent3D minImageTransferGranularity;
}

/// <summary>
/// 3D extent structure.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkExtent3D
{
    public uint width;
    public uint height;
    public uint depth;
}

/// <summary>
/// Device queue creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkDeviceQueueCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public uint queueFamilyIndex;
    public uint queueCount;
    public float* pQueuePriorities;
}

/// <summary>
/// Device creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkDeviceCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public uint queueCreateInfoCount;
    public VkDeviceQueueCreateInfo* pQueueCreateInfos;
    public uint enabledLayerCount;
    public byte** ppEnabledLayerNames;
    public uint enabledExtensionCount;
    public byte** ppEnabledExtensionNames;
    public void* pEnabledFeatures;
}

/// <summary>
/// Buffer creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkBufferCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public ulong size;
    public uint usage;
    public int sharingMode;
    public uint queueFamilyIndexCount;
    public uint* pQueueFamilyIndices;
}

/// <summary>
/// Memory requirements for a buffer or image.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryRequirements
{
    public ulong size;
    public ulong alignment;
    public uint memoryTypeBits;
}

/// <summary>
/// Memory allocation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkMemoryAllocateInfo
{
    public int sType;
    public void* pNext;
    public ulong allocationSize;
    public uint memoryTypeIndex;
}

/// <summary>
/// Memory type description.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryType
{
    public uint propertyFlags;
    public uint heapIndex;
}

/// <summary>
/// Memory heap description.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkMemoryHeap
{
    public ulong size;
    public uint flags;
    // Padding for 16-byte alignment as per Vulkan spec
    private uint _padding;
}

/// <summary>
/// Physical device memory properties.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkPhysicalDeviceMemoryProperties
{
    public uint memoryTypeCount;
    public fixed byte memoryTypes[32 * 8]; // VkMemoryType[32] - 8 bytes each
    public uint memoryHeapCount;
    public fixed byte memoryHeaps[16 * 16]; // VkMemoryHeap[16] - 16 bytes each

    public VkMemoryType GetMemoryType(int index)
    {
        fixed (byte* ptr = memoryTypes)
        {
            return ((VkMemoryType*)ptr)[index];
        }
    }

    public VkMemoryHeap GetMemoryHeap(int index)
    {
        fixed (byte* ptr = memoryHeaps)
        {
            return ((VkMemoryHeap*)ptr)[index];
        }
    }
}

/// <summary>
/// Mapped memory range for flushing.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkMappedMemoryRange
{
    public int sType;
    public void* pNext;
    public IntPtr memory;
    public ulong offset;
    public ulong size;
}

/// <summary>
/// Shader module creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkShaderModuleCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public nuint codeSize;
    public uint* pCode;
}

/// <summary>
/// Specialization map entry.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkSpecializationMapEntry
{
    public uint constantID;
    public uint offset;
    public nuint size;
}

/// <summary>
/// Specialization info for shader constants.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkSpecializationInfo
{
    public uint mapEntryCount;
    public VkSpecializationMapEntry* pMapEntries;
    public nuint dataSize;
    public void* pData;
}

/// <summary>
/// Pipeline shader stage creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkPipelineShaderStageCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public uint stage;
    public IntPtr module;
    public byte* pName;
    public VkSpecializationInfo* pSpecializationInfo;
}

/// <summary>
/// Descriptor set layout binding.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkDescriptorSetLayoutBinding
{
    public uint binding;
    public int descriptorType;
    public uint descriptorCount;
    public uint stageFlags;
    public IntPtr* pImmutableSamplers;
}

/// <summary>
/// Descriptor set layout creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkDescriptorSetLayoutCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public uint bindingCount;
    public VkDescriptorSetLayoutBinding* pBindings;
}

/// <summary>
/// Push constant range.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkPushConstantRange
{
    public uint stageFlags;
    public uint offset;
    public uint size;
}

/// <summary>
/// Pipeline layout creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkPipelineLayoutCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public uint setLayoutCount;
    public IntPtr* pSetLayouts;
    public uint pushConstantRangeCount;
    public VkPushConstantRange* pPushConstantRanges;
}

/// <summary>
/// Compute pipeline creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkComputePipelineCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public VkPipelineShaderStageCreateInfo stage;
    public IntPtr layout;
    public IntPtr basePipelineHandle;
    public int basePipelineIndex;
}

/// <summary>
/// Descriptor pool size.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorPoolSize
{
    public int type;
    public uint descriptorCount;
}

/// <summary>
/// Descriptor pool creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkDescriptorPoolCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public uint maxSets;
    public uint poolSizeCount;
    public VkDescriptorPoolSize* pPoolSizes;
}

/// <summary>
/// Descriptor set allocation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkDescriptorSetAllocateInfo
{
    public int sType;
    public void* pNext;
    public IntPtr descriptorPool;
    public uint descriptorSetCount;
    public IntPtr* pSetLayouts;
}

/// <summary>
/// Descriptor buffer information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkDescriptorBufferInfo
{
    public IntPtr buffer;
    public ulong offset;
    public ulong range;
}

/// <summary>
/// Write descriptor set information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkWriteDescriptorSet
{
    public int sType;
    public void* pNext;
    public IntPtr dstSet;
    public uint dstBinding;
    public uint dstArrayElement;
    public uint descriptorCount;
    public int descriptorType;
    public void* pImageInfo;
    public VkDescriptorBufferInfo* pBufferInfo;
    public void* pTexelBufferView;
}

/// <summary>
/// Command pool creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkCommandPoolCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public uint queueFamilyIndex;
}

/// <summary>
/// Command buffer allocation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkCommandBufferAllocateInfo
{
    public int sType;
    public void* pNext;
    public IntPtr commandPool;
    public int level;
    public uint commandBufferCount;
}

/// <summary>
/// Command buffer begin information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkCommandBufferBeginInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
    public void* pInheritanceInfo;
}

/// <summary>
/// Buffer copy region.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct VkBufferCopy
{
    public ulong srcOffset;
    public ulong dstOffset;
    public ulong size;
}

/// <summary>
/// Buffer memory barrier.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkBufferMemoryBarrier
{
    public int sType;
    public void* pNext;
    public uint srcAccessMask;
    public uint dstAccessMask;
    public uint srcQueueFamilyIndex;
    public uint dstQueueFamilyIndex;
    public IntPtr buffer;
    public ulong offset;
    public ulong size;
}

/// <summary>
/// Submit information for queue submission.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkSubmitInfo
{
    public int sType;
    public void* pNext;
    public uint waitSemaphoreCount;
    public IntPtr* pWaitSemaphores;
    public uint* pWaitDstStageMask;
    public uint commandBufferCount;
    public IntPtr* pCommandBuffers;
    public uint signalSemaphoreCount;
    public IntPtr* pSignalSemaphores;
}

/// <summary>
/// Fence creation information.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VkFenceCreateInfo
{
    public int sType;
    public void* pNext;
    public uint flags;
}

/// <summary>
/// Pipeline stage flags for synchronization.
/// </summary>
public static class VkPipelineStageFlags
{
    public const uint VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001;
    public const uint VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT = 0x00000002;
    public const uint VK_PIPELINE_STAGE_VERTEX_INPUT_BIT = 0x00000004;
    public const uint VK_PIPELINE_STAGE_VERTEX_SHADER_BIT = 0x00000008;
    public const uint VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT = 0x00000080;
    public const uint VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800;
    public const uint VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000;
    public const uint VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x00002000;
    public const uint VK_PIPELINE_STAGE_HOST_BIT = 0x00004000;
    public const uint VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT = 0x00008000;
    public const uint VK_PIPELINE_STAGE_ALL_COMMANDS_BIT = 0x00010000;
}

/// <summary>
/// Access flags for memory barriers.
/// </summary>
public static class VkAccessFlags
{
    public const uint VK_ACCESS_INDIRECT_COMMAND_READ_BIT = 0x00000001;
    public const uint VK_ACCESS_INDEX_READ_BIT = 0x00000002;
    public const uint VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT = 0x00000004;
    public const uint VK_ACCESS_UNIFORM_READ_BIT = 0x00000008;
    public const uint VK_ACCESS_INPUT_ATTACHMENT_READ_BIT = 0x00000010;
    public const uint VK_ACCESS_SHADER_READ_BIT = 0x00000020;
    public const uint VK_ACCESS_SHADER_WRITE_BIT = 0x00000040;
    public const uint VK_ACCESS_TRANSFER_READ_BIT = 0x00000800;
    public const uint VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000;
    public const uint VK_ACCESS_HOST_READ_BIT = 0x00002000;
    public const uint VK_ACCESS_HOST_WRITE_BIT = 0x00004000;
    public const uint VK_ACCESS_MEMORY_READ_BIT = 0x00008000;
    public const uint VK_ACCESS_MEMORY_WRITE_BIT = 0x00010000;
}

/// <summary>
/// Command pool flags.
/// </summary>
public static class VkCommandPoolCreateFlags
{
    public const uint VK_COMMAND_POOL_CREATE_TRANSIENT_BIT = 0x00000001;
    public const uint VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002;
}

/// <summary>
/// Command buffer usage flags.
/// </summary>
public static class VkCommandBufferUsageFlags
{
    public const uint VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001;
    public const uint VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT = 0x00000002;
    public const uint VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT = 0x00000004;
}

/// <summary>
/// Fence create flags.
/// </summary>
public static class VkFenceCreateFlags
{
    public const uint VK_FENCE_CREATE_SIGNALED_BIT = 0x00000001;
}

/// <summary>
/// Physical device types.
/// </summary>
public static class VkPhysicalDeviceType
{
    public const int VK_PHYSICAL_DEVICE_TYPE_OTHER = 0;
    public const int VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1;
    public const int VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2;
    public const int VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3;
    public const int VK_PHYSICAL_DEVICE_TYPE_CPU = 4;
}

/// <summary>
/// Descriptor pool create flags.
/// </summary>
public static class VkDescriptorPoolCreateFlags
{
    public const uint VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT = 0x00000001;
    public const uint VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT = 0x00000002;
}
