// Copyright (c) AiDotNet. All rights reserved.
// Vulkan GPU buffer implementation with efficient memory management.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Represents a GPU buffer allocated in Vulkan device memory.
/// </summary>
/// <remarks>
/// <para><b>Memory Architecture:</b></para>
/// <para>
/// Vulkan buffers require explicit memory allocation and binding. This implementation
/// uses a staging buffer pattern for efficient CPU-GPU data transfer:
/// </para>
/// <list type="bullet">
/// <item><b>Device Local Memory</b>: Fast GPU-only memory for compute operations</item>
/// <item><b>Host Visible Memory</b>: CPU-accessible memory for data transfer</item>
/// <item><b>Staging Buffer</b>: Temporary host-visible buffer for upload/download</item>
/// </list>
/// <para><b>Transfer Operations:</b></para>
/// <para>
/// Data transfer follows this pattern:
/// 1. Write data to staging buffer (host visible)
/// 2. Copy staging buffer to device buffer via command buffer
/// 3. Execute transfer command and wait for completion
/// </para>
/// </remarks>
public sealed unsafe class VulkanBuffer : IDisposable
{
    private readonly VulkanDevice _device;
    private IntPtr _buffer;
    private IntPtr _memory;
    private readonly ulong _size;
    private readonly int _elementCount;
    private readonly uint _usage;
    private readonly bool _isHostVisible;
    private bool _disposed;

    /// <summary>
    /// Gets the Vulkan buffer handle.
    /// </summary>
    public IntPtr Handle => _buffer;

    /// <summary>
    /// Gets the device memory handle.
    /// </summary>
    public IntPtr Memory => _memory;

    /// <summary>
    /// Gets the buffer size in bytes.
    /// </summary>
    public ulong SizeInBytes => _size;

    /// <summary>
    /// Gets the element count.
    /// </summary>
    public int ElementCount => _elementCount;

    /// <summary>
    /// Gets whether this buffer uses host-visible memory.
    /// </summary>
    public bool IsHostVisible => _isHostVisible;

    /// <summary>
    /// Creates a new Vulkan buffer.
    /// </summary>
    /// <param name="device">The Vulkan device.</param>
    /// <param name="elementCount">Number of float elements.</param>
    /// <param name="usage">Buffer usage flags.</param>
    /// <param name="hostVisible">Whether to use host-visible memory.</param>
    private VulkanBuffer(VulkanDevice device, int elementCount, uint usage, bool hostVisible)
    {
        if (elementCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(elementCount), elementCount, "Element count must be positive.");
        }

        _device = device;
        _elementCount = elementCount;
        _size = (ulong)elementCount * sizeof(float);
        _usage = usage;
        _isHostVisible = hostVisible;
    }

    /// <summary>
    /// Creates a device-local storage buffer for compute operations.
    /// </summary>
    /// <param name="elementCount">Number of float elements.</param>
    /// <returns>A new VulkanBuffer, or null if creation failed.</returns>
    public static VulkanBuffer? CreateStorageBuffer(int elementCount)
    {
        var device = VulkanDevice.Instance;
        if (!device.IsInitialized)
        {
            return null;
        }

        var buffer = new VulkanBuffer(
            device,
            elementCount,
            VulkanNativeBindings.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VulkanNativeBindings.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VulkanNativeBindings.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            hostVisible: false);

        if (!buffer.AllocateDeviceLocal())
        {
            buffer.Dispose();
            return null;
        }

        return buffer;
    }

    /// <summary>
    /// Creates a host-visible staging buffer for data transfer.
    /// </summary>
    /// <param name="elementCount">Number of float elements.</param>
    /// <returns>A new VulkanBuffer, or null if creation failed.</returns>
    public static VulkanBuffer? CreateStagingBuffer(int elementCount)
    {
        var device = VulkanDevice.Instance;
        if (!device.IsInitialized)
        {
            return null;
        }

        var buffer = new VulkanBuffer(
            device,
            elementCount,
            VulkanNativeBindings.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VulkanNativeBindings.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            hostVisible: true);

        if (!buffer.AllocateHostVisible())
        {
            buffer.Dispose();
            return null;
        }

        return buffer;
    }

    /// <summary>
    /// Creates a uniform buffer for shader parameters.
    /// </summary>
    /// <param name="sizeBytes">Size in bytes.</param>
    /// <returns>A new VulkanBuffer, or null if creation failed.</returns>
    public static VulkanBuffer? CreateUniformBuffer(int sizeBytes)
    {
        var device = VulkanDevice.Instance;
        if (!device.IsInitialized)
        {
            return null;
        }

        int elementCount = (sizeBytes + sizeof(float) - 1) / sizeof(float);
        var buffer = new VulkanBuffer(
            device,
            elementCount,
            VulkanNativeBindings.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
            VulkanNativeBindings.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            hostVisible: true);

        if (!buffer.AllocateHostVisible())
        {
            buffer.Dispose();
            return null;
        }

        return buffer;
    }

    private bool AllocateDeviceLocal()
    {
        return CreateBuffer() && AllocateMemory(
            VulkanNativeBindings.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    private bool AllocateHostVisible()
    {
        return CreateBuffer() && AllocateMemory(
            VulkanNativeBindings.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VulkanNativeBindings.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    private bool CreateBuffer()
    {
        var createInfo = new VkBufferCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext = null,
            flags = 0,
            size = _size,
            usage = _usage,
            sharingMode = 0, // VK_SHARING_MODE_EXCLUSIVE
            queueFamilyIndexCount = 0,
            pQueueFamilyIndices = null
        };

        var result = VulkanNativeBindings.vkCreateBuffer(
            _device.Device, &createInfo, IntPtr.Zero, out _buffer);

        return result == VulkanNativeBindings.VK_SUCCESS && _buffer != IntPtr.Zero;
    }

    private bool AllocateMemory(uint memoryProperties)
    {
        // Get memory requirements
        VkMemoryRequirements memReqs;
        VulkanNativeBindings.vkGetBufferMemoryRequirements(_device.Device, _buffer, &memReqs);

        // Find suitable memory type
        int memoryTypeIndex = _device.FindMemoryType(memReqs.memoryTypeBits, memoryProperties);
        if (memoryTypeIndex < 0)
        {
            return false;
        }

        // Allocate memory
        var allocInfo = new VkMemoryAllocateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext = null,
            allocationSize = memReqs.size,
            memoryTypeIndex = (uint)memoryTypeIndex
        };

        var result = VulkanNativeBindings.vkAllocateMemory(
            _device.Device, &allocInfo, IntPtr.Zero, out _memory);

        if (result != VulkanNativeBindings.VK_SUCCESS || _memory == IntPtr.Zero)
        {
            return false;
        }

        // Bind memory to buffer
        result = VulkanNativeBindings.vkBindBufferMemory(_device.Device, _buffer, _memory, 0);
        return result == VulkanNativeBindings.VK_SUCCESS;
    }

    /// <summary>
    /// Writes data to a host-visible buffer.
    /// </summary>
    /// <param name="data">The data to write.</param>
    public void WriteData(ReadOnlySpan<float> data)
    {
        if (!_isHostVisible || _disposed)
        {
            return;
        }

        int copyCount = Math.Min(data.Length, _elementCount);
        if (copyCount == 0)
        {
            return;
        }

        ulong copySize = (ulong)copyCount * sizeof(float);

        var result = VulkanNativeBindings.vkMapMemory(
            _device.Device, _memory, 0, copySize, 0, out IntPtr mappedPtr);

        if (result != VulkanNativeBindings.VK_SUCCESS || mappedPtr == IntPtr.Zero)
        {
            return;
        }

        try
        {
            fixed (float* srcPtr = data)
            {
                Buffer.MemoryCopy(srcPtr, (void*)mappedPtr, (long)copySize, (long)copySize);
            }
        }
        finally
        {
            VulkanNativeBindings.vkUnmapMemory(_device.Device, _memory);
        }
    }

    /// <summary>
    /// Reads data from a host-visible buffer.
    /// </summary>
    /// <param name="data">The destination span.</param>
    public void ReadData(Span<float> data)
    {
        if (!_isHostVisible || _disposed)
        {
            return;
        }

        int copyCount = Math.Min(data.Length, _elementCount);
        if (copyCount == 0)
        {
            return;
        }

        ulong copySize = (ulong)copyCount * sizeof(float);

        var result = VulkanNativeBindings.vkMapMemory(
            _device.Device, _memory, 0, copySize, 0, out IntPtr mappedPtr);

        if (result != VulkanNativeBindings.VK_SUCCESS || mappedPtr == IntPtr.Zero)
        {
            return;
        }

        try
        {
            fixed (float* dstPtr = data)
            {
                Buffer.MemoryCopy((void*)mappedPtr, dstPtr, (long)copySize, (long)copySize);
            }
        }
        finally
        {
            VulkanNativeBindings.vkUnmapMemory(_device.Device, _memory);
        }
    }

    /// <summary>
    /// Writes data directly to device-local memory via mapped pointer.
    /// For host-visible buffers only.
    /// </summary>
    /// <typeparam name="T">The data type.</typeparam>
    /// <param name="data">The data to write.</param>
    public void WriteRawData<T>(ReadOnlySpan<T> data) where T : unmanaged
    {
        if (!_isHostVisible || _disposed || data.Length == 0)
        {
            return;
        }

        ulong copySize = (ulong)data.Length * (ulong)sizeof(T);
        if (copySize > _size)
        {
            copySize = _size;
        }

        var result = VulkanNativeBindings.vkMapMemory(
            _device.Device, _memory, 0, copySize, 0, out IntPtr mappedPtr);

        if (result != VulkanNativeBindings.VK_SUCCESS || mappedPtr == IntPtr.Zero)
        {
            return;
        }

        try
        {
            fixed (T* srcPtr = data)
            {
                Buffer.MemoryCopy(srcPtr, (void*)mappedPtr, (long)copySize, (long)copySize);
            }
        }
        finally
        {
            VulkanNativeBindings.vkUnmapMemory(_device.Device, _memory);
        }
    }

    /// <summary>
    /// Creates a descriptor buffer info for this buffer.
    /// </summary>
    /// <returns>The descriptor buffer info.</returns>
    public VkDescriptorBufferInfo GetDescriptorInfo()
    {
        return new VkDescriptorBufferInfo
        {
            buffer = _buffer,
            offset = 0,
            range = _size
        };
    }

    /// <summary>
    /// Disposes the buffer and frees GPU resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_buffer != IntPtr.Zero)
        {
            VulkanNativeBindings.vkDestroyBuffer(_device.Device, _buffer, IntPtr.Zero);
            _buffer = IntPtr.Zero;
        }

        if (_memory != IntPtr.Zero)
        {
            VulkanNativeBindings.vkFreeMemory(_device.Device, _memory, IntPtr.Zero);
            _memory = IntPtr.Zero;
        }
    }
}

/// <summary>
/// Manages buffer transfer operations between host and device memory.
/// </summary>
public sealed unsafe class VulkanBufferTransfer : IDisposable
{
    private readonly VulkanDevice _device;
    private IntPtr _commandBuffer;
    private bool _disposed;

    /// <summary>
    /// Creates a new buffer transfer manager.
    /// </summary>
    public VulkanBufferTransfer()
    {
        _device = VulkanDevice.Instance;
        AllocateCommandBuffer();
    }

    private void AllocateCommandBuffer()
    {
        if (!_device.IsInitialized)
        {
            return;
        }

        var allocInfo = new VkCommandBufferAllocateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = null,
            commandPool = _device.CommandPool,
            level = VulkanNativeBindings.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount = 1
        };

        IntPtr cmdBuffer;
        int result = VulkanNativeBindings.vkAllocateCommandBuffers(_device.Device, &allocInfo, &cmdBuffer);
        if (result != VulkanNativeBindings.VK_SUCCESS || cmdBuffer == IntPtr.Zero)
        {
            throw new InvalidOperationException($"Failed to allocate Vulkan transfer command buffer: {result}");
        }

        _commandBuffer = cmdBuffer;
    }

    /// <summary>
    /// Copies data from a staging buffer to a device buffer.
    /// </summary>
    /// <param name="staging">The source staging buffer.</param>
    /// <param name="device">The destination device buffer.</param>
    public void CopyToDevice(VulkanBuffer staging, VulkanBuffer device)
    {
        if (_commandBuffer == IntPtr.Zero || _disposed)
        {
            return;
        }

        ulong copySize = Math.Min(staging.SizeInBytes, device.SizeInBytes);

        // Reset and begin command buffer
        VulkanNativeBindings.vkResetCommandBuffer(_commandBuffer, 0);

        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = null,
            flags = VkCommandBufferUsageFlags.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo = null
        };

        VulkanNativeBindings.vkBeginCommandBuffer(_commandBuffer, &beginInfo);

        // Record copy command
        var copyRegion = new VkBufferCopy
        {
            srcOffset = 0,
            dstOffset = 0,
            size = copySize
        };

        VulkanNativeBindings.vkCmdCopyBuffer(
            _commandBuffer, staging.Handle, device.Handle, 1, &copyRegion);

        // Add memory barrier to ensure copy completes before compute
        var barrier = new VkBufferMemoryBarrier
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            pNext = null,
            srcAccessMask = VkAccessFlags.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask = VkAccessFlags.VK_ACCESS_SHADER_READ_BIT | VkAccessFlags.VK_ACCESS_SHADER_WRITE_BIT,
            srcQueueFamilyIndex = VulkanNativeBindings.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex = VulkanNativeBindings.VK_QUEUE_FAMILY_IGNORED,
            buffer = device.Handle,
            offset = 0,
            size = VulkanNativeBindings.VK_WHOLE_SIZE
        };

        VulkanNativeBindings.vkCmdPipelineBarrier(
            _commandBuffer,
            VkPipelineStageFlags.VK_PIPELINE_STAGE_TRANSFER_BIT,
            VkPipelineStageFlags.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, IntPtr.Zero, 1, &barrier, 0, IntPtr.Zero);

        VulkanNativeBindings.vkEndCommandBuffer(_commandBuffer);

        // Submit and wait
        _device.SubmitAndWait(_commandBuffer);
    }

    /// <summary>
    /// Copies data from a device buffer to a staging buffer.
    /// </summary>
    /// <param name="device">The source device buffer.</param>
    /// <param name="staging">The destination staging buffer.</param>
    public void CopyFromDevice(VulkanBuffer device, VulkanBuffer staging)
    {
        if (_commandBuffer == IntPtr.Zero || _disposed)
        {
            return;
        }

        ulong copySize = Math.Min(device.SizeInBytes, staging.SizeInBytes);

        // Reset and begin command buffer
        VulkanNativeBindings.vkResetCommandBuffer(_commandBuffer, 0);

        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = null,
            flags = VkCommandBufferUsageFlags.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo = null
        };

        VulkanNativeBindings.vkBeginCommandBuffer(_commandBuffer, &beginInfo);

        // Add memory barrier to ensure compute writes are visible
        var preBarrier = new VkBufferMemoryBarrier
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            pNext = null,
            srcAccessMask = VkAccessFlags.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask = VkAccessFlags.VK_ACCESS_TRANSFER_READ_BIT,
            srcQueueFamilyIndex = VulkanNativeBindings.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex = VulkanNativeBindings.VK_QUEUE_FAMILY_IGNORED,
            buffer = device.Handle,
            offset = 0,
            size = VulkanNativeBindings.VK_WHOLE_SIZE
        };

        VulkanNativeBindings.vkCmdPipelineBarrier(
            _commandBuffer,
            VkPipelineStageFlags.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VkPipelineStageFlags.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, IntPtr.Zero, 1, &preBarrier, 0, IntPtr.Zero);

        // Record copy command
        var copyRegion = new VkBufferCopy
        {
            srcOffset = 0,
            dstOffset = 0,
            size = copySize
        };

        VulkanNativeBindings.vkCmdCopyBuffer(
            _commandBuffer, device.Handle, staging.Handle, 1, &copyRegion);

        // Add post-transfer barrier to ensure copy completes before host read
        var postBarrier = new VkBufferMemoryBarrier
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            pNext = null,
            srcAccessMask = VkAccessFlags.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask = VkAccessFlags.VK_ACCESS_HOST_READ_BIT,
            srcQueueFamilyIndex = VulkanNativeBindings.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex = VulkanNativeBindings.VK_QUEUE_FAMILY_IGNORED,
            buffer = staging.Handle,
            offset = 0,
            size = VulkanNativeBindings.VK_WHOLE_SIZE
        };

        VulkanNativeBindings.vkCmdPipelineBarrier(
            _commandBuffer,
            VkPipelineStageFlags.VK_PIPELINE_STAGE_TRANSFER_BIT,
            VkPipelineStageFlags.VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, IntPtr.Zero, 1, &postBarrier, 0, IntPtr.Zero);

        VulkanNativeBindings.vkEndCommandBuffer(_commandBuffer);

        // Submit and wait
        _device.SubmitAndWait(_commandBuffer);
    }

    /// <summary>
    /// Disposes the transfer manager.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_commandBuffer != IntPtr.Zero && _device.IsInitialized)
        {
            var cmdPtr = _commandBuffer;
            VulkanNativeBindings.vkFreeCommandBuffers(
                _device.Device, _device.CommandPool, 1, &cmdPtr);
            _commandBuffer = IntPtr.Zero;
        }
    }
}
