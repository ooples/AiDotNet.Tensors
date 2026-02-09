// Copyright (c) AiDotNet. All rights reserved.
// Vulkan device initialization and management for GPU compute.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Manages Vulkan device initialization and provides access to GPU resources.
/// </summary>
/// <remarks>
/// <para><b>Initialization Sequence:</b></para>
/// <list type="number">
/// <item>Create Vulkan instance with application info</item>
/// <item>Enumerate physical devices and select best GPU</item>
/// <item>Find compute queue family</item>
/// <item>Create logical device with compute queue</item>
/// <item>Create command pool for command buffer allocation</item>
/// </list>
/// <para><b>Device Selection:</b></para>
/// <para>
/// Prefers discrete GPUs over integrated GPUs. Falls back to any available device.
/// </para>
/// </remarks>
public sealed unsafe class VulkanDevice : IDisposable
{
    private static VulkanDevice? _instance;
    private static readonly object _lock = new();

    private IntPtr _instance_vk;
    private IntPtr _physicalDevice;
    private IntPtr _device;
    private IntPtr _computeQueue;
    private IntPtr _commandPool;
    private IntPtr _fence;
    private uint _computeQueueFamily;
    private bool _initialized;
    private bool _disposed;

    private string _deviceName = string.Empty;
    private string _vendorName = string.Empty;
    private VkPhysicalDeviceLimits _limits;
    private VkPhysicalDeviceMemoryProperties _memoryProperties;

    /// <summary>
    /// Gets the singleton device instance.
    /// </summary>
    public static VulkanDevice Instance
    {
        get
        {
            if (_instance is null)
            {
                lock (_lock)
                {
                    _instance ??= new VulkanDevice();
                }
            }
            return _instance;
        }
    }

    /// <summary>
    /// Gets whether the device is initialized and ready.
    /// </summary>
    public bool IsInitialized => _initialized && !_disposed;

    /// <summary>
    /// Gets the Vulkan logical device handle.
    /// </summary>
    public IntPtr Device => _device;

    /// <summary>
    /// Gets the compute queue handle.
    /// </summary>
    public IntPtr ComputeQueue => _computeQueue;

    /// <summary>
    /// Gets the command pool handle.
    /// </summary>
    public IntPtr CommandPool => _commandPool;

    /// <summary>
    /// Gets the reusable fence for synchronization.
    /// </summary>
    public IntPtr Fence => _fence;

    /// <summary>
    /// Gets the compute queue family index.
    /// </summary>
    public uint ComputeQueueFamily => _computeQueueFamily;

    /// <summary>
    /// Gets the device name.
    /// </summary>
    public string DeviceName => _deviceName;

    /// <summary>
    /// Gets the vendor name.
    /// </summary>
    public string VendorName => _vendorName;

    /// <summary>
    /// Gets the physical device limits.
    /// </summary>
    public VkPhysicalDeviceLimits Limits => _limits;

    /// <summary>
    /// Gets the memory properties.
    /// </summary>
    public VkPhysicalDeviceMemoryProperties MemoryProperties => _memoryProperties;

    /// <summary>
    /// Gets the maximum compute workgroup size.
    /// </summary>
    public uint MaxWorkgroupSize => Math.Min(_limits.maxComputeWorkGroupSizeX,
        Math.Min(_limits.maxComputeWorkGroupSizeY, _limits.maxComputeWorkGroupSizeZ));

    /// <summary>
    /// Gets the maximum compute shared memory size.
    /// </summary>
    public uint MaxSharedMemorySize => _limits.maxComputeSharedMemorySize;

    /// <summary>
    /// Gets the maximum storage buffer range.
    /// </summary>
    public uint MaxStorageBufferRange => _limits.maxStorageBufferRange;

    private VulkanDevice()
    {
    }

    /// <summary>
    /// Initializes the Vulkan device.
    /// </summary>
    /// <returns>True if initialization succeeded.</returns>
    public bool Initialize()
    {
        if (_initialized)
        {
            return true;
        }

        if (!VulkanNativeBindings.IsPlatformSupported)
        {
            return false;
        }

        try
        {
            if (!CreateInstance())
            {
                return false;
            }

            if (!SelectPhysicalDevice())
            {
                Cleanup();
                return false;
            }

            if (!CreateLogicalDevice())
            {
                Cleanup();
                return false;
            }

            if (!CreateCommandPool())
            {
                Cleanup();
                return false;
            }

            if (!CreateFence())
            {
                Cleanup();
                return false;
            }

            _initialized = true;
            return true;
        }
        catch
        {
            Cleanup();
            return false;
        }
    }

    private bool CreateInstance()
    {
        var appNameBytes = Encoding.UTF8.GetBytes("AiDotNet.Tensors\0");
        var engineNameBytes = Encoding.UTF8.GetBytes("DirectGpu\0");

        fixed (byte* pAppName = appNameBytes)
        fixed (byte* pEngineName = engineNameBytes)
        {
            var appInfo = new VkApplicationInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pNext = null,
                pApplicationName = pAppName,
                applicationVersion = 1,
                pEngineName = pEngineName,
                engineVersion = 1,
                apiVersion = (uint)VulkanNativeBindings.VK_API_VERSION_1_0
            };

            var createInfo = new VkInstanceCreateInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pNext = null,
                flags = 0,
                pApplicationInfo = &appInfo,
                enabledLayerCount = 0,
                ppEnabledLayerNames = null,
                enabledExtensionCount = 0,
                ppEnabledExtensionNames = null
            };

            var result = VulkanNativeBindings.vkCreateInstance(&createInfo, IntPtr.Zero, out _instance_vk);
            return result == VulkanNativeBindings.VK_SUCCESS;
        }
    }

    private bool SelectPhysicalDevice()
    {
        uint deviceCount = 0;
        var result = VulkanNativeBindings.vkEnumeratePhysicalDevices(_instance_vk, ref deviceCount, null);
        if (result != VulkanNativeBindings.VK_SUCCESS || deviceCount == 0)
        {
            return false;
        }

        var devices = stackalloc IntPtr[(int)deviceCount];
        result = VulkanNativeBindings.vkEnumeratePhysicalDevices(_instance_vk, ref deviceCount, devices);
        if (result != VulkanNativeBindings.VK_SUCCESS)
        {
            return false;
        }

        // Find best device - prefer discrete GPU
        IntPtr bestDevice = IntPtr.Zero;
        int bestScore = -1;

        for (int i = 0; i < deviceCount; i++)
        {
            var device = devices[i];
            VkPhysicalDeviceProperties props;
            VulkanNativeBindings.vkGetPhysicalDeviceProperties(device, &props);

            // Check for compute queue
            if (!HasComputeQueue(device))
            {
                continue;
            }

            int score = 0;
            switch (props.deviceType)
            {
                case VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                    score = 1000;
                    break;
                case VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                    score = 100;
                    break;
                case VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                    score = 50;
                    break;
                default:
                    score = 10;
                    break;
            }

            // Add score based on limits
            score += (int)(props.limits.maxComputeWorkGroupInvocations / 256);

            if (score > bestScore)
            {
                bestScore = score;
                bestDevice = device;
            }
        }

        if (bestDevice == IntPtr.Zero)
        {
            return false;
        }

        _physicalDevice = bestDevice;

        // Get device properties
        VkPhysicalDeviceProperties deviceProps;
        VulkanNativeBindings.vkGetPhysicalDeviceProperties(_physicalDevice, &deviceProps);
        _limits = deviceProps.limits;

        // Extract device name
        _deviceName = GetDeviceName(&deviceProps);
        _vendorName = GetVendorName(deviceProps.vendorID);

        // Get memory properties
        fixed (VkPhysicalDeviceMemoryProperties* pMemProps = &_memoryProperties)
        {
            VulkanNativeBindings.vkGetPhysicalDeviceMemoryProperties(_physicalDevice, pMemProps);
        }

        // Find compute queue family
        _computeQueueFamily = FindComputeQueueFamily(_physicalDevice);

        return true;
    }

    private static string GetDeviceName(VkPhysicalDeviceProperties* props)
    {
        // Find null terminator
        int length = 0;
        for (int i = 0; i < 256; i++)
        {
            if (props->deviceName[i] == 0)
            {
                break;
            }
            length++;
        }

        if (length == 0)
        {
            return string.Empty;
        }

        // Copy to managed array for .NET Framework compatibility
        var nameBytes = new byte[length];
        for (int i = 0; i < length; i++)
        {
            nameBytes[i] = props->deviceName[i];
        }

        return Encoding.UTF8.GetString(nameBytes);
    }

    private static string GetVendorName(uint vendorId) => vendorId switch
    {
        0x1002 => "AMD",
        0x1010 => "ImgTec",
        0x10DE => "NVIDIA",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x8086 => "Intel",
        _ => $"Unknown (0x{vendorId:X4})"
    };

    private bool HasComputeQueue(IntPtr device)
    {
        uint queueFamilyCount = 0;
        VulkanNativeBindings.vkGetPhysicalDeviceQueueFamilyProperties(device, ref queueFamilyCount, null);
        if (queueFamilyCount == 0)
        {
            return false;
        }

        var queueFamilies = stackalloc VkQueueFamilyProperties[(int)queueFamilyCount];
        VulkanNativeBindings.vkGetPhysicalDeviceQueueFamilyProperties(device, ref queueFamilyCount, queueFamilies);

        for (int i = 0; i < queueFamilyCount; i++)
        {
            if ((queueFamilies[i].queueFlags & VulkanNativeBindings.VK_QUEUE_COMPUTE_BIT) != 0)
            {
                return true;
            }
        }

        return false;
    }

    private uint FindComputeQueueFamily(IntPtr device)
    {
        uint queueFamilyCount = 0;
        VulkanNativeBindings.vkGetPhysicalDeviceQueueFamilyProperties(device, ref queueFamilyCount, null);

        var queueFamilies = stackalloc VkQueueFamilyProperties[(int)queueFamilyCount];
        VulkanNativeBindings.vkGetPhysicalDeviceQueueFamilyProperties(device, ref queueFamilyCount, queueFamilies);

        // Prefer dedicated compute queue (no graphics)
        for (uint i = 0; i < queueFamilyCount; i++)
        {
            if ((queueFamilies[i].queueFlags & VulkanNativeBindings.VK_QUEUE_COMPUTE_BIT) != 0 &&
                (queueFamilies[i].queueFlags & VulkanNativeBindings.VK_QUEUE_GRAPHICS_BIT) == 0)
            {
                return i;
            }
        }

        // Fall back to any compute queue
        for (uint i = 0; i < queueFamilyCount; i++)
        {
            if ((queueFamilies[i].queueFlags & VulkanNativeBindings.VK_QUEUE_COMPUTE_BIT) != 0)
            {
                return i;
            }
        }

        return 0;
    }

    private bool CreateLogicalDevice()
    {
        float queuePriority = 1.0f;

        var queueCreateInfo = new VkDeviceQueueCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext = null,
            flags = 0,
            queueFamilyIndex = _computeQueueFamily,
            queueCount = 1,
            pQueuePriorities = &queuePriority
        };

        var deviceCreateInfo = new VkDeviceCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext = null,
            flags = 0,
            queueCreateInfoCount = 1,
            pQueueCreateInfos = &queueCreateInfo,
            enabledLayerCount = 0,
            ppEnabledLayerNames = null,
            enabledExtensionCount = 0,
            ppEnabledExtensionNames = null,
            pEnabledFeatures = null
        };

        var result = VulkanNativeBindings.vkCreateDevice(_physicalDevice, &deviceCreateInfo, IntPtr.Zero, out _device);
        if (result != VulkanNativeBindings.VK_SUCCESS)
        {
            return false;
        }

        VulkanNativeBindings.vkGetDeviceQueue(_device, _computeQueueFamily, 0, out _computeQueue);
        return _computeQueue != IntPtr.Zero;
    }

    private bool CreateCommandPool()
    {
        var createInfo = new VkCommandPoolCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext = null,
            flags = VkCommandPoolCreateFlags.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex = _computeQueueFamily
        };

        var result = VulkanNativeBindings.vkCreateCommandPool(_device, &createInfo, IntPtr.Zero, out _commandPool);
        return result == VulkanNativeBindings.VK_SUCCESS;
    }

    private bool CreateFence()
    {
        var createInfo = new VkFenceCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            pNext = null,
            flags = 0
        };

        var result = VulkanNativeBindings.vkCreateFence(_device, &createInfo, IntPtr.Zero, out _fence);
        return result == VulkanNativeBindings.VK_SUCCESS;
    }

    /// <summary>
    /// Finds a suitable memory type for allocation.
    /// </summary>
    /// <param name="typeFilter">Memory type bits from requirements.</param>
    /// <param name="properties">Required memory property flags.</param>
    /// <returns>Memory type index, or -1 if not found.</returns>
    public int FindMemoryType(uint typeFilter, uint properties)
    {
        for (int i = 0; i < _memoryProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1u << i)) != 0)
            {
                var memType = _memoryProperties.GetMemoryType(i);
                if ((memType.propertyFlags & properties) == properties)
                {
                    return i;
                }
            }
        }
        return -1;
    }

    /// <summary>
    /// Waits for the device to become idle.
    /// </summary>
    public void WaitIdle()
    {
        if (_device != IntPtr.Zero)
        {
            VulkanNativeBindings.vkDeviceWaitIdle(_device);
        }
    }

    /// <summary>
    /// Submits a command buffer and waits for completion.
    /// </summary>
    public void SubmitAndWait(IntPtr commandBuffer)
    {
        if (_device == IntPtr.Zero || _computeQueue == IntPtr.Zero || _fence == IntPtr.Zero)
        {
            return;
        }

        // Reset fence
        var fencePtr = _fence;
        VulkanNativeBindings.vkResetFences(_device, 1, &fencePtr);

        var submitInfo = new VkSubmitInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext = null,
            waitSemaphoreCount = 0,
            pWaitSemaphores = null,
            pWaitDstStageMask = null,
            commandBufferCount = 1,
            pCommandBuffers = &commandBuffer,
            signalSemaphoreCount = 0,
            pSignalSemaphores = null
        };

        VulkanNativeBindings.vkQueueSubmit(_computeQueue, 1, &submitInfo, _fence);
        VulkanNativeBindings.vkWaitForFences(_device, 1, &fencePtr, 1, ulong.MaxValue);
    }

    private void Cleanup()
    {
        if (_fence != IntPtr.Zero)
        {
            VulkanNativeBindings.vkDestroyFence(_device, _fence, IntPtr.Zero);
            _fence = IntPtr.Zero;
        }

        if (_commandPool != IntPtr.Zero)
        {
            VulkanNativeBindings.vkDestroyCommandPool(_device, _commandPool, IntPtr.Zero);
            _commandPool = IntPtr.Zero;
        }

        if (_device != IntPtr.Zero)
        {
            VulkanNativeBindings.vkDestroyDevice(_device, IntPtr.Zero);
            _device = IntPtr.Zero;
        }

        if (_instance_vk != IntPtr.Zero)
        {
            VulkanNativeBindings.vkDestroyInstance(_instance_vk, IntPtr.Zero);
            _instance_vk = IntPtr.Zero;
        }
    }

    /// <summary>
    /// Disposes the Vulkan device and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        WaitIdle();
        Cleanup();
        _initialized = false;
    }

    public override string ToString()
    {
        return $"VulkanDevice[{_vendorName} {_deviceName}]";
    }
}
