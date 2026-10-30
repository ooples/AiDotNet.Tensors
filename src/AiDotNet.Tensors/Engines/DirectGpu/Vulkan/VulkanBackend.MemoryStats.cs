// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.DevicePrimitives;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IDeviceMemoryAccounted
{
    /// <inheritdoc/>
    DeviceMemoryAccount IDeviceMemoryAccounted.MemoryAccount => VulkanGpuBuffer.Account;
}
