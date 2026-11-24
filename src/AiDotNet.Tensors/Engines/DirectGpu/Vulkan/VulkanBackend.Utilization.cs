// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// No device utilization counter is bound for this backend's platform (CUDA reads NVML), so it reports none.
public sealed unsafe partial class VulkanBackend : IDeviceUtilization
{
    bool IDeviceUtilization.TryGetDeviceUtilization(out int gpuPercent, out int memoryPercent)
    {
        gpuPercent = memoryPercent = 0;
        return false;
    }
}
