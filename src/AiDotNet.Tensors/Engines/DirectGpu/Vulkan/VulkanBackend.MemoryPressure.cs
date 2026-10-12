// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Free memory needs VK_EXT_memory_budget, which this backend does not enable. Pressure-gated reclamation therefore never triggers; the unconditional sweep still runs.
public sealed unsafe partial class VulkanBackend : IDeviceMemoryPressure
{
    bool IDeviceMemoryPressure.TryGetDeviceMemory(out ulong free, out ulong total)
    {
        free = 0; total = 0;
        return false;
    }

    void IDeviceMemoryPressure.ReclaimUnderPressure()
    {
    }

    void IDeviceMemoryPressure.ReclaimNow() => DeviceMemorySweep.Run();

    bool IDeviceMemoryPressure.ReclaimsOnAllocationFailure => false;
}
