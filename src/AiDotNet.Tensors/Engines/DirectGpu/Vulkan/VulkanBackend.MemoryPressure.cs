// Copyright (c) CrowdTrainer LLC. All rights reserved.
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

    // The Vulkan buffer pool is process-wide; wait for pending submissions and drain finalizer-queued frees first so
    // in-flight released buffers are reclaimed too.
    int IDeviceMemoryPressure.ReleaseCachedBuffers()
    {
        if (IsAvailable) _device.WaitForPendingSubmissions();
        VulkanGpuBuffer.PendingFrees.Drain();
        return VulkanGpuBuffer.Pool.DrainAll();
    }

    /// <remarks>VulkanGpuBuffer.Rent collects unreachable buffers, drains the pool and retries once on out-of-memory.</remarks>
    bool IDeviceMemoryPressure.ReclaimsOnAllocationFailure => true;
}
