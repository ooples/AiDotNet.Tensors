// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// No device utilization counter is bound for this backend's platform (CUDA reads NVML), so it reports none.
public sealed partial class MetalBackend : IDeviceUtilization
{
    bool IDeviceUtilization.TryGetDeviceUtilization(out int gpuPercent, out int memoryPercent)
    {
        gpuPercent = memoryPercent = 0;
        return false;
    }

    bool IDeviceUtilization.TryGetDeviceMemoryInUse(out long usedBytes, out long totalBytes)
    {
        usedBytes = totalBytes = 0;
        return false;
    }
}
