// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The device's utilization as its management library samples it (NVML on CUDA, as <c>nvidia-smi</c> shows): the
/// percentage of the last sample period in which a kernel ran, and in which device memory was read or written. These
/// cover every process on the device. A backend whose platform exposes no such counter answers false.
/// </summary>
internal interface IDeviceUtilization
{
    /// <summary>The device's kernel and memory-traffic utilization in percent, or false when it cannot be read.</summary>
    bool TryGetDeviceUtilization(out int gpuPercent, out int memoryPercent);

    /// <summary>
    /// The device's memory in use by every process and its total in bytes, as the management library reports them
    /// (NVML on CUDA: nvidia-smi's memory.used), or false when it cannot be read.
    /// </summary>
    bool TryGetDeviceMemoryInUse(out long usedBytes, out long totalBytes);
}
