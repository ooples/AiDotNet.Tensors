// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.DevicePrimitives;

/// <summary>
/// The device's utilization as its management library samples it: on CUDA, NVML's utilization rates (the figures
/// <c>nvidia-smi</c> prints as GPU-Util and memory utilization), over the driver's last sample period and every process
/// on the device.
/// </summary>
public static class GpuUtilization
{
    /// <summary>
    /// Reads <paramref name="backend"/>'s device utilization: <paramref name="gpuPercent"/>, the share of the sample
    /// period in which a kernel was running, and <paramref name="memoryPercent"/>, the share in which device memory was
    /// read or written. False when the platform exposes no counter (every backend but CUDA) or NVML is not installed.
    /// </summary>
    public static bool TryRead(IDirectGpuBackend backend, out int gpuPercent, out int memoryPercent)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        gpuPercent = memoryPercent = 0;
        return backend is IDeviceUtilization utilization && utilization.TryGetDeviceUtilization(out gpuPercent, out memoryPercent);
    }

    /// <summary>
    /// Reads <paramref name="backend"/>'s device memory in use by every process and its total, in bytes, as the
    /// management library reports them: on CUDA, NVML (the memory.used and memory.total <c>nvidia-smi</c> prints).
    /// The driver's free-memory query (<see cref="GpuMemoryStats.TryGetDeviceMemory"/>) is not that figure: under
    /// Windows' display driver model it answers this process's budget, about 1.2 GB off nvidia-smi on an RTX 3080.
    /// False when the platform exposes no counter (every backend but CUDA) or NVML is not installed.
    /// </summary>
    public static bool TryReadDeviceMemory(IDirectGpuBackend backend, out long usedBytes, out long totalBytes)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        usedBytes = totalBytes = 0;
        return backend is IDeviceUtilization utilization && utilization.TryGetDeviceMemoryInUse(out usedBytes, out totalBytes);
    }
}
