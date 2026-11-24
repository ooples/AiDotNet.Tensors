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
}
