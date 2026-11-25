// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// NVML's utilization rates for this backend's device, found by the device's UUID.
public sealed partial class CudaBackend : IDeviceUtilization
{
    private IntPtr _nvmlDevice;
    private int _nvmlState;   // 0 not looked up, 1 found, -1 unavailable

    bool IDeviceUtilization.TryGetDeviceUtilization(out int gpuPercent, out int memoryPercent)
    {
        gpuPercent = memoryPercent = 0;
        return HasNvmlDevice() && NvmlNative.TryGetUtilization(_nvmlDevice, out gpuPercent, out memoryPercent);
    }

    bool IDeviceUtilization.TryGetDeviceMemoryInUse(out long usedBytes, out long totalBytes)
    {
        usedBytes = totalBytes = 0;
        return HasNvmlDevice() && NvmlNative.TryGetMemory(_nvmlDevice, out usedBytes, out totalBytes);
    }

    /// <summary>This backend's device UUID ("GPU-..."), or null (tests that query nvidia-smi for the same device).</summary>
    internal string? DeviceUuidForTests => TryDeviceUuid(out string uuid) ? uuid : null;

    private bool HasNvmlDevice()
    {
        if (_nvmlState == 0)
            _nvmlState = TryDeviceUuid(out string uuid) && NvmlNative.TryGetDevice(uuid, out _nvmlDevice) ? 1 : -1;
        return _nvmlState > 0;
    }

    // "GPU-" and the 16 UUID bytes as 8-4-4-4-12 hex digits, the form nvidia-smi and NVML use.
    private unsafe bool TryDeviceUuid(out string uuid)
    {
        uuid = "";
        if (!IsAvailable) return false;
        try
        {
            if (CudaNativeBindings.cuDeviceGetUuidV2(out CudaDeviceUuid raw, _cuDevice) != CudaResult.Success) return false;
            var hex = new System.Text.StringBuilder(36);
            for (int i = 0; i < 16; i++)
            {
                if (i == 4 || i == 6 || i == 8 || i == 10) hex.Append('-');
                hex.Append(raw.Bytes[i].ToString("x2"));
            }
            uuid = "GPU-" + hex;
            return true;
        }
        catch (EntryPointNotFoundException ex)
        {
            // A driver older than cuDeviceGetUuid_v2: no NVML telemetry for this device.
            System.Diagnostics.Trace.TraceInformation($"The CUDA driver cannot report a device UUID ({ex.GetType().Name}); NVML telemetry is off.");
            return false;
        }
    }
}
