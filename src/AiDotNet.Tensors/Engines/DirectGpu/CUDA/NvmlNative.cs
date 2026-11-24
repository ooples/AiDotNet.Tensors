// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The NVIDIA Management Library calls the utilization read needs. NVML ships with the driver (nvml.dll on Windows,
/// libnvidia-ml.so.1 on Linux); a missing library or entry point reads as "not available". Devices are looked up by
/// UUID, which names the same device under any CUDA_VISIBLE_DEVICES ordering (NVML indices are not CUDA ordinals).
/// </summary>
internal static class NvmlNative
{
    [StructLayout(LayoutKind.Sequential)]
    private struct NvmlUtilization
    {
        public uint Gpu;
        public uint Memory;
    }

    private static class Windows
    {
        [DllImport("nvml", EntryPoint = "nvmlInit_v2")]
        public static extern int Init();

        [DllImport("nvml", EntryPoint = "nvmlDeviceGetHandleByUUID")]
        public static extern int HandleByUuid([MarshalAs(UnmanagedType.LPStr)] string uuid, out IntPtr device);

        [DllImport("nvml", EntryPoint = "nvmlDeviceGetUtilizationRates")]
        public static extern int Utilization(IntPtr device, out NvmlUtilization rates);
    }

    private static class Linux
    {
        [DllImport("libnvidia-ml.so.1", EntryPoint = "nvmlInit_v2")]
        public static extern int Init();

        [DllImport("libnvidia-ml.so.1", EntryPoint = "nvmlDeviceGetHandleByUUID")]
        public static extern int HandleByUuid([MarshalAs(UnmanagedType.LPStr)] string uuid, out IntPtr device);

        [DllImport("libnvidia-ml.so.1", EntryPoint = "nvmlDeviceGetUtilizationRates")]
        public static extern int Utilization(IntPtr device, out NvmlUtilization rates);
    }

    private static readonly bool s_windows = Environment.OSVersion.Platform == PlatformID.Win32NT;
    private static readonly object s_gate = new();
    private static int s_state;   // 0 not tried, 1 initialized, -1 unavailable

    private static bool IsLoadFailure(Exception ex) =>
        ex is DllNotFoundException || ex is EntryPointNotFoundException || ex is BadImageFormatException;

    /// <summary>The NVML handle of the device with this UUID ("GPU-xxxxxxxx-..."), or false when NVML is unavailable.</summary>
    internal static bool TryGetDevice(string uuid, out IntPtr device)
    {
        device = IntPtr.Zero;
        if (!EnsureInitialized()) return false;
        try { return (s_windows ? Windows.HandleByUuid(uuid, out device) : Linux.HandleByUuid(uuid, out device)) == 0; }
        catch (Exception ex) when (IsLoadFailure(ex)) { return false; }
    }

    /// <summary>The device's kernel and memory utilization over NVML's last sample period, in percent.</summary>
    internal static bool TryGetUtilization(IntPtr device, out int gpuPercent, out int memoryPercent)
    {
        gpuPercent = memoryPercent = 0;
        try
        {
            NvmlUtilization rates;
            int status = s_windows ? Windows.Utilization(device, out rates) : Linux.Utilization(device, out rates);
            if (status != 0) return false;
            gpuPercent = (int)rates.Gpu;
            memoryPercent = (int)rates.Memory;
            return true;
        }
        catch (Exception ex) when (IsLoadFailure(ex)) { return false; }
    }

    private static bool EnsureInitialized()
    {
        if (s_state != 0) return s_state > 0;
        lock (s_gate)
        {
            if (s_state != 0) return s_state > 0;
            try { s_state = (s_windows ? Windows.Init() : Linux.Init()) == 0 ? 1 : -1; }
            catch (Exception ex) when (IsLoadFailure(ex)) { s_state = -1; }
            return s_state > 0;
        }
    }
}
