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

    [StructLayout(LayoutKind.Sequential)]
    private struct NvmlMemory
    {
        public ulong Total;
        public ulong Free;
        public ulong Used;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NvmlMemoryV2
    {
        public uint Version;
        public ulong Total;
        public ulong Reserved;
        public ulong Free;
        public ulong Used;
    }

    // NVML_STRUCT_VERSION(Memory, 2): the struct size with the version in the top byte.
    private static readonly uint s_memoryV2Version = (uint)Marshal.SizeOf<NvmlMemoryV2>() | (2u << 24);

    private static class Windows
    {
        [DllImport("nvml", EntryPoint = "nvmlDeviceGetMemoryInfo")]
        public static extern int MemoryInfo(IntPtr device, out NvmlMemory memory);

        [DllImport("nvml", EntryPoint = "nvmlDeviceGetMemoryInfo_v2")]
        public static extern int MemoryInfoV2(IntPtr device, ref NvmlMemoryV2 memory);

        [DllImport("nvml", EntryPoint = "nvmlInit_v2")]
        public static extern int Init();

        [DllImport("nvml", EntryPoint = "nvmlDeviceGetHandleByUUID")]
        public static extern int HandleByUuid([MarshalAs(UnmanagedType.LPStr)] string uuid, out IntPtr device);

        [DllImport("nvml", EntryPoint = "nvmlDeviceGetUtilizationRates")]
        public static extern int Utilization(IntPtr device, out NvmlUtilization rates);
    }

    private static class Linux
    {
        [DllImport("libnvidia-ml.so.1", EntryPoint = "nvmlDeviceGetMemoryInfo")]
        public static extern int MemoryInfo(IntPtr device, out NvmlMemory memory);

        [DllImport("libnvidia-ml.so.1", EntryPoint = "nvmlDeviceGetMemoryInfo_v2")]
        public static extern int MemoryInfoV2(IntPtr device, ref NvmlMemoryV2 memory);

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

    /// <summary>
    /// The device's memory in use by every process and its total, in bytes, as <c>nvidia-smi</c> prints them
    /// (memory.used and memory.total). NVML's version 2 query leaves the driver's own reservation out of "used", as
    /// nvidia-smi does; a driver without it answers the version 1 figure.
    /// </summary>
    internal static bool TryGetMemory(IntPtr device, out long usedBytes, out long totalBytes)
    {
        usedBytes = totalBytes = 0;
        try
        {
            var v2 = new NvmlMemoryV2 { Version = s_memoryV2Version };
            int status;
            try { status = s_windows ? Windows.MemoryInfoV2(device, ref v2) : Linux.MemoryInfoV2(device, ref v2); }
            catch (EntryPointNotFoundException) { status = -1; }
            if (status == 0)
            {
                usedBytes = (long)v2.Used;
                totalBytes = (long)v2.Total;
                return true;
            }
            NvmlMemory v1;
            if ((s_windows ? Windows.MemoryInfo(device, out v1) : Linux.MemoryInfo(device, out v1)) != 0) return false;
            usedBytes = (long)v1.Used;
            totalBytes = (long)v1.Total;
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
