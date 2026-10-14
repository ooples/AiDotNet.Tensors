// Copyright (c) AiDotNet. All rights reserved.
// CLBlast device info mapping (vendor/architecture/device name normalization).
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

internal readonly struct ClBlastDeviceInfo
{
    public string Type { get; }
    public string Vendor { get; }
    public string Architecture { get; }
    public string DeviceName { get; }

    internal ClBlastDeviceInfo(string type, string vendor, string architecture, string deviceName)
    {
        Type = type;
        Vendor = vendor;
        Architecture = architecture;
        DeviceName = deviceName;
    }

    public static ClBlastDeviceInfo FromContext(DirectOpenClContext context)
    {
        string vendor = NormalizeVendor(context.DeviceVendor);
        string type = NormalizeDeviceType(context.DeviceType);

        bool hasAmdAttributes = HasExtension(context.Extensions, "cl_amd_device_attribute_query");
        bool hasNvAttributes = HasExtension(context.Extensions, "cl_nv_device_attribute_query");

        string rawDeviceName = context.DeviceName ?? string.Empty;
        string architecture = string.Empty;
        if (hasNvAttributes)
        {
            uint major = OpenClNativeBindings.GetDeviceInfoUInt(context.Device, OpenClNativeBindings.CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV);
            uint minor = OpenClNativeBindings.GetDeviceInfoUInt(context.Device, OpenClNativeBindings.CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV);
            if (major > 0 || minor > 0)
                architecture = $"SM{major}.{minor}";
        }
        else if (hasAmdAttributes)
        {
            architecture = rawDeviceName;
        }

        architecture = NormalizeArchitecture(architecture);

        string deviceName = rawDeviceName;
        if (hasAmdAttributes && !string.IsNullOrWhiteSpace(context.DeviceBoardName))
            deviceName = context.DeviceBoardName;

        deviceName = NormalizeDeviceName(deviceName);
        return new ClBlastDeviceInfo(type, vendor, architecture, deviceName);
    }

    /// <summary>
    /// The same database key an AMD device gets through OpenCL (vendor "AMD", the gfx target as the architecture, the board
    /// name as the device), so a HIP backend on that device reads the parameters CLBlast tuned for it (#1266).
    /// </summary>
    internal static ClBlastDeviceInfo ForAmdGpu(string gfxTarget, string boardName)
    {
        string architecture = NormalizeArchitecture(gfxTarget ?? string.Empty);
        string deviceName = NormalizeDeviceName(string.IsNullOrWhiteSpace(boardName) ? gfxTarget ?? string.Empty : boardName);
        return new ClBlastDeviceInfo("GPU", "AMD", architecture, deviceName);
    }
    private static bool HasExtension(string extensions, string extension)
    {
        return extensions.IndexOf(extension, StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private static string NormalizeVendor(string vendor)
    {
        return vendor switch
        {
            "Intel(R) Corporation" => "Intel",
            "GenuineIntel" => "Intel",
            "Advanced Micro Devices, Inc." => "AMD",
            "NVIDIA Corporation" => "NVIDIA",
            _ => string.IsNullOrWhiteSpace(vendor) ? "default" : vendor
        };
    }

    private static string NormalizeArchitecture(string architecture)
    {
        if (string.IsNullOrWhiteSpace(architecture))
            return string.Empty;

        return architecture switch
        {
            "gfx803" => "Fiji",
            "gfx900" => "Vega",
            _ => architecture
        };
    }

    private static string NormalizeDeviceName(string deviceName)
    {
        if (string.IsNullOrEmpty(deviceName))
            return deviceName;

        // net471 compatible: Contains and Replace with StringComparison are .NET Core 2.1+
        if (deviceName.IndexOf("pthread-", StringComparison.Ordinal) >= 0)
            deviceName = deviceName.Replace("pthread-", string.Empty);

        return deviceName.Trim();
    }

    private static string NormalizeDeviceType(ulong deviceType)
    {
        if ((deviceType & OpenClNativeBindings.CL_DEVICE_TYPE_GPU) != 0)
            return "GPU";
        if ((deviceType & OpenClNativeBindings.CL_DEVICE_TYPE_CPU) != 0)
            return "CPU";
        return "default";
    }
}
