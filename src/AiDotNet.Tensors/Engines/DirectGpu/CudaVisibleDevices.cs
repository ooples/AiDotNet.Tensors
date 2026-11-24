using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The NVIDIA GPUs <c>CUDA_VISIBLE_DEVICES</c> leaves visible, as device UUIDs, for backends whose APIs do not read the
/// variable themselves (Vulkan enumerates every GPU in the machine). CUDA and NVIDIA's OpenCL honour it already.
/// </summary>
internal static class CudaVisibleDevices
{
    /// <summary>NVIDIA's PCI vendor id, as Vulkan reports it.</summary>
    internal const uint NvidiaVendorId = 0x10DE;

    private static readonly Lazy<HashSet<string>?> s_visible = new(() => Resolve(
        Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES"), TryCudaVisibleUuids));

    /// <summary>
    /// The visible NVIDIA devices' UUIDs (32 lowercase hex digits), or null when the variable is unset (every device is
    /// visible). An empty set hides every NVIDIA device.
    /// </summary>
    internal static HashSet<string>? VisibleNvidiaUuids => s_visible.Value;

    /// <summary>True when an NVIDIA device with this 16-byte UUID may be used.</summary>
    internal static bool IsVisible(ReadOnlySpan<byte> uuid) => VisibleNvidiaUuids is not { } visible || visible.Contains(Hex(uuid));

    /// <summary>
    /// Resolves the variable's value: through the CUDA driver when it is present (it applies the variable itself, so
    /// indices, full or prefix UUIDs and MIG names all mean what they mean to CUDA), else from the full <c>GPU-</c> UUIDs
    /// in the value. Indices cannot be mapped without the driver; then nothing is filtered.
    /// </summary>
    internal static HashSet<string>? Resolve(string? value, Func<List<string>?> cudaVisibleUuids)
    {
        if (value is null) return null;
        if (cudaVisibleUuids() is { } fromDriver) return new HashSet<string>(fromDriver, StringComparer.Ordinal);
        var fromValue = new HashSet<string>(StringComparer.Ordinal);
        foreach (var raw in value.Split(','))
        {
            var entry = raw.Trim();
            if (entry.Length == 0) continue;
            if (!entry.StartsWith("GPU-", StringComparison.OrdinalIgnoreCase)) return null;
            var hex = entry.Substring(4).Replace("-", string.Empty).ToLowerInvariant();
            if (hex.Length != 32) return null;
            fromValue.Add(hex);
        }
        return fromValue;
    }

    internal static string Hex(ReadOnlySpan<byte> uuid)
    {
        var chars = new char[uuid.Length * 2];
        for (int i = 0; i < uuid.Length; i++)
        {
            chars[2 * i] = "0123456789abcdef"[uuid[i] >> 4];
            chars[2 * i + 1] = "0123456789abcdef"[uuid[i] & 15];
        }
        return new string(chars);
    }

    private static unsafe List<string>? TryCudaVisibleUuids()
    {
        try
        {
            if (CuBlasNative.cuInit(0) != CudaResult.Success) return null;
            if (CuBlasNative.cuDeviceGetCount(out int count) != CudaResult.Success) return null;
            var uuids = new List<string>(count);
            for (int ordinal = 0; ordinal < count; ordinal++)
            {
                if (CuBlasNative.cuDeviceGet(out int device, ordinal) != CudaResult.Success) return null;
                if (CudaNativeBindings.cuDeviceGetUuidV2(out var uuid, device) != CudaResult.Success) return null;
                uuids.Add(Hex(new ReadOnlySpan<byte>(uuid.Bytes, 16)));
            }
            return uuids;
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            return null;
        }
    }
}
