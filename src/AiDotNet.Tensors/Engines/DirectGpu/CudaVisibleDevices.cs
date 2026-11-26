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
        // MIG instances have UUIDs of their own, which never match a Vulkan device's whole-GPU UUID: not filtered.
        if (value.IndexOf("MIG-", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            // A MIG instance is a slice of a device, not a device: the backends that select by device UUID cannot apply it.
            System.Diagnostics.Trace.TraceWarning(
                $"CUDA_VISIBLE_DEVICES='{value}' names MIG instances; backends that select NVIDIA devices by device UUID "
                + "(Vulkan) cannot be restricted to a MIG instance and see every device.");
            return null;
        }
        if (cudaVisibleUuids() is { } fromDriver) return new HashSet<string>(fromDriver, StringComparer.Ordinal);
        var fromValue = new HashSet<string>(StringComparer.Ordinal);
        foreach (var raw in value.Split(','))
        {
            var entry = raw.Trim();
            if (entry.Length == 0) continue;
            if (!entry.StartsWith("GPU-", StringComparison.OrdinalIgnoreCase)) return Unresolved(value);
            var hex = entry.Substring(4).Replace("-", string.Empty).ToLowerInvariant();
            if (hex.Length != 32) return Unresolved(value);
            fromValue.Add(hex);
        }
        return fromValue;
    }

    // The CUDA driver could not say which devices the value names, and it lists ordinals (or malformed UUIDs) that only
    // the driver can map: every NVIDIA device stays visible to the backends that filter by UUID, which is reported.
    private static HashSet<string>? Unresolved(string value)
    {
        System.Diagnostics.Trace.TraceWarning(
            $"CUDA_VISIBLE_DEVICES='{value}' could not be resolved to device UUIDs without the CUDA driver; backends that "
            + "select NVIDIA devices by UUID (Vulkan) are not restricted by it. Set it to GPU-<uuid> entries to apply it.");
        return null;
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
            // No visible device (CUDA_VISIBLE_DEVICES=-1, or ids that name none): every NVIDIA device is hidden.
            var init = CuBlasNative.cuInit(0);
            if (init == CudaResult.NoDevice) return new List<string>();
            if (init != CudaResult.Success) return null;
            var counted = CuBlasNative.cuDeviceGetCount(out int count);
            if (counted == CudaResult.NoDevice) return new List<string>();
            if (counted != CudaResult.Success) return null;
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
            // No CUDA driver to ask: the caller reads full GPU- UUIDs from the variable itself.
            System.Diagnostics.Trace.TraceInformation(
                $"CUDA_VISIBLE_DEVICES could not be resolved through the CUDA driver ({ex.GetType().Name}); only GPU- UUIDs in it are applied to Vulkan.");
            return null;
        }
    }
}
