// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Globalization;

namespace AiDotNet.Tensors.Engines.DevicePrimitives;

/// <summary>
/// A per-process cap on the GPU memory this library reserves on a device -- PyTorch's
/// <c>torch.cuda.set_per_process_memory_fraction</c>. Two trainings can then share one GPU without either taking the
/// other's memory: an allocation that would take this process's reserved memory past its budget first releases cached
/// and unreachable memory (as a driver out-of-memory does) and, still over, fails with the backend's out-of-memory
/// error instead of growing. On Windows that matters beyond fairness: past the device's memory the driver silently
/// backs allocations with system memory, and both processes slow down instead of one failing.
/// </summary>
/// <remarks>
/// <para>The budget counts reserved memory, as PyTorch's does: memory in use plus what the library's pools keep for
/// reuse (CUDA's stream-ordered pool included). It does not count the driver context or library handles, nor memory
/// other processes hold, and it partitions memory only: kernels of processes sharing a GPU still take turns on it.</para>
/// <para>A request is held against the budget from the moment it is admitted, so threads allocating at the same
/// moment cannot together pass it.</para>
/// <para>Devices are named <c>backend:index</c> in lower case, as <see cref="GpuMemoryStats"/> names them
/// (<c>cuda:0</c>, <c>hip:1</c>, <c>opencl:0</c>, <c>vulkan:0</c>, <c>metal:0</c>, <c>webgpu:0</c>). The environment
/// variable <c>AIDOTNET_GPU_MEMORY_FRACTION</c> (a fraction in (0, 1]) sets a fraction for every device, so a launcher
/// can budget a process it does not otherwise configure; a limit set through this class takes precedence.</para>
/// </remarks>
public static class GpuMemoryBudget
{
    private static readonly ConcurrentDictionary<string, long> s_limits = new(StringComparer.Ordinal);
    private static readonly ConcurrentDictionary<string, double> s_fractions = new(StringComparer.Ordinal);
    // Parsed on first use; a malformed value throws there, every time, rather than failing the type initializer.
    private static readonly Lazy<double> s_environmentFraction = new(ReadEnvironmentFraction);

    /// <summary>The environment variable that sets a fraction budget for every device.</summary>
    public const string FractionEnvironmentVariable = "AIDOTNET_GPU_MEMORY_FRACTION";

    /// <summary>
    /// Caps this process's reserved memory on <paramref name="device"/> at <paramref name="fraction"/> of the
    /// device's total memory (as its backend reports it). Replaces any byte limit set for the device.
    /// </summary>
    /// <param name="fraction">In (0, 1].</param>
    /// <param name="device">A device name such as <c>cuda:0</c>.</param>
    public static void SetPerProcessMemoryFraction(double fraction, string device = "cuda:0")
    {
        if (!(fraction > 0 && fraction <= 1)) throw new ArgumentOutOfRangeException(nameof(fraction), fraction, "A fraction in (0, 1].");
        // A fraction too small to be one byte of the device still is a budget: one that admits nothing.
        string key = Key(device);
        s_limits.TryRemove(key, out _);
        s_fractions[key] = fraction;
    }

    /// <summary>Caps this process's reserved memory on <paramref name="device"/> at <paramref name="bytes"/>.
    /// Replaces any fraction set for the device.</summary>
    public static void SetMemoryLimit(long bytes, string device = "cuda:0")
    {
        if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes), bytes, "A positive number of bytes.");
        string key = Key(device);
        s_fractions.TryRemove(key, out _);
        s_limits[key] = bytes;
    }

    /// <summary>Removes the budget set for <paramref name="device"/> through this class (the environment's fraction,
    /// when set, applies again).</summary>
    public static void ClearLimit(string device = "cuda:0")
    {
        string key = Key(device);
        s_limits.TryRemove(key, out _);
        s_fractions.TryRemove(key, out _);
    }

    /// <summary>
    /// The budget in bytes on <paramref name="device"/>, or null when it has none or its fraction cannot be resolved
    /// yet (no backend on the device has reported its total memory). Zero is a budget: a fraction smaller than one
    /// byte of the device, under which every allocation is refused.
    /// </summary>
    public static long? GetLimitBytes(string device = "cuda:0")
    {
        string key = Key(device);
        long total = DeviceMemoryAccount.TryGet(key, out var account) ? account.DeviceTotalBytes : 0;
        return LimitFor(key, total);
    }

    /// <summary>This process's reserved memory on <paramref name="device"/> now, in bytes (0 for a device no backend
    /// has used).</summary>
    public static long GetReservedBytes(string device = "cuda:0")
        => DeviceMemoryAccount.TryGet(Key(device), out var account) ? account.ReservedBytes : 0;

    // The limit for a device: an explicit byte limit, else a fraction (this class's, else the environment's) of the
    // device's total memory. Null: none, or a fraction with no total to resolve it against. Zero is a limit (a
    // fraction of less than a byte), not the absence of one.
    internal static long? LimitFor(string device, long deviceTotalBytes)
    {
        if (s_limits.TryGetValue(device, out long bytes)) return bytes;
        double fraction = s_fractions.TryGetValue(device, out double f) ? f : s_environmentFraction.Value;
        if (fraction <= 0 || deviceTotalBytes <= 0) return null;
        return (long)Math.Floor(deviceTotalBytes * fraction);
    }

    private static string Key(string device)
    {
        if (string.IsNullOrWhiteSpace(device)) throw new ArgumentException("A device name such as cuda:0.", nameof(device));
        return device.Trim().ToLowerInvariant();
    }

    private static double ReadEnvironmentFraction()
    {
        string? text = Environment.GetEnvironmentVariable(FractionEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(text)) return 0;
        if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double fraction)
            && fraction > 0 && fraction <= 1)
            return fraction;
        throw new InvalidOperationException(
            $"{FractionEnvironmentVariable}={text} is not a fraction in (0, 1].");
    }
}
