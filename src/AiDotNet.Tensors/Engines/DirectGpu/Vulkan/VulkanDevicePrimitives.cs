// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Cached probe for a Vulkan loader on this host.
/// </summary>
internal static class VulkanLoaderProbe
{
    private const string Lib = "vulkan-1";

#if NET5_0_OR_GREATER
    static VulkanLoaderProbe()
    {
        NativeLibrary.SetDllImportResolver(typeof(VulkanLoaderProbe).Assembly, ResolveLib);
    }

    private static IntPtr ResolveLib(string name, System.Reflection.Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (NativeLibrary.TryLoad("libvulkan.so.1", asm, path, out var h)) return h;
            if (NativeLibrary.TryLoad("libvulkan.so", asm, path, out h)) return h;
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            if (NativeLibrary.TryLoad("libvulkan.1.dylib", asm, path, out var h)) return h;
            if (NativeLibrary.TryLoad("libMoltenVK.dylib", asm, path, out h)) return h;
        }
        return IntPtr.Zero;
    }
#endif

    [DllImport(Lib)]
    private static extern int vkEnumerateInstanceVersion(out uint pApiVersion);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            // vkEnumerateInstanceVersion exists on every modern Vulkan loader.
            // We don't need to actually create an instance — successful loader
            // load + entry-point resolution is enough to confirm the runtime
            // is wired.
            int err = vkEnumerateInstanceVersion(out _);
            return err == 0;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}

/// <summary>
/// Vulkan-backed <see cref="IDevicePrimitives"/>. Delegates to
/// <see cref="CpuDevicePrimitives"/> until the SPIR-V kernel layer for
/// device-wide reduce / scan / sort / histogram lands. The
/// <see cref="VulkanBackend"/> infrastructure (queue, descriptor sets,
/// SPIR-V dispatch via <see cref="SpirvBuilder"/>) is in the repo
/// already; this class wires the device-primitives surface against it.
/// </summary>
public sealed class VulkanDevicePrimitives : IDevicePrimitives
{
    private readonly CpuDevicePrimitives _cpuFallback = new();

    /// <summary>True when a Vulkan loader is registered on this host.</summary>
    public static bool IsAvailable => VulkanLoaderProbe.IsAvailable;

    /// <inheritdoc/>
    public Tensor<T> Reduce<T>(Tensor<T> input, int axis = -1, ReductionKind kind = ReductionKind.Sum)
        => _cpuFallback.Reduce(input, axis, kind);

    /// <inheritdoc/>
    public Tensor<T> Scan<T>(Tensor<T> input, int axis = -1, ReductionKind kind = ReductionKind.Sum, bool exclusive = false)
        => _cpuFallback.Scan(input, axis, kind, exclusive);

    /// <inheritdoc/>
    public Tensor<T> Sort<T>(Tensor<T> input, int axis = -1, bool descending = false)
        => _cpuFallback.Sort(input, axis, descending);

    /// <inheritdoc/>
    public Tensor<int> ArgSort<T>(Tensor<T> input, int axis = -1, bool descending = false)
        => _cpuFallback.ArgSort(input, axis, descending);

    /// <inheritdoc/>
    public Tensor<int> Histogram<T>(Tensor<T> input, int bins, T lo, T hi)
        => _cpuFallback.Histogram(input, bins, lo, hi);

    /// <inheritdoc/>
    public (Tensor<T> Values, Tensor<int> Counts) RunLengthEncode<T>(Tensor<T> input)
        => _cpuFallback.RunLengthEncode(input);
}

/// <summary>Vulkan-backed <see cref="IDeviceRng"/>.</summary>
public sealed class VulkanRng : IDeviceRng
{
    private readonly CpuPhiloxGenerator _cpu;
    /// <inheritdoc/>
    public DeviceRngAlgorithm Algorithm { get; }
    /// <inheritdoc/>
    public ulong Seed => _cpu.Seed;
    /// <inheritdoc/>
    public ulong Offset { get => _cpu.Offset; set => _cpu.Offset = value; }
    /// <inheritdoc/>
    public ulong Subsequence { get => _cpu.Subsequence; set => _cpu.Subsequence = value; }
    /// <summary>True when a Vulkan loader is available.</summary>
    public static bool IsAvailable => VulkanLoaderProbe.IsAvailable;
    /// <summary>Constructs a Vulkan-backed RNG.</summary>
    public VulkanRng(ulong seed, DeviceRngAlgorithm algo = DeviceRngAlgorithm.Philox,
        ulong subsequence = 0, ulong offset = 0)
    {
        if (algo != DeviceRngAlgorithm.Philox)
            throw new System.NotSupportedException($"VulkanRng currently implements only {DeviceRngAlgorithm.Philox}; requested {algo}.");
        Algorithm = algo;
        _cpu = new CpuPhiloxGenerator(seed, subsequence, offset);
    }
    /// <inheritdoc/>
    public void Uniform(Tensor<float> output) => _cpu.Uniform(output);
    /// <inheritdoc/>
    public void Uniform(Tensor<double> output) => _cpu.Uniform(output);
    /// <inheritdoc/>
    public void Normal(Tensor<float> output, float mean = 0f, float stddev = 1f) => _cpu.Normal(output, mean, stddev);
    /// <inheritdoc/>
    public void Normal(Tensor<double> output, double mean = 0, double stddev = 1) => _cpu.Normal(output, mean, stddev);
    /// <inheritdoc/>
    public void Bernoulli(Tensor<float> output, float p) => _cpu.Bernoulli(output, p);
}
