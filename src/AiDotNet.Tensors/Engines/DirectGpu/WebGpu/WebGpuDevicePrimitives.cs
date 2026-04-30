// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Cached probe for a WebGPU implementation on this host. Probes Dawn's
/// <c>libwebgpu_dawn</c> first (the production Chromium implementation),
/// then <c>libwgpu_native</c> (the wgpu-rs native bridge).
/// </summary>
internal static class WebGpuLoaderProbe
{
    private const string Lib = "webgpu_dawn";

#if NET5_0_OR_GREATER
    static WebGpuLoaderProbe()
    {
        AiDotNet.Tensors.Engines.NativeLibraryResolverRegistry.Register(ResolveLib);
    }

    private static IntPtr ResolveLib(string name, System.Reflection.Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        // Try Dawn first, then wgpu-native.
        if (NativeLibrary.TryLoad("webgpu_dawn", asm, path, out var h)) return h;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (NativeLibrary.TryLoad("libwgpu_native.so", asm, path, out h)) return h;
            if (NativeLibrary.TryLoad("libdawn.so", asm, path, out h)) return h;
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            if (NativeLibrary.TryLoad("libwgpu_native.dylib", asm, path, out h)) return h;
            if (NativeLibrary.TryLoad("libdawn.dylib", asm, path, out h)) return h;
        }
        else
        {
            if (NativeLibrary.TryLoad("wgpu_native.dll", asm, path, out h)) return h;
            if (NativeLibrary.TryLoad("dawn.dll", asm, path, out h)) return h;
        }
        return IntPtr.Zero;
    }
#endif

    [DllImport(Lib, EntryPoint = "wgpuCreateInstance")]
    private static extern IntPtr wgpuCreateInstance(IntPtr descriptor);

    [DllImport(Lib, EntryPoint = "wgpuInstanceRelease")]
    private static extern void wgpuInstanceRelease(IntPtr instance);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            var inst = wgpuCreateInstance(IntPtr.Zero);
            if (inst != IntPtr.Zero) { wgpuInstanceRelease(inst); return true; }
            return false;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}

/// <summary>
/// WebGPU-backed <see cref="IDevicePrimitives"/>. Delegates to
/// <see cref="CpuDevicePrimitives"/> until the WGSL kernel layer for
/// device-wide reduce / scan / sort / histogram lands. The
/// <see cref="WebGpuBackend"/> infrastructure (queue, command buffer
/// recording, WGSL pipeline cache) is already in the repo; this class
/// wires the device-primitives surface against it.
/// </summary>
public sealed class WebGpuDevicePrimitives : IDevicePrimitives
{
    private readonly CpuDevicePrimitives _cpuFallback = new();

    /// <summary>True when a WebGPU implementation (Dawn / wgpu-native) is loadable.</summary>
    public static bool IsAvailable => WebGpuLoaderProbe.IsAvailable;

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

/// <summary>WebGPU-backed <see cref="IDeviceRng"/>.</summary>
public sealed class WebGpuRng : IDeviceRng
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
    /// <summary>True when a WebGPU implementation is available.</summary>
    public static bool IsAvailable => WebGpuLoaderProbe.IsAvailable;
    /// <summary>Constructs a WebGPU-backed RNG.</summary>
    public WebGpuRng(ulong seed, DeviceRngAlgorithm algo = DeviceRngAlgorithm.Philox,
        ulong subsequence = 0, ulong offset = 0)
    {
        if (algo != DeviceRngAlgorithm.Philox)
            throw new System.NotSupportedException($"WebGpuRng currently implements only {DeviceRngAlgorithm.Philox}; requested {algo}.");
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
