// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// OpenCL-backed <see cref="IDevicePrimitives"/>. Delegates to
/// <see cref="CpuDevicePrimitives"/> until the OpenCL kernel layer for
/// device-wide reduce / scan / sort / histogram lands. The
/// <see cref="OpenClBackend"/> infrastructure (queue, device buffer
/// allocation, ClBlast integration) is already in the repo; this class
/// wires the device-primitives surface against it so callers compile
/// against the same <see cref="IDevicePrimitives"/> regardless of
/// backend choice.
///
/// <para>Cross-platform compute test runners (the cross-vendor
/// IDevicePrimitives parity sweep across CUDA / ROCm / MPS / OpenCL /
/// Vulkan / WebGPU) are tracked in #219's continuous-CI follow-up.</para>
/// </summary>
public sealed class OpenClDevicePrimitives : IDevicePrimitives
{
    private readonly CpuDevicePrimitives _cpuFallback = new();

    /// <summary>Whether an OpenCL platform with a usable device is loadable
    /// on this host. Probes via the existing <see cref="OpenClBackend"/>.</summary>
    public static bool IsAvailable => OpenClPlatformProbe.IsAvailable;

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

/// <summary>OpenCL-backed <see cref="IDeviceRng"/> — Philox 4x32-10 via
/// CPU fallback today, with the CL kernel dispatch path wiring in once
/// the device-tensor pipeline lands.</summary>
public sealed class OpenClRng : IDeviceRng
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
    /// <summary>True when an OpenCL device is loadable.</summary>
    public static bool IsAvailable => OpenClPlatformProbe.IsAvailable;
    /// <summary>Constructs an OpenCL-backed RNG. Currently only
    /// <see cref="DeviceRngAlgorithm.Philox"/> is supported.</summary>
    public OpenClRng(ulong seed, DeviceRngAlgorithm algo = DeviceRngAlgorithm.Philox,
        ulong subsequence = 0, ulong offset = 0)
    {
        if (algo != DeviceRngAlgorithm.Philox)
            throw new System.NotSupportedException($"OpenClRng currently implements only {DeviceRngAlgorithm.Philox}; requested {algo}.");
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
