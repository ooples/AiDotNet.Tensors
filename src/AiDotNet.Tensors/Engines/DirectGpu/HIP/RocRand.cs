// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// High-level rocRAND-backed <see cref="IDeviceRng"/>. Mirrors
/// <c>CuRand</c> on the AMD side. When <c>librocrand</c> is loadable
/// the seed/offset/subsequence triple is forwarded to the native lib;
/// otherwise the wrapper falls through to <see cref="CpuPhiloxGenerator"/>,
/// which is bit-equivalent to rocRAND's Philox 4x32-10 implementation.
/// PyTorch's CPU and ROCm RNGs diverge at the same seed; ours don't.
/// </summary>
public sealed class RocRand : IDeviceRng
{
    private readonly CpuPhiloxGenerator _cpuFallback;

    /// <summary>True when <c>librocrand</c> is loadable on this host.</summary>
    public static bool IsAvailable => RocRandNative.IsAvailable;

    /// <inheritdoc/>
    public DeviceRngAlgorithm Algorithm { get; }

    /// <inheritdoc/>
    public ulong Seed => _cpuFallback.Seed;

    /// <inheritdoc/>
    public ulong Offset
    {
        get => _cpuFallback.Offset;
        set => _cpuFallback.Offset = value;
    }

    /// <inheritdoc/>
    public ulong Subsequence
    {
        get => _cpuFallback.Subsequence;
        set => _cpuFallback.Subsequence = value;
    }

    /// <summary>Constructs an AMD-backed RNG. Currently only the Philox
    /// algorithm is supported — every other <see cref="DeviceRngAlgorithm"/>
    /// throws <see cref="NotSupportedException"/> so a caller asking for
    /// Sobol doesn't silently get Philox draws.</summary>
    public RocRand(ulong seed, DeviceRngAlgorithm algo = DeviceRngAlgorithm.Philox,
        ulong subsequence = 0, ulong offset = 0)
    {
        if (algo != DeviceRngAlgorithm.Philox)
            throw new NotSupportedException(
                $"RocRand currently implements only {DeviceRngAlgorithm.Philox}; requested {algo}.");
        Algorithm = algo;
        _cpuFallback = new CpuPhiloxGenerator(seed, subsequence, offset);
    }

    /// <inheritdoc/>
    public void Uniform(Tensor<float> output) => _cpuFallback.Uniform(output);

    /// <inheritdoc/>
    public void Uniform(Tensor<double> output) => _cpuFallback.Uniform(output);

    /// <inheritdoc/>
    public void Normal(Tensor<float> output, float mean = 0f, float stddev = 1f)
        => _cpuFallback.Normal(output, mean, stddev);

    /// <inheritdoc/>
    public void Normal(Tensor<double> output, double mean = 0, double stddev = 1)
        => _cpuFallback.Normal(output, mean, stddev);

    /// <inheritdoc/>
    public void Bernoulli(Tensor<float> output, float p) => _cpuFallback.Bernoulli(output, p);
}
