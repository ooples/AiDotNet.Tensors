// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// MPS-backed <see cref="IDeviceRng"/>. Apple-platform mirror of
/// <c>CuRand</c> / <c>RocRand</c>. The MPS Philox RNG kernel produces
/// Philox 4x32-10 bits identical to ours when seeded the same way, so
/// the cross-vendor determinism property holds: the same
/// (seed, subsequence, offset) triple yields the same draws on
/// CPU / CUDA / ROCm / MPS.
/// </summary>
public sealed class MpsRng : IDeviceRng
{
    private readonly CpuPhiloxGenerator _cpuFallback;

    /// <summary>True when running on macOS and MPS is available.</summary>
    public static bool IsAvailable => MpsRngNative.IsAvailable;

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

    /// <summary>Constructs an MPS-backed RNG.</summary>
    public MpsRng(ulong seed, DeviceRngAlgorithm algo = DeviceRngAlgorithm.Philox,
        ulong subsequence = 0, ulong offset = 0)
    {
        if (algo != DeviceRngAlgorithm.Philox)
            throw new NotSupportedException(
                $"MpsRng currently implements only {DeviceRngAlgorithm.Philox}; requested {algo}.");
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
