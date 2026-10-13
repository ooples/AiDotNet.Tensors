// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// The CPU generator for each <see cref="DeviceRngAlgorithm"/>. Every backend's RNG wrapper creates its generator
/// here, so one (algorithm, seed, subsequence, offset) gives the same numbers on every backend.
/// </summary>
public static class CpuDeviceRng
{
    /// <summary>Whether <paramref name="algorithm"/> has a generator.</summary>
    public static bool IsSupported(DeviceRngAlgorithm algorithm)
        => algorithm is DeviceRngAlgorithm.Philox or DeviceRngAlgorithm.Mrg32K3A;

    /// <summary>Creates the generator for <paramref name="algorithm"/>.</summary>
    /// <exception cref="NotSupportedException">The algorithm has no generator yet.</exception>
    public static IDeviceRng Create(DeviceRngAlgorithm algorithm, ulong seed, ulong subsequence = 0, ulong offset = 0)
        => algorithm switch
        {
            DeviceRngAlgorithm.Philox => new CpuPhiloxGenerator(seed, subsequence, offset),
            DeviceRngAlgorithm.Mrg32K3A => new CpuMrg32k3aGenerator(seed, subsequence, offset),
            _ => throw new NotSupportedException(
                $"No generator for {algorithm} yet; {DeviceRngAlgorithm.Philox} and {DeviceRngAlgorithm.Mrg32K3A} are implemented."),
        };
}
