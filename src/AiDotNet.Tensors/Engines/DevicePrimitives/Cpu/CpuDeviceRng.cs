// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// The CPU generator for each <see cref="DeviceRngAlgorithm"/>. Every backend's RNG wrapper creates its generator
/// here, so one (algorithm, seed, subsequence, offset, dimensions) gives the same numbers on every backend.
/// </summary>
public static class CpuDeviceRng
{
    /// <summary>Whether <paramref name="algorithm"/> has a generator.</summary>
    public static bool IsSupported(DeviceRngAlgorithm algorithm)
        => algorithm is DeviceRngAlgorithm.Philox or DeviceRngAlgorithm.Mrg32K3A
            or DeviceRngAlgorithm.Sobol or DeviceRngAlgorithm.ScrambledSobol;

    /// <summary>Creates the generator for <paramref name="algorithm"/>.</summary>
    /// <param name="algorithm">The algorithm.</param>
    /// <param name="seed">The seed. The plain Sobol sequence has none and ignores it.</param>
    /// <param name="subsequence">The subsequence; for the Sobol generators, the first dimension used.</param>
    /// <param name="offset">The starting offset; for the Sobol generators, the index of the first point.</param>
    /// <param name="dimensions">Dimensions per point for the Sobol generators. The pseudo-random generators have
    /// no dimensions and accept only 1.</param>
    public static IDeviceRng Create(DeviceRngAlgorithm algorithm, ulong seed, ulong subsequence = 0, ulong offset = 0, int dimensions = 1)
    {
        if (dimensions != 1 && algorithm is DeviceRngAlgorithm.Philox or DeviceRngAlgorithm.Mrg32K3A)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, $"{algorithm} is not a quasi-random sequence and has no dimensions.");
        return algorithm switch
        {
            DeviceRngAlgorithm.Philox => new CpuPhiloxGenerator(seed, subsequence, offset),
            DeviceRngAlgorithm.Mrg32K3A => new CpuMrg32k3aGenerator(seed, subsequence, offset),
            DeviceRngAlgorithm.Sobol => new CpuSobolGenerator(dimensions, scrambled: false, seed, subsequence, offset),
            DeviceRngAlgorithm.ScrambledSobol => new CpuSobolGenerator(dimensions, scrambled: true, seed, subsequence, offset),
            _ => throw new NotSupportedException($"No generator for {algorithm}."),
        };
    }
}
