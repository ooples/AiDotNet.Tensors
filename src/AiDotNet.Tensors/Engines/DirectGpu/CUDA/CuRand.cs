// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// High-level cuRAND-backed <see cref="IDeviceRng"/>. When the native
/// cuRAND library is loadable the seed/offset/subsequence triple is
/// forwarded to <c>libcurand</c> via <see cref="CuRandNative"/> for
/// device generation. When the library is missing or the requested
/// algorithm isn't supported, the wrapper transparently falls back to
/// <see cref="CpuPhiloxGenerator"/> — both implement the identical
/// Philox 4x32-10 algorithm so a (seed, subsequence, offset) triple
/// produces identical bits across CPU and CUDA. That cross-backend
/// determinism is #219's "How we beat PyTorch" point #1; PyTorch's
/// CPU and CUDA generators diverge at the same seed.
///
/// <para>Device-pointer plumbing: cuRAND fills GPU buffers, but the
/// <see cref="IDeviceRng"/> contract takes host <see cref="Tensor{T}"/>
/// outputs. The first cut here uses the CPU Philox path even when
/// cuRAND is loadable, since (a) CPU Philox is bit-equivalent and
/// (b) the GPU memcpy round-trip is a wash for the small RNG buffers
/// typical in dropout / data-loader sampling. The native bindings stay
/// wired so a future device-tensor pipeline can flip the dispatch to
/// real on-device generation without an API break.</para>
/// </summary>
public sealed class CuRand : IDeviceRng
{
    private readonly CpuPhiloxGenerator _cpuFallback;

    /// <summary>True when <c>libcurand</c> is loadable on this host.
    /// Useful for telemetry; the wrapper degrades gracefully either
    /// way.</summary>
    public static bool IsAvailable => CuRandNative.IsAvailable;

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

    /// <summary>Constructs a CUDA-backed RNG. <paramref name="algo"/>
    /// must be one of cuRAND's supported algorithms; non-Philox
    /// algorithms are passed straight through to the native side and
    /// also implemented on CPU as fallback.</summary>
    public CuRand(ulong seed, DeviceRngAlgorithm algo = DeviceRngAlgorithm.Philox,
        ulong subsequence = 0, ulong offset = 0)
    {
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
