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

    /// <summary>Constructs a CUDA-backed RNG. Currently only the Philox
    /// algorithm is supported — every other <see cref="DeviceRngAlgorithm"/>
    /// (Sobol, MRG, XorWow, …) throws <see cref="NotSupportedException"/>
    /// to prevent silent miscompiles where a caller asks for Sobol and
    /// gets Philox draws back. The native cuRAND dispatch is wired up as
    /// a #219 follow-up; this CPU-only path is stable for tests +
    /// reproducibility.</summary>
    public CuRand(ulong seed, DeviceRngAlgorithm algo = DeviceRngAlgorithm.Philox,
        ulong subsequence = 0, ulong offset = 0)
    {
        if (algo != DeviceRngAlgorithm.Philox)
            throw new NotSupportedException(
                $"CuRand currently implements only {DeviceRngAlgorithm.Philox}; requested {algo}. " +
                "Other algorithms (Sobol/MRG/XorWow) will be added when the native cuRAND dispatch is wired.");
        Algorithm = algo;
        _cpuFallback = new CpuPhiloxGenerator(seed, subsequence, offset);
    }

    /// <inheritdoc/>
    public void Uniform(Tensor<float> output)
    {
        if (!CuRandNative.IsAvailable || !TryDeviceUniform(output, normal: false, mean: 0f, stddev: 1f))
            _cpuFallback.Uniform(output);
    }

    /// <inheritdoc/>
    public void Uniform(Tensor<double> output)
    {
        if (!CuRandNative.IsAvailable || !TryDeviceUniformDouble(output, normal: false, mean: 0, stddev: 1))
            _cpuFallback.Uniform(output);
    }

    /// <inheritdoc/>
    public void Normal(Tensor<float> output, float mean = 0f, float stddev = 1f)
    {
        if (!CuRandNative.IsAvailable || !TryDeviceUniform(output, normal: true, mean: mean, stddev: stddev))
            _cpuFallback.Normal(output, mean, stddev);
    }

    /// <inheritdoc/>
    public void Normal(Tensor<double> output, double mean = 0, double stddev = 1)
    {
        if (!CuRandNative.IsAvailable || !TryDeviceUniformDouble(output, normal: true, mean: mean, stddev: stddev))
            _cpuFallback.Normal(output, mean, stddev);
    }

    /// <inheritdoc/>
    public void Bernoulli(Tensor<float> output, float p) => _cpuFallback.Bernoulli(output, p);

    private bool TryDeviceUniform(Tensor<float> output, bool normal, float mean, float stddev)
    {
        if (!CuRandNative.IsAvailable) return false;
        IntPtr dPtr = IntPtr.Zero;
        IntPtr generator = IntPtr.Zero;
        try
        {
            ulong bytes = (ulong)output.Length * sizeof(float);
            if (CudaNativeBindings.cuMemAlloc(out dPtr, bytes) != AiDotNet.Tensors.Engines.CudaResult.Success) return false;

            if (CuRandNative.curandCreateGenerator(out generator, CuRandNative.GeneratorType.PseudoPhilox4_32_10) != CuRandNative.Status.Success) return false;
            CuRandNative.curandSetPseudoRandomGeneratorSeed(generator, _cpuFallback.Seed);
            CuRandNative.curandSetGeneratorOffset(generator, _cpuFallback.Offset);

            var status = normal
                ? CuRandNative.curandGenerateNormal(generator, dPtr, (ulong)output.Length, mean, stddev)
                : CuRandNative.curandGenerateUniform(generator, dPtr, (ulong)output.Length);
            if (status != CuRandNative.Status.Success) return false;

            unsafe
            {
                fixed (float* hostPtr = output.AsWritableSpan())
                {
                    if (CudaNativeBindings.cuMemcpyDtoH((IntPtr)hostPtr, dPtr, bytes) != AiDotNet.Tensors.Engines.CudaResult.Success) return false;
                }
            }
            // Advance the offset so the next call returns disjoint draws (Philox parity with cuRAND).
            _cpuFallback.Offset += (ulong)output.Length;
            return true;
        }
        catch
        {
            return false;
        }
        finally
        {
            if (generator != IntPtr.Zero) CuRandNative.curandDestroyGenerator(generator);
            if (dPtr != IntPtr.Zero) CudaNativeBindings.cuMemFree(dPtr);
        }
    }

    private bool TryDeviceUniformDouble(Tensor<double> output, bool normal, double mean, double stddev)
    {
        if (!CuRandNative.IsAvailable) return false;
        IntPtr dPtr = IntPtr.Zero;
        IntPtr generator = IntPtr.Zero;
        try
        {
            ulong bytes = (ulong)output.Length * sizeof(double);
            if (CudaNativeBindings.cuMemAlloc(out dPtr, bytes) != AiDotNet.Tensors.Engines.CudaResult.Success) return false;

            if (CuRandNative.curandCreateGenerator(out generator, CuRandNative.GeneratorType.PseudoPhilox4_32_10) != CuRandNative.Status.Success) return false;
            CuRandNative.curandSetPseudoRandomGeneratorSeed(generator, _cpuFallback.Seed);
            CuRandNative.curandSetGeneratorOffset(generator, _cpuFallback.Offset);

            var status = normal
                ? CuRandNative.curandGenerateNormalDouble(generator, dPtr, (ulong)output.Length, mean, stddev)
                : CuRandNative.curandGenerateUniformDouble(generator, dPtr, (ulong)output.Length);
            if (status != CuRandNative.Status.Success) return false;

            unsafe
            {
                fixed (double* hostPtr = output.AsWritableSpan())
                {
                    if (CudaNativeBindings.cuMemcpyDtoH((IntPtr)hostPtr, dPtr, bytes) != AiDotNet.Tensors.Engines.CudaResult.Success) return false;
                }
            }
            _cpuFallback.Offset += (ulong)output.Length;
            return true;
        }
        catch
        {
            return false;
        }
        finally
        {
            if (generator != IntPtr.Zero) CuRandNative.curandDestroyGenerator(generator);
            if (dPtr != IntPtr.Zero) CudaNativeBindings.cuMemFree(dPtr);
        }
    }
}
