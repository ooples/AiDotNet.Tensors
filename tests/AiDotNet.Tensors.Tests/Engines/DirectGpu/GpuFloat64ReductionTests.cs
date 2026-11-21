// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Double-precision device reductions (<see cref="IGpuFloat64Reductions"/>): the device result must be a double result,
/// not a float one widened, at every size that crosses a work-group boundary.
/// </summary>
public class GpuFloat64ReductionTests
{
    private static readonly int[] Sizes = { 1, 2, 3, 31, 255, 256, 257, 4097, 65_536, 100_003 };

    private static (IDirectGpuBackend Backend, IGpuFloat64Reductions Reductions)? Device()
    {
        if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpu) return null;
        if (gpu.GetBackend() is not { } backend) return null;
        if (backend is not IGpuFloat64Reductions reductions || !reductions.SupportsFloat64Reductions) return null;
        return backend.SupportsElementType(GpuElementType.Float64) ? (backend, reductions) : null;
    }

    [SkippableFact]
    public void Sum_MatchesTheHostDoubleSum_AtEverySize()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision reductions on this host.");
        var (backend, reductions) = device.Value;
        foreach (int size in Sizes)
        {
            var values = new double[size];
            double expected = 0.0;
            for (int i = 0; i < size; i++)
            {
                // Mixed signs and magnitudes, each exactly representable, so the exact sum is a host-computable integer
                // multiple of 2^-10 and the result does not depend on the order of accumulation.
                values[i] = ((i * 37 % 2001) - 1000) / 1024.0;
                expected += values[i];
            }

            using var buffer = backend.AllocateBuffer(size, GpuElementType.Float64);
            backend.UploadBuffer<double>(buffer, values);
            Assert.Equal(expected, reductions.SumFloat64(buffer, size));
        }
    }

    [SkippableFact]
    public void Sum_KeepsThePrecisionAFloatSumLoses()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision reductions on this host.");
        var (backend, reductions) = device.Value;
        const int size = 1_000_000;
        var values = new double[size];
        for (int i = 0; i < size; i++) values[i] = 0.1;
        using var buffer = backend.AllocateBuffer(size, GpuElementType.Float64);
        backend.UploadBuffer<double>(buffer, values);
        // A single-precision accumulation of a million 0.1f is off by hundreds; double is within rounding of 100000.
        Assert.Equal(100_000.0, reductions.SumFloat64(buffer, size), 6);
    }

    [SkippableFact]
    public void Sum_OfAPrefix_IgnoresTheRestOfTheBuffer()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision reductions on this host.");
        var (backend, reductions) = device.Value;
        var values = new double[1000];
        for (int i = 0; i < values.Length; i++) values[i] = i < 300 ? 2.0 : 1e300;
        using var buffer = backend.AllocateBuffer(values.Length, GpuElementType.Float64);
        backend.UploadBuffer<double>(buffer, values);
        Assert.Equal(600.0, reductions.SumFloat64(buffer, 300));
        Assert.Equal(0.0, reductions.SumFloat64(buffer, 0));
    }

    [SkippableFact]
    public void Dot_MatchesTheHostDoubleDot_AtEverySize()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision reductions on this host.");
        var (backend, reductions) = device.Value;
        foreach (int size in Sizes)
        {
            var left = new double[size];
            var right = new double[size];
            double expected = 0.0;
            for (int i = 0; i < size; i++)
            {
                // Small integers over powers of two: every product and every partial sum is exact in double.
                left[i] = ((i * 13 % 257) - 128) / 16.0;
                right[i] = ((i * 7 % 129) - 64) / 8.0;
                expected += left[i] * right[i];
            }

            using var a = backend.AllocateBuffer(size, GpuElementType.Float64);
            using var b = backend.AllocateBuffer(size, GpuElementType.Float64);
            backend.UploadBuffer<double>(a, left);
            backend.UploadBuffer<double>(b, right);
            Assert.Equal(expected, reductions.DotFloat64(a, b, size));
        }
    }

    [SkippableFact]
    public void Reductions_RefuseABufferThatIsNotFloat64()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision reductions on this host.");
        var (backend, reductions) = device.Value;
        using var floats = backend.AllocateBuffer(8);
        using var doubles = backend.AllocateBuffer(8, GpuElementType.Float64);
        Assert.Throws<ArgumentException>(() => reductions.SumFloat64(floats, 8));
        Assert.Throws<ArgumentException>(() => reductions.DotFloat64(doubles, floats, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => reductions.SumFloat64(doubles, 9));
        Assert.Throws<ArgumentOutOfRangeException>(() => reductions.DotFloat64(doubles, doubles, -1));
    }
}
