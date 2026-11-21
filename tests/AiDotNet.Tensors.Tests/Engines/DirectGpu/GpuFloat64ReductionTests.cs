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
[Collection("DirectGpuSerial")]
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

    // Engine level. A vector's values live on the host, so the engine sends a double reduction to the device only from
    // DirectGpuTensorEngine.Float64ReductionMinLength; either side of it the answer is the CPU engine's.
    [SkippableTheory]
    [InlineData(1000)]
    [InlineData(DirectGpuTensorEngine.Float64ReductionMinLength - 1)]
    [InlineData(DirectGpuTensorEngine.Float64ReductionMinLength)]
    [InlineData(DirectGpuTensorEngine.Float64ReductionMinLength + 4097)]
    public void Engine_DoubleVectorReductions_EqualTheCpuEngine(int length)
    {
        Skip.If(AiDotNetEngine.Current is not DirectGpuTensorEngine, "Not a DirectGpu engine.");
        var gpu = (IEngine)AiDotNetEngine.Current;
        var cpu = (IEngine)new CpuEngine();
        var left = new double[length];
        var right = new double[length];
        for (int i = 0; i < length; i++)
        {
            // Dyadic values whose sums and products are exact in double, so the device's tree order and the CPU's
            // running order must agree to the last bit.
            left[i] = ((i * 13 % 257) - 128) / 16.0;
            right[i] = ((i * 7 % 129) - 64) / 8.0;
        }

        var a = new AiDotNet.Tensors.LinearAlgebra.Vector<double>(left);
        var b = new AiDotNet.Tensors.LinearAlgebra.Vector<double>(right);
        Assert.Equal(cpu.Sum(a), gpu.Sum(a));
        Assert.Equal(cpu.Mean(a), gpu.Mean(a));
        Assert.Equal(cpu.DotProduct(a, b), gpu.DotProduct(a, b));
        Assert.Equal(cpu.DotProduct(a, a), gpu.DotProduct(a, a));
    }

    [SkippableFact]
    public void Engine_VectorReductions_OfOtherElementTypes_AnswerAsTheCpuEngine()
    {
        Skip.If(AiDotNetEngine.Current is not DirectGpuTensorEngine, "Not a DirectGpu engine.");
        var gpu = (IEngine)AiDotNetEngine.Current;
        var cpu = (IEngine)new CpuEngine();
        var ints = new AiDotNet.Tensors.LinearAlgebra.Vector<int>(new[] { 3, -4, 5, 100 });
        var longs = new AiDotNet.Tensors.LinearAlgebra.Vector<long>(new[] { 1L << 40, -7L, 9L });
        var floats = new AiDotNet.Tensors.LinearAlgebra.Vector<float>(new[] { 1.5f, -2.25f, 4f, 0.125f });
        Assert.Equal(cpu.Sum(ints), gpu.Sum(ints));
        Assert.Equal(cpu.DotProduct(ints, ints), gpu.DotProduct(ints, ints));
        Assert.Equal(cpu.Sum(longs), gpu.Sum(longs));
        Assert.Equal(cpu.Mean(longs), gpu.Mean(longs));
        Assert.Equal(cpu.Sum(floats), gpu.Sum(floats));
        Assert.Equal(cpu.DotProduct(floats, floats), gpu.DotProduct(floats, floats));
        Assert.Throws<ArgumentNullException>(() => gpu.Sum<double>(null!));
    }

    // The equality tests pass whichever side computes. This one pins the routing: the device answers from the threshold
    // up (its scalar comes back as a readback) and the CPU answers below it (nothing is read back).
    [SkippableFact]
    public void Engine_DoubleVectorSum_RunsOnTheDevice_FromTheThresholdUp_AndOnTheCpuBelowIt()
    {
        Skip.If(Device() is null, "No device with double-precision reductions on this host.");
        var gpu = (IEngine)AiDotNetEngine.Current;
        var large = new AiDotNet.Tensors.LinearAlgebra.Vector<double>(new double[DirectGpuTensorEngine.Float64ReductionMinLength]);
        var small = new AiDotNet.Tensors.LinearAlgebra.Vector<double>(new double[DirectGpuTensorEngine.Float64ReductionMinLength - 1]);

        long before = GpuLaunchProbe.Readbacks;
        gpu.Sum(large);
        long afterLarge = GpuLaunchProbe.Readbacks;
        gpu.Sum(small);
        long afterSmall = GpuLaunchProbe.Readbacks;

        Assert.True(afterLarge > before, "A double sum at the threshold did not read a result back from the device.");
        Assert.Equal(afterLarge, afterSmall);
    }
}
