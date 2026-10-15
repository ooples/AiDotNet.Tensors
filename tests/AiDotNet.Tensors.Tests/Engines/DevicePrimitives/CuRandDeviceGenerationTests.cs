using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DevicePrimitives;

/// <summary>
/// #1204: <see cref="CuRand"/> generates MRG32k3a and plain Sobol32 uniforms with cuRAND on a CUDA device, and the
/// result must be bit-identical to the CPU generator, with the CPU generator's position kept in lockstep. The
/// device tests skip (not fail) where there is no cuRAND library or no CUDA device; each one asserts the call really
/// ran on the device, so a silent fallback to the CPU cannot pass them by comparing the CPU with itself.
/// </summary>
public class CuRandDeviceGenerationTests
{
    private static void SkipWithoutDevice()
        => Skip.IfNot(CuRand.DeviceGenerationAvailable, "No cuRAND library or no CUDA device on this host.");

    /// <summary>A wrapper that sends every eligible call to the device, whatever its size.</summary>
    private static CuRand DeviceRng(DeviceRngAlgorithm algorithm, ulong seed, ulong offset, int dimensions = 1, ulong subsequence = 0)
        => new(seed, algorithm, subsequence, offset, dimensions) { SobolDeviceMinimumLength = 0, Mrg32k3aSetupBreakEven = 0 };

    private static float[] Uniform(IDeviceRng rng, int n)
    {
        var t = new Tensor<float>(new[] { n });
        rng.Uniform(t);
        return t.ToArray();
    }

    private static double[] UniformDouble(IDeviceRng rng, int n)
    {
        var t = new Tensor<double>(new[] { n });
        rng.Uniform(t);
        return t.ToArray();
    }

    private static float[] Normal(IDeviceRng rng, int n)
    {
        var t = new Tensor<float>(new[] { n });
        rng.Normal(t);
        return t.ToArray();
    }

    private static float[] Bernoulli(IDeviceRng rng, int n, float p)
    {
        var t = new Tensor<float>(new[] { n });
        rng.Bernoulli(t, p);
        return t.ToArray();
    }

    private static void AssertSameBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            int e = BitConverter.ToInt32(BitConverter.GetBytes(expected[i]), 0);
            int a = BitConverter.ToInt32(BitConverter.GetBytes(actual[i]), 0);
            if (e != a) Assert.Fail($"Value {i}: expected {expected[i]:R} (0x{e:X8}), device gave {actual[i]:R} (0x{a:X8}).");
        }
    }

    private static void AssertSameBits(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            long e = BitConverter.DoubleToInt64Bits(expected[i]);
            long a = BitConverter.DoubleToInt64Bits(actual[i]);
            if (e != a) Assert.Fail($"Value {i}: expected {expected[i]:R} (0x{e:X16}), device gave {actual[i]:R} (0x{a:X16}).");
        }
    }

    // ---------------------------------------------------------------- MRG32k3a

    public static IEnumerable<object[]> MrgCases() => new[]
    {
        // seed, offset, length: inside one row of streams, across row boundaries, unaligned starts, big offsets.
        new object[] { 0UL, 0UL, 1 },
        new object[] { 12345UL, 0UL, 1000 },
        new object[] { 12345UL, 7UL, 81920 + 13 },
        new object[] { 0x123456789UL, 81919UL, 3 * 81920 + 5 },
        new object[] { ulong.MaxValue, 1_000_003UL, 250_000 },
        new object[] { 42UL, (1UL << 40) + 11, 4097 },
        new object[] { 7UL, 0UL, 1 << 20 },
    };

    [SkippableTheory]
    [MemberData(nameof(MrgCases))]
    public void Mrg32k3a_DeviceUniforms_AreTheCpuGeneratorsBits(ulong seed, ulong offset, int length)
    {
        SkipWithoutDevice();

        var device = DeviceRng(DeviceRngAlgorithm.Mrg32K3A, seed, offset);
        var actual = Uniform(device, length);
        Assert.True(device.LastUniformUsedDevice);
        AssertSameBits(Uniform(new CpuMrg32k3aGenerator(seed, 0, offset), length), actual);
        Assert.Equal(offset + (ulong)length, device.Offset);

        var deviceD = DeviceRng(DeviceRngAlgorithm.Mrg32K3A, seed, offset);
        var actualD = UniformDouble(deviceD, length);
        Assert.True(deviceD.LastUniformUsedDevice);
        AssertSameBits(UniformDouble(new CpuMrg32k3aGenerator(seed, 0, offset), length), actualD);
        Assert.Equal(offset + (ulong)length, deviceD.Offset);
    }

    [SkippableFact]
    public void Mrg32k3a_MixedCalls_StayInLockstepWithTheCpuGenerator()
    {
        SkipWithoutDevice();
        const ulong Seed = 2024;
        var device = DeviceRng(DeviceRngAlgorithm.Mrg32K3A, Seed, 5);
        var cpu = new CpuMrg32k3aGenerator(Seed, 0, 5);

        // Device, then device again: the second call starts where the first ended.
        AssertSameBits(Uniform(cpu, 100_000), Uniform(device, 100_000));
        Assert.True(device.LastUniformUsedDevice);
        AssertSameBits(UniformDouble(cpu, 81_925), UniformDouble(device, 81_925));
        Assert.True(device.LastUniformUsedDevice);

        // Bernoulli consumes uniforms only, so the next uniform call can still use the device.
        Assert.Equal(Bernoulli(cpu, 999, 0.25f), Bernoulli(device, 999, 0.25f));
        AssertSameBits(Uniform(cpu, 50_000), Uniform(device, 50_000));
        Assert.True(device.LastUniformUsedDevice);

        // A normal call moves individual streams; cuRAND has no offset for that, so uniforms stay on the CPU...
        Assert.Equal(Normal(cpu, 1000), Normal(device, 1000));
        AssertSameBits(Uniform(cpu, 70_000), Uniform(device, 70_000));
        Assert.False(device.LastUniformUsedDevice);
        Assert.Equal(cpu.Offset, device.Offset);

        // ...until the offset is set again.
        cpu.Offset = 123_456_789;
        device.Offset = 123_456_789;
        AssertSameBits(UniformDouble(cpu, 200_000), UniformDouble(device, 200_000));
        Assert.True(device.LastUniformUsedDevice);
        Assert.Equal(cpu.Offset, device.Offset);
        AssertSameBits(Uniform(cpu, 10), Uniform(device, 10));

        // Disposing frees the device generator; the next call makes a new one at the right place.
        device.Dispose();
        AssertSameBits(Uniform(cpu, 81_921), Uniform(device, 81_921));
        Assert.True(device.LastUniformUsedDevice);
        Assert.Equal(cpu.Offset, device.Offset);
    }

    // ---------------------------------------------------------------- Sobol32

    public static IEnumerable<object[]> SobolCases() => new[]
    {
        // dimensions, offset, points
        new object[] { 1, 0UL, 1 },
        new object[] { 1, 0UL, 1 << 20 },
        new object[] { 1, 1UL, 4095 },
        new object[] { 3, 0UL, 1000 },
        new object[] { 3, 17UL, 33_333 },
        new object[] { 7, 1_000_000UL, 5_000 },
        new object[] { 64, 255UL, 1024 },
        new object[] { 20_000, 2UL, 3 },
        new object[] { 2, (1UL << 32) - 100, 100 },      // ends exactly at the period
    };

    [SkippableTheory]
    [MemberData(nameof(SobolCases))]
    public void Sobol_DeviceUniforms_AreTheCpuGeneratorsBits(int dimensions, ulong offset, int points)
    {
        SkipWithoutDevice();
        int length = dimensions * points;

        var device = DeviceRng(DeviceRngAlgorithm.Sobol, 0, offset, dimensions);
        var actual = Uniform(device, length);
        Assert.True(device.LastUniformUsedDevice);
        AssertSameBits(Uniform(new CpuSobolGenerator(dimensions, scrambled: false, offset: offset), length), actual);
        Assert.Equal(offset + (ulong)points, device.Offset);

        var deviceD = DeviceRng(DeviceRngAlgorithm.Sobol, 0, offset, dimensions);
        var actualD = UniformDouble(deviceD, length);
        Assert.True(deviceD.LastUniformUsedDevice);
        AssertSameBits(UniformDouble(new CpuSobolGenerator(dimensions, scrambled: false, offset: offset), length), actualD);
        Assert.Equal(offset + (ulong)points, deviceD.Offset);
    }

    [SkippableFact]
    public void Sobol_MixedCalls_StayInLockstepWithTheCpuGenerator()
    {
        SkipWithoutDevice();
        const int D = 5;
        var device = DeviceRng(DeviceRngAlgorithm.Sobol, 0, 3, D);
        var cpu = new CpuSobolGenerator(D, offset: 3);

        AssertSameBits(Uniform(cpu, D * 1000), Uniform(device, D * 1000));
        Assert.True(device.LastUniformUsedDevice);
        // Consecutive device calls continue the kept generator, in float or double.
        AssertSameBits(UniformDouble(cpu, D * 333), UniformDouble(device, D * 333));
        Assert.True(device.LastUniformUsedDevice);
        AssertSameBits(Uniform(cpu, D * 2), Uniform(device, D * 2));
        Assert.True(device.LastUniformUsedDevice);
        // Sobol normals and Bernoulli draws run on the CPU and advance the same point counter.
        Assert.Equal(Normal(cpu, D * 10), Normal(device, D * 10));
        Assert.Equal(Bernoulli(cpu, D * 7, 0.5f), Bernoulli(device, D * 7, 0.5f));
        AssertSameBits(UniformDouble(cpu, D * 4000), UniformDouble(device, D * 4000));
        Assert.True(device.LastUniformUsedDevice);
        Assert.Equal(cpu.Offset, device.Offset);
        Assert.Equal(3UL + 1000 + 333 + 2 + 10 + 7 + 4000, device.Offset);
    }

    // ---------------------------------------------------------------- calls that stay on the CPU

    [SkippableFact]
    public void CallsTheDeviceCannotReproduce_RunOnTheCpuGenerator()
    {
        SkipWithoutDevice();

        // MRG32k3a subsequences are this library's extension; cuRAND has none.
        var mrgSub = DeviceRng(DeviceRngAlgorithm.Mrg32K3A, 9, 4, subsequence: 2);
        AssertSameBits(Uniform(new CpuMrg32k3aGenerator(9, 2, 4), 5000), Uniform(mrgSub, 5000));
        Assert.False(mrgSub.LastUniformUsedDevice);

        // A Sobol subsequence is a first dimension other than 0, which cuRAND cannot start from.
        var sobolSub = DeviceRng(DeviceRngAlgorithm.Sobol, 0, 0, dimensions: 2, subsequence: 3);
        AssertSameBits(Uniform(new CpuSobolGenerator(2, subsequence: 3), 2000), Uniform(sobolSub, 2000));
        Assert.False(sobolSub.LastUniformUsedDevice);

        // The scramble is this library's own.
        var scrambled = DeviceRng(DeviceRngAlgorithm.ScrambledSobol, 77, 0, dimensions: 2);
        AssertSameBits(Uniform(new CpuSobolGenerator(2, scrambled: true, seed: 77), 2000), Uniform(scrambled, 2000));
        Assert.False(scrambled.LastUniformUsedDevice);

        // A call that would cross point 2^32 (where the CPU generator wraps).
        var wrapping = DeviceRng(DeviceRngAlgorithm.Sobol, 0, (1UL << 32) - 10);
        AssertSameBits(Uniform(new CpuSobolGenerator(1, offset: (1UL << 32) - 10), 20), Uniform(wrapping, 20));
        Assert.False(wrapping.LastUniformUsedDevice);

        // Partial points are rejected exactly as the CPU generator rejects them.
        Assert.Throws<ArgumentException>(() => Uniform(DeviceRng(DeviceRngAlgorithm.Sobol, 0, 0, dimensions: 3), 10));

        // Below the size threshold the CPU generator is as fast, so it runs.
        var small = new CuRand(5, DeviceRngAlgorithm.Sobol);
        AssertSameBits(Uniform(new CpuSobolGenerator(1), 1000), Uniform(small, 1000));
        Assert.False(small.LastUniformUsedDevice);
    }

    [SkippableFact]
    public void Mrg32k3a_SetsUpTheDeviceOnceDemandCoversTheSetUpCost_ThenKeepsContinuing()
    {
        SkipWithoutDevice();
        var rng = new CuRand(99, DeviceRngAlgorithm.Mrg32K3A) { Mrg32k3aSetupBreakEven = 3000 };
        var cpu = new CpuMrg32k3aGenerator(99);

        AssertSameBits(Uniform(cpu, 1000), Uniform(rng, 1000));
        Assert.False(rng.LastUniformUsedDevice);             // demand 1000
        AssertSameBits(Uniform(cpu, 1999), Uniform(rng, 1999));
        Assert.False(rng.LastUniformUsedDevice);             // demand 2999
        AssertSameBits(Uniform(cpu, 1), Uniform(rng, 1));
        Assert.True(rng.LastUniformUsedDevice);              // demand 3000: set up
        // Continuing calls stay on the device at any size.
        AssertSameBits(UniformDouble(cpu, 3), UniformDouble(rng, 3));
        Assert.True(rng.LastUniformUsedDevice);
        AssertSameBits(Uniform(cpu, 1), Uniform(rng, 1));
        Assert.True(rng.LastUniformUsedDevice);

        // A Bernoulli call runs on the CPU and leaves the device behind; demand counts again from zero.
        Assert.Equal(Bernoulli(cpu, 10, 0.5f), Bernoulli(rng, 10, 0.5f));
        AssertSameBits(Uniform(cpu, 2999), Uniform(rng, 2999));
        Assert.False(rng.LastUniformUsedDevice);
        AssertSameBits(Uniform(cpu, 1), Uniform(rng, 1));
        Assert.True(rng.LastUniformUsedDevice);
        Assert.Equal(cpu.Offset, rng.Offset);

        // With the default break-even a single small call stays on the CPU.
        var fresh = new CuRand(5, DeviceRngAlgorithm.Mrg32K3A);
        AssertSameBits(Uniform(new CpuMrg32k3aGenerator(5), 100_000), Uniform(fresh, 100_000));
        Assert.False(fresh.LastUniformUsedDevice);
    }

    [Fact]
    public void WithOrWithoutADevice_TheNumbersAreTheCpuGenerators()
    {
        // Runs everywhere: with no device the CPU generator produces the call, with one the device does, and the
        // values are the same either way.
        var rng = DeviceRng(DeviceRngAlgorithm.Mrg32K3A, 31, 100);
        AssertSameBits(Uniform(new CpuMrg32k3aGenerator(31, 0, 100), 4096), Uniform(rng, 4096));
        Assert.Equal(CuRand.DeviceGenerationAvailable, rng.LastUniformUsedDevice);
        Assert.Equal(4196UL, rng.Offset);

        var sobol = DeviceRng(DeviceRngAlgorithm.Sobol, 0, 0, dimensions: 4);
        AssertSameBits(UniformDouble(new CpuSobolGenerator(4), 4096), UniformDouble(sobol, 4096));
        Assert.Equal(CuRand.DeviceGenerationAvailable, sobol.LastUniformUsedDevice);
        Assert.Equal(1024UL, sobol.Offset);
    }
}