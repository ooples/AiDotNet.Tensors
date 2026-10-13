using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DevicePrimitives;

/// <summary>
/// #1204: <see cref="CpuMrg32k3aGenerator"/> against cuRAND's MRG32k3a. Where the cuRAND library is present the
/// comparison is made live, through cuRAND's host generator (no GPU or CUDA context needed). The known-answer cases
/// hold values recorded from cuRAND 10 (CUDA 13.3), so the equality is also pinned where cuRAND is absent.
/// </summary>
public class CpuMrg32k3aGeneratorTests
{
    private const int S = CpuMrg32k3aGenerator.StreamCount;

    // ---------------------------------------------------------------- cuRAND host generator

    private sealed class CuRandHost : IDisposable
    {
        private readonly IntPtr _generator;
        private CuRandHost(IntPtr generator) => _generator = generator;

        public static CuRandHost? TryCreate(ulong seed, ulong? offset = null)
        {
            try
            {
                if (CuRandNative.curandCreateGeneratorHost(out var g, CuRandNative.GeneratorType.PseudoMrg32K3A) != CuRandNative.Status.Success)
                    return null;
                var host = new CuRandHost(g);
                Assert.Equal(CuRandNative.Status.Success, CuRandNative.curandSetPseudoRandomGeneratorSeed(g, seed));
                if (offset is { } o) Assert.Equal(CuRandNative.Status.Success, CuRandNative.curandSetGeneratorOffset(g, o));
                return host;
            }
            catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
            {
                return null;
            }
        }

        public unsafe uint[] Raw(int n)
        {
            var a = new uint[n];
            fixed (uint* p = a) Assert.Equal(CuRandNative.Status.Success, CuRandNative.curandGenerate(_generator, (IntPtr)p, (ulong)n));
            return a;
        }

        public unsafe float[] Uniform(int n)
        {
            var a = new float[n];
            fixed (float* p = a) Assert.Equal(CuRandNative.Status.Success, CuRandNative.curandGenerateUniform(_generator, (IntPtr)p, (ulong)n));
            return a;
        }

        public unsafe double[] UniformDouble(int n)
        {
            var a = new double[n];
            fixed (double* p = a) Assert.Equal(CuRandNative.Status.Success, CuRandNative.curandGenerateUniformDouble(_generator, (IntPtr)p, (ulong)n));
            return a;
        }

        public unsafe float[] Normal(int n, float mean, float stddev)
        {
            var a = new float[n];
            fixed (float* p = a) Assert.Equal(CuRandNative.Status.Success, CuRandNative.curandGenerateNormal(_generator, (IntPtr)p, (ulong)n, mean, stddev));
            return a;
        }

        public unsafe double[] NormalDouble(int n, double mean, double stddev)
        {
            var a = new double[n];
            fixed (double* p = a) Assert.Equal(CuRandNative.Status.Success, CuRandNative.curandGenerateNormalDouble(_generator, (IntPtr)p, (ulong)n, mean, stddev));
            return a;
        }

        public void Dispose() => CuRandNative.curandDestroyGenerator(_generator);
    }

    private static CuRandHost HostOrSkip(ulong seed, ulong? offset = null)
    {
        var host = CuRandHost.TryCreate(seed, offset);
        Skip.If(host is null, "The cuRAND library is not available.");
        return host!;
    }

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

    private static float[] Normal(IDeviceRng rng, int n, float mean = 0f, float stddev = 1f)
    {
        var t = new Tensor<float>(new[] { n });
        rng.Normal(t, mean, stddev);
        return t.ToArray();
    }

    private static double[] NormalDouble(IDeviceRng rng, int n, double mean = 0, double stddev = 1)
    {
        var t = new Tensor<double>(new[] { n });
        rng.Normal(t, mean, stddev);
        return t.ToArray();
    }

    private static uint[] Raw(CpuMrg32k3aGenerator rng, int n)
    {
        var a = new uint[n];
        for (int i = 0; i < n; i++) a[i] = rng.NextUInt32();
        return a;
    }

    public static IEnumerable<object[]> Seeds() => new[]
    {
        new object[] { 0UL }, new object[] { 1UL }, new object[] { 12345UL }, new object[] { 0x123456789UL },
        new object[] { ulong.MaxValue }, new object[] { 0xAAAAAAAA55555555UL },
    };

    // ---------------------------------------------------------------- live against cuRAND

    [SkippableTheory]
    [MemberData(nameof(Seeds))]
    public void RawOutputs_AreCuRands_AcrossSeveralRowsOfStreams(ulong seed)
    {
        using var host = HostOrSkip(seed);
        int n = 3 * S + 7;
        Assert.Equal(host.Raw(n), Raw(new CpuMrg32k3aGenerator(seed), n));
    }

    [SkippableTheory]
    [InlineData(0UL)]
    [InlineData(1UL)]
    [InlineData(5UL)]
    [InlineData(81919UL)]
    [InlineData(81920UL)]
    [InlineData(81923UL)]
    [InlineData(1_000_000UL)]
    [InlineData(123_456_789_012UL)]
    public void UniformFloatsAndDoubles_AreCuRands_BitForBit_AtAnOffset(ulong offset)
    {
        const ulong seed = 7;
        int n = S + 100;
        using (var host = HostOrSkip(seed, offset))
            Assert.Equal(host.Uniform(n), Uniform(new CpuMrg32k3aGenerator(seed, offset: offset), n));
        using (var host = HostOrSkip(seed, offset))
            Assert.Equal(host.UniformDouble(n), UniformDouble(new CpuMrg32k3aGenerator(seed, offset: offset), n));
    }

    [SkippableTheory]
    [InlineData(0UL)]
    [InlineData(3UL)]
    [InlineData(163840UL)]
    public void Normals_MatchCuRand_ToTheLastFewBits(ulong offset)
    {
        const ulong seed = 7;
        int n = 4 * S + 10;   // more than two full rows of pairs
        float[] expected, actual;
        using (var host = HostOrSkip(seed, offset))
            expected = host.Normal(n, 1.5f, 2.5f);
        actual = Normal(new CpuMrg32k3aGenerator(seed, offset: offset), n, 1.5f, 2.5f);
        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 4e-6f * Math.Max(1f, Math.Abs(expected[i])), $"float [{i}] cuRAND {expected[i]:R}, ours {actual[i]:R}");

        double[] expectedD, actualD;
        using (var host = HostOrSkip(seed, offset))
            expectedD = host.NormalDouble(n, -0.5, 3.0);
        actualD = NormalDouble(new CpuMrg32k3aGenerator(seed, offset: offset), n, -0.5, 3.0);
        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs(expectedD[i] - actualD[i]) <= 1e-13 * Math.Max(1.0, Math.Abs(expectedD[i])), $"double [{i}] cuRAND {expectedD[i]:R}, ours {actualD[i]:R}");
    }

    [SkippableFact]
    public void AMixedSequenceOfCalls_ContinuesExactlyAsCuRandDoes()
    {
        // cuRAND keeps a position per stream: a normal call advances the streams it used by two elements, and the
        // next call carries on from the following stream. Uniform values must stay bit-identical through that.
        const ulong seed = 99;
        using var host = HostOrSkip(seed);
        var ours = new CpuMrg32k3aGenerator(seed);

        Assert.Equal(host.Uniform(3), Uniform(ours, 3));
        AssertNear(host.Normal(4, 0f, 1f), Normal(ours, 4));
        Assert.Equal(host.Uniform(5), Uniform(ours, 5));
        AssertNear(host.Normal(2 * S, 0f, 1f), Normal(ours, 2 * S));      // every stream once: wraps the stream pointer
        Assert.Equal(host.UniformDouble(S + 9), UniformDouble(ours, S + 9));
        AssertNear(host.Normal(6, 0f, 1f), Normal(ours, 6));
        Assert.Equal(host.Raw(2 * S + 1), Raw(ours, 2 * S + 1));

        static void AssertNear(float[] expected, float[] actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 4e-6f * Math.Max(1f, Math.Abs(expected[i])), $"[{i}] cuRAND {expected[i]:R}, ours {actual[i]:R}");
        }
    }

    // ---------------------------------------------------------------- recorded from cuRAND (runs everywhere)

    [Theory]
    [InlineData(0UL, 0UL, new uint[] { 0x2083cd07, 0x14537e00, 0x43115831, 0x80d28544, 0xf132d19e, 0xab8caf71, 0x7f700c9c, 0x6b0ccb1b })]
    [InlineData(1UL, 0UL, new uint[] { 0x6edeeb53, 0x2d10a9e6, 0x4c18beb4, 0xe4da73e4, 0x2bfab304, 0xf6947b2f, 0xfdafb691, 0xbe6d369e })]
    [InlineData(12345UL, 0UL, new uint[] { 0xb776a57d, 0x21e73d4a, 0x11630f0b, 0xa0901887, 0x1d8a6a09, 0x740cd09a, 0x40611877, 0x360bdbde })]
    [InlineData(0x123456789UL, 0UL, new uint[] { 0xd9408162, 0x18c6d631, 0xef8a83df, 0xdd9979ef, 0xf8e41ffe, 0x42c5be05, 0xc2852bd1, 0x9c647ff9 })]
    [InlineData(7UL, 1UL, new uint[] { 0xfe386b54, 0xead22169, 0xe91c9910 })]
    [InlineData(7UL, 5UL, new uint[] { 0x33894cce, 0x76f188ef, 0xae3696f8 })]
    [InlineData(7UL, 81920UL, new uint[] { 0x25729746, 0x6e7afb92, 0x3042e974 })]
    [InlineData(7UL, 81923UL, new uint[] { 0x8d846b01, 0xc93d1b6d, 0x827618c6 })]
    [InlineData(7UL, 1_000_000UL, new uint[] { 0xf1e3104b, 0x7c53bfbd, 0xeaf83cfa })]
    public void RawOutputs_EqualTheValuesRecordedFromCuRand(ulong seed, ulong offset, uint[] recorded)
        => Assert.Equal(recorded, Raw(new CpuMrg32k3aGenerator(seed, offset: offset), recorded.Length));

    [Fact]
    public void UniformValues_EqualTheValuesRecordedFromCuRand()
    {
        // cuRAND, seed 0: the first four floats and doubles.
        Assert.Equal(new[] { 0.12701112031936646f, 0.0793989896774292f, 0.26198339462280273f, 0.5032122731208801f },
            Uniform(new CpuMrg32k3aGenerator(0), 4));
        Assert.Equal(new[] { 0.12701112207614923, 0.07939898981582115, 0.26198340620718247, 0.5032122889932638 },
            UniformDouble(new CpuMrg32k3aGenerator(0), 4));
    }

    // ---------------------------------------------------------------- the generator's own contract

    [Fact]
    public void Offset_AdvancesByTheUniformCount_AndRestoringItResumesTheSequence()
    {
        var whole = Uniform(new CpuMrg32k3aGenerator(42), 2 * S + 50);

        var rng = new CpuMrg32k3aGenerator(42);
        var first = Uniform(rng, S - 3);
        Assert.Equal((ulong)(S - 3), rng.Offset);
        var second = Uniform(rng, S + 53);
        Assert.Equal((ulong)(2 * S + 50), rng.Offset);
        Assert.Equal(whole, first.Concat(second).ToArray());

        var resumed = new CpuMrg32k3aGenerator(42) { Offset = (ulong)(S - 3) };
        Assert.Equal(second, Uniform(resumed, S + 53));
        var constructed = new CpuMrg32k3aGenerator(42, offset: (ulong)(S + 10));
        Assert.Equal(whole.Skip(S + 10).Take(20).ToArray(), Uniform(constructed, 20));
    }

    [Fact]
    public void RestoringTheOffsetAfterNormals_NeverReplaysAValue()
    {
        var rng = new CpuMrg32k3aGenerator(5);
        var before = Uniform(rng, 10);
        Normal(rng, 6);                              // streams 10, 11 and 12 advance two elements each
        ulong offset = rng.Offset;
        Assert.Equal(0UL, offset % (ulong)S);        // the start of a row
        Assert.Equal(2UL * (ulong)S, offset);        // the first row no stream has reached

        // Everything produced so far lies in rows 0 and 1; a generator restored at the offset starts in row 2.
        var seen = new HashSet<uint>(Raw(new CpuMrg32k3aGenerator(5), 2 * S));
        var after = Raw(new CpuMrg32k3aGenerator(5) { Offset = offset }, 64);
        Assert.Equal(Raw(new CpuMrg32k3aGenerator(5, offset: 2UL * (ulong)S), 64), after);
        Assert.True(after.Count(seen.Contains) <= 1, "values after the restored offset repeat earlier ones");
        Assert.Equal(10, before.Length);
    }

    [Fact]
    public void Subsequences_AreDeterministic_AndDifferent()
    {
        var zero = Uniform(new CpuMrg32k3aGenerator(3), 64);
        var one = Uniform(new CpuMrg32k3aGenerator(3, subsequence: 1), 64);
        var two = Uniform(new CpuMrg32k3aGenerator(3, subsequence: 2), 64);
        Assert.Equal(one, Uniform(new CpuMrg32k3aGenerator(3, subsequence: 1), 64));
        Assert.Equal(0, zero.Intersect(one).Count());
        Assert.Equal(0, one.Intersect(two).Count());

        // Setting the property moves an existing generator to the same place, keeping its offset.
        var moved = new CpuMrg32k3aGenerator(3, offset: 17) { Subsequence = 2 };
        Assert.Equal(17UL, moved.Offset);
        Assert.Equal(Uniform(new CpuMrg32k3aGenerator(3, subsequence: 2, offset: 17), 32), Uniform(moved, 32));
    }

    [Fact]
    public void Distributions_HaveTheRightMoments()
    {
        const int n = 400_000;
        var u = UniformDouble(new CpuMrg32k3aGenerator(11), n);
        Assert.All(u, v => Assert.True(v > 0.0 && v <= 1.0));
        Assert.InRange(u.Average(), 0.497, 0.503);
        Assert.InRange(u.Select(v => (v - 0.5) * (v - 0.5)).Average(), 1.0 / 12 - 0.001, 1.0 / 12 + 0.001);

        var normal = NormalDouble(new CpuMrg32k3aGenerator(11), n, 2.0, 3.0);
        double mean = normal.Average();
        double variance = normal.Select(v => (v - mean) * (v - mean)).Average();
        Assert.InRange(mean, 1.97, 2.03);
        Assert.InRange(variance, 8.85, 9.15);

        var odd = Normal(new CpuMrg32k3aGenerator(11), 7);     // cuRAND rejects an odd count; this fills all seven
        Assert.All(odd, v => Assert.True(!float.IsNaN(v) && Math.Abs(v) < 10f));

        var coin = new Tensor<float>(new[] { n });
        new CpuMrg32k3aGenerator(11).Bernoulli(coin, 0.3f);
        var flips = coin.ToArray();
        Assert.All(flips, v => Assert.True(v == 0f || v == 1f));
        Assert.InRange(flips.Average(), 0.295, 0.305);
    }

    [Fact]
    public void EveryBackendWrapper_ProducesTheCpuGeneratorsNumbers()
    {
        var expected = Uniform(new CpuMrg32k3aGenerator(77, subsequence: 0, offset: 9), 300);
        var wrappers = new IDeviceRng[]
        {
            CpuDeviceRng.Create(DeviceRngAlgorithm.Mrg32K3A, 77, 0, 9),
            new CuRand(77, DeviceRngAlgorithm.Mrg32K3A, 0, 9),
            new AiDotNet.Tensors.Engines.DirectGpu.HIP.RocRand(77, DeviceRngAlgorithm.Mrg32K3A, 0, 9),
            new AiDotNet.Tensors.Engines.DirectGpu.Metal.MpsRng(77, DeviceRngAlgorithm.Mrg32K3A, 0, 9),
            new AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClRng(77, DeviceRngAlgorithm.Mrg32K3A, 0, 9),
            new AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanRng(77, DeviceRngAlgorithm.Mrg32K3A, 0, 9),
        };
        foreach (var rng in wrappers)
        {
            Assert.Equal(DeviceRngAlgorithm.Mrg32K3A, rng.Algorithm);
            Assert.Equal(77UL, rng.Seed);
            Assert.Equal(expected, Uniform(rng, 300));
            Assert.Equal(309UL, rng.Offset);
        }

        Assert.True(CpuDeviceRng.IsSupported(DeviceRngAlgorithm.Philox));
        Assert.True(CpuDeviceRng.IsSupported(DeviceRngAlgorithm.Mrg32K3A));
        // Not yet: the two Sobol generators (tracked in #1204).
        Assert.False(CpuDeviceRng.IsSupported(DeviceRngAlgorithm.Sobol));
        Assert.Throws<NotSupportedException>(() => new CuRand(1, DeviceRngAlgorithm.Sobol));
        Assert.Throws<NotSupportedException>(() => CpuDeviceRng.Create(DeviceRngAlgorithm.ScrambledSobol, 1));
    }
}
