using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DevicePrimitives;

/// <summary>
/// #1204: <see cref="CpuSobolGenerator"/>. The plain sequence is compared with cuRAND's Sobol32 generator, live
/// through its host generator where the library is present, and against values recorded from cuRAND 10 (CUDA 13.3)
/// everywhere. The scrambled sequence is this library's own, so it is tested for what a scramble must keep.
/// </summary>
public class CpuSobolGeneratorTests
{
    // ---------------------------------------------------------------- cuRAND host generator

    private sealed class CuRandSobol : IDisposable
    {
        private readonly IntPtr _generator;
        private CuRandSobol(IntPtr generator) => _generator = generator;

        public static CuRandSobol? TryCreate(int dimensions, ulong offset)
        {
            try
            {
                if (CuRandNative.curandCreateGeneratorHost(out var g, CuRandNative.GeneratorType.QuasiSobol32) != CuRandNative.Status.Success)
                    return null;
                Assert.Equal(CuRandNative.Status.Success, CuRandNative.curandSetQuasiRandomGeneratorDimensions(g, (uint)dimensions));
                Assert.Equal(CuRandNative.Status.Success, CuRandNative.curandSetGeneratorOffset(g, offset));
                return new CuRandSobol(g);
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

    private static CuRandSobol HostOrSkip(int dimensions, ulong offset = 0)
    {
        var host = CuRandSobol.TryCreate(dimensions, offset);
        Skip.If(host is null, "The cuRAND library is not available.");
        return host!;
    }

    private static float[] Uniform(IDeviceRng rng, int n) { var t = new Tensor<float>(new[] { n }); rng.Uniform(t); return t.ToArray(); }
    private static double[] UniformDouble(IDeviceRng rng, int n) { var t = new Tensor<double>(new[] { n }); rng.Uniform(t); return t.ToArray(); }
    private static float[] Normal(IDeviceRng rng, int n, float mean = 0f, float stddev = 1f) { var t = new Tensor<float>(new[] { n }); rng.Normal(t, mean, stddev); return t.ToArray(); }
    private static double[] NormalDouble(IDeviceRng rng, int n, double mean = 0, double stddev = 1) { var t = new Tensor<double>(new[] { n }); rng.Normal(t, mean, stddev); return t.ToArray(); }

    // ---------------------------------------------------------------- live against cuRAND

    [SkippableFact]
    public void EveryDirectionVector_OfAll20000Dimensions_IsCuRands()
    {
        // The embedded Joe-Kuo data, expanded, against the table cuRAND hands out (CURAND_DIRECTION_VECTORS_32_JOEKUO6).
        IntPtr table;
        try
        {
            Skip.IfNot(CuRandNative.curandGetDirectionVectors32(out table, 101) == CuRandNative.Status.Success, "cuRAND did not return its direction vectors.");
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            Skip.If(true, "The cuRAND library is not available.");
            return;
        }
        var theirs = new int[32];
        for (int d = 0; d < CpuSobolGenerator.MaxDimensions; d++)
        {
            Marshal.Copy(IntPtr.Add(table, d * 32 * sizeof(uint)), theirs, 0, 32);
            uint[] ours = SobolDirectionNumbers.Vectors(d);
            for (int k = 0; k < 32; k++)
                Assert.True(ours[k] == unchecked((uint)theirs[k]), $"dimension {d}, vector {k}: ours 0x{ours[k]:x8}, cuRAND 0x{unchecked((uint)theirs[k]):x8}");
        }
    }

    [SkippableTheory]
    [InlineData(1, 0UL, 4096)]
    [InlineData(1, 5UL, 1000)]
    [InlineData(2, 0UL, 2000)]
    [InlineData(3, 7UL, 999)]
    [InlineData(64, 1_000_003UL, 6400)]
    [InlineData(1000, 0UL, 5000)]
    [InlineData(20000, 123UL, 40000)]            // every dimension cuRAND has
    [InlineData(1, 4294967290UL, 12)]            // across the 2^32 wrap
    public void RawAndUniformOutputs_AreCuRands_BitForBit(int dimensions, ulong offset, int n)
    {
        using (var host = HostOrSkip(dimensions, offset))
            Assert.Equal(host.Raw(n), new CpuSobolGenerator(dimensions, offset: offset).NextRaw(n));
        using (var host = HostOrSkip(dimensions, offset))
            Assert.Equal(host.Uniform(n), Uniform(new CpuSobolGenerator(dimensions, offset: offset), n));
        using (var host = HostOrSkip(dimensions, offset))
            Assert.Equal(host.UniformDouble(n), UniformDouble(new CpuSobolGenerator(dimensions, offset: offset), n));
    }

    [SkippableTheory]
    [InlineData(1, 0UL, 4096)]
    [InlineData(5, 11UL, 5000)]
    public void Normals_MatchCuRand_ToRounding(int dimensions, ulong offset, int n)
    {
        float[] expected, actual;
        using (var host = HostOrSkip(dimensions, offset))
            expected = host.Normal(n, 1.5f, 2.5f);
        actual = Normal(new CpuSobolGenerator(dimensions, offset: offset), n, 1.5f, 2.5f);
        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])), $"float [{i}] cuRAND {expected[i]:R}, ours {actual[i]:R}");

        double[] expectedD, actualD;
        using (var host = HostOrSkip(dimensions, offset))
            expectedD = host.NormalDouble(n, -0.5, 3.0);
        actualD = NormalDouble(new CpuSobolGenerator(dimensions, offset: offset), n, -0.5, 3.0);
        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs(expectedD[i] - actualD[i]) <= 1e-12 * Math.Max(1.0, Math.Abs(expectedD[i])), $"double [{i}] cuRAND {expectedD[i]:R}, ours {actualD[i]:R}");
    }

    [SkippableFact]
    public void ConsecutiveCalls_ContinueAsCuRandDoes()
    {
        using var host = HostOrSkip(3);
        var ours = new CpuSobolGenerator(3);
        Assert.Equal(host.Raw(9), ours.NextRaw(9));
        Assert.Equal(host.Uniform(30), Uniform(ours, 30));
        Assert.Equal(host.Raw(3), ours.NextRaw(3));
        Assert.Equal(14UL, ours.Offset);     // 3 + 10 + 1 points
    }

    // ---------------------------------------------------------------- recorded from cuRAND (runs everywhere)

    [Fact]
    public void Outputs_EqualTheValuesRecordedFromCuRand()
    {
        Assert.Equal(new uint[] { 0x0, 0x80000000, 0xc0000000, 0x40000000, 0x60000000, 0xe0000000, 0xa0000000, 0x20000000, 0x30000000, 0xb0000000, 0xf0000000, 0x70000000 },
            new CpuSobolGenerator(1).NextRaw(12));
        // Two dimensions, six points: all of dimension 0, then all of dimension 1.
        Assert.Equal(new uint[] { 0x0, 0x80000000, 0xc0000000, 0x40000000, 0x60000000, 0xe0000000, 0x0, 0x80000000, 0x40000000, 0xc0000000, 0x60000000, 0xe0000000 },
            new CpuSobolGenerator(2).NextRaw(12));
        Assert.Equal(new uint[] { 0x0, 0x80000000, 0xc0000000, 0x40000000, 0x0, 0x80000000, 0x40000000, 0xc0000000, 0x0, 0x80000000, 0x40000000, 0xc0000000 },
            new CpuSobolGenerator(3).NextRaw(12));
        Assert.Equal(new uint[] { 0xe0000000, 0xa0000000, 0x20000000, 0x30000000, 0xb0000000, 0xf0000000 },
            new CpuSobolGenerator(1, offset: 5).NextRaw(6));

        Assert.Equal(new[] { 1.1641532182693481e-10f, 0.5f, 0.75f, 0.25f, 0.375f, 0.875f }, Uniform(new CpuSobolGenerator(1), 6));
        Assert.Equal(new[] { 2.3283064365386963e-10, 0.5000000002328306, 0.7500000002328306, 0.25000000023283064, 0.37500000023283064, 0.8750000002328306 },
            UniformDouble(new CpuSobolGenerator(1), 6));

        // The first six direction vectors of dimensions 1 and 2, as cuRAND returns them.
        Assert.Equal(new uint[] { 0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000 }, SobolDirectionNumbers.Vectors(1).Take(6).ToArray());
        Assert.Equal(new uint[] { 0x80000000, 0xc0000000, 0x60000000, 0x90000000, 0xe8000000, 0x5c000000 }, SobolDirectionNumbers.Vectors(2).Take(6).ToArray());
    }

    // ---------------------------------------------------------------- the generator's own contract

    [Fact]
    public void ALengthThatIsNotWholePoints_IsRejected_AndDimensionsAreBounded()
    {
        Assert.Throws<ArgumentException>(() => new CpuSobolGenerator(3).NextRaw(10));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CpuSobolGenerator(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CpuSobolGenerator(CpuSobolGenerator.MaxDimensions + 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CpuSobolGenerator(2, subsequence: (ulong)CpuSobolGenerator.MaxDimensions - 1));
        Assert.Equal(2 * CpuSobolGenerator.MaxDimensions, new CpuSobolGenerator(CpuSobolGenerator.MaxDimensions).NextRaw(2 * CpuSobolGenerator.MaxDimensions).Length);
    }

    [Fact]
    public void Subsequence_SelectsTheFirstDimension()
    {
        // Dimensions 5..7 of an 8-dimensional generator are dimensions 0..2 of one that starts at 5.
        const int points = 50;
        var wide = new CpuSobolGenerator(8).NextRaw(8 * points);
        var shifted = new CpuSobolGenerator(3, subsequence: 5).NextRaw(3 * points);
        Assert.Equal(wide.Skip(5 * points).Take(3 * points).ToArray(), shifted);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EveryPowerOfTwoBlock_IsStratified_InEachDimension(bool scrambled)
    {
        // The defining property of a Sobol sequence, and what a scramble must keep: the first 2^m points of any
        // dimension put exactly one point in each of the 2^m equal intervals.
        const int m = 10, points = 1 << m, dimensions = 6;
        var raw = new CpuSobolGenerator(dimensions, scrambled, seed: 31).NextRaw(dimensions * points);
        for (int d = 0; d < dimensions; d++)
        {
            var cells = new HashSet<uint>();
            for (int i = 0; i < points; i++) cells.Add(raw[d * points + i] >> (32 - m));
            Assert.Equal(points, cells.Count);
        }
        // And pairs of dimensions fill a coarse grid evenly: 2^10 points over a 16 x 16 grid, 4 per cell at best.
        // The first two dimensions form a (0, m, 2)-net, so every cell holds exactly 4.
        var grid = new int[16 * 16];
        for (int i = 0; i < points; i++) grid[(raw[i] >> 28) * 16 + (raw[points + i] >> 28)]++;
        Assert.All(grid, count => Assert.Equal(4, count));
    }

    [Fact]
    public void TheScramble_IsSeeded_Deterministic_AndChangesThePoints()
    {
        const int n = 4 * 256;
        var plain = new CpuSobolGenerator(4).NextRaw(n);
        var a = new CpuSobolGenerator(4, scrambled: true, seed: 1).NextRaw(n);
        var b = new CpuSobolGenerator(4, scrambled: true, seed: 2).NextRaw(n);
        Assert.Equal(a, new CpuSobolGenerator(4, scrambled: true, seed: 1).NextRaw(n));
        Assert.NotEqual(plain, a);
        Assert.NotEqual(a, b);
        Assert.NotEqual(0u, a[0]);                                  // the first point is no longer the origin
        Assert.Equal(DeviceRngAlgorithm.ScrambledSobol, new CpuSobolGenerator(4, scrambled: true).Algorithm);
        Assert.Equal(DeviceRngAlgorithm.Sobol, new CpuSobolGenerator(4).Algorithm);

        // A dimension's scramble depends on the seed and the dimension only, so it does not change with how many
        // dimensions are generated alongside it.
        var two = new CpuSobolGenerator(2, scrambled: true, seed: 1).NextRaw(2 * 256);
        Assert.Equal(a.Take(2 * 256).ToArray(), two);

        // Resuming at an offset continues the same scrambled sequence.
        var resumed = new CpuSobolGenerator(4, scrambled: true, seed: 1, offset: 100).NextRaw(4 * 156);
        for (int d = 0; d < 4; d++)
            Assert.Equal(a.Skip(d * 256 + 100).Take(156).ToArray(), resumed.Skip(d * 156).Take(156).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnIntegral_ConvergesFasterThanRandomSampling(bool scrambled)
    {
        // The mean of prod_d (1 + (u_d - 1/2)) over [0,1]^5 is exactly 1. With 2^14 points a Sobol estimate is
        // within 1e-3; random sampling would be around 5e-3 (standard error 0.66 / sqrt(n)).
        const int dimensions = 5, points = 1 << 14;
        var u = UniformDouble(new CpuSobolGenerator(dimensions, scrambled, seed: 9), dimensions * points);
        double sum = 0;
        for (int i = 0; i < points; i++)
        {
            double product = 1;
            for (int d = 0; d < dimensions; d++) product *= 0.5 + u[d * points + i];
            sum += product;
        }
        Assert.InRange(sum / points, 1.0 - 1e-3, 1.0 + 1e-3);
    }

    [Fact]
    public void Normals_AreTheInverseCdfOfThePoints()
    {
        const int n = 1 << 12;
        var normal = NormalDouble(new CpuSobolGenerator(1, offset: 1), n);       // skip the origin: its normal is far in the tail
        double mean = normal.Average();
        double variance = normal.Select(v => (v - mean) * (v - mean)).Average();
        Assert.InRange(mean, -0.01, 0.01);
        Assert.InRange(variance, 0.97, 1.03);
        // Point 0x80000000 is one half: its normal is 0 to rounding.
        var half = NormalDouble(new CpuSobolGenerator(1, offset: 1), 1);
        Assert.InRange(half[0], -1e-9, 1e-9);
        var scaled = Normal(new CpuSobolGenerator(1, offset: 1), 4, mean: 10f, stddev: 2f);
        Assert.All(scaled, v => Assert.InRange(v, 0f, 20f));
    }
}
