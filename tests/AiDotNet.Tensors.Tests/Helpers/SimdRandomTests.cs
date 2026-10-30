using System;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class SimdRandomTests
{
    // An independent statement of SimdRandom's stream: four xoshiro256** lanes seeded with SplitMix64 in lane
    // order; a call of L >= 4 values takes floor(L/4) rounds (value 4r + k from lane k) then a lane-0 tail, and a
    // shorter call uses lane 0. The AVX2 path and the scalar path must both produce exactly this, so the same
    // seed gives the same values whatever the instruction set (run this suite with DOTNET_EnableAVX2=0 too).
    private sealed class ReferenceStream
    {
        private readonly ulong[,] _s = new ulong[4, 4];

        public ReferenceStream(int seed)
        {
            ulong state = (ulong)seed;
            for (int lane = 0; lane < 4; lane++)
                for (int word = 0; word < 4; word++) _s[word, lane] = SplitMix64(ref state);
        }

        public double[] Take(int length)
        {
            var values = new double[length];
            int rounds = length >= 4 ? length / 4 : 0, i = 0;
            for (int r = 0; r < rounds; r++)
                for (int lane = 0; lane < 4; lane++) values[i++] = Next(lane);
            for (; i < length; i++) values[i] = Next(0);
            return values;
        }

        private double Next(int lane)
        {
            ulong result = Rotl(_s[1, lane] * 5, 7) * 9;
            ulong t = _s[1, lane] << 17;
            _s[2, lane] ^= _s[0, lane];
            _s[3, lane] ^= _s[1, lane];
            _s[1, lane] ^= _s[2, lane];
            _s[0, lane] ^= _s[3, lane];
            _s[2, lane] ^= t;
            _s[3, lane] = Rotl(_s[3, lane], 45);
            return BitConverter.Int64BitsToDouble((long)((result >> 12) | 0x3FF0000000000000UL)) - 1.0;
        }

        private static ulong Rotl(ulong x, int k) => (x << k) | (x >> (64 - k));

        private static ulong SplitMix64(ref ulong state)
        {
            ulong z = state += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    [Theory]
    [InlineData(new[] { 37 })]
    [InlineData(new[] { 3, 4, 9, 1, 16, 2, 4097 })]
    [InlineData(new[] { 1, 1, 1, 1, 1 })]
    public void NextDoubles_IsTheFourLaneStream_OnEveryInstructionSet(int[] callLengths)
    {
        var rng = new SimdRandom(1234);
        var reference = new ReferenceStream(1234);
        foreach (int length in callLengths)
        {
            var actual = new double[length];
            rng.NextDoubles(actual);
            var expected = reference.Take(length);
            for (int i = 0; i < length; i++)
                Assert.True(expected[i] == actual[i],
                    $"call of {length}: value {i} is {actual[i]:R}, the four-lane stream gives {expected[i]:R}.");
        }
    }

    [Fact]
    public void NextFloats_IsTheFourLaneStream_OnEveryInstructionSet()
    {
        var rng = new SimdRandom(99);
        var reference = new ReferenceStream(99);
        foreach (int length in new[] { 10, 3, 33 })
        {
            var actual = new float[length];
            rng.NextFloats(actual);
            // NextFloats draws one four-value round at a time, then a lane-0 tail, like NextDoubles.
            int rounds = length / 4;
            var expected = new System.Collections.Generic.List<double>();
            for (int r = 0; r < rounds; r++) expected.AddRange(reference.Take(4));
            for (int i = rounds * 4; i < length; i++) expected.AddRange(reference.Take(1));
            for (int i = 0; i < length; i++)
                Assert.Equal(Math.Min((float)expected[i], 0.99999994f), actual[i]);
        }
    }

    [Fact]
    public void NextDoubles_AllValuesInRange()
    {
        var rng = new SimdRandom(42);
        var buf = new double[10000];
        rng.NextDoubles(buf);

        foreach (double v in buf)
        {
            Assert.True(v >= 0.0 && v < 1.0, $"Value {v} out of [0, 1) range");
        }
    }

    [Fact]
    public void NextDoubles_Deterministic_SameSeed()
    {
        var rng1 = new SimdRandom(123);
        var rng2 = new SimdRandom(123);
        var buf1 = new double[1000];
        var buf2 = new double[1000];

        rng1.NextDoubles(buf1);
        rng2.NextDoubles(buf2);

        for (int i = 0; i < buf1.Length; i++)
        {
            Assert.Equal(buf1[i], buf2[i]);
        }
    }

    [Fact]
    public void NextDoubles_DifferentSeeds_DifferentValues()
    {
        var rng1 = new SimdRandom(42);
        var rng2 = new SimdRandom(99);
        var buf1 = new double[100];
        var buf2 = new double[100];

        rng1.NextDoubles(buf1);
        rng2.NextDoubles(buf2);

        bool anyDifferent = false;
        for (int i = 0; i < buf1.Length; i++)
        {
            if (buf1[i] != buf2[i]) { anyDifferent = true; break; }
        }
        Assert.True(anyDifferent);
    }

    [Fact]
    public void NextDoubles_UniformDistribution()
    {
        var rng = new SimdRandom(42);
        var buf = new double[100000];
        rng.NextDoubles(buf);

        double mean = buf.Average();
        // Uniform [0,1) should have mean ≈ 0.5
        Assert.True(Math.Abs(mean - 0.5) < 0.01, $"Mean {mean} too far from 0.5");

        // Check quartiles are roughly equal
        int q1 = buf.Count(v => v < 0.25);
        int q2 = buf.Count(v => v >= 0.25 && v < 0.5);
        int q3 = buf.Count(v => v >= 0.5 && v < 0.75);
        int q4 = buf.Count(v => v >= 0.75);

        foreach (int q in new[] { q1, q2, q3, q4 })
        {
            Assert.True(q > 20000 && q < 30000, $"Quartile count {q} indicates poor uniformity");
        }
    }

    [Fact]
    public void NextFloats_AllValuesInRange()
    {
        var rng = new SimdRandom(42);
        var buf = new float[10000];
        rng.NextFloats(buf);

        foreach (float v in buf)
        {
            Assert.True(v >= 0f && v < 1f, $"Value {v} out of [0, 1) range");
        }
    }

    [Fact]
    public void NextNormalDoubles_MeanAndStdDev()
    {
        var rng = new SimdRandom(42);
        var buf = new double[100000];
        rng.NextNormalDoubles(buf);

        double mean = buf.Average();
        double variance = buf.Select(x => (x - mean) * (x - mean)).Average();
        double stddev = Math.Sqrt(variance);

        // Normal(0, 1) should have mean ≈ 0, stddev ≈ 1
        Assert.True(Math.Abs(mean) < 0.02, $"Normal mean {mean} too far from 0");
        Assert.True(Math.Abs(stddev - 1.0) < 0.02, $"Normal stddev {stddev} too far from 1");
    }

    [Fact]
    public void NextDoubles_ScalarTail_HandledCorrectly()
    {
        // Test with sizes that don't align to SIMD width
        var rng = new SimdRandom(42);
        foreach (int size in new[] { 1, 2, 3, 5, 7, 13 })
        {
            var buf = new double[size];
            rng.NextDoubles(buf);
            foreach (double v in buf)
            {
                Assert.True(v >= 0.0 && v < 1.0, $"Size={size}: value {v} out of range");
            }
        }
    }

    [Fact]
    public void NextNormalDoubles_OddLength_NoException()
    {
        var rng = new SimdRandom(42);
        var buf = new double[7]; // odd length for Box-Muller pairs
        rng.NextNormalDoubles(buf);
        Assert.Equal(7, buf.Length);
    }
}
