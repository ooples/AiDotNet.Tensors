using System;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class SimdRandomTests
{
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
