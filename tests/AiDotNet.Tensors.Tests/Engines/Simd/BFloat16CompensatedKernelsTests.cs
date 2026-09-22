using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

public class BFloat16CompensatedKernelsTests
{
    [Theory]
    [InlineData(24)]
    [InlineData(25)]
    [InlineData(31)]
    [InlineData(64)]
    public void Dot_PreservesSmallTermsBetweenCancellingLargeTerms(int length)
    {
        var x = new BFloat16[length];
        var y = new BFloat16[length];
        for (int i = 0; i < length; i++) x[i] = BFloat16.FromFloat(1f);
        for (int i = 0; i < 8; i++)
        {
            y[i] = BFloat16.FromFloat(16777216f);
            y[i + 8] = BFloat16.FromFloat(1f);
            y[i + 16] = BFloat16.FromFloat(-16777216f);
        }
        for (int i = 24; i < length; i++) y[i] = BFloat16.FromFloat(0.5f);
        Assert.Equal(8f + (length - 24) * 0.5f, BFloat16CompensatedKernels.Dot(x, y));
        Array.Reverse(x);
        Array.Reverse(y);
        Assert.Equal(8f + (length - 24) * 0.5f, BFloat16CompensatedKernels.Dot(x, y));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(64)]
    [InlineData(257)]
    public void Dot_MatchesExactDyadicReferenceIncludingScalarTails(int length)
    {
        var x = new BFloat16[length];
        var y = new BFloat16[length];
        double expected = 0;
        for (int i = 0; i < length; i++)
        {
            x[i] = BFloat16.FromFloat((i % 31 - 15) / 8f);
            y[i] = BFloat16.FromFloat((i % 17 - 8) / 16f);
            expected += (double)(float)x[i] * (float)y[i];
        }
        Assert.Equal((float)expected, BFloat16CompensatedKernels.Dot(x, y));
    }

    [Fact]
    public void Dot_ValidatesLengthsAndAcceptsEmptyInputs()
    {
        Assert.Equal(0f, BFloat16CompensatedKernels.Dot(Array.Empty<BFloat16>(), Array.Empty<BFloat16>()));
        Assert.Throws<ArgumentException>(() => BFloat16CompensatedKernels.Dot(new BFloat16[8], new BFloat16[7]));
    }

    [Fact]
    public void Dot_PreservesNonfiniteArithmeticInsteadOfCorruptingInfinityWithCompensation()
    {
        var x = new BFloat16[16];
        var y = new BFloat16[16];
        for (int i = 0; i < x.Length; i++) { x[i] = BFloat16.FromFloat(1f); y[i] = BFloat16.FromFloat(1f); }
        x[0] = BFloat16.FromFloat(float.PositiveInfinity);
        Assert.True(float.IsPositiveInfinity(BFloat16CompensatedKernels.Dot(x, y)));
        x[1] = BFloat16.FromFloat(float.NegativeInfinity);
        Assert.True(float.IsNaN(BFloat16CompensatedKernels.Dot(x, y)));
        x[0] = BFloat16.FromFloat(float.NaN);
        Assert.True(float.IsNaN(BFloat16CompensatedKernels.Dot(x, y)));
    }
}
