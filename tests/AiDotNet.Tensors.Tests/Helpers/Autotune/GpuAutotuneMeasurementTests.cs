using System;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

public sealed class GpuAutotuneMeasurementTests
{
    [Fact]
    public void StableMedianMilliseconds_UsesDistributionMedian()
    {
        double median = GpuAutotuneMeasurement.StableMedianMilliseconds(
            new[] { 10.2f, 9.9f, 10.0f, 10.1f, 10.0f, 9.8f, 10.1f },
            maxP95ToMedian: 1.05);

        Assert.Equal(10.0, median, 3);
    }

    [Fact]
    public void StableMedianMilliseconds_RejectsNoisyCandidate()
    {
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            GpuAutotuneMeasurement.StableMedianMilliseconds(
                new[] { 10.0f, 10.1f, 9.9f, 10.0f, 14.0f },
                maxP95ToMedian: 1.05));

        Assert.Contains("unstable", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void StableMedianMilliseconds_RejectsInvalidSample(float invalid)
    {
        Assert.Throws<InvalidOperationException>(() =>
            GpuAutotuneMeasurement.StableMedianMilliseconds(new[] { 1f, invalid, 1f }));
    }

    [Fact]
    public void StableGflops_UsesMedianRatherThanFastestSample()
    {
        double gflops = GpuAutotuneMeasurement.StableGflops(
            new[] { 2.0f, 2.1f, 1.9f, 2.0f, 2.0f }, operations: 2_000_000);

        Assert.Equal(1.0, gflops, 3);
    }
}
