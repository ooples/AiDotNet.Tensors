using System;
using System.Collections.Generic;
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
    public void SortedDistributionStatistics_DerivesMedianAndP95FromLength()
    {
        (double oddMedian, double oddP95) =
            GpuAutotuneMeasurement.SortedDistributionStatistics(new[] { 1f, 2f, 3f, 4f, 5f });
        var even = new float[20];
        for (int i = 0; i < even.Length; i++) even[i] = i + 1;
        (double evenMedian, double evenP95) =
            GpuAutotuneMeasurement.SortedDistributionStatistics(even);

        Assert.Equal(3.0, oddMedian);
        Assert.Equal(5.0, oddP95);
        Assert.Equal(10.5, evenMedian);
        Assert.Equal(19.0, evenP95);
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

    [Fact]
    public void AdaptiveStableGflops_IncreasesLaunchGroupingUntilStable()
    {
        var launchGroups = new List<int>();
        double gflops = GpuAutotuneMeasurement.AdaptiveStableGflops(
            launches =>
            {
                launchGroups.Add(launches);
                return launches == 8
                    ? new[] { 1.0f, 1.0f, 1.0f, 1.5f }
                    : new[] { 2.0f, 2.0f, 2.01f, 1.99f };
            },
            operations: 2_000_000);

        Assert.Equal(new[] { 8, 32 }, launchGroups);
        Assert.Equal(1.0, gflops, 2);
    }

    [Fact]
    public void AdaptiveStableMedian_DoesNotRetryLaunchFailure()
    {
        int calls = 0;
        Assert.Throws<NotSupportedException>(() =>
            GpuAutotuneMeasurement.AdaptiveStableMedianMilliseconds(
                _ =>
                {
                    calls++;
                    throw new NotSupportedException("candidate cannot launch");
                }));
        Assert.Equal(1, calls);
    }
}
