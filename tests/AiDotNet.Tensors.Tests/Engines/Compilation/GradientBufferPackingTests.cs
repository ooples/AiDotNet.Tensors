using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class GradientBufferPackingTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SameActionReadersAndWritersNeverShareStorage(bool longestFirst)
    {
        var intervals = new List<GradientBufferPacking.Interval>
        { new(0, 1024, 0, 3), new(1, 1024, 3, 6), new(2, 512, 7, 8) };
        var result = GradientBufferPacking.Pack(3, intervals, longestFirst);
        Assert.NotEqual(result.BankOf[0], result.BankOf[1]);
        Assert.Equal(2, result.Capacities.Count);
        Validate(intervals, result);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LongSmallTenantDoesNotOccupyEveryLargeBank(bool longestFirst)
    {
        var intervals = new List<GradientBufferPacking.Interval>
        { new(0, 1024, 0, 1), new(1, 16, 2, 20), new(2, 1024, 3, 4), new(3, 1024, 6, 7) };
        var result = GradientBufferPacking.Pack(4, intervals, longestFirst);
        Assert.Equal(1040, result.Capacities.Sum());
        Assert.Equal(result.BankOf[0], result.BankOf[2]);
        Assert.Equal(result.BankOf[2], result.BankOf[3]);
        Validate(intervals, result);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BoundedRandomLifetimesPreserveEveryConcurrentValueAndDeterministicAssignment(bool longestFirst)
    {
        var random = new Random(7351);
        for (int trial = 0; trial < 32; trial++)
        {
            var intervals = Enumerable.Range(0, 64).Select(id =>
            {
                int start = random.Next(80);
                return new GradientBufferPacking.Interval(id, 1 << random.Next(0, 11), start, start + random.Next(15));
            }).ToList();
            var result = GradientBufferPacking.Pack(64, intervals, longestFirst);
            Validate(intervals, result);
            var repeated = GradientBufferPacking.Pack(64, intervals, longestFirst);
            Assert.Equal(result.BankOf, repeated.BankOf);
            Assert.Equal(result.Capacities, repeated.Capacities);
            var banks = result.Capacities.Select(size => new int[size]).ToArray();
            for (int action = 0; action < 96; action++)
            {
                foreach (var interval in intervals.Where(value => value.First == action))
                    banks[result.BankOf[interval.Id]].AsSpan(0, interval.Elements).Fill(interval.Id + 1);
                foreach (var interval in intervals.Where(value => value.First <= action && action <= value.Last))
                    Assert.True(banks[result.BankOf[interval.Id]].Take(interval.Elements).All(value => value == interval.Id + 1));
            }
        }
    }

    private static void Validate(List<GradientBufferPacking.Interval> intervals,
        (int[] BankOf, List<int> Capacities) result)
    {
        foreach (var a in intervals)
        {
            Assert.True(result.Capacities[result.BankOf[a.Id]] >= a.Elements);
            foreach (var b in intervals)
                if (a.Id != b.Id && a.First <= b.Last && b.First <= a.Last)
                    Assert.NotEqual(result.BankOf[a.Id], result.BankOf[b.Id]);
        }
    }
}
