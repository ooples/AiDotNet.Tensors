// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// WarmWindow converts between TimeSpan and Stopwatch ticks. It used to truncate twice (to whole microseconds,
/// then through a double), so a small positive window became 0 and silently parked workers immediately.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class PersistentParallelExecutorWarmWindowTests
{
    [Theory]
    [InlineData(1L)]            // 100 ns: below one microsecond
    [InlineData(9L)]
    [InlineData(10L)]           // exactly 1 µs
    [InlineData(2_000_000L)]    // the 200 ms default
    public void APositiveWindow_ReadsBackAtLeastAsLongAndNeverZero(long ticks)
    {
        var original = PersistentParallelExecutor.WarmWindow;
        try
        {
            var set = TimeSpan.FromTicks(ticks);
            PersistentParallelExecutor.WarmWindow = set;
            var read = PersistentParallelExecutor.WarmWindow;

            Assert.True(read > TimeSpan.Zero, $"A {set.Ticks}-tick window read back as zero.");
            Assert.True(read >= set, $"Set {set.Ticks} ticks, read back {read.Ticks}.");
            // Rounding up to whole Stopwatch ticks can add at most one of them.
            long oneStopwatchTick = (long)Math.Ceiling((double)TimeSpan.TicksPerSecond / System.Diagnostics.Stopwatch.Frequency);
            Assert.True(read.Ticks - set.Ticks <= oneStopwatchTick, $"Set {set.Ticks} ticks, read back {read.Ticks}.");
        }
        finally
        {
            PersistentParallelExecutor.WarmWindow = original;
        }
    }

    [Fact]
    public void AZeroWindow_StaysZero()
    {
        var original = PersistentParallelExecutor.WarmWindow;
        try
        {
            PersistentParallelExecutor.WarmWindow = TimeSpan.Zero;
            Assert.Equal(TimeSpan.Zero, PersistentParallelExecutor.WarmWindow);
        }
        finally
        {
            PersistentParallelExecutor.WarmWindow = original;
        }
    }
}
