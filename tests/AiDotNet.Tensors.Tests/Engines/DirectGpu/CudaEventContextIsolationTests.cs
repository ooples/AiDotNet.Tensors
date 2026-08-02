// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class CudaEventContextIsolationTests
{
    [SkippableFact]
    public void AlternatingBackendsKeepEventsOnTheirOwningContexts()
    {
        using var first = new CudaBackend();
        using var second = new CudaBackend();
        Skip.IfNot(first.IsAvailable && second.IsAvailable,
            "Requires two available NVIDIA CUDA backend contexts.");

        using var firstStart = first.CreateEvent(enableTiming: true);
        using var secondStart = second.CreateEvent(enableTiming: true);
        using var firstEnd = first.CreateEvent(enableTiming: true);
        using var secondEnd = second.CreateEvent(enableTiming: true);

        first.RecordEvent(firstStart, first.DefaultStream);
        second.RecordEvent(secondStart, second.DefaultStream);
        first.RecordEvent(firstEnd, first.DefaultStream);
        second.RecordEvent(secondEnd, second.DefaultStream);

        firstEnd.Synchronize();
        secondEnd.Synchronize();
        Assert.True(first.GetEventElapsedTime(firstStart, firstEnd) >= 0);
        Assert.True(second.GetEventElapsedTime(secondStart, secondEnd) >= 0);

        Assert.Throws<ArgumentException>(() =>
            first.RecordEvent(firstStart, second.DefaultStream));
        Assert.Throws<ArgumentException>(() =>
            first.GetEventElapsedTime(firstStart, secondEnd));
    }
}
