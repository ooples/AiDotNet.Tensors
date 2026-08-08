// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CudaEventContextIsolationTests
{
    [Fact]
    public async Task ConcurrentAvailabilityReadsPublishOneStableResult()
    {
        var readers = Enumerable.Range(0, 64)
            .Select(_ => Task.Run(() => CudaNativeBindings.IsAvailable))
            .ToArray();

        bool[] results = await Task.WhenAll(readers);

        Assert.All(results, value => Assert.Equal(results[0], value));
    }

    [SkippableFact]
    public void AlternatingBackendsKeepEventsOnTheirOwningContexts()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable,
            "Requires an NVIDIA CUDA driver and GPU.");
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
