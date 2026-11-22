// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// IBatchedCopyBackend.CopyMany on CUDA (copy_many8): eight copies per launch, a ragged last launch whose unused slots
/// copy nothing, and the contract the concurrent copies rely on: every buffer holds its copy's length, and no copy
/// writes memory another copy reads or writes.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CudaCopyManyTests
{
    private static CudaBackend RequireBackend()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "Requires an NVIDIA CUDA GPU.");
        }
        return backend;
    }

    private static float[] Values(int count, int seed) => Enumerable.Range(0, count).Select(i => seed * 1000f + i + 0.5f).ToArray();

    [SkippableFact]
    public void Eleven_copies_across_two_launches_copy_exactly_their_lengths()
    {
        using var backend = RequireBackend();
        // Lengths straddle the 256-thread block and the 1024-block stride; a zero-length copy and one past a launch's
        // grid (1024 x 256) included. The destinations are one longer than the copy, so a write past it shows.
        int[] lengths = { 1, 0, 255, 256, 257, 4099, 31, 1024 * 256 + 3, 7, 513, 2 };
        var sources = new List<IGpuBuffer>();
        var destinations = new List<IGpuBuffer>();
        try
        {
            for (int i = 0; i < lengths.Length; i++)
            {
                sources.Add(backend.AllocateBuffer(Values(Math.Max(1, lengths[i]), i + 1)));
                destinations.Add(backend.AllocateBuffer(Enumerable.Repeat(-7f, lengths[i] + 1).ToArray()));
            }
            ((IBatchedCopyBackend)backend).CopyMany(sources, destinations, lengths);
            backend.Synchronize();
            for (int i = 0; i < lengths.Length; i++)
            {
                var got = backend.DownloadBuffer(destinations[i]);
                var want = Values(Math.Max(1, lengths[i]), i + 1);
                for (int j = 0; j < lengths[i]; j++)
                    Assert.True(got[j] == want[j], $"copy {i} element {j}: {got[j]} != {want[j]}");
                Assert.Equal(-7f, got[lengths[i]]);
            }
        }
        finally
        {
            foreach (var b in sources.Concat(destinations)) b.Dispose();
        }
    }

    [SkippableFact]
    public void Overlapping_copies_and_short_buffers_are_refused_before_launch()
    {
        using var backend = RequireBackend();
        var copy = (IBatchedCopyBackend)backend;
        using var a = backend.AllocateBuffer(Values(64, 1));
        using var b = backend.AllocateBuffer(Values(64, 2));
        using var c = backend.AllocateBuffer(Values(64, 3));
        using var shortBuffer = backend.AllocateBuffer(Values(8, 4));

        // Two copies into one destination race; a destination another copy reads races with that read.
        Assert.Throws<ArgumentException>(() => copy.CopyMany(new[] { a, b }, new[] { c, c }, new[] { 64, 64 }));
        Assert.Throws<ArgumentException>(() => copy.CopyMany(new[] { a, b }, new[] { b, c }, new[] { 64, 64 }));
        // A length past either buffer would read or write outside it.
        Assert.Throws<ArgumentException>(() => copy.CopyMany(new[] { a }, new[] { shortBuffer }, new[] { 64 }));
        Assert.Throws<ArgumentException>(() => copy.CopyMany(new[] { shortBuffer }, new[] { a }, new[] { 64 }));
        backend.Synchronize();
        Assert.Equal(Values(64, 1), backend.DownloadBuffer(a));
        Assert.Equal(Values(64, 2), backend.DownloadBuffer(b));
        Assert.Equal(Values(64, 3), backend.DownloadBuffer(c));

        // Allowed: one source read by two copies, a copy onto its own buffer, and a zero-length copy into anything.
        using var d = backend.AllocateBuffer(Values(64, 5));
        copy.CopyMany(new[] { a, a, d, c }, new[] { b, c, d, a }, new[] { 64, 64, 64, 0 });
        backend.Synchronize();
        Assert.Equal(Values(64, 1), backend.DownloadBuffer(b));
        Assert.Equal(Values(64, 1), backend.DownloadBuffer(c));
        Assert.Equal(Values(64, 5), backend.DownloadBuffer(d));
        Assert.Equal(Values(64, 1), backend.DownloadBuffer(a));
    }
}
