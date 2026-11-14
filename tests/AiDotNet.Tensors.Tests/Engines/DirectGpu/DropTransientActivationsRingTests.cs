using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// DropTransientActivationsCreatedAfter finds the entries cached after its snapshot through a ring of recently cached
/// keys instead of copying the whole activation cache. These pin that it still frees exactly those entries (keeping
/// the live result and everything older) when the cache holds far more older entries than new ones, and when more
/// entries were cached since the snapshot than the ring holds (it then scans the whole cache).
/// </summary>
[Collection("DirectGpuSerial")]
public class DropTransientActivationsRingTests
{
    private sealed class TrackingGpuBuffer : IGpuBuffer
    {
        public int DisposeCount;
        public int Size => 4;
        public long SizeInBytes => 16;
        public IntPtr Handle => IntPtr.Zero;
        public void Dispose() => DisposeCount++;
    }

    private static readonly MethodInfo CacheActivationMethod = typeof(DirectGpuTensorEngine).GetMethod(
        "CacheActivation", BindingFlags.NonPublic | BindingFlags.Instance, null,
        new[] { typeof(object), typeof(IGpuBuffer), typeof(int[]), typeof(IDirectGpuBackend), typeof(bool), typeof(int), typeof(int) },
        null)!;

    private static (object Key, TrackingGpuBuffer Buffer) Cache(DirectGpuTensorEngine engine)
    {
        var key = new float[4];
        var buffer = new TrackingGpuBuffer();
        CacheActivationMethod.Invoke(engine, new object?[] { key, buffer, new[] { 4 }, null, false, 0, 0 });
        return (key, buffer);
    }

    [Theory]
    [InlineData(10000, 3)]     // many older entries, a few new ones: the ring walk
    [InlineData(10, 6000)]     // more new entries than the ring keeps: the full-cache fallback
    public void Drop_FreesExactlyTheEntriesCachedAfterTheSnapshot_ExceptTheKeptOne(int before, int after)
    {
        using var engine = new DirectGpuTensorEngine();
        Assert.NotNull(CacheActivationMethod);

        var older = new List<TrackingGpuBuffer>();
        for (int i = 0; i < before; i++) older.Add(Cache(engine).Buffer);

        long snapshot = engine.ActivationCacheTimestampSnapshot();
        var newer = new List<TrackingGpuBuffer>();
        for (int i = 0; i < after - 1; i++) newer.Add(Cache(engine).Buffer);
        var kept = Cache(engine);

        engine.DropTransientActivationsCreatedAfter(snapshot, kept.Key);

        Assert.All(older, b => Assert.Equal(0, b.DisposeCount));
        Assert.All(newer, b => Assert.Equal(1, b.DisposeCount));
        Assert.Equal(0, kept.Buffer.DisposeCount);
    }
}
