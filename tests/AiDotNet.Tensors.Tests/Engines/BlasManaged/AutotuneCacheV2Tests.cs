using System;
using System.Threading;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Autotune-Serial")]
public class AutotuneCacheV2Tests
{
    [Fact]
    public void Cache_FirstCall_RegistersWarmup()
    {
        var cache = new AutotuneCacheV2();
        var key = new AutotuneCacheV2.ShapeKey(64, 64, 64, false, false, DType.Single);
        Assert.False(cache.TryGet(key, out _));
        cache.RecordWarmupSample(key, PackingMode.ForceStreaming, numThreads: 8, mc: 64, nc: 64, kc: 64, mr: 8, nr: 8, measuredMs: 0.1);
        cache.RecordWarmupSample(key, PackingMode.ForcePackAOnly, numThreads: 16, mc: 64, nc: 64, kc: 64, mr: 8, nr: 8, measuredMs: 0.5);
        cache.RecordWarmupSample(key, PackingMode.ForcePackBoth, numThreads: 16, mc: 64, nc: 64, kc: 64, mr: 8, nr: 8, measuredMs: 0.2);
        cache.FinalizeWarmup(key);
        Assert.True(cache.TryGet(key, out var entry));
        // Min measured: Streaming at 0.1ms.
        Assert.Equal(PackingMode.ForceStreaming, entry!.Mode);
        Assert.Equal(8, entry.NumThreads);
    }

    [Fact]
    public void Cache_ReTuneTrigger_FiresAfter10ConsecutiveSlowCalls()
    {
        var cache = new AutotuneCacheV2();
        var key = new AutotuneCacheV2.ShapeKey(64, 64, 64, false, false, DType.Single);
        cache.RecordWarmupSample(key, PackingMode.ForceStreaming, 8, 64, 64, 64, 8, 8, measuredMs: 0.1);
        cache.FinalizeWarmup(key);
        for (int i = 0; i < 9; i++)
            Assert.False(cache.ObserveAndMaybeReTune(key, measuredMs: 0.2));  // 2× cached but only 9 in a row
        Assert.True(cache.ObserveAndMaybeReTune(key, measuredMs: 0.2));  // 10th: triggers re-tune
    }

    [Fact]
    public void Cache_ReTuneCounter_ResetsOnFastObservation()
    {
        var cache = new AutotuneCacheV2();
        var key = new AutotuneCacheV2.ShapeKey(64, 64, 64, false, false, DType.Single);
        cache.RecordWarmupSample(key, PackingMode.ForceStreaming, 8, 64, 64, 64, 8, 8, 0.1);
        cache.FinalizeWarmup(key);
        for (int i = 0; i < 5; i++)
            Assert.False(cache.ObserveAndMaybeReTune(key, 0.2));
        // A fast observation resets the counter.
        Assert.False(cache.ObserveAndMaybeReTune(key, 0.1));
        for (int i = 0; i < 9; i++)
            Assert.False(cache.ObserveAndMaybeReTune(key, 0.2));
        Assert.True(cache.ObserveAndMaybeReTune(key, 0.2));
    }
}
