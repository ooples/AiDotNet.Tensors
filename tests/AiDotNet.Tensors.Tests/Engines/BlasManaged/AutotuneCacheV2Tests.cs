using System;
using System.Threading;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged.Autotune;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
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

    [Fact]
    public void EnableAutotuneV2_Gemm_ProducesCorrectResultAndCachesShape()
    {
        // End-to-end: with EnableAutotuneV2 on, a GEMM above the work floor runs
        // a warmup sweep (cache miss), caches the winner, and produces a correct
        // result. A second call hits the cache. Both must match the naive ref.
        const int M = 128, K = 128, N = 128;  // work = 2.1M ≥ 100K floor
        var rng = new Random(7);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var reference = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int kk = 0; kk < K; kk++) sum += a[i * K + kk] * b[kk * N + j];
                reference[i * N + j] = sum;
            }

        bool prev = BlasManagedLib.EnableAutotuneV2;
        BlasManagedLib.EnableAutotuneV2 = true;
        try
        {
            int countBefore = BlasManagedLib.AutotuneV2ShapeCount;

            var c1 = new float[M * N];
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c1, N, M, N, K);

            // Cache must now hold this shape (sweep finalized).
            Assert.True(BlasManagedLib.AutotuneV2ShapeCount > countBefore,
                "Expected the autotune cache to gain the swept shape.");

            // Second call: cache hit, same correct result.
            var c2 = new float[M * N];
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c2, N, M, N, K);

            for (int i = 0; i < reference.Length; i++)
            {
                Assert.True(Math.Abs(reference[i] - c1[i]) < 1e-2,
                    $"first-call mismatch at {i}: ref {reference[i]} got {c1[i]}");
                Assert.True(Math.Abs(reference[i] - c2[i]) < 1e-2,
                    $"cached-call mismatch at {i}: ref {reference[i]} got {c2[i]}");
            }
        }
        finally
        {
            BlasManagedLib.EnableAutotuneV2 = prev;
        }
    }
}
