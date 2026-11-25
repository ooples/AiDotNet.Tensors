#if NET5_0_OR_GREATER
// The packed-weight cache is part of the .NET 5+ SIMD GEMM (SimdGemm's packed kernels); .NET Framework has no cache.
using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class PackedWeightCacheSerial
{
    public const string Name = "PackedWeightCacheSerial";
}

/// <summary>
/// The packed-weight cache behind <see cref="SimdGemm.SgemmWithCachedB"/> kept a packed copy of every weight it ever
/// multiplied, so a model served from the CPU held its weights twice (a 2-layer cut of a 6.9B-parameter mixture-of-experts model peaked at 21.1 GB
/// private with the copies, 15.5 GB without). The cache is now bounded: weights past the budget take the per-call
/// packing route, keep no copy, and still compute the same product.
/// </summary>
[Collection(PackedWeightCacheSerial.Name)]
public sealed class PackedWeightCacheBoundTests
{
    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return d;
    }

    private static void AssertDirectSum(float[] a, float[] b, float[] c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0, magnitude = 0;
                for (int p = 0; p < k; p++)
                {
                    double term = (double)a[i * k + p] * b[p * n + j];
                    sum += term; magnitude += Math.Abs(term);
                }
                Assert.True(Math.Abs(sum - c[i * n + j]) <= k * Math.Pow(2, -24) * magnitude,
                    $"[{i},{j}]: {c[i * n + j]} vs {sum}");
            }
    }

    [Fact]
    public void Weights_past_the_budget_keep_no_packed_copy_and_still_multiply_exactly()
    {
        int m = 65, k = 768, n = 192;
        var a = Rand(3, m * k);
        long saved = SimdGemm.PackedWeightCacheBudgetBytes;
        long held = SimdGemm.PackedWeightCacheBytes;
        long one = SimdGemm.PrePackedBBytes(k, n, m);
        // Room for at most four more of these weights; each is under a quarter of the budget, so size alone admits it.
        SimdGemm.PackedWeightCacheBudgetBytes = held + 4 * one + one / 2;
        var weights = new float[8][];
        try
        {
            for (int w = 0; w < weights.Length; w++)
            {
                weights[w] = Rand(10 + w, k * n);
                var c = new float[m * n];
                SimdGemm.SgemmWithCachedB(a, weights[w], c, m, k, n);
                AssertDirectSum(a, weights[w], c, m, k, n);
                Assert.True(SimdGemm.PackedWeightCacheBytes <= SimdGemm.PackedWeightCacheBudgetBytes,
                    $"after weight {w}: cache holds {SimdGemm.PackedWeightCacheBytes} B, budget {SimdGemm.PackedWeightCacheBudgetBytes} B");
            }
            // Unbounded, the eight weights would add 8 copies; bounded, at most four were admitted.
            Assert.True(SimdGemm.PackedWeightCacheBytes - held <= 4 * one + one / 2,
                $"cache grew by {SimdGemm.PackedWeightCacheBytes - held} B for eight {one} B weights");
        }
        finally
        {
            SimdGemm.PackedWeightCacheBudgetBytes = saved;
            GC.KeepAlive(weights);
        }
    }

    [Fact]
    public void A_weight_larger_than_a_quarter_of_the_budget_is_never_cached()
    {
        int m = 65, k = 768, n = 192;
        var a = Rand(5, m * k);
        var b = Rand(6, k * n);
        long saved = SimdGemm.PackedWeightCacheBudgetBytes;
        SimdGemm.PackedWeightCacheBudgetBytes = SimdGemm.PrePackedBBytes(k, n, m) * 2;   // the weight is half the budget
        try
        {
            long held = SimdGemm.PackedWeightCacheBytes;
            int tiledBefore = SimdGemm.CachedBTiledCallsOnThisThread;
            var c = new float[m * n];
            SimdGemm.SgemmWithCachedB(a, b, c, m, k, n);
            Assert.Equal(tiledBefore, SimdGemm.CachedBTiledCallsOnThisThread);
            Assert.Equal(held, SimdGemm.PackedWeightCacheBytes);
            AssertDirectSum(a, b, c, m, k, n);
        }
        finally { SimdGemm.PackedWeightCacheBudgetBytes = saved; GC.KeepAlive(b); }
    }
    [Fact]
    public void Concurrent_first_uses_admit_no_more_than_the_budget()
    {
        int m = 65, k = 768, n = 192;
        var a = Rand(7, m * k);
        long saved = SimdGemm.PackedWeightCacheBudgetBytes;
        long held = SimdGemm.PackedWeightCacheBytes;
        long one = SimdGemm.PrePackedBBytes(k, n, m);
        // Room for at most four more of these weights (a weight must also be under a quarter of the budget). Eight threads
        // multiply eight new weights at once: each sees room before any is admitted, so the budget holds only if
        // admission is decided where admissions are serialized.
        SimdGemm.PackedWeightCacheBudgetBytes = Math.Max(held + one + one / 2, 4 * one + 4);
        long room = SimdGemm.PackedWeightCacheBudgetBytes - held;
        const int threads = 8;
        var weights = new float[threads][];
        for (int w = 0; w < threads; w++) weights[w] = Rand(30 + w, k * n);
        var products = new float[threads][];
        using var start = new System.Threading.Barrier(threads);
        try
        {
            var workers = new System.Threading.Thread[threads];
            for (int w = 0; w < threads; w++)
            {
                int id = w;
                workers[w] = new System.Threading.Thread(() =>
                {
                    var c = new float[m * n];
                    start.SignalAndWait();
                    SimdGemm.SgemmWithCachedB(a, weights[id], c, m, k, n);
                    products[id] = c;
                });
                workers[w].Start();
            }
            foreach (var t in workers) t.Join();
            for (int w = 0; w < threads; w++) AssertDirectSum(a, weights[w], products[w], m, k, n);
            long grown = SimdGemm.PackedWeightCacheBytes - held;
            Assert.True(grown <= room, $"the cache grew by {grown} B with room for {room} B ({one} B per weight)");
        }
        finally
        {
            SimdGemm.PackedWeightCacheBudgetBytes = saved;
            GC.KeepAlive(weights);
        }
    }
}
#endif
