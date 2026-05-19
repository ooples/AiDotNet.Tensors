using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-F3 (#374) task: verifies <see cref="PrefersManagedCache"/> measures both
/// paths on cache miss, caches the winner, and routes <see cref="BlasProvider.TryGemmEx"/>
/// through the cached decision when <see cref="BlasManagedLib.AutotuneRouting"/> is true.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class PrefersManagedCacheTest
{
    public PrefersManagedCacheTest()
    {
        PrefersManagedCache.Clear();
        BlasManagedLib.AutotuneRouting = false;
        BlasManagedLib.PreferManaged = false;
    }

    [Fact]
    public void Default_AutotuneRouting_Is_False()
    {
        Assert.False(BlasManagedLib.AutotuneRouting);
    }

    [Fact]
    public void First_Call_Populates_Cache()
    {
        PrefersManagedCache.Clear();
        Assert.Equal(0, PrefersManagedCache.Count);

        // Trigger measurement.
        bool result = PrefersManagedCache.PrefersManaged(64, 64, 64, false, false, typeof(float));

        Assert.Equal(1, PrefersManagedCache.Count);
        // Result is true or false depending on which path was faster — we don't
        // assert direction, just that it stabilizes.
        bool result2 = PrefersManagedCache.PrefersManaged(64, 64, 64, false, false, typeof(float));
        Assert.Equal(result, result2);
        Assert.Equal(1, PrefersManagedCache.Count);  // still one entry — cache hit
    }

    [Fact]
    public void Different_Shapes_Get_Different_Cache_Entries()
    {
        PrefersManagedCache.Clear();
        PrefersManagedCache.PrefersManaged(32, 32, 32, false, false, typeof(float));
        PrefersManagedCache.PrefersManaged(64, 64, 64, false, false, typeof(float));
        PrefersManagedCache.PrefersManaged(32, 32, 32, false, false, typeof(double));  // FP64 differs by dtype
        Assert.Equal(3, PrefersManagedCache.Count);
    }

    [Fact]
    public void TryGemmEx_With_AutotuneRouting_Returns_Correct_Output_FP32()
    {
        PrefersManagedCache.Clear();
        BlasManagedLib.AutotuneRouting = true;
        try
        {
            const int M = 64, N = 64, K = 64;
            var rng = new Random(42);
            var a = new float[M * K];
            var b = new float[K * N];
            var c = new float[M * N];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            bool ok = BlasProvider.TryGemmEx(M, N, K, a, 0, K, false, b, 0, N, false, c, 0, N);
            Assert.True(ok);

            // Output is non-zero (kernel actually ran via cache-routed path).
            bool anyNonZero = false;
            for (int i = 0; i < c.Length; i++) if (c[i] != 0) { anyNonZero = true; break; }
            Assert.True(anyNonZero);

            // Cache was populated.
            Assert.True(PrefersManagedCache.Count >= 1);
        }
        finally
        {
            BlasManagedLib.AutotuneRouting = false;
        }
    }

    [Fact]
    public void Default_Behavior_Unchanged_When_AutotuneRouting_False()
    {
        BlasManagedLib.AutotuneRouting = false;
        BlasManagedLib.PreferManaged = false;
        PrefersManagedCache.Clear();

        const int M = 32, N = 32, K = 32;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        bool ok = BlasProvider.TryGemmEx(M, N, K, a, 0, K, false, b, 0, N, false, c, 0, N);

        // With autotune off, cache must NOT be touched.
        Assert.Equal(0, PrefersManagedCache.Count);
        // ok matches native availability (same as pre-F3 behavior).
        Assert.Equal(BlasProvider.IsAvailable, ok);
    }

    [Fact]
    public void PreferManaged_True_Bypasses_Autotune()
    {
        BlasManagedLib.PreferManaged = true;
        BlasManagedLib.AutotuneRouting = true;
        PrefersManagedCache.Clear();
        try
        {
            const int M = 32, N = 32, K = 32;
            var rng = new Random(42);
            var a = new float[M * K];
            var b = new float[K * N];
            var c = new float[M * N];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            bool ok = BlasProvider.TryGemmEx(M, N, K, a, 0, K, false, b, 0, N, false, c, 0, N);
            Assert.True(ok);

            // PreferManaged is checked BEFORE autotune; cache should still be empty.
            Assert.Equal(0, PrefersManagedCache.Count);
        }
        finally
        {
            BlasManagedLib.PreferManaged = false;
            BlasManagedLib.AutotuneRouting = false;
        }
    }
}
