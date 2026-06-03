using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Stale-weight invalidation for the identity-keyed inference caches
/// (AiDotNet#1296 grad-accum probe root cause): SgemmWithCachedB keys its
/// persistent pre-packed B on the weight ARRAY's object identity and never
/// re-reads contents, so an in-place weight mutation (optimizer step,
/// SetParameters/WithParameters bulk load) silently served the OLD weights
/// on every subsequent inference GEMM. <see cref="InferenceWeightCache.InvalidateAll"/>
/// bumps a global epoch that every hit path (ConditionalWeakTable AND the
/// per-thread MRU slots) checks, forcing a re-pack from live contents.
/// </summary>
public class InferenceWeightCacheInvalidationTests
{
    [Fact]
    public void SgemmWithCachedB_AfterInPlaceWeightMutation_InvalidateAll_UsesFreshWeights()
    {
        // Shape chosen to take the cached-B packed path (not the JIT/small-K
        // or direct fast paths): large K so directWork exceeds the direct
        // thresholds, n modest so the single-jc cache assumption holds.
        const int m = 8, k = 512, n = 64;
        var rng = new Random(1296);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // First call populates the identity-keyed pre-packed cache (or takes
        // a non-cached path on AVX-512 machines — invalidation must be
        // correct either way).
        var c1 = new float[m * n];
        SimdGemm.SgemmWithCachedB(a, b, c1, m, k, n);
        AssertMatchesNaive(a, b, c1, m, k, n, "initial call");

        // Mutate B IN PLACE — same array identity, new contents. This is
        // exactly what an optimizer step or a bulk parameter load does.
        for (int i = 0; i < b.Length; i++) b[i] = -b[i] * 0.5f + 0.25f;

        // The documented contract: in-place mutators must invalidate.
        InferenceWeightCache.InvalidateAll();

        var c2 = new float[m * n];
        SimdGemm.SgemmWithCachedB(a, b, c2, m, k, n);
        AssertMatchesNaive(a, b, c2, m, k, n, "post-mutation call");
    }

    [Fact]
    public void SgemmWithCachedB_RepeatedCallsSameWeights_StayCorrect()
    {
        // Companion guard: the epoch gate must not break the steady-state
        // cache-hit path — repeated calls with UNCHANGED weights must keep
        // producing correct results (and may consume the cached pack).
        const int m = 8, k = 512, n = 64;
        var rng = new Random(7);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        for (int call = 0; call < 3; call++)
        {
            var c = new float[m * n];
            SimdGemm.SgemmWithCachedB(a, b, c, m, k, n);
            AssertMatchesNaive(a, b, c, m, k, n, $"call {call}");
        }
    }

    private static void AssertMatchesNaive(
        float[] a, float[] b, float[] c, int m, int k, int n, string label)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int p = 0; p < k; p++) acc += (double)a[i * k + p] * b[p * n + j];
                double got = c[i * n + j];
                double tol = 1e-3 * Math.Max(1.0, Math.Abs(acc));
                Assert.True(Math.Abs(got - acc) <= tol,
                    $"{label}: C[{i},{j}] = {got}, expected {acc} (stale packed weights?)");
            }
        }
    }
}
