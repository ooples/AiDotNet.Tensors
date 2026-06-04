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
    public void SgemmWithCachedB_AfterInPlaceWeightMutation_TargetedInvalidate_UsesFreshWeights()
    {
        // Per-array invalidation (review AiDotNet#1488): mutating ONE weight
        // and invalidating ONLY that array must produce fresh results for it
        // — while a second, untouched weight keeps producing correct results
        // throughout (its cached pack stays valid; no global eviction).
        const int m = 8, k = 512, n = 64;
        var rng = new Random(1488);
        var a = new float[m * k];
        var b1 = new float[k * n];
        var b2 = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b1.Length; i++) b1[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b2.Length; i++) b2[i] = (float)(rng.NextDouble() * 2 - 1);

        // Populate caches for BOTH weights.
        var c1 = new float[m * n];
        var c2 = new float[m * n];
        SimdGemm.SgemmWithCachedB(a, b1, c1, m, k, n);
        SimdGemm.SgemmWithCachedB(a, b2, c2, m, k, n);
        AssertMatchesNaive(a, b1, c1, m, k, n, "b1 initial");
        AssertMatchesNaive(a, b2, c2, m, k, n, "b2 initial");

        // Mutate ONLY b1; invalidate ONLY b1.
        for (int i = 0; i < b1.Length; i++) b1[i] = -b1[i] * 0.5f + 0.25f;
        InferenceWeightCache.Invalidate(b1);

        var c1b = new float[m * n];
        var c2b = new float[m * n];
        SimdGemm.SgemmWithCachedB(a, b1, c1b, m, k, n);
        SimdGemm.SgemmWithCachedB(a, b2, c2b, m, k, n);
        AssertMatchesNaive(a, b1, c1b, m, k, n, "b1 post-targeted-invalidate");
        AssertMatchesNaive(a, b2, c2b, m, k, n, "b2 untouched");
    }

    [Fact]
    public void Invalidate_NullOrEmpty_IsSafeNoOp()
    {
        InferenceWeightCache.Invalidate(null);
        InferenceWeightCache.Invalidate(Array.Empty<float>());
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

#if NET5_0_OR_GREATER
    // SgemmWithInt8CachedB (and the int8 pre-packed cache) only exist on
    // net5+ — the whole Path-D region in SimdGemm is NET5_0_OR_GREATER.
    [Fact]
    public void SgemmWithInt8CachedB_AfterInPlaceWeightMutation_InvalidateAll_UsesFreshWeights()
    {
        // Same contract as the float path, on the weight-only int8 cache.
        // Tolerance is the int8 quantization error (per-tensor symmetric,
        // 35-40 dB SNR), not float rounding.
        const int m = 8, k = 512, n = 64;
        var rng = new Random(540);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var c1 = new float[m * n];
        SimdGemm.SgemmWithInt8CachedB(a, b, c1, m, k, n);
        AssertMatchesNaiveQuantized(a, b, c1, m, k, n, "initial int8 call");

        for (int i = 0; i < b.Length; i++) b[i] = -b[i] * 0.5f + 0.25f;
        InferenceWeightCache.InvalidateAll();

        var c2 = new float[m * n];
        SimdGemm.SgemmWithInt8CachedB(a, b, c2, m, k, n);
        AssertMatchesNaiveQuantized(a, b, c2, m, k, n, "post-mutation int8 call");
    }

    private static void AssertMatchesNaiveQuantized(
        float[] a, float[] b, float[] c, int m, int k, int n, string label)
    {
        // Per-element bound: int8 symmetric quantization of B contributes
        // |err| <= scale/2 per product term; accumulate over K with the A
        // magnitudes. A loose absolute bound of 2% of the accumulated
        // |a|·|b| magnitude comfortably covers it while still failing hard
        // on stale weights (B's mutation flips sign and shifts by 0.25 —
        // stale results differ by O(1), not O(0.02)).
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double acc = 0, mag = 0;
                for (int p = 0; p < k; p++)
                {
                    acc += (double)a[i * k + p] * b[p * n + j];
                    mag += Math.Abs((double)a[i * k + p] * b[p * n + j]);
                }
                double got = c[i * n + j];
                double tol = 0.02 * Math.Max(1.0, mag);
                Assert.True(Math.Abs(got - acc) <= tol,
                    $"{label}: C[{i},{j}] = {got}, expected {acc} ± {tol} (stale packed int8 weights?)");
            }
        }
    }
#endif

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
