// Copyright (c) AiDotNet. All rights reserved.
// FftPlanCache contract tests: the cache memoizes Bluestein plans per
// (n, inverse) key and returns bit-identical results across cold/warm calls.
// Test strategy: probe GetOrCreateBluestein directly rather than counting
// entries (which is race-prone with concurrent FFT tests in the same process).

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Fft;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.FftTests;

public class FftPlanCacheTests
{
    // Repeated lookups with the same key return the SAME plan instance.
    // This asserts reuse directly instead of depending on a global counter
    // that concurrent tests can perturb.
    [Fact]
    public void GetOrCreateBluestein_ReturnsSameInstanceForSameKey()
    {
        var a = FftPlanCache.GetOrCreateBluestein(83, inverse: false);
        var b = FftPlanCache.GetOrCreateBluestein(83, inverse: false);
        Assert.Same(a, b);

        // Different direction → different plan.
        var c = FftPlanCache.GetOrCreateBluestein(83, inverse: true);
        Assert.NotSame(a, c);

        // Different size → different plan.
        var d = FftPlanCache.GetOrCreateBluestein(89, inverse: false);
        Assert.NotSame(a, d);
    }

    [Fact]
    public void GetOrCreateBluestein_ParametersAreCorrect()
    {
        var plan = FftPlanCache.GetOrCreateBluestein(7, inverse: false);
        Assert.Equal(7, plan.N);
        Assert.False(plan.Inverse);
        Assert.Equal(7, plan.ChirpRe.Length);
        Assert.Equal(7, plan.ChirpIm.Length);
        // M must be a power of 2 ≥ 2N−1 = 13 → 16.
        Assert.Equal(16, plan.M);
        Assert.Equal(16, plan.BSpectrumRe.Length);
    }

    [Fact]
    public void BluesteinPlan_BitIdenticalWarmColdResults()
    {
        // Independent of global cache state: two identical inputs must produce
        // byte-identical outputs whether the plan was freshly constructed or
        // retrieved from cache.
        var x = new Tensor<double>(new[] { 2 * 13 }); // n=13 complex → Bluestein
        var d = x.GetDataArray();
        var rng = new Random(42);
        for (int i = 0; i < d.Length; i++) d[i] = rng.NextDouble();

        var cold = Fft.Fft1(x);
        var warm = Fft.Fft1(x);
        var coldD = cold.GetDataArray();
        var warmD = warm.GetDataArray();
        for (int i = 0; i < coldD.Length; i++)
            Assert.Equal(coldD[i], warmD[i]);
    }
}
