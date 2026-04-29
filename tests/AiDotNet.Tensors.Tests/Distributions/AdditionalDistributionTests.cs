using System;
using AiDotNet.Tensors.Distributions;
using AiDotNet.Tensors.Distributions.Helpers;
using AiDotNet.Tensors.Distributions.Transforms;
using Xunit;

namespace AiDotNet.Tensors.Tests.Distributions;

/// <summary>
/// Additional tests for the gap-fill distributions/transforms/RNG and full
/// MC-vs-analytical KL cross-checks for every registered KL pair.
/// </summary>
public class AdditionalDistributionTests
{
    private const int N = 50_000;

    // -------------------- New distributions: moments --------------------

    [Fact]
    public void FisherSnedecor_Moments_Match()
    {
        // F(d1=10, d2=20): mean = 20/(20-2) = 10/9 ≈ 1.111;
        // var  = 2 · 20² · (10+20-2) / (10 · 18² · 16) = 22400 / 51840 ≈ 0.4321.
        var d = new FisherSnedecorDistribution(new[] { 10f }, new[] { 20f });
        double sum = 0, sum2 = 0;
        var rng = new Random(7);
        for (int i = 0; i < N; i++) { float x = d.Sample(rng)[0]; sum += x; sum2 += x * x; }
        double mean = sum / N; double var = sum2 / N - mean * mean;
        Assert.InRange(mean, 1.111 - 0.05, 1.111 + 0.05);
        Assert.InRange(var, 0.4321 - 0.10, 0.4321 + 0.10);
    }

    [Fact]
    public void Kumaraswamy_CdfIcdf_RoundTrip()
    {
        // 4-batch Kumaraswamy probes 4 quantiles.
        var d = new KumaraswamyDistribution(
            concentration1: new[] { 2f, 2f, 2f, 2f },
            concentration0: new[] { 5f, 5f, 5f, 5f });
        var p = new[] { 0.1f, 0.3f, 0.5f, 0.9f };
        var x = d.Icdf(p);
        var p2 = d.Cdf(x);
        for (int i = 0; i < 4; i++) Assert.InRange(p2[i] - p[i], -1e-4f, 1e-4f);
    }

    [Fact]
    public void VonMises_SamplesInsideTwoPiInterval()
    {
        var d = new VonMisesDistribution(new[] { 0f }, new[] { 4f });
        var rng = new Random(0);
        for (int i = 0; i < 1000; i++)
        {
            float x = d.Sample(rng)[0];
            Assert.InRange(x, -MathF.PI - 1e-3f, MathF.PI + 1e-3f);
        }
    }

    [Fact]
    public void VonMises_LowKappa_CloseToUniform()
    {
        // κ → 0 ⇒ uniform on (-π, π], variance → 1 (1 - I1/I0 → 1 as κ → 0).
        var d = new VonMisesDistribution(new[] { 0f }, new[] { 0.001f });
        Assert.InRange(d.Variance[0], 0.99f, 1.0001f);
    }

    [Fact]
    public void ContinuousBernoulli_LogProb_Lambda05_IsZero()
    {
        // At λ=0.5 the density f(x) = C(0.5) · 0.5^x · 0.5^(1-x) = 2 · 0.5 = 1, so log p = 0.
        var d = new ContinuousBernoulliDistribution(new[] { 0.5f });
        Assert.Equal(0f, d.LogProb(new[] { 0.3f })[0], precision: 4);
        Assert.Equal(0f, d.LogProb(new[] { 0.7f })[0], precision: 4);
    }

    [Fact]
    public void LowRankMVN_Variance_EqualsDiagPlusFactorOuterProduct()
    {
        // Σ = D + W·Wᵀ; Var_i = D_i + Σ_k W_{ik}².
        var d = new LowRankMultivariateNormalDistribution(
            loc: new[] { 0f, 0f },
            covDiag: new[] { 1f, 2f },
            covFactor: new[] { 1f, 0.5f, -1f, 0.25f },  // [2, 2] flattened
            d: 2, k: 2);
        var v = d.Variance;
        Assert.Equal(1f + 1f + 0.25f, v[0], precision: 4);
        Assert.Equal(2f + 1f + 0.0625f, v[1], precision: 4);
    }

    [Fact]
    public void LowRankMVN_LogProb_AgreesWithFullMVN()
    {
        // Full Σ = diag([1, 2]) + W Wᵀ where W = [[1, 0.5], [-1, 0.25]].
        var lr = new LowRankMultivariateNormalDistribution(
            loc: new[] { 0f, 0f },
            covDiag: new[] { 1f, 2f },
            covFactor: new[] { 1f, 0.5f, -1f, 0.25f },
            d: 2, k: 2);
        // Construct equivalent full MVN:
        var full = new MultivariateNormalDistribution(
            loc: new[] { 0f, 0f },
            covariance: new[]
            {
                1f + 1f + 0.25f,        -1f + 0.125f,
                -1f + 0.125f,            2f + 1f + 0.0625f,
            }, d: 2);
        var x = new[] { 0.3f, -0.7f };
        Assert.Equal(full.LogProb(x)[0], lr.LogProb(x)[0], precision: 3);
    }

    // -------------------- New transforms --------------------

    [Fact]
    public void SoftmaxTransform_OutputIsSimplex()
    {
        var t = new SoftmaxTransform(3);
        var x = new[] { 1f, 2f, 3f };
        var y = t.Forward(x);
        float sum = 0f; foreach (var v in y) { Assert.True(v >= 0); sum += v; }
        Assert.InRange(sum, 1f - 1e-4f, 1f + 1e-4f);
    }

    [Fact]
    public void StickBreakingTransform_OutputIsSimplex()
    {
        var t = new StickBreakingTransform(4);
        var x = new[] { -0.5f, 0.5f, 1.5f };  // K-1 = 3 entries
        var y = t.Forward(x);
        Assert.Equal(4, y.Length);
        float sum = 0f; foreach (var v in y) { Assert.True(v >= 0); sum += v; }
        Assert.InRange(sum, 1f - 1e-4f, 1f + 1e-4f);
    }

    [Fact]
    public void StickBreakingTransform_ForwardInverse_RoundTrips()
    {
        var t = new StickBreakingTransform(4);
        var x = new[] { -0.5f, 0.7f, 1.3f };
        var y = t.Forward(x);
        var x2 = t.Inverse(y);
        for (int i = 0; i < 3; i++) Assert.InRange(x2[i] - x[i], -1e-3f, 1e-3f);
    }

    [Fact]
    public void LowerCholeskyTransform_HasZerosAboveDiagAndPositiveDiag()
    {
        var t = new LowerCholeskyTransform(3);
        var x = new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f };
        var y = t.Forward(x);
        Assert.Equal(0f, y[1]); Assert.Equal(0f, y[2]); Assert.Equal(0f, y[5]);
        Assert.True(y[0] > 0); Assert.True(y[4] > 0); Assert.True(y[8] > 0);
    }

    [Fact]
    public void CorrCholeskyTransform_OutputProducesValidCorrelationMatrix()
    {
        // L · Lᵀ should have unit diagonal.
        var t = new CorrCholeskyTransform(3);
        var x = new[] { 0.5f, -0.3f, 0.7f };  // N(N-1)/2 = 3
        var L = t.Forward(x);
        // Compute (L · Lᵀ)_ii.
        for (int i = 0; i < 3; i++)
        {
            float diag = 0f;
            for (int k = 0; k <= i; k++) diag += L[i * 3 + k] * L[i * 3 + k];
            Assert.InRange(diag, 1f - 1e-3f, 1f + 1e-3f);
        }
    }

    [Fact]
    public void CdfTransform_RoundTripsWithIcdf()
    {
        var nd = new NormalDistribution(new[] { 0f, 0f, 0f }, new[] { 1f, 1f, 1f });
        var t = new CumulativeDistributionTransform(nd);
        var x = new[] { -1f, 0f, 1f };
        var u = t.Forward(x);
        var x2 = t.Inverse(u);
        for (int i = 0; i < 3; i++) Assert.InRange(x2[i] - x[i], -2e-3f, 2e-3f);
    }

    // -------------------- ExpandingDistribution --------------------

    [Fact]
    public void ExpandingDistribution_ReplicatesSampleAcrossExpandedBatch()
    {
        var inner = new NormalDistribution(new[] { 5f }, new[] { 0.0001f });  // tiny var
        var expanded = new ExpandingDistribution(inner, batchSize: 8);
        var rng = new Random(0);
        var s = expanded.Sample(rng);
        Assert.Equal(8, s.Length);
        // All samples should be ≈ 5 because inner's variance is tiny.
        foreach (var v in s) Assert.InRange(v, 4.99f, 5.01f);
    }

    // -------------------- Philox RNG --------------------

    [Fact]
    public void Philox_SameSeedSameSubKey_IsDeterministic()
    {
        var a = new PhiloxGenerator(seed: 42);
        var b = new PhiloxGenerator(seed: 42);
        for (int i = 0; i < 100; i++) Assert.Equal(a.NextDouble(), b.NextDouble());
    }

    [Fact]
    public void Philox_DifferentSubKeys_ProduceIndependentStreams()
    {
        var a = new PhiloxGenerator(seed: 42, subKey: 0);
        var b = new PhiloxGenerator(seed: 42, subKey: 1);
        int diff = 0;
        for (int i = 0; i < 100; i++) if (a.NextDouble() != b.NextDouble()) diff++;
        Assert.True(diff > 95, $"streams should differ on almost every draw; got {diff}/100");
    }

    [Fact]
    public void Philox_BatchRowHashing_ProducesIndependentPerRow()
    {
        // Setting the same seed and stepping per-row via batch-row hashing must produce
        // distinct streams that nonetheless reproduce on the same (seed, row) pair.
        var rngA = new PhiloxGenerator(seed: 100); rngA.SetBatchRow(7);
        var rngB = new PhiloxGenerator(seed: 100); rngB.SetBatchRow(7);
        var rngC = new PhiloxGenerator(seed: 100); rngC.SetBatchRow(8);
        Assert.Equal(rngA.NextDouble(), rngB.NextDouble());
        Assert.NotEqual(rngA.NextDouble(), rngC.NextDouble());
    }

    [Fact]
    public void Philox_DistributionsWorkWithPhilox()
    {
        var rng = new PhiloxGenerator(seed: 1234);
        var d = new NormalDistribution(0f, 1f);
        // Just check sampling does not throw and produces finite values.
        for (int i = 0; i < 100; i++)
        {
            var x = d.Sample(rng)[0];
            Assert.False(float.IsNaN(x));
            Assert.False(float.IsInfinity(x));
        }
    }

    // -------------------- KL: MC cross-check for every registered pair --------------------

    private static double MonteCarloKl(IDistribution p, IDistribution q, int nSamples = 30000, int seed = 17)
    {
        var rng = new Random(seed);
        double s = 0;
        for (int i = 0; i < nSamples; i++)
        {
            var x = p.Sample(rng);
            s += p.LogProb(x)[0] - q.LogProb(x)[0];
        }
        return s / nSamples;
    }

    [Fact]
    public void KL_ExpExp_AnalyticalMatchesMonteCarlo()
    {
        var p = new ExponentialDistribution(new[] { 2f });
        var q = new ExponentialDistribution(new[] { 0.5f });
        double kl = KLDivergence.Compute(p, q)[0];
        Assert.InRange(MonteCarloKl(p, q), kl - 0.05, kl + 0.05);
    }

    [Fact]
    public void KL_LaplaceLaplace_AnalyticalMatchesMonteCarlo()
    {
        var p = new LaplaceDistribution(new[] { 0f }, new[] { 1f });
        var q = new LaplaceDistribution(new[] { 1f }, new[] { 1.5f });
        double kl = KLDivergence.Compute(p, q)[0];
        Assert.InRange(MonteCarloKl(p, q), kl - 0.06, kl + 0.06);
    }

    [Fact]
    public void KL_GammaGamma_AnalyticalMatchesMonteCarlo()
    {
        var p = new GammaDistribution(new[] { 2f }, new[] { 1f });
        var q = new GammaDistribution(new[] { 3f }, new[] { 0.5f });
        double kl = KLDivergence.Compute(p, q)[0];
        Assert.InRange(MonteCarloKl(p, q), kl - 0.1, kl + 0.1);
    }

    [Fact]
    public void KL_BetaBeta_AnalyticalMatchesMonteCarlo()
    {
        var p = new BetaDistribution(new[] { 2f }, new[] { 5f });
        var q = new BetaDistribution(new[] { 3f }, new[] { 3f });
        double kl = KLDivergence.Compute(p, q)[0];
        Assert.InRange(MonteCarloKl(p, q), kl - 0.05, kl + 0.05);
    }

    [Fact]
    public void KL_DirichletDirichlet_AnalyticalMatchesMonteCarlo()
    {
        var p = new DirichletDistribution(new[] { 1f, 2f, 3f }, k: 3);
        var q = new DirichletDistribution(new[] { 2f, 2f, 2f }, k: 3);
        double kl = KLDivergence.Compute(p, q)[0];
        // Dirichlet log-probs require simplex inputs from samples; we evaluate them as length-K vectors.
        var rng = new Random(11);
        double s = 0;
        const int n = 30000;
        for (int i = 0; i < n; i++)
        {
            var x = p.Sample(rng);
            s += p.LogProb(x)[0] - q.LogProb(x)[0];
        }
        double mc = s / n;
        Assert.InRange(mc, kl - 0.1, kl + 0.1);
    }

    [Fact]
    public void KL_UniformUniform_PSubsetQ_HasExpectedValue()
    {
        // KL(U(0, 1) || U(-1, 2)) = log(3 / 1) = log 3.
        var p = new UniformDistribution(new[] { 0f }, new[] { 1f });
        var q = new UniformDistribution(new[] { -1f }, new[] { 2f });
        Assert.Equal(MathF.Log(3f), KLDivergence.Compute(p, q)[0], precision: 4);
    }

    [Fact]
    public void KL_UniformUniform_PNotSubset_IsInfinity()
    {
        var p = new UniformDistribution(new[] { -2f }, new[] { 2f });
        var q = new UniformDistribution(new[] { -1f }, new[] { 1f });
        Assert.True(float.IsPositiveInfinity(KLDivergence.Compute(p, q)[0]));
    }

    [Fact]
    public void KL_DiagMVN_MonteCarloMatchesAnalytical()
    {
        var p = new DiagonalMultivariateNormalDistribution(new[] { 0f, 0f }, new[] { 1f, 1f }, d: 2);
        var q = new DiagonalMultivariateNormalDistribution(new[] { 1f, -1f }, new[] { 2f, 2f }, d: 2);
        double kl = KLDivergence.Compute(p, q)[0];
        var rng = new Random(21);
        double s = 0;
        for (int i = 0; i < 30000; i++)
        {
            var x = p.Sample(rng);
            s += p.LogProb(x)[0] - q.LogProb(x)[0];
        }
        Assert.InRange(s / 30000, kl - 0.05, kl + 0.05);
    }

    [Fact]
    public void KL_RegistrySupportsCustomTypes()
    {
        // Register a NEW pair (Normal -> Uniform) that has no built-in formula. Avoids polluting
        // the registry's default entries that other tests rely on, and uses try/finally to
        // unregister the sentinel handler before returning so parallel-running tests can't
        // observe the test's transient state.
        bool used = false;
        KLDivergence.Register<NormalDistribution, UniformDistribution>((a, b) =>
        {
            used = true;
            return new[] { 42f };  // sentinel
        });
        try
        {
            var p = new NormalDistribution(0f, 1f);
            var q = new UniformDistribution(new[] { -1f }, new[] { 1f });
            var kl = KLDivergence.Compute(p, q)[0];
            Assert.True(used);
            Assert.Equal(42f, kl);
        }
        finally
        {
            KLDivergence.Unregister<NormalDistribution, UniformDistribution>();
        }
    }

    // -------------------- Reference-value log_prob spot checks --------------------

    [Fact]
    public void LogProb_ReferenceValues_NormalAtZero()
    {
        // Standard Normal at x = 0: log p(0) = -0.5 · log(2π) ≈ -0.9189.
        var d = new NormalDistribution(0f, 1f);
        Assert.Equal(-0.91893853f, d.LogProb(new[] { 0f })[0], precision: 4);
    }

    [Fact]
    public void LogProb_ReferenceValues_PoissonAtMean()
    {
        // Poisson(λ=4) at x=4: log p = 4·log(4) − 4 − log(4!) = 4·log4 − 4 − log24 ≈ -1.6322.
        var d = new PoissonDistribution(new[] { 4f });
        var expected = (float)(4 * Math.Log(4) - 4 - Math.Log(24));
        Assert.Equal(expected, d.LogProb(new[] { 4f })[0], precision: 3);
    }

    [Fact]
    public void LogProb_ReferenceValues_BetaAtMode()
    {
        // Beta(α=2, β=2) at x=0.5: f = 6 · 0.5 · 0.5 = 1.5; log = log 1.5.
        var d = new BetaDistribution(new[] { 2f }, new[] { 2f });
        Assert.Equal(MathF.Log(1.5f), d.LogProb(new[] { 0.5f })[0], precision: 4);
    }

    [Fact]
    public void LogProb_ReferenceValues_GammaAtMean()
    {
        // Gamma(α=3, β=1) at x=3: log p = 3·log(1) + 2·log(3) − 3 − log Γ(3)
        //                                = 0 + 2 log 3 − 3 − log 2 ≈ -1.5028.
        var d = new GammaDistribution(new[] { 3f }, new[] { 1f });
        var expected = (float)(2 * Math.Log(3) - 3 - Math.Log(2));
        Assert.Equal(expected, d.LogProb(new[] { 3f })[0], precision: 3);
    }
}
