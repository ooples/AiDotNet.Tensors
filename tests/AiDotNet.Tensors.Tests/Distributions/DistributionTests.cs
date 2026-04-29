using System;
using AiDotNet.Tensors.Distributions;
using AiDotNet.Tensors.Distributions.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Distributions;

/// <summary>
/// Property tests for every distribution: empirical sample mean ≈ analytical mean,
/// empirical variance ≈ analytical variance, log-prob density values match
/// hand-computed reference values, CDF/ICDF round-trip.
/// </summary>
public class DistributionTests
{
    private const int N = 50_000;

    private static (double mean, double variance) MomentsOfSamples(IDistribution d, int seed = 1234)
    {
        var rng = new Random(seed);
        double sum = 0, sum2 = 0;
        for (int i = 0; i < N; i++)
        {
            float x = d.Sample(rng)[0];
            sum += x; sum2 += (double)x * x;
        }
        double mean = sum / N;
        double var = sum2 / N - mean * mean;
        return (mean, var);
    }

    // -------------------- Continuous --------------------

    [Fact]
    public void Normal_Moments_Match()
    {
        var d = new NormalDistribution(loc: 1.5f, scale: 2.0f);
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 1.5 - 0.05, 1.5 + 0.05);
        Assert.InRange(var, 4.0 - 0.15, 4.0 + 0.15);
        // log-prob at mean: -0.5·log(2π·σ²)
        var lp = d.LogProb(new[] { 1.5f })[0];
        Assert.InRange(lp, -1.613f, -1.611f);
    }

    [Fact]
    public void Normal_CdfIcdf_RoundTrip()
    {
        // A batch of 4 standard normals so we can probe four x values in parallel.
        var d = new NormalDistribution(
            loc: new[] { 0f, 0f, 0f, 0f },
            scale: new[] { 1f, 1f, 1f, 1f });
        var x = new[] { -1.5f, 0f, 0.7f, 2f };
        var c = d.Cdf(x);
        var x2 = d.Icdf(c);
        for (int i = 0; i < x.Length; i++) Assert.InRange(x2[i] - x[i], -2e-3f, 2e-3f);
    }

    [Fact]
    public void Uniform_Moments_Match()
    {
        var d = new UniformDistribution(new[] { 0f }, new[] { 4f });
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 2.0 - 0.05, 2.0 + 0.05);
        Assert.InRange(var, 16.0 / 12 - 0.1, 16.0 / 12 + 0.1);
    }

    [Fact]
    public void Exponential_Moments_Match()
    {
        var d = new ExponentialDistribution(new[] { 2f });  // mean 0.5, var 0.25
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 0.5 - 0.02, 0.5 + 0.02);
        Assert.InRange(var, 0.25 - 0.02, 0.25 + 0.02);
    }

    [Fact]
    public void LogNormal_Moments_Match()
    {
        var d = new LogNormalDistribution(new[] { 0f }, new[] { 1f });
        var (mean, _) = MomentsOfSamples(d, seed: 7);
        // E[X] = exp(0 + 0.5) ≈ 1.6487
        Assert.InRange(mean, 1.6487 - 0.1, 1.6487 + 0.1);
    }

    [Fact]
    public void Gamma_Moments_Match()
    {
        var d = new GammaDistribution(new[] { 3f }, new[] { 1.5f });  // mean 2, var 4/3
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 2.0 - 0.05, 2.0 + 0.05);
        Assert.InRange(var, 4.0 / 3 - 0.1, 4.0 / 3 + 0.1);
    }

    [Fact]
    public void Beta_Moments_Match()
    {
        var d = new BetaDistribution(new[] { 2f }, new[] { 5f });  // mean 2/7, var 10/(49·8)
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 2.0 / 7 - 0.01, 2.0 / 7 + 0.01);
        Assert.InRange(var, 10.0 / (49 * 8) - 0.005, 10.0 / (49 * 8) + 0.005);
    }

    [Fact]
    public void Chi2_Moments_Match()
    {
        var d = new Chi2Distribution(new[] { 5f });  // mean 5, var 10
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 5.0 - 0.1, 5.0 + 0.1);
        Assert.InRange(var, 10.0 - 0.5, 10.0 + 0.5);
    }

    [Fact]
    public void Laplace_Moments_Match()
    {
        var d = new LaplaceDistribution(new[] { 2f }, new[] { 1.5f });  // mean 2, var 2·1.5²=4.5
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 2.0 - 0.05, 2.0 + 0.05);
        Assert.InRange(var, 4.5 - 0.2, 4.5 + 0.2);
    }

    [Fact]
    public void Gumbel_Moments_Match()
    {
        var d = new GumbelDistribution(new[] { 0f }, new[] { 1f });  // mean γ ≈ 0.5772, var π²/6
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 0.5772 - 0.05, 0.5772 + 0.05);
        Assert.InRange(var, Math.PI * Math.PI / 6 - 0.1, Math.PI * Math.PI / 6 + 0.1);
    }

    [Fact]
    public void StudentT_HeavyTail_FiniteAt5Df()
    {
        var d = new StudentTDistribution(new[] { 5f }, new[] { 0f }, new[] { 1f });
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, -0.05, 0.05);
        // For df > 2, var = df / (df - 2). For df=5, var = 5/3 ≈ 1.667.
        Assert.InRange(var, 1.5, 1.85);
    }

    [Fact]
    public void HalfNormal_Moments_Match()
    {
        var d = new HalfNormalDistribution(new[] { 1f });
        var (mean, _) = MomentsOfSamples(d);
        // E[|X|] for X ∼ N(0,1) = sqrt(2/π) ≈ 0.7979.
        Assert.InRange(mean, 0.7979 - 0.02, 0.7979 + 0.02);
    }

    // -------------------- Discrete --------------------

    [Fact]
    public void Bernoulli_Moments_Match()
    {
        var d = new BernoulliDistribution(new[] { 0.3f });
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 0.3 - 0.01, 0.3 + 0.01);
        Assert.InRange(var, 0.21 - 0.01, 0.21 + 0.01);
    }

    [Fact]
    public void Categorical_Sampling_FollowsProbs()
    {
        var d = new CategoricalDistribution(new[] { 0.1f, 0.3f, 0.6f }, k: 3);
        var rng = new Random(42);
        int[] counts = new int[3];
        for (int i = 0; i < N; i++) counts[(int)d.Sample(rng)[0]]++;
        Assert.InRange((double)counts[0] / N, 0.1 - 0.01, 0.1 + 0.01);
        Assert.InRange((double)counts[1] / N, 0.3 - 0.02, 0.3 + 0.02);
        Assert.InRange((double)counts[2] / N, 0.6 - 0.02, 0.6 + 0.02);
    }

    [Fact]
    public void Poisson_Moments_Match()
    {
        var d = new PoissonDistribution(new[] { 4f });  // mean = var = 4
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 4.0 - 0.05, 4.0 + 0.05);
        Assert.InRange(var, 4.0 - 0.2, 4.0 + 0.2);
    }

    [Fact]
    public void Binomial_Moments_Match()
    {
        var d = new BinomialDistribution(new[] { 10 }, new[] { 0.4f });  // mean 4, var 2.4
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 4.0 - 0.05, 4.0 + 0.05);
        Assert.InRange(var, 2.4 - 0.1, 2.4 + 0.1);
    }

    [Fact]
    public void Geometric_Moments_Match()
    {
        var d = new GeometricDistribution(new[] { 0.25f });  // mean (1-p)/p = 3, var (1-p)/p² = 12
        var (mean, var) = MomentsOfSamples(d);
        Assert.InRange(mean, 3.0 - 0.1, 3.0 + 0.1);
        Assert.InRange(var, 12.0 - 0.6, 12.0 + 0.6);
    }

    // -------------------- Multivariate --------------------

    [Fact]
    public void DiagonalMVN_SampleMean_NearLoc()
    {
        var d = new DiagonalMultivariateNormalDistribution(
            new[] { 1f, -2f, 3f }, new[] { 1f, 2f, 0.5f }, d: 3);
        var rng = new Random(11);
        double[] sums = new double[3];
        for (int i = 0; i < N; i++)
        {
            var s = d.Sample(rng);
            for (int j = 0; j < 3; j++) sums[j] += s[j];
        }
        Assert.InRange(sums[0] / N, 1.0 - 0.05, 1.0 + 0.05);
        Assert.InRange(sums[1] / N, -2.0 - 0.05, -2.0 + 0.05);
        Assert.InRange(sums[2] / N, 3.0 - 0.05, 3.0 + 0.05);
    }

    [Fact]
    public void Dirichlet_SamplesAreInTheSimplex()
    {
        var d = new DirichletDistribution(new[] { 1f, 2f, 3f }, k: 3);
        var rng = new Random(99);
        for (int i = 0; i < 100; i++)
        {
            var s = d.Sample(rng);
            float sum = 0f;
            foreach (var v in s) { Assert.True(v >= 0); sum += v; }
            Assert.InRange(sum, 1f - 1e-4f, 1f + 1e-4f);
        }
    }

    [Fact]
    public void MVN_Full_LogProbMatchesDiagWhenCovIsDiag()
    {
        var diag = new DiagonalMultivariateNormalDistribution(
            new[] { 0f, 0f }, new[] { 1f, 2f }, d: 2);
        var full = new MultivariateNormalDistribution(
            new[] { 0f, 0f },
            new[] { 1f, 0f, 0f, 4f },  // diag(1, 4)
            d: 2);
        var x = new[] { 0.5f, -1f };
        Assert.Equal(diag.LogProb(x)[0], full.LogProb(x)[0], precision: 4);
    }

    // -------------------- KL Divergence --------------------

    [Fact]
    public void KL_NormalNormal_AnalyticalMatchesMonteCarlo()
    {
        var p = new NormalDistribution(new[] { 1f }, new[] { 1f });
        var q = new NormalDistribution(new[] { 0f }, new[] { 2f });
        var kl = KLDivergence.Compute(p, q)[0];

        // KL(N(1,1)||N(0,4)) = log(2/1) + (1+1)/8 - 0.5 = ln 2 + 0.25 - 0.5 ≈ 0.4431
        Assert.Equal(0.4431f, kl, precision: 3);

        // Monte-Carlo cross-check.
        var rng = new Random(7);
        double mc = 0;
        for (int i = 0; i < 20000; i++)
        {
            float x = p.Sample(rng)[0];
            mc += p.LogProb(new[] { x })[0] - q.LogProb(new[] { x })[0];
        }
        mc /= 20000;
        Assert.InRange(mc, kl - 0.05, kl + 0.05);
    }

    [Fact]
    public void KL_BernoulliBernoulli_AnalyticalMatchesEmpirical()
    {
        var p = new BernoulliDistribution(new[] { 0.7f });
        var q = new BernoulliDistribution(new[] { 0.5f });
        var kl = KLDivergence.Compute(p, q)[0];
        // 0.7 log(0.7/0.5) + 0.3 log(0.3/0.5) ≈ 0.0822
        Assert.Equal(0.0822f, kl, precision: 3);
    }

    [Fact]
    public void KL_CategoricalCategorical_Symmetry()
    {
        var p = new CategoricalDistribution(new[] { 0.5f, 0.5f }, k: 2);
        var q = new CategoricalDistribution(new[] { 0.5f, 0.5f }, k: 2);
        Assert.Equal(0f, KLDivergence.Compute(p, q)[0], precision: 5);
    }

    [Fact]
    public void KL_DiagMvnDiagMvn_ReducesToSumOfPerDimNormalKL()
    {
        // KL between diag MVNs equals the sum of per-dim Normal KLs.
        var p = new DiagonalMultivariateNormalDistribution(
            new[] { 1f, -1f }, new[] { 1f, 2f }, d: 2);
        var q = new DiagonalMultivariateNormalDistribution(
            new[] { 0f, 0f }, new[] { 2f, 1f }, d: 2);
        var kl = KLDivergence.Compute(p, q)[0];
        var n1 = new NormalDistribution(1f, 1f);
        var n2 = new NormalDistribution(0f, 2f);
        var n3 = new NormalDistribution(-1f, 2f);
        var n4 = new NormalDistribution(0f, 1f);
        var sum = KLDivergence.Compute(n1, n2)[0] + KLDivergence.Compute(n3, n4)[0];
        Assert.Equal(sum, kl, precision: 4);
    }

    // -------------------- Special functions sanity --------------------

    [Fact]
    public void SpecialFunctions_Lgamma_MatchesKnownValues()
    {
        Assert.Equal(0f,            SpecialFunctions.Lgamma(1f), precision: 3);
        Assert.Equal(0f,            SpecialFunctions.Lgamma(2f), precision: 3);
        Assert.Equal(MathF.Log(2f), SpecialFunctions.Lgamma(3f), precision: 3); // log(2!)
        Assert.Equal(MathF.Log(6f), SpecialFunctions.Lgamma(4f), precision: 3); // log(3!)
    }

    [Fact]
    public void SpecialFunctions_Erf_MatchesKnownValues()
    {
        Assert.Equal(0f,         SpecialFunctions.Erf(0f), precision: 4);
        Assert.Equal(0.8427008f, SpecialFunctions.Erf(1f), precision: 4);
        Assert.Equal(-0.8427008f, SpecialFunctions.Erf(-1f), precision: 4);
    }

    [Fact]
    public void SpecialFunctions_NormalIcdf_RoundTripsCdf()
    {
        for (float p = 0.01f; p < 1f; p += 0.1f)
        {
            float x = SpecialFunctions.NormalIcdf(p);
            float p2 = SpecialFunctions.NormalCdf(x);
            Assert.InRange(p2 - p, -1e-3f, 1e-3f);
        }
    }

    // -------------------- Transforms --------------------

    [Fact]
    public void AffineTransform_RoundTrips()
    {
        var t = new AiDotNet.Tensors.Distributions.Transforms.AffineTransform(loc: 2f, scale: 3f);
        var x = new[] { -1f, 0f, 1f, 5f };
        var y = t.Forward(x);
        var x2 = t.Inverse(y);
        for (int i = 0; i < x.Length; i++) Assert.Equal(x[i], x2[i], precision: 5);
        Assert.Equal(MathF.Log(3f), t.LogAbsDetJacobian(x, y)[0], precision: 4);
    }

    [Fact]
    public void SigmoidTransform_RoundTrips()
    {
        var t = AiDotNet.Tensors.Distributions.Transforms.SigmoidTransform.Instance;
        var x = new[] { -2f, -0.5f, 0.5f, 2f };
        var y = t.Forward(x);
        var x2 = t.Inverse(y);
        for (int i = 0; i < x.Length; i++) Assert.InRange(x2[i] - x[i], -2e-3f, 2e-3f);
    }

    [Fact]
    public void TransformedDistribution_LogProb_AccountsForJacobian()
    {
        // exp(N(0, 1)) === LogNormal(0, 1). Their log-probs at the same point should match.
        var baseDist = new NormalDistribution(0f, 1f);
        var transformed = new TransformedDistribution(
            baseDist, AiDotNet.Tensors.Distributions.Transforms.ExpTransform.Instance);
        var logNormal = new LogNormalDistribution(new[] { 0f }, new[] { 1f });
        var x = new[] { 1.5f };
        Assert.Equal(logNormal.LogProb(x)[0], transformed.LogProb(x)[0], precision: 4);
    }

    [Fact]
    public void RelaxedOneHotCategorical_ReturnsSimplexSamples()
    {
        var d = new RelaxedOneHotCategoricalDistribution(
            logits: new[] { 0f, 1f, 2f }, temperature: new[] { 0.5f }, k: 3);
        var rng = new Random(0);
        var s = d.Sample(rng);
        float sum = 0f; foreach (var v in s) { Assert.True(v >= 0); sum += v; }
        Assert.InRange(sum, 1f - 1e-4f, 1f + 1e-4f);
    }
}
