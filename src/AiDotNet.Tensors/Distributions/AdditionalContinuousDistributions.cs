using System;
using AiDotNet.Tensors.Distributions.Constraints;
using AiDotNet.Tensors.Distributions.Helpers;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Fisher-Snedecor (F) distribution: ratio of two scaled chi-squared variables.
/// X ∼ F(d1, d2) is equivalent to X = (U/d1)/(V/d2) where U ∼ Chi²(d1), V ∼ Chi²(d2).
/// </summary>
public sealed class FisherSnedecorDistribution : DistributionBase
{
    /// <summary>Numerator degrees of freedom.</summary>
    public float[] Df1 { get; }
    /// <summary>Denominator degrees of freedom.</summary>
    public float[] Df2 { get; }
    /// <inheritdoc />
    public override int BatchSize => Df1.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => PositiveConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Build an F distribution from per-batch (df1, df2).</summary>
    public FisherSnedecorDistribution(float[] df1, float[] df2)
    {
        if (df1.Length != df2.Length) throw new ArgumentException();
        for (int i = 0; i < df1.Length; i++)
        {
            if (!(df1[i] > 0f)) throw new ArgumentException("df1 > 0.");
            if (!(df2[i] > 0f)) throw new ArgumentException("df2 > 0.");
        }
        Df1 = df1; Df2 = df2;
    }

    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);

    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float u = 2f * GammaDistribution.MarsagliaTsang(rng, Df1[i] * 0.5f);  // Chi²(df1)
            float v = 2f * GammaDistribution.MarsagliaTsang(rng, Df2[i] * 0.5f);  // Chi²(df2)
            x[i] = (u / Df1[i]) / (v / Df2[i]);
        }
        return x;
    }

    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            if (value[i] <= 0) { lp[i] = float.NegativeInfinity; continue; }
            float d1 = Df1[i], d2 = Df2[i], v = value[i];
            float logZ = SpecialFunctions.Lgamma((d1 + d2) * 0.5f)
                       - SpecialFunctions.Lgamma(d1 * 0.5f)
                       - SpecialFunctions.Lgamma(d2 * 0.5f)
                       + (d1 * 0.5f) * MathF.Log(d1)
                       + (d2 * 0.5f) * MathF.Log(d2);
            lp[i] = logZ + (d1 * 0.5f - 1f) * MathF.Log(v)
                  - ((d1 + d2) * 0.5f) * MathF.Log(d1 * v + d2);
        }
        return lp;
    }

    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) h[i] = float.NaN;
        return h;
    }

    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            var m = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++)
                m[i] = Df2[i] > 2f ? Df2[i] / (Df2[i] - 2f) : float.PositiveInfinity;
            return m;
        }
    }

    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var v = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++)
            {
                float d1 = Df1[i], d2 = Df2[i];
                if (d2 <= 4f) { v[i] = float.PositiveInfinity; continue; }
                v[i] = 2f * d2 * d2 * (d1 + d2 - 2f) / (d1 * (d2 - 2f) * (d2 - 2f) * (d2 - 4f));
            }
            return v;
        }
    }
}

/// <summary>
/// Kumaraswamy distribution on (0, 1). f(x) = a·b·x^(a-1) · (1 - x^a)^(b-1).
/// Cousin of Beta with closed-form CDF and inverse CDF — the latter making sampling
/// reparameterisable in closed form (no Gamma rejection needed).
/// </summary>
public sealed class KumaraswamyDistribution : DistributionBase
{
    /// <summary>Concentration α &gt; 0.</summary>
    public float[] Concentration1 { get; }
    /// <summary>Concentration β &gt; 0.</summary>
    public float[] Concentration0 { get; }
    /// <inheritdoc />
    public override int BatchSize => Concentration1.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => UnitIntervalConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Build a Kumaraswamy from per-batch (a, b).</summary>
    public KumaraswamyDistribution(float[] concentration1, float[] concentration0)
    {
        if (concentration1.Length != concentration0.Length) throw new ArgumentException();
        for (int i = 0; i < concentration1.Length; i++)
        {
            if (!(concentration1[i] > 0f)) throw new ArgumentException("α > 0.");
            if (!(concentration0[i] > 0f)) throw new ArgumentException("β > 0.");
        }
        Concentration1 = concentration1; Concentration0 = concentration0;
    }

    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);

    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        // Closed-form ICDF: x = (1 - (1 - u)^(1/b))^(1/a).
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            double u = rng.NextDouble();
            if (u < 1e-12) u = 1e-12; if (u > 1 - 1e-12) u = 1 - 1e-12;
            x[i] = MathF.Pow(1f - MathF.Pow(1f - (float)u, 1f / Concentration0[i]), 1f / Concentration1[i]);
        }
        return x;
    }

    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float v = value[i];
            if (v <= 0f || v >= 1f) { lp[i] = float.NegativeInfinity; continue; }
            float a = Concentration1[i], b = Concentration0[i];
            lp[i] = MathF.Log(a) + MathF.Log(b)
                  + (a - 1f) * MathF.Log(v)
                  + (b - 1f) * MathF.Log(1f - MathF.Pow(v, a));
        }
        return lp;
    }

    /// <inheritdoc />
    public override float[] Entropy()
    {
        // Closed-form: H = (1 - 1/b) + (1 - 1/a)·H_b - log(a·b), where H_b is the b-th harmonic-like.
        // We use Kumaraswamy's analytical form (Jones, 2009): see arXiv 1005.0287.
        var h = new float[BatchSize];
        const float EulerMascheroni = 0.5772156649f;
        for (int i = 0; i < BatchSize; i++)
        {
            float a = Concentration1[i], b = Concentration0[i];
            // H = (1 − 1/b) + (1 − 1/a)·(ψ(b+1) − ψ(1)) − log(a·b)
            float hb = SpecialFunctions.Digamma(b + 1f) + EulerMascheroni;  // ψ(b+1) − ψ(1)
            h[i] = (1f - 1f / b) + (1f - 1f / a) * hb - MathF.Log(a * b);
        }
        return h;
    }

    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            var m = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++)
            {
                float a = Concentration1[i], b = Concentration0[i];
                m[i] = b * MathF.Exp(SpecialFunctions.LogBeta(1f + 1f / a, b));
            }
            return m;
        }
    }

    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var v = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++)
            {
                float a = Concentration1[i], b = Concentration0[i];
                float m1 = b * MathF.Exp(SpecialFunctions.LogBeta(1f + 1f / a, b));
                float m2 = b * MathF.Exp(SpecialFunctions.LogBeta(1f + 2f / a, b));
                v[i] = m2 - m1 * m1;
            }
            return v;
        }
    }

    /// <inheritdoc />
    public override float[] Cdf(float[] value)
    {
        EnsureValueShape(value);
        var c = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float v = value[i];
            if (v <= 0f) { c[i] = 0f; continue; }
            if (v >= 1f) { c[i] = 1f; continue; }
            c[i] = 1f - MathF.Pow(1f - MathF.Pow(v, Concentration1[i]), Concentration0[i]);
        }
        return c;
    }

    /// <inheritdoc />
    public override float[] Icdf(float[] probability)
    {
        EnsureValueShape(probability);
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            x[i] = MathF.Pow(1f - MathF.Pow(1f - probability[i], 1f / Concentration0[i]), 1f / Concentration1[i]);
        return x;
    }
}

/// <summary>
/// Von Mises distribution on (-π, π]: circular analogue of the normal distribution,
/// f(x) = exp(κ·cos(x − μ)) / (2π·I₀(κ)).
/// </summary>
public sealed class VonMisesDistribution : DistributionBase
{
    /// <summary>Per-batch mean direction μ ∈ (-π, π].</summary>
    public float[] Loc { get; }
    /// <summary>Per-batch concentration κ ≥ 0.</summary>
    public float[] Concentration { get; }
    /// <inheritdoc />
    public override int BatchSize => Loc.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => new IntervalConstraint(-MathF.PI, MathF.PI);

    /// <summary>Build a von Mises from per-batch (μ, κ).</summary>
    public VonMisesDistribution(float[] loc, float[] concentration)
    {
        if (loc.Length != concentration.Length) throw new ArgumentException();
        for (int i = 0; i < concentration.Length; i++)
            if (!(concentration[i] >= 0f)) throw new ArgumentException("κ ≥ 0.");
        Loc = loc; Concentration = concentration;
    }

    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        // Best-Fisher rejection sampler (Best & Fisher, 1979).
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) x[i] = SampleOne(rng, Concentration[i], Loc[i]);
        return x;
    }

    private static float SampleOne(Random rng, float kappa, float mu)
    {
        if (kappa < 1e-6f)
            return (float)(2.0 * Math.PI * rng.NextDouble() - Math.PI);
        double tau = 1.0 + Math.Sqrt(1.0 + 4.0 * kappa * kappa);
        double rho = (tau - Math.Sqrt(2.0 * tau)) / (2.0 * kappa);
        double r = (1.0 + rho * rho) / (2.0 * rho);
        while (true)
        {
            double u1 = rng.NextDouble();
            double z = Math.Cos(Math.PI * u1);
            double f = (1.0 + r * z) / (r + z);
            double c = kappa * (r - f);
            double u2 = rng.NextDouble();
            if (u2 < c * (2.0 - c) || Math.Log(c / u2) + 1.0 - c >= 0.0)
            {
                double sign = rng.NextDouble() < 0.5 ? -1.0 : 1.0;
                double theta = sign * Math.Acos(f) + mu;
                // Reduce to (-π, π].
                theta = (theta + Math.PI) % (2.0 * Math.PI) - Math.PI;
                return (float)theta;
            }
        }
    }

    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float k = Concentration[i];
            float logI0 = LogBesselI0(k);
            lp[i] = k * MathF.Cos(value[i] - Loc[i]) - MathF.Log(2f * MathF.PI) - logI0;
        }
        return lp;
    }

    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float k = Concentration[i];
            // H = log(2π·I₀(κ)) − κ·I₁(κ)/I₀(κ)
            float logI0 = LogBesselI0(k);
            float ratioI1I0 = BesselI1OverI0(k);
            h[i] = MathF.Log(2f * MathF.PI) + logI0 - k * ratioI1I0;
        }
        return h;
    }

    /// <inheritdoc />
    public override float[] Mean => (float[])Loc.Clone();

    /// <inheritdoc />
    public override float[] Variance
    {
        // Circular variance: 1 − I₁(κ)/I₀(κ).
        get
        {
            var v = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++) v[i] = 1f - BesselI1OverI0(Concentration[i]);
            return v;
        }
    }

    /// <summary>Asymptotic + series approximation to log I₀(κ).</summary>
    private static float LogBesselI0(float k)
    {
        if (k < 3.75f)
        {
            // Series I₀(k) = Σ (k/2)^(2n) / (n!)²
            double term = 1.0, sum = 1.0; double half = k * 0.5;
            for (int n = 1; n < 50; n++)
            {
                term *= (half * half) / (n * n);
                sum += term;
                if (term < sum * 1e-12) break;
            }
            return (float)Math.Log(sum);
        }
        // Asymptotic for large κ: log I₀(k) ≈ k − 0.5·log(2πk) + log(1 + 1/(8k) + ...)
        double inv = 1.0 / k;
        double s = 1.0 + inv / 8.0 + 9.0 * inv * inv / 128.0;
        return (float)(k - 0.5 * Math.Log(2 * Math.PI * k) + Math.Log(s));
    }

    /// <summary>I₁(κ) / I₀(κ) using the same series + asymptotic split.</summary>
    private static float BesselI1OverI0(float k)
    {
        if (k < 3.75f)
        {
            double t1 = 0.0, sum1 = 0.0; double half = k * 0.5;
            double t0 = 1.0, sum0 = 1.0;
            // I_v(k) = (k/2)^v · Σ (k/2)^(2n) / (n!·(n+v)!)
            // We compute both series in a single pass.
            double term0 = 1.0; double term1 = half;
            sum0 = term0; sum1 = term1;
            for (int n = 1; n < 50; n++)
            {
                term0 *= (half * half) / (n * n);
                term1 *= (half * half) / (n * (n + 1));
                sum0 += term0; sum1 += term1;
                if (term0 < sum0 * 1e-12 && term1 < sum1 * 1e-12) break;
            }
            _ = t0; _ = t1;
            return (float)(sum1 / sum0);
        }
        // Asymptotic ratio: I₁/I₀ → 1 − 1/(2k) − 1/(8k²) + ...
        double inv = 1.0 / k;
        return (float)(1.0 - 0.5 * inv - 0.125 * inv * inv);
    }
}

/// <summary>
/// Continuous Bernoulli (Loaiza-Ganem &amp; Cunningham, 2019): a continuous distribution
/// on [0, 1] that's the natural reparameterisation for VAE pixel decoders.
/// f(x; λ) = C(λ)·λ^x·(1-λ)^(1-x), where C(λ) = 2·atanh(1-2λ)/(1-2λ) for λ ≠ 0.5.
/// </summary>
public sealed class ContinuousBernoulliDistribution : DistributionBase
{
    /// <summary>Per-batch parameter λ ∈ (0, 1).</summary>
    public float[] Probs { get; }
    /// <inheritdoc />
    public override int BatchSize => Probs.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => UnitIntervalConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Build a continuous Bernoulli from per-batch λ.</summary>
    public ContinuousBernoulliDistribution(float[] probs)
    {
        for (int i = 0; i < probs.Length; i++)
            if (!(probs[i] > 0f && probs[i] < 1f)) throw new ArgumentException("λ ∈ (0, 1).");
        Probs = probs;
    }

    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);

    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        // Inverse-CDF sampling via the closed-form quantile from Loaiza-Ganem & Cunningham.
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            double u = rng.NextDouble();
            float lambda = Probs[i];
            if (MathF.Abs(lambda - 0.5f) < 1e-4f)
            {
                x[i] = (float)u;  // uniform when λ ≈ 0.5
            }
            else
            {
                float numer = (float)u * (2f * lambda - 1f) + (1f - lambda);
                float denom = 1f - lambda;
                x[i] = MathF.Log(numer / denom) / MathF.Log(lambda / (1f - lambda));
            }
        }
        return x;
    }

    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float v = value[i];
            if (v < 0f || v > 1f) { lp[i] = float.NegativeInfinity; continue; }
            float lambda = Probs[i];
            lp[i] = v * MathF.Log(lambda) + (1f - v) * MathF.Log(1f - lambda) + LogNormalizer(lambda);
        }
        return lp;
    }

    /// <summary>log C(λ) — the log of the normalising constant.</summary>
    private static float LogNormalizer(float lambda)
    {
        // Numerically stable form: when λ ≈ 0.5, expand around 0 to avoid 0/0.
        float diff = 1f - 2f * lambda;
        if (MathF.Abs(diff) < 1e-3f) return MathF.Log(2f);
        return MathF.Log(2f * MathF.Atanh(diff)) - MathF.Log(diff);
    }

    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float lambda = Probs[i];
            float m = MeanOf(lambda);
            h[i] = -m * MathF.Log(lambda) - (1f - m) * MathF.Log(1f - lambda) - LogNormalizer(lambda);
        }
        return h;
    }

    /// <inheritdoc />
    public override float[] Mean
    {
        get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = MeanOf(Probs[i]); return m; }
    }

    private static float MeanOf(float lambda)
    {
        if (MathF.Abs(lambda - 0.5f) < 1e-4f) return 0.5f;
        return lambda / (2f * lambda - 1f) + 1f / (2f * MathF.Atanh(1f - 2f * lambda));
    }

    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var v = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++)
            {
                float lambda = Probs[i];
                if (MathF.Abs(lambda - 0.5f) < 1e-4f) { v[i] = 1f / 12f; continue; }
                // Closed form (Loaiza-Ganem & Cunningham eq. 8).
                float t = 1f - 2f * lambda;
                float t2 = 1f / (t * t);
                float at = MathF.Atanh(t);
                v[i] = t2 + 1f / (4f * at * at) - 1f / (t * at);
            }
            return v;
        }
    }
}

/// <summary>
/// Low-rank multivariate normal: Σ = D + W·Wᵀ where D is diagonal and W has shape [D, K].
/// Sampling: x = μ + sqrt(D)·z₁ + W·z₂ where z₁ ∼ N(0, I_D), z₂ ∼ N(0, I_K).
/// log_prob uses the Sherman-Morrison-Woodbury identity for the inverse and the matrix
/// determinant lemma for log|Σ|.
/// </summary>
public sealed class LowRankMultivariateNormalDistribution : DistributionBase
{
    /// <summary>Per-batch loc, layout <c>[batch, D]</c>.</summary>
    public float[] Loc { get; }
    /// <summary>Per-batch diagonal of <c>D</c> (length D per batch element). Must be &gt; 0.</summary>
    public float[] CovDiag { get; }
    /// <summary>Per-batch W matrix, layout <c>[batch, D, K]</c>.</summary>
    public float[] CovFactor { get; }
    /// <summary>Event dimension D.</summary>
    public int D { get; }
    /// <summary>Low-rank dimension K.</summary>
    public int K { get; }
    /// <inheritdoc />
    public override int BatchSize => Loc.Length / D;
    /// <inheritdoc />
    public override int EventSize => D;
    /// <inheritdoc />
    public override IConstraint Support => RealConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Build a low-rank MVN from loc, diag(D), and W.</summary>
    public LowRankMultivariateNormalDistribution(float[] loc, float[] covDiag, float[] covFactor, int d, int k)
    {
        if (d <= 0 || k <= 0) throw new ArgumentException();
        if (loc.Length % d != 0) throw new ArgumentException();
        int batch = loc.Length / d;
        if (covDiag.Length != batch * d) throw new ArgumentException();
        if (covFactor.Length != batch * d * k) throw new ArgumentException();
        for (int i = 0; i < covDiag.Length; i++) if (!(covDiag[i] > 0f)) throw new ArgumentException("covDiag > 0.");
        Loc = loc; CovDiag = covDiag; CovFactor = covFactor; D = d; K = k;
    }

    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);

    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        int batch = BatchSize;
        var x = new float[batch * D];
        var z2 = new float[K];
        for (int b = 0; b < batch; b++)
        {
            for (int k = 0; k < K; k++) z2[k] = (float)NormalDistribution.Gaussian(rng);
            for (int i = 0; i < D; i++)
            {
                float diagPart = MathF.Sqrt(CovDiag[b * D + i]) * (float)NormalDistribution.Gaussian(rng);
                float lowRankPart = 0f;
                for (int k = 0; k < K; k++) lowRankPart += CovFactor[b * D * K + i * K + k] * z2[k];
                x[b * D + i] = Loc[b * D + i] + diagPart + lowRankPart;
            }
        }
        return x;
    }

    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        int batch = BatchSize;
        var lp = new float[batch];
        const float HalfLog2Pi = 0.91893853321f;
        // Σ⁻¹ = D⁻¹ − D⁻¹·W·(I_K + Wᵀ·D⁻¹·W)⁻¹·Wᵀ·D⁻¹  (Woodbury)
        // |Σ| = |D|·|I_K + Wᵀ·D⁻¹·W|                    (matrix determinant lemma)
        for (int b = 0; b < batch; b++)
        {
            // Build M = I_K + Wᵀ·D⁻¹·W  (size K×K).
            var M = new float[K * K];
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                {
                    float s = 0f;
                    for (int i = 0; i < D; i++)
                        s += CovFactor[b * D * K + i * K + p] * CovFactor[b * D * K + i * K + q] / CovDiag[b * D + i];
                    M[p * K + q] = (p == q ? 1f : 0f) + s;
                }
            // Cholesky M = L·Lᵀ.
            var L = new float[K * K];
            CholeskyKxK(M, L, K);

            // diff = value - loc.
            var diff = new float[D];
            for (int i = 0; i < D; i++) diff[i] = value[b * D + i] - Loc[b * D + i];
            // Quadratic form: diffᵀ Σ⁻¹ diff = diffᵀ D⁻¹ diff − ‖L⁻¹ Wᵀ D⁻¹ diff‖²
            float diagQuad = 0f;
            var rhs = new float[K];
            for (int i = 0; i < D; i++)
            {
                float di = diff[i] / CovDiag[b * D + i];
                diagQuad += diff[i] * di;
                for (int p = 0; p < K; p++) rhs[p] += CovFactor[b * D * K + i * K + p] * di;
            }
            // Forward-substitute L · y = rhs.
            var y = new float[K];
            for (int p = 0; p < K; p++)
            {
                float s = rhs[p];
                for (int q = 0; q < p; q++) s -= L[p * K + q] * y[q];
                y[p] = s / L[p * K + p];
            }
            float lowRankQuad = 0f;
            for (int p = 0; p < K; p++) lowRankQuad += y[p] * y[p];

            // log|Σ| = Σ log d_i + 2 Σ log L_pp.
            float logDet = 0f;
            for (int i = 0; i < D; i++) logDet += MathF.Log(CovDiag[b * D + i]);
            for (int p = 0; p < K; p++) logDet += 2f * MathF.Log(L[p * K + p]);

            lp[b] = -0.5f * (diagQuad - lowRankQuad) - 0.5f * logDet - D * HalfLog2Pi;
        }
        return lp;
    }

    private static void CholeskyKxK(float[] src, float[] dst, int n)
    {
        for (int i = 0; i < n * n; i++) dst[i] = 0f;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double s = src[i * n + j];
                for (int k = 0; k < j; k++) s -= dst[i * n + k] * dst[j * n + k];
                if (i == j)
                {
                    if (s <= 0) throw new ArgumentException("M = I + Wᵀ D⁻¹ W is not positive definite.");
                    dst[i * n + i] = (float)Math.Sqrt(s);
                }
                else dst[i * n + j] = (float)(s / dst[j * n + j]);
            }
        }
    }

    /// <inheritdoc />
    public override float[] Entropy()
    {
        int batch = BatchSize;
        var h = new float[batch];
        const float HalfLog2PiE = 1.41893853321f;
        for (int b = 0; b < batch; b++)
        {
            // Same logDet computation as LogProb.
            var M = new float[K * K];
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                {
                    float s = 0f;
                    for (int i = 0; i < D; i++)
                        s += CovFactor[b * D * K + i * K + p] * CovFactor[b * D * K + i * K + q] / CovDiag[b * D + i];
                    M[p * K + q] = (p == q ? 1f : 0f) + s;
                }
            var L = new float[K * K];
            CholeskyKxK(M, L, K);
            float logDet = 0f;
            for (int i = 0; i < D; i++) logDet += MathF.Log(CovDiag[b * D + i]);
            for (int p = 0; p < K; p++) logDet += 2f * MathF.Log(L[p * K + p]);
            h[b] = D * HalfLog2PiE + 0.5f * logDet;
        }
        return h;
    }

    /// <inheritdoc />
    public override float[] Mean => (float[])Loc.Clone();

    /// <inheritdoc />
    public override float[] Variance
    {
        // Variance per dim = D_i + Σ_k W_{i,k}².
        get
        {
            int batch = BatchSize;
            var v = new float[batch * D];
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < D; i++)
                {
                    float s = CovDiag[b * D + i];
                    for (int k = 0; k < K; k++)
                    {
                        float w = CovFactor[b * D * K + i * K + k];
                        s += w * w;
                    }
                    v[b * D + i] = s;
                }
            return v;
        }
    }
}
