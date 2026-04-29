using System;
using AiDotNet.Tensors.Distributions.Constraints;
using AiDotNet.Tensors.Distributions.Helpers;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Gamma distribution Γ(α, β). Marsaglia-Tsang sampler.
/// Reparameterisable via implicit reparameterisation (Figurnov et al., 2018) — but here
/// we expose the rejection-based <see cref="Sample"/> only; <see cref="HasRSample"/> is
/// true and <see cref="RSample"/> uses the same path (the gradient w.r.t. shape requires
/// implicit reparam machinery and is implemented in the autograd layer).
/// </summary>
public sealed class GammaDistribution : DistributionBase
{
    /// <summary>Shape parameter α &gt; 0.</summary>
    public float[] Concentration { get; }
    /// <summary>Rate parameter β &gt; 0 (1 / scale).</summary>
    public float[] Rate { get; }
    /// <inheritdoc />
    public override int BatchSize => Concentration.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => PositiveConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Build a Gamma from per-batch concentration and rate.</summary>
    public GammaDistribution(float[] concentration, float[] rate)
    {
        if (concentration.Length != rate.Length) throw new ArgumentException();
        for (int i = 0; i < concentration.Length; i++)
        {
            if (!(concentration[i] > 0f)) throw new ArgumentException("concentration > 0.");
            if (!(rate[i] > 0f)) throw new ArgumentException("rate > 0.");
        }
        Concentration = (float[])concentration.Clone(); Rate = (float[])rate.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            x[i] = MarsagliaTsang(rng, Concentration[i]) / Rate[i];
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
            float a = Concentration[i]; float b = Rate[i];
            lp[i] = a * MathF.Log(b) + (a - 1f) * MathF.Log(value[i]) - b * value[i] - SpecialFunctions.Lgamma(a);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float a = Concentration[i];
            h[i] = a - MathF.Log(Rate[i]) + SpecialFunctions.Lgamma(a) + (1f - a) * SpecialFunctions.Digamma(a);
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    { get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = Concentration[i] / Rate[i]; return m; } }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = Concentration[i] / (Rate[i] * Rate[i]); return v; } }
    /// <inheritdoc />
    public override float[] Cdf(float[] value)
    {
        EnsureValueShape(value);
        var c = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            c[i] = SpecialFunctions.GammaP(Concentration[i], Rate[i] * value[i]);
        return c;
    }

    /// <summary>Marsaglia-Tsang rejection sampler for Gamma(α, 1).</summary>
    internal static float MarsagliaTsang(Random rng, float alpha)
    {
        if (alpha < 1f)
        {
            // Boost: G(α) = G(α + 1) · U^(1/α).
            float g = MarsagliaTsang(rng, alpha + 1f);
            double u = rng.NextDouble();
            if (u < 1e-12) u = 1e-12;
            return g * MathF.Pow((float)u, 1f / alpha);
        }
        double d = alpha - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double z = NormalDistribution.Gaussian(rng);
            double v = 1.0 + c * z;
            if (v <= 0) continue;
            v = v * v * v;
            double u = rng.NextDouble();
            if (u < 1.0 - 0.0331 * z * z * z * z) return (float)(d * v);
            if (Math.Log(u) < 0.5 * z * z + d * (1.0 - v + Math.Log(v))) return (float)(d * v);
        }
    }
}

/// <summary>Inverse Gamma: 1/X for X ∼ Gamma(α, β).</summary>
public sealed class InverseGammaDistribution : DistributionBase
{
    /// <summary>Shape α.</summary>
    public float[] Concentration { get; }
    /// <summary>Scale β (note: PyTorch's InverseGamma uses scale, not rate).</summary>
    public float[] Scale { get; }
    /// <inheritdoc />
    public override int BatchSize => Concentration.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => PositiveConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build an Inverse Gamma from per-batch shape/scale.</summary>
    public InverseGammaDistribution(float[] concentration, float[] scale)
    {
        if (concentration.Length != scale.Length) throw new ArgumentException();
        for (int i = 0; i < concentration.Length; i++)
        {
            if (!(concentration[i] > 0f)) throw new ArgumentException("concentration > 0.");
            if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
        }
        Concentration = (float[])concentration.Clone(); Scale = (float[])scale.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float g = GammaDistribution.MarsagliaTsang(rng, Concentration[i]);
            x[i] = Scale[i] / g;
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
            float a = Concentration[i]; float s = Scale[i];
            lp[i] = a * MathF.Log(s) - SpecialFunctions.Lgamma(a)
                  - (a + 1f) * MathF.Log(value[i]) - s / value[i];
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float a = Concentration[i];
            h[i] = a + MathF.Log(Scale[i]) + SpecialFunctions.Lgamma(a) - (1f + a) * SpecialFunctions.Digamma(a);
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
                m[i] = Concentration[i] > 1f ? Scale[i] / (Concentration[i] - 1f) : float.PositiveInfinity;
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
                float a = Concentration[i]; float s = Scale[i];
                v[i] = a > 2f ? s * s / ((a - 1f) * (a - 1f) * (a - 2f)) : float.PositiveInfinity;
            }
            return v;
        }
    }
}

/// <summary>Chi-squared distribution: χ²(k) = Gamma(k/2, 1/2).</summary>
public sealed class Chi2Distribution : DistributionBase
{
    /// <summary>Degrees of freedom (must be &gt; 0).</summary>
    public float[] Df { get; }
    /// <inheritdoc />
    public override int BatchSize => Df.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => PositiveConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a chi-squared from per-batch degrees of freedom.</summary>
    public Chi2Distribution(float[] df)
    {
        for (int i = 0; i < df.Length; i++) if (!(df[i] > 0f)) throw new ArgumentException("df > 0.");
        Df = (float[])df.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            x[i] = 2f * GammaDistribution.MarsagliaTsang(rng, Df[i] * 0.5f);
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
            float k = Df[i];
            lp[i] = -k * 0.5f * MathF.Log(2f) - SpecialFunctions.Lgamma(k * 0.5f)
                  + (k * 0.5f - 1f) * MathF.Log(value[i]) - 0.5f * value[i];
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float k = Df[i] * 0.5f;
            h[i] = k + MathF.Log(2f) + SpecialFunctions.Lgamma(k) + (1f - k) * SpecialFunctions.Digamma(k);
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean => (float[])Df.Clone();
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = 2f * Df[i]; return v; } }
}

/// <summary>Beta distribution Beta(α, β) on (0, 1).</summary>
public sealed class BetaDistribution : DistributionBase
{
    /// <summary>α concentration parameter.</summary>
    public float[] Concentration1 { get; }
    /// <summary>β concentration parameter.</summary>
    public float[] Concentration0 { get; }
    /// <inheritdoc />
    public override int BatchSize => Concentration1.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => UnitIntervalConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a Beta from per-batch (α, β).</summary>
    public BetaDistribution(float[] concentration1, float[] concentration0)
    {
        if (concentration1.Length != concentration0.Length) throw new ArgumentException();
        for (int i = 0; i < concentration1.Length; i++)
        {
            if (!(concentration1[i] > 0f)) throw new ArgumentException("α > 0.");
            if (!(concentration0[i] > 0f)) throw new ArgumentException("β > 0.");
        }
        Concentration1 = (float[])concentration1.Clone(); Concentration0 = (float[])concentration0.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float ga = GammaDistribution.MarsagliaTsang(rng, Concentration1[i]);
            float gb = GammaDistribution.MarsagliaTsang(rng, Concentration0[i]);
            x[i] = ga / (ga + gb);
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
            float a = Concentration1[i]; float b = Concentration0[i];
            lp[i] = (a - 1f) * MathF.Log(v) + (b - 1f) * MathF.Log(1f - v) - SpecialFunctions.LogBeta(a, b);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float a = Concentration1[i]; float b = Concentration0[i];
            float ab = a + b;
            h[i] = SpecialFunctions.LogBeta(a, b)
                 - (a - 1f) * SpecialFunctions.Digamma(a)
                 - (b - 1f) * SpecialFunctions.Digamma(b)
                 + (ab - 2f) * SpecialFunctions.Digamma(ab);
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            var m = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++) m[i] = Concentration1[i] / (Concentration1[i] + Concentration0[i]);
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
                float a = Concentration1[i]; float b = Concentration0[i]; float ab = a + b;
                v[i] = a * b / (ab * ab * (ab + 1f));
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
            c[i] = SpecialFunctions.BetaI(Concentration1[i], Concentration0[i], value[i]);
        return c;
    }
}

/// <summary>Student's t distribution.</summary>
public sealed class StudentTDistribution : DistributionBase
{
    /// <summary>Degrees of freedom.</summary>
    public float[] Df { get; }
    /// <summary>Per-batch location.</summary>
    public float[] Loc { get; }
    /// <summary>Per-batch scale.</summary>
    public float[] Scale { get; }
    /// <inheritdoc />
    public override int BatchSize => Df.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => RealConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a Student's-t from df, loc, scale.</summary>
    public StudentTDistribution(float[] df, float[] loc, float[] scale)
    {
        if (df.Length != loc.Length || df.Length != scale.Length) throw new ArgumentException();
        for (int i = 0; i < df.Length; i++)
        {
            if (!(df[i] > 0f)) throw new ArgumentException("df > 0.");
            if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
        }
        Df = (float[])df.Clone(); Loc = (float[])loc.Clone(); Scale = (float[])scale.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        // X = μ + σ · Z / sqrt(V/df) where Z ∼ N(0,1), V ∼ Chi²(df).
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            double z = NormalDistribution.Gaussian(rng);
            float v = 2f * GammaDistribution.MarsagliaTsang(rng, Df[i] * 0.5f);
            x[i] = Loc[i] + Scale[i] * (float)z / MathF.Sqrt(v / Df[i]);
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
            float df = Df[i];
            float z = (value[i] - Loc[i]) / Scale[i];
            float c = SpecialFunctions.Lgamma(0.5f * (df + 1f))
                    - SpecialFunctions.Lgamma(0.5f * df)
                    - 0.5f * MathF.Log(df * MathF.PI)
                    - MathF.Log(Scale[i]);
            lp[i] = c - 0.5f * (df + 1f) * MathF.Log(1f + z * z / df);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float df = Df[i];
            h[i] = MathF.Log(Scale[i])
                 + 0.5f * (df + 1f) * (SpecialFunctions.Digamma(0.5f * (df + 1f)) - SpecialFunctions.Digamma(0.5f * df))
                 + 0.5f * MathF.Log(df) + SpecialFunctions.LogBeta(0.5f * df, 0.5f);
        }
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            var m = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++) m[i] = Df[i] > 1f ? Loc[i] : float.NaN;
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
                float df = Df[i];
                v[i] = df > 2f ? Scale[i] * Scale[i] * df / (df - 2f) :
                       df > 1f ? float.PositiveInfinity : float.NaN;
            }
            return v;
        }
    }
}

/// <summary>Pareto Type-I distribution.</summary>
public sealed class ParetoDistribution : DistributionBase
{
    /// <summary>Per-batch scale (minimum value).</summary>
    public float[] Scale { get; }
    /// <summary>Per-batch shape α.</summary>
    public float[] Alpha { get; }
    /// <inheritdoc />
    public override int BatchSize => Scale.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support
    {
        get
        {
            float minScale = Scale[0];
            for (int i = 1; i < Scale.Length; i++) if (Scale[i] < minScale) minScale = Scale[i];
            return new GreaterThanConstraint(minScale - 1e-6f);
        }
    }
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a Pareto from per-batch scale/alpha.</summary>
    public ParetoDistribution(float[] scale, float[] alpha)
    {
        if (scale.Length != alpha.Length) throw new ArgumentException();
        for (int i = 0; i < scale.Length; i++)
        {
            if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
            if (!(alpha[i] > 0f)) throw new ArgumentException("alpha > 0.");
        }
        Scale = (float[])scale.Clone(); Alpha = (float[])alpha.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            double u = rng.NextDouble();
            if (u < 1e-12) u = 1e-12;
            x[i] = Scale[i] * MathF.Pow(1f - (float)u, -1f / Alpha[i]);
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
            if (value[i] < Scale[i]) { lp[i] = float.NegativeInfinity; continue; }
            lp[i] = MathF.Log(Alpha[i]) + Alpha[i] * MathF.Log(Scale[i]) - (Alpha[i] + 1f) * MathF.Log(value[i]);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            h[i] = MathF.Log(Scale[i] / Alpha[i]) + 1f / Alpha[i] + 1f;
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            var m = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++)
                m[i] = Alpha[i] > 1f ? Alpha[i] * Scale[i] / (Alpha[i] - 1f) : float.PositiveInfinity;
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
                float a = Alpha[i]; float s = Scale[i];
                v[i] = a > 2f ? s * s * a / ((a - 1f) * (a - 1f) * (a - 2f)) : float.PositiveInfinity;
            }
            return v;
        }
    }
}

/// <summary>Weibull distribution.</summary>
public sealed class WeibullDistribution : DistributionBase
{
    /// <summary>Scale λ &gt; 0.</summary>
    public float[] Scale { get; }
    /// <summary>Shape k &gt; 0 (concentration).</summary>
    public float[] Concentration { get; }
    /// <inheritdoc />
    public override int BatchSize => Scale.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a Weibull from per-batch scale/concentration.</summary>
    public WeibullDistribution(float[] scale, float[] concentration)
    {
        if (scale.Length != concentration.Length) throw new ArgumentException();
        for (int i = 0; i < scale.Length; i++)
        {
            if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
            if (!(concentration[i] > 0f)) throw new ArgumentException("concentration > 0.");
        }
        Scale = (float[])scale.Clone(); Concentration = (float[])concentration.Clone();
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            double u = rng.NextDouble();
            if (u < 1e-12) u = 1e-12;
            x[i] = Scale[i] * MathF.Pow(-MathF.Log(1f - (float)u), 1f / Concentration[i]);
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
            if (value[i] < 0) { lp[i] = float.NegativeInfinity; continue; }
            float k = Concentration[i]; float l = Scale[i];
            // x = 0 edge case: density value depends on the shape k.
            // k > 1: density is 0 at x=0 ⇒ log p = -∞.
            // k = 1: exponential, density at 0 is 1/λ ⇒ log p = -log(l).
            // k < 1: density diverges to +∞ at x=0; report +∞ rather than NaN from log(0).
            if (value[i] == 0f)
            {
                lp[i] = k > 1f ? float.NegativeInfinity
                      : k == 1f ? -MathF.Log(l)
                      : float.PositiveInfinity;
                continue;
            }
            float zk = MathF.Pow(value[i] / l, k);
            lp[i] = MathF.Log(k) - MathF.Log(l) + (k - 1f) * MathF.Log(value[i] / l) - zk;
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        const float EulerMascheroni = 0.5772156649f;
        for (int i = 0; i < BatchSize; i++)
        {
            float k = Concentration[i];
            h[i] = EulerMascheroni * (1f - 1f / k) + MathF.Log(Scale[i] / k) + 1f;
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
                m[i] = Scale[i] * MathF.Exp(SpecialFunctions.Lgamma(1f + 1f / Concentration[i]));
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
                float k = Concentration[i]; float l = Scale[i];
                float g1 = MathF.Exp(SpecialFunctions.Lgamma(1f + 1f / k));
                float g2 = MathF.Exp(SpecialFunctions.Lgamma(1f + 2f / k));
                v[i] = l * l * (g2 - g1 * g1);
            }
            return v;
        }
    }
}
