using System;
using AiDotNet.Tensors.Distributions.Constraints;
using AiDotNet.Tensors.Distributions.Helpers;

namespace AiDotNet.Tensors.Distributions;

/// <summary>LogNormal distribution: X = exp(μ + σ·Z), Z ∼ N(0, 1). Domain (0, ∞).</summary>
public sealed class LogNormalDistribution : DistributionBase
{
    /// <summary>Mean of the underlying normal.</summary>
    public float[] Loc { get; }
    /// <summary>Std of the underlying normal.</summary>
    public float[] Scale { get; }
    /// <inheritdoc />
    public override int BatchSize => Loc.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => PositiveConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Build a LogNormal from per-batch loc/scale.</summary>
    public LogNormalDistribution(float[] loc, float[] scale)
    {
        if (loc.Length != scale.Length) throw new ArgumentException("loc and scale must match.");
        for (int i = 0; i < scale.Length; i++) if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
        Loc = loc; Scale = scale;
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            x[i] = MathF.Exp(Loc[i] + Scale[i] * (float)NormalDistribution.Gaussian(rng));
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        const float LogTwoPi = 1.83787706641f;
        for (int i = 0; i < BatchSize; i++)
        {
            float lnX = MathF.Log(value[i]);
            float z = (lnX - Loc[i]) / Scale[i];
            lp[i] = -0.5f * (z * z + LogTwoPi) - MathF.Log(Scale[i]) - lnX;
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        const float HalfLogTwoPiE = 1.41893853321f;
        for (int i = 0; i < BatchSize; i++) h[i] = HalfLogTwoPiE + MathF.Log(Scale[i]) + Loc[i];
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            var m = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++)
                m[i] = MathF.Exp(Loc[i] + 0.5f * Scale[i] * Scale[i]);
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
                float s2 = Scale[i] * Scale[i];
                v[i] = (MathF.Exp(s2) - 1f) * MathF.Exp(2f * Loc[i] + s2);
            }
            return v;
        }
    }
}

/// <summary>Half-normal distribution: |Z|·σ for Z ∼ N(0, 1).</summary>
public sealed class HalfNormalDistribution : DistributionBase
{
    /// <summary>Per-batch scale.</summary>
    public float[] Scale { get; }
    /// <inheritdoc />
    public override int BatchSize => Scale.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Build a half-normal from per-batch scale.</summary>
    public HalfNormalDistribution(float[] scale)
    {
        for (int i = 0; i < scale.Length; i++) if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
        Scale = scale;
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            x[i] = MathF.Abs(Scale[i] * (float)NormalDistribution.Gaussian(rng));
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        const float LogTwoOverPi = 0.45158270528f; // 0.5·log(2/π)
        for (int i = 0; i < BatchSize; i++)
        {
            if (value[i] < 0) { lp[i] = float.NegativeInfinity; continue; }
            float z = value[i] / Scale[i];
            lp[i] = LogTwoOverPi - 0.5f * z * z - MathF.Log(Scale[i]);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        const float HalfLog2EOverPi = 0.72579135264f; // 0.5·log(πe/2)
        for (int i = 0; i < BatchSize; i++) h[i] = HalfLog2EOverPi + MathF.Log(Scale[i]);
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            var m = new float[BatchSize];
            const float SqrtTwoOverPi = 0.79788456080f;
            for (int i = 0; i < BatchSize; i++) m[i] = Scale[i] * SqrtTwoOverPi;
            return m;
        }
    }
    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var v = new float[BatchSize];
            const float OneMinusTwoOverPi = 0.36338022763f;
            for (int i = 0; i < BatchSize; i++) v[i] = Scale[i] * Scale[i] * OneMinusTwoOverPi;
            return v;
        }
    }
}

/// <summary>Cauchy distribution. Heavy-tailed; mean and variance are undefined.</summary>
public sealed class CauchyDistribution : DistributionBase
{
    /// <summary>Per-batch location (median).</summary>
    public float[] Loc { get; }
    /// <summary>Per-batch scale (half-width at half-maximum).</summary>
    public float[] Scale { get; }
    /// <inheritdoc />
    public override int BatchSize => Loc.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => RealConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Build a Cauchy from per-batch loc/scale.</summary>
    public CauchyDistribution(float[] loc, float[] scale)
    {
        if (loc.Length != scale.Length) throw new ArgumentException();
        for (int i = 0; i < scale.Length; i++) if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
        Loc = loc; Scale = scale;
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
            x[i] = Loc[i] + Scale[i] * (float)Math.Tan(Math.PI * (u - 0.5));
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
            float z = (value[i] - Loc[i]) / Scale[i];
            lp[i] = -MathF.Log(MathF.PI) - MathF.Log(Scale[i]) - MathF.Log(1f + z * z);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        float lnFourPi = MathF.Log(4f * MathF.PI);
        for (int i = 0; i < BatchSize; i++) h[i] = lnFourPi + MathF.Log(Scale[i]);
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = float.NaN; return m; }
    }
    /// <inheritdoc />
    public override float[] Variance
    {
        get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = float.NaN; return v; }
    }
    /// <inheritdoc />
    public override float[] Cdf(float[] value)
    {
        EnsureValueShape(value);
        var c = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            float z = (value[i] - Loc[i]) / Scale[i];
            c[i] = 0.5f + MathF.Atan(z) / MathF.PI;
        }
        return c;
    }
    /// <inheritdoc />
    public override float[] Icdf(float[] probability)
    {
        EnsureValueShape(probability);
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            x[i] = Loc[i] + Scale[i] * MathF.Tan(MathF.PI * (probability[i] - 0.5f));
        return x;
    }
}

/// <summary>Half-Cauchy distribution: |X| where X ∼ Cauchy(0, scale).</summary>
public sealed class HalfCauchyDistribution : DistributionBase
{
    /// <summary>Per-batch scale.</summary>
    public float[] Scale { get; }
    /// <inheritdoc />
    public override int BatchSize => Scale.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a half-Cauchy from scale.</summary>
    public HalfCauchyDistribution(float[] scale)
    {
        for (int i = 0; i < scale.Length; i++) if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
        Scale = scale;
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
            x[i] = MathF.Abs(Scale[i] * (float)Math.Tan(Math.PI * (u - 0.5)));
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
            float z = value[i] / Scale[i];
            lp[i] = MathF.Log(2f / MathF.PI) - MathF.Log(Scale[i]) - MathF.Log(1f + z * z);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        float c = MathF.Log(2f * MathF.PI);
        for (int i = 0; i < BatchSize; i++) h[i] = c + MathF.Log(Scale[i]);
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    { get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = float.PositiveInfinity; return m; } }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = float.PositiveInfinity; return v; } }
}

/// <summary>Laplace distribution: f(x) = (1/2b)·exp(-|x-μ|/b).</summary>
public sealed class LaplaceDistribution : DistributionBase
{
    /// <summary>Per-batch location.</summary>
    public float[] Loc { get; }
    /// <summary>Per-batch scale b.</summary>
    public float[] Scale { get; }
    /// <inheritdoc />
    public override int BatchSize => Loc.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => RealConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a Laplace from per-batch loc/scale.</summary>
    public LaplaceDistribution(float[] loc, float[] scale)
    {
        if (loc.Length != scale.Length) throw new ArgumentException();
        for (int i = 0; i < scale.Length; i++) if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
        Loc = loc; Scale = scale;
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            double u = rng.NextDouble() - 0.5;
            x[i] = Loc[i] - Scale[i] * MathF.Sign((float)u) * MathF.Log(1f - 2f * MathF.Abs((float)u));
        }
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            lp[i] = -MathF.Log(2f * Scale[i]) - MathF.Abs(value[i] - Loc[i]) / Scale[i];
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        float c = 1f + MathF.Log(2f);
        for (int i = 0; i < BatchSize; i++) h[i] = c + MathF.Log(Scale[i]);
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean => (float[])Loc.Clone();
    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var v = new float[BatchSize];
            for (int i = 0; i < BatchSize; i++) v[i] = 2f * Scale[i] * Scale[i];
            return v;
        }
    }
}

/// <summary>Uniform distribution on [low, high).</summary>
public sealed class UniformDistribution : DistributionBase
{
    /// <summary>Per-batch lower bound (inclusive).</summary>
    public float[] Low { get; }
    /// <summary>Per-batch upper bound (exclusive).</summary>
    public float[] High { get; }
    /// <inheritdoc />
    public override int BatchSize => Low.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support
    {
        get
        {
            float lo = Low[0], hi = High[0];
            for (int i = 1; i < Low.Length; i++)
            {
                if (Low[i] < lo) lo = Low[i];
                if (High[i] > hi) hi = High[i];
            }
            return new IntervalConstraint(lo, hi);
        }
    }
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a uniform from per-batch low/high.</summary>
    public UniformDistribution(float[] low, float[] high)
    {
        if (low.Length != high.Length) throw new ArgumentException();
        for (int i = 0; i < low.Length; i++)
            if (!(high[i] > low[i])) throw new ArgumentException("high > low.");
        Low = low; High = high;
    }
    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            x[i] = Low[i] + (High[i] - Low[i]) * (float)rng.NextDouble();
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            lp[i] = (value[i] >= Low[i] && value[i] < High[i])
                ? -MathF.Log(High[i] - Low[i])
                : float.NegativeInfinity;
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) h[i] = MathF.Log(High[i] - Low[i]);
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = 0.5f * (Low[i] + High[i]); return m; }
    }
    /// <inheritdoc />
    public override float[] Variance
    {
        get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) { var w = High[i] - Low[i]; v[i] = w * w / 12f; } return v; }
    }
    /// <inheritdoc />
    public override float[] Cdf(float[] value)
    {
        EnsureValueShape(value);
        var c = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            if (value[i] < Low[i]) c[i] = 0f;
            else if (value[i] >= High[i]) c[i] = 1f;
            else c[i] = (value[i] - Low[i]) / (High[i] - Low[i]);
        }
        return c;
    }
    /// <inheritdoc />
    public override float[] Icdf(float[] probability)
    {
        EnsureValueShape(probability);
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            x[i] = Low[i] + probability[i] * (High[i] - Low[i]);
        return x;
    }
}

/// <summary>Exponential distribution f(x) = λ·exp(-λ·x), x ≥ 0.</summary>
public sealed class ExponentialDistribution : DistributionBase
{
    /// <summary>Per-batch rate parameter (1 / mean).</summary>
    public float[] Rate { get; }
    /// <inheritdoc />
    public override int BatchSize => Rate.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => NonNegativeConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build an exponential from per-batch rate.</summary>
    public ExponentialDistribution(float[] rate)
    {
        for (int i = 0; i < rate.Length; i++) if (!(rate[i] > 0f)) throw new ArgumentException("rate > 0.");
        Rate = rate;
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
            x[i] = -MathF.Log((float)u) / Rate[i];
        }
        return x;
    }
    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            lp[i] = value[i] < 0 ? float.NegativeInfinity : MathF.Log(Rate[i]) - Rate[i] * value[i];
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) h[i] = 1f - MathF.Log(Rate[i]);
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    { get { var m = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) m[i] = 1f / Rate[i]; return m; } }
    /// <inheritdoc />
    public override float[] Variance
    { get { var v = new float[BatchSize]; for (int i = 0; i < BatchSize; i++) v[i] = 1f / (Rate[i] * Rate[i]); return v; } }
    /// <inheritdoc />
    public override float[] Cdf(float[] value)
    {
        EnsureValueShape(value);
        var c = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            c[i] = value[i] < 0 ? 0f : 1f - MathF.Exp(-Rate[i] * value[i]);
        return c;
    }
    /// <inheritdoc />
    public override float[] Icdf(float[] probability)
    {
        EnsureValueShape(probability);
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++) x[i] = -MathF.Log(1f - probability[i]) / Rate[i];
        return x;
    }
}

/// <summary>Gumbel distribution F(x) = exp(-exp(-(x-μ)/β)).</summary>
public sealed class GumbelDistribution : DistributionBase
{
    /// <summary>Per-batch location.</summary>
    public float[] Loc { get; }
    /// <summary>Per-batch scale.</summary>
    public float[] Scale { get; }
    /// <inheritdoc />
    public override int BatchSize => Loc.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => RealConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;
    /// <summary>Build a Gumbel from per-batch loc/scale.</summary>
    public GumbelDistribution(float[] loc, float[] scale)
    {
        if (loc.Length != scale.Length) throw new ArgumentException();
        for (int i = 0; i < scale.Length; i++) if (!(scale[i] > 0f)) throw new ArgumentException("scale > 0.");
        Loc = loc; Scale = scale;
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
            x[i] = Loc[i] - Scale[i] * MathF.Log(-MathF.Log((float)u));
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
            float z = (value[i] - Loc[i]) / Scale[i];
            lp[i] = -MathF.Log(Scale[i]) - z - MathF.Exp(-z);
        }
        return lp;
    }
    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        const float EulerMascheroni = 0.5772156649f;
        for (int i = 0; i < BatchSize; i++) h[i] = MathF.Log(Scale[i]) + EulerMascheroni + 1f;
        return h;
    }
    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            var m = new float[BatchSize];
            const float EulerMascheroni = 0.5772156649f;
            for (int i = 0; i < BatchSize; i++) m[i] = Loc[i] + EulerMascheroni * Scale[i];
            return m;
        }
    }
    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var v = new float[BatchSize];
            float pi2over6 = MathF.PI * MathF.PI / 6f;
            for (int i = 0; i < BatchSize; i++) v[i] = pi2over6 * Scale[i] * Scale[i];
            return v;
        }
    }
}
