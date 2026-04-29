using System;
using AiDotNet.Tensors.Distributions.Constraints;
using AiDotNet.Tensors.Distributions.Helpers;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Normal (Gaussian) distribution N(μ, σ²). Reparameterisable via the
/// location-scale trick: <c>x = μ + σ·z, z ∼ N(0, 1)</c>.
/// </summary>
public sealed class NormalDistribution : DistributionBase
{
    /// <summary>Per-batch means.</summary>
    public float[] Loc { get; }
    /// <summary>Per-batch standard deviations (must be &gt; 0).</summary>
    public float[] Scale { get; }

    /// <inheritdoc />
    public override int BatchSize => Loc.Length;
    /// <inheritdoc />
    public override int EventSize => 1;
    /// <inheritdoc />
    public override IConstraint Support => RealConstraint.Instance;
    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Build a normal distribution from a per-batch <paramref name="loc"/> and <paramref name="scale"/>.</summary>
    public NormalDistribution(float[] loc, float[] scale)
    {
        if (loc == null) throw new ArgumentNullException(nameof(loc));
        if (scale == null) throw new ArgumentNullException(nameof(scale));
        if (loc.Length != scale.Length)
            throw new ArgumentException("loc and scale must have the same length.");
        for (int i = 0; i < scale.Length; i++)
            if (!(scale[i] > 0f)) throw new ArgumentException("scale must be > 0.");
        Loc = loc; Scale = scale;
    }

    /// <summary>Build a scalar (single-element batch) normal.</summary>
    public NormalDistribution(float loc, float scale) : this(new[] { loc }, new[] { scale }) { }

    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);

    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            x[i] = Loc[i] + Scale[i] * (float)Gaussian(rng);
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
            float z = (value[i] - Loc[i]) / Scale[i];
            lp[i] = -0.5f * (z * z + LogTwoPi) - MathF.Log(Scale[i]);
        }
        return lp;
    }

    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        const float HalfLogTwoPiE = 1.41893853321f; // 0.5·log(2πe)
        for (int i = 0; i < BatchSize; i++) h[i] = HalfLogTwoPiE + MathF.Log(Scale[i]);
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
            for (int i = 0; i < BatchSize; i++) v[i] = Scale[i] * Scale[i];
            return v;
        }
    }
    /// <inheritdoc />
    public override float[] StdDev => (float[])Scale.Clone();

    /// <inheritdoc />
    public override float[] Cdf(float[] value)
    {
        EnsureValueShape(value);
        var c = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            c[i] = SpecialFunctions.NormalCdf((value[i] - Loc[i]) / Scale[i]);
        return c;
    }

    /// <inheritdoc />
    public override float[] Icdf(float[] probability)
    {
        EnsureValueShape(probability);
        var x = new float[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            x[i] = Loc[i] + Scale[i] * SpecialFunctions.NormalIcdf(probability[i]);
        return x;
    }

    /// <summary>Box-Muller standard-normal sampler. Used by every distribution that needs Gaussians.</summary>
    internal static double Gaussian(Random rng)
    {
        double u1 = rng.NextDouble(); double u2 = rng.NextDouble();
        if (u1 < 1e-12) u1 = 1e-12;
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
