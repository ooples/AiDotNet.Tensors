using System;
using AiDotNet.Tensors.Distributions.Constraints;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Default implementations for the boilerplate parts of <see cref="IDistribution"/>:
/// <see cref="StdDev"/> derives from <see cref="Variance"/>, <see cref="RSample"/>
/// throws when <see cref="HasRSample"/> is false, CDF/ICDF default to throwing for
/// distributions without a closed-form expression.
/// </summary>
public abstract class DistributionBase : IDistribution
{
    /// <inheritdoc />
    public abstract int BatchSize { get; }
    /// <inheritdoc />
    public abstract int EventSize { get; }
    /// <inheritdoc />
    public abstract IConstraint Support { get; }
    /// <inheritdoc />
    public virtual bool HasRSample => false;

    /// <inheritdoc />
    public abstract float[] Sample(Random rng);

    /// <inheritdoc />
    public virtual float[] RSample(Random rng) =>
        throw new NotSupportedException(GetType().Name + " is not reparameterisable.");

    /// <inheritdoc />
    public abstract float[] LogProb(float[] value);

    /// <inheritdoc />
    public abstract float[] Entropy();

    /// <inheritdoc />
    public abstract float[] Mean { get; }
    /// <inheritdoc />
    public abstract float[] Variance { get; }

    /// <inheritdoc />
    public virtual float[] StdDev
    {
        get
        {
            var v = Variance;
            var s = new float[v.Length];
            for (int i = 0; i < v.Length; i++) s[i] = MathF.Sqrt(v[i]);
            return s;
        }
    }

    /// <inheritdoc />
    public virtual float[] Cdf(float[] value) =>
        throw new NotSupportedException(GetType().Name + " has no closed-form CDF.");

    /// <inheritdoc />
    public virtual float[] Icdf(float[] probability) =>
        throw new NotSupportedException(GetType().Name + " has no closed-form inverse CDF.");

    /// <summary>Validate that <paramref name="value"/>'s length is compatible with this distribution's shape.</summary>
    protected void EnsureValueShape(float[] value)
    {
        int expected = BatchSize * EventSize;
        if (value.Length != expected)
            throw new ArgumentException(
                $"value length {value.Length} != BatchSize·EventSize {expected}.");
    }
}
