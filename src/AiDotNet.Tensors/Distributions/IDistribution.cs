using System;
using AiDotNet.Tensors.Distributions.Constraints;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Common contract for every probability distribution in the tensor library.
///
/// Mirrors the shape of <c>torch.distributions.Distribution</c>: one instance
/// represents a (possibly batched) family of distributions sharing a common
/// parameterisation. <see cref="BatchSize"/> is the number of independent
/// distributions packed in this instance, and <see cref="EventSize"/> is the
/// dimension of a single sample (1 for scalar distributions, N for an
/// N-dimensional MVN, etc.). All array shapes follow row-major
/// <c>[batch, event]</c> layout.
/// </summary>
public interface IDistribution
{
    /// <summary>Number of independent distributions in this batch.</summary>
    int BatchSize { get; }

    /// <summary>Length of a single sample. Scalar distributions have <c>EventSize == 1</c>.</summary>
    int EventSize { get; }

    /// <summary>Set of values at which the distribution puts non-zero probability mass / density.</summary>
    IConstraint Support { get; }

    /// <summary>True if <see cref="RSample"/> produces values whose dependence on the
    /// distribution parameters is differentiable (i.e. the reparameterisation trick applies).</summary>
    bool HasRSample { get; }

    /// <summary>Draw a single (batch of) samples without tracking gradients.</summary>
    float[] Sample(Random rng);

    /// <summary>Draw a single (batch of) reparameterised samples. Throws when <see cref="HasRSample"/> is false.</summary>
    float[] RSample(Random rng);

    /// <summary>Log probability mass / density at the supplied values. Length = <c>BatchSize · EventSize</c>.</summary>
    float[] LogProb(float[] value);

    /// <summary>Differential entropy (continuous) or Shannon entropy (discrete), per batch element. Length = <c>BatchSize</c>.</summary>
    float[] Entropy();

    /// <summary>Mean per batch element, flattened. Length = <c>BatchSize · EventSize</c>.</summary>
    float[] Mean { get; }

    /// <summary>Variance per batch element, flattened. Length = <c>BatchSize · EventSize</c>.</summary>
    float[] Variance { get; }

    /// <summary>Standard deviation per batch element. Length = <c>BatchSize · EventSize</c>.</summary>
    float[] StdDev { get; }

    /// <summary>Cumulative distribution function (univariate only). Length = <c>BatchSize</c>.</summary>
    float[] Cdf(float[] value);

    /// <summary>Inverse CDF (univariate only). <paramref name="probability"/> in (0, 1).</summary>
    float[] Icdf(float[] probability);
}
