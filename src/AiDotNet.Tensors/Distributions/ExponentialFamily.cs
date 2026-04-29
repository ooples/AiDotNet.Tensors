using System;
using AiDotNet.Tensors.Distributions.Constraints;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Abstract base for exponential-family distributions with density of the form
///   f(x; η) = h(x) · exp(η·T(x) − A(η))
/// where η are the natural parameters, T(x) is the sufficient statistic, A(η) is the
/// log-partition / cumulant function, and h(x) is the carrier measure.
///
/// Provides analytical helpers used by entropy/KL formulas:
///  * <see cref="NaturalParameters"/> — η
///  * <see cref="LogNormalizer"/>      — A(η)
///  * <see cref="MeanCarrierMeasure"/> — E[log h(X)]
///
/// Concrete subclasses still implement <see cref="DistributionBase.Sample"/>,
/// <see cref="DistributionBase.LogProb"/>, etc. — this base just provides the
/// extra hooks for code that wants to exploit exponential-family structure
/// (e.g. natural-gradient optimization, conjugate-prior updates).
/// </summary>
public abstract class ExponentialFamily : DistributionBase
{
    /// <summary>Natural parameters η. Layout <c>[batch, num_params]</c> (subclass-defined).</summary>
    public abstract float[] NaturalParameters { get; }

    /// <summary>Log-partition A(η) per batch element. Length = <see cref="DistributionBase.BatchSize"/>.</summary>
    public abstract float[] LogNormalizer { get; }

    /// <summary>Sufficient statistic T(x) for a supplied <paramref name="value"/>.
    /// Layout <c>[batch, num_stats]</c>.</summary>
    public abstract float[] SufficientStatistic(float[] value);

    /// <summary>Mean of log h(X) — usually a constant per batch element.</summary>
    public abstract float[] MeanCarrierMeasure { get; }
}
