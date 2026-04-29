using System;

namespace AiDotNet.Tensors.Distributions.Constraints;

/// <summary>
/// A constraint describes the set of values a variable can take. Constraints are used
/// by <see cref="IDistribution.Support"/> to advertise where a sample can land, by
/// <see cref="Transforms.ITransform"/> to describe its <c>codomain</c>, and by validators
/// to bounds-check user-supplied parameters.
/// </summary>
public interface IConstraint
{
    /// <summary>Returns true at indices where the constraint is satisfied.</summary>
    bool[] Check(float[] values);
}

/// <summary>Real line: every finite value satisfies the constraint.</summary>
public sealed class RealConstraint : IConstraint
{
    /// <summary>Singleton instance.</summary>
    public static readonly RealConstraint Instance = new RealConstraint();
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        var ok = new bool[values.Length];
        for (int i = 0; i < values.Length; i++) ok[i] = !float.IsNaN(values[i]) && !float.IsInfinity(values[i]);
        return ok;
    }
}

/// <summary>Strictly positive: x &gt; 0.</summary>
public sealed class PositiveConstraint : IConstraint
{
    /// <summary>Singleton instance.</summary>
    public static readonly PositiveConstraint Instance = new PositiveConstraint();
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        var ok = new bool[values.Length];
        for (int i = 0; i < values.Length; i++) ok[i] = values[i] > 0f;
        return ok;
    }
}

/// <summary>Non-negative: x ≥ 0.</summary>
public sealed class NonNegativeConstraint : IConstraint
{
    /// <summary>Singleton instance.</summary>
    public static readonly NonNegativeConstraint Instance = new NonNegativeConstraint();
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        var ok = new bool[values.Length];
        for (int i = 0; i < values.Length; i++) ok[i] = values[i] >= 0f;
        return ok;
    }
}

/// <summary>Open unit interval: 0 &lt; x &lt; 1.</summary>
public sealed class UnitIntervalConstraint : IConstraint
{
    /// <summary>Singleton instance.</summary>
    public static readonly UnitIntervalConstraint Instance = new UnitIntervalConstraint();
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        var ok = new bool[values.Length];
        for (int i = 0; i < values.Length; i++) ok[i] = values[i] > 0f && values[i] < 1f;
        return ok;
    }
}

/// <summary>Closed interval [low, high].</summary>
public sealed class IntervalConstraint : IConstraint
{
    /// <summary>Lower bound, inclusive.</summary>
    public float Low { get; }
    /// <summary>Upper bound, inclusive.</summary>
    public float High { get; }
    /// <summary>Build a closed interval constraint.</summary>
    public IntervalConstraint(float low, float high)
    {
        if (high < low) throw new ArgumentException("high must be >= low.");
        Low = low; High = high;
    }
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        var ok = new bool[values.Length];
        for (int i = 0; i < values.Length; i++) ok[i] = values[i] >= Low && values[i] <= High;
        return ok;
    }
}

/// <summary>Half-open less-than constraint: x &lt; upper.</summary>
public sealed class LessThanConstraint : IConstraint
{
    /// <summary>Strict upper bound.</summary>
    public float Upper { get; }
    /// <summary>Build a strict-less-than constraint.</summary>
    public LessThanConstraint(float upper) { Upper = upper; }
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        var ok = new bool[values.Length];
        for (int i = 0; i < values.Length; i++) ok[i] = values[i] < Upper;
        return ok;
    }
}

/// <summary>Half-open greater-than constraint: x &gt; lower.</summary>
public sealed class GreaterThanConstraint : IConstraint
{
    /// <summary>Strict lower bound.</summary>
    public float Lower { get; }
    /// <summary>Build a strict-greater-than constraint.</summary>
    public GreaterThanConstraint(float lower) { Lower = lower; }
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        var ok = new bool[values.Length];
        for (int i = 0; i < values.Length; i++) ok[i] = values[i] > Lower;
        return ok;
    }
}

/// <summary>Integer interval [low, high] inclusive.</summary>
public sealed class IntegerIntervalConstraint : IConstraint
{
    /// <summary>Lower bound, inclusive.</summary>
    public int Low { get; }
    /// <summary>Upper bound, inclusive.</summary>
    public int High { get; }
    /// <summary>Build an integer interval constraint.</summary>
    public IntegerIntervalConstraint(int low, int high)
    {
        if (high < low) throw new ArgumentException("high must be >= low.");
        Low = low; High = high;
    }
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        var ok = new bool[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            var v = values[i];
            ok[i] = v == MathF.Floor(v) && v >= Low && v <= High;
        }
        return ok;
    }
}

/// <summary>Non-negative integers: x ≥ 0 and x is a whole, finite number.</summary>
public sealed class NonNegativeIntegerConstraint : IConstraint
{
    /// <summary>Singleton instance.</summary>
    public static readonly NonNegativeIntegerConstraint Instance = new NonNegativeIntegerConstraint();
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        var ok = new bool[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            var v = values[i];
            ok[i] = !float.IsNaN(v) && !float.IsInfinity(v) && v >= 0f && v == MathF.Floor(v);
        }
        return ok;
    }
}

/// <summary>One-hot vector of length K: exactly one entry equals 1 and the rest equal 0.</summary>
public sealed class OneHotConstraint : IConstraint
{
    /// <summary>Length of each one-hot vector.</summary>
    public int K { get; }
    /// <summary>Build a one-hot constraint of width K.</summary>
    public OneHotConstraint(int k)
    {
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        K = k;
    }
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        if (values.Length % K != 0)
            throw new ArgumentException($"values.Length ({values.Length}) must be a multiple of K ({K}).");
        int batch = values.Length / K;
        var ok = new bool[batch];
        for (int b = 0; b < batch; b++)
        {
            int hits = 0; bool good = true;
            for (int i = 0; i < K; i++)
            {
                var v = values[b * K + i];
                if (v == 1f) hits++;
                else if (v != 0f) { good = false; break; }
            }
            ok[b] = good && hits == 1;
        }
        return ok;
    }
}

/// <summary>Non-negative integer count vectors that sum to a fixed total per batch row.
/// If <see cref="Totals"/> is null the per-row sum is unconstrained; otherwise each row's
/// sum must equal the corresponding entry. Used by <c>MultinomialDistribution.Support</c>.</summary>
public sealed class IntegerSimplexConstraint : IConstraint
{
    /// <summary>Length of each count vector.</summary>
    public int K { get; }
    /// <summary>Per-batch required totals (null = no sum constraint).</summary>
    public int[]? Totals { get; }
    /// <summary>Build an integer-simplex constraint.</summary>
    public IntegerSimplexConstraint(int k, int[]? totals = null)
    {
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        K = k; Totals = totals;
    }
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        if (values.Length % K != 0)
            throw new ArgumentException($"values.Length ({values.Length}) must be a multiple of K ({K}).");
        int batch = values.Length / K;
        if (Totals != null && Totals.Length != batch)
            throw new ArgumentException("Totals length must equal batch count.");
        var ok = new bool[batch];
        for (int b = 0; b < batch; b++)
        {
            long sum = 0; bool good = true;
            for (int i = 0; i < K; i++)
            {
                var v = values[b * K + i];
                if (float.IsNaN(v) || float.IsInfinity(v) || v < 0f || v != MathF.Floor(v))
                { good = false; break; }
                sum += (long)v;
            }
            ok[b] = good && (Totals == null || sum == Totals[b]);
        }
        return ok;
    }
}

/// <summary>Probability simplex: x_i ≥ 0 and sum_i x_i = 1 (within tolerance).</summary>
public sealed class SimplexConstraint : IConstraint
{
    /// <summary>Length of each simplex element (i.e. the number of categories).</summary>
    public int EventSize { get; }
    /// <summary>Tolerance on the sum-to-one check.</summary>
    public float Tolerance { get; }
    /// <summary>Build a simplex constraint.</summary>
    public SimplexConstraint(int eventSize, float tolerance = 1e-5f)
    {
        if (eventSize <= 0) throw new ArgumentOutOfRangeException(nameof(eventSize));
        EventSize = eventSize; Tolerance = tolerance;
    }
    /// <inheritdoc />
    public bool[] Check(float[] values)
    {
        if (values.Length % EventSize != 0)
            throw new ArgumentException($"values.Length ({values.Length}) must be a multiple of EventSize ({EventSize}).");
        int batch = values.Length / EventSize;
        var ok = new bool[batch];
        for (int b = 0; b < batch; b++)
        {
            float sum = 0f; bool good = true;
            for (int i = 0; i < EventSize; i++)
            {
                var v = values[b * EventSize + i];
                if (v < 0f) { good = false; break; }
                sum += v;
            }
            ok[b] = good && MathF.Abs(sum - 1f) <= Tolerance;
        }
        return ok;
    }
}
