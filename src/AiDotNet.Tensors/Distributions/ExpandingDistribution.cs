using System;
using AiDotNet.Tensors.Distributions.Constraints;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Wraps an existing distribution and broadcasts its batch dimension to a larger size.
/// Useful when a single set of parameters needs to be reused across many simulations.
/// Mirrors PyTorch <c>Distribution.expand(batch_shape)</c>.
///
/// Parameters are SHARED across the expanded batch but each repeated block draws
/// independently — calling <see cref="Sample"/> / <see cref="RSample"/> invokes the
/// underlying base sampler once per repeat factor so Monte-Carlo estimators see fresh
/// (i.i.d.) samples rather than copies of a single draw.
/// </summary>
public sealed class ExpandingDistribution : DistributionBase
{
    /// <summary>Underlying base distribution.</summary>
    public IDistribution Base { get; }
    /// <summary>Expanded batch size.</summary>
    public override int BatchSize { get; }
    /// <inheritdoc />
    public override int EventSize => Base.EventSize;
    /// <inheritdoc />
    public override IConstraint Support => Base.Support;
    /// <inheritdoc />
    public override bool HasRSample => Base.HasRSample;

    /// <summary>Build an expanded view of <paramref name="@base"/> with new batch size <paramref name="batchSize"/>.
    /// Must be a multiple of the base's batch size so each base param is replicated an integer number of times.</summary>
    public ExpandingDistribution(IDistribution @base, int batchSize)
    {
        Base = @base ?? throw new ArgumentNullException(nameof(@base));
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (batchSize % @base.BatchSize != 0)
            throw new ArgumentException(
                $"new batchSize ({batchSize}) must be a multiple of base.BatchSize ({@base.BatchSize}).");
        BatchSize = batchSize;
    }

    private int RepeatFactor => BatchSize / Base.BatchSize;

    private float[] Replicate(float[] perBaseBatch)
    {
        int baseLen = perBaseBatch.Length;
        int rep = RepeatFactor;
        var dst = new float[baseLen * rep];
        for (int r = 0; r < rep; r++)
            Array.Copy(perBaseBatch, 0, dst, r * baseLen, baseLen);
        return dst;
    }

    /// <inheritdoc />
    public override float[] Sample(Random rng)
    {
        // PyTorch parity: expand SHARES parameters but draws fresh samples for each
        // expanded block. Replicating the same draw across the expanded batch would
        // break Monte-Carlo estimators that rely on independent samples.
        int blockLen = Base.BatchSize * Base.EventSize;
        var dst = new float[BatchSize * EventSize];
        for (int r = 0; r < RepeatFactor; r++)
        {
            var sample = Base.Sample(rng);
            Array.Copy(sample, 0, dst, r * blockLen, blockLen);
        }
        return dst;
    }
    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        int blockLen = Base.BatchSize * Base.EventSize;
        var dst = new float[BatchSize * EventSize];
        for (int r = 0; r < RepeatFactor; r++)
        {
            var sample = Base.RSample(rng);
            Array.Copy(sample, 0, dst, r * blockLen, blockLen);
        }
        return dst;
    }

    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        // Reduce the expanded value back to base shape by summing matching slices? No — log_prob
        // is per batch element. We pull the first base.BatchSize·EventSize entries to evaluate,
        // then replicate. This matches PyTorch's expand semantics where the parameters are shared.
        int baseLen = Base.BatchSize * Base.EventSize;
        var slice = new float[baseLen];
        Array.Copy(value, 0, slice, 0, baseLen);
        var lpBase = Base.LogProb(slice);
        var lp = new float[BatchSize];
        int rep = RepeatFactor;
        for (int r = 0; r < rep; r++)
            Array.Copy(lpBase, 0, lp, r * Base.BatchSize, Base.BatchSize);
        // For repeats r > 0, recompute against the corresponding value slice for correctness.
        for (int r = 1; r < rep; r++)
        {
            Array.Copy(value, r * baseLen, slice, 0, baseLen);
            var lpr = Base.LogProb(slice);
            Array.Copy(lpr, 0, lp, r * Base.BatchSize, Base.BatchSize);
        }
        return lp;
    }

    /// <inheritdoc />
    public override float[] Entropy() => Replicate(Base.Entropy());
    /// <inheritdoc />
    public override float[] Mean => Replicate(Base.Mean);
    /// <inheritdoc />
    public override float[] Variance => Replicate(Base.Variance);
}
