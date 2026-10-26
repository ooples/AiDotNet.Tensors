using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Data.Samplers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Data;

/// <summary>
/// Unseeded samplers draw from RandomHelper's secure generator, not a clock-seeded <c>new Random()</c> (#1555). On
/// .NET Framework the clock seed is <c>Environment.TickCount</c>, so samplers built within one tick -- a loader's
/// workers, created together -- shuffled identically. Seeded samplers keep the sequence <c>new Random(seed)</c> gave.
/// </summary>
public class SamplerSeedingTests
{
    [Fact]
    public void UnseededSamplersBuiltTogetherShuffleIndependently()
    {
        var orders = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < 16; i++)
            orders.Add(string.Join(",", new RandomSampler(64).GetIndices()));
        // 64! orders: sixteen independent shuffles collide with negligible probability, clock-seeded ones always did.
        Assert.Equal(16, orders.Count);
    }

    [Fact]
    public void UnseededWeightedAndSubsetSamplersBuiltTogetherDrawIndependently()
    {
        var weights = Enumerable.Repeat(1.0, 64).ToArray();
        var subset = Enumerable.Range(0, 64).ToArray();
        var weighted = new HashSet<string>(StringComparer.Ordinal);
        var subsets = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < 16; i++)
        {
            weighted.Add(string.Join(",", new WeightedRandomSampler(weights, 64).GetIndices()));
            subsets.Add(string.Join(",", new SubsetRandomSampler(subset).GetIndices()));
        }
        Assert.Equal(16, weighted.Count);
        Assert.Equal(16, subsets.Count);
    }

    /// <summary>A seeded sampler's order is the one <c>new Random(foldedSeed)</c> produced, so seeds stay reproducible.</summary>
    [Fact]
    public void ASeededSamplerKeepsTheSequenceOfTheSameSeededRandom()
    {
        const int Count = 32, Seed = 17;
        // The sampler folds its epoch into the seed; epoch 0 leaves Seed * 0x9E3779B1.
        var reference = new Random(unchecked((int)(Seed * 0x9E3779B1u)));
        var perm = Enumerable.Range(0, Count).ToArray();
        for (int i = Count - 1; i > 0; i--)
        {
            int j = reference.Next(i + 1);
            (perm[i], perm[j]) = (perm[j], perm[i]);
        }
        Assert.Equal(perm, new RandomSampler(Count, seed: Seed).GetIndices().ToArray());
    }
}
