using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Data;
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

    private sealed class Range : IDataset<int>, IIterableDataset<int>
    {
        public Range(int count) => Count = count;
        public int Count { get; }
        public int this[int index] => index;
        public IEnumerator<int> GetEnumerator() => Enumerable.Range(0, Count).GetEnumerator();
    }

    /// <summary>The reservoir shuffle and the random split draw unseeded from the secure generator too (#1555).</summary>
    [Fact]
    public void UnseededShufflePipesAndRandomSplitsBuiltTogetherDiffer()
    {
        var source = new Range(64);
        var shuffles = new HashSet<string>(StringComparer.Ordinal);
        var splits = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < 16; i++)
        {
            var e = source.Shuffle(bufferSize: 64).GetEnumerator();
            var order = new List<int>();
            while (e.MoveNext()) order.Add(e.Current);
            Assert.Equal(Enumerable.Range(0, 64), order.OrderBy(x => x));
            shuffles.Add(string.Join(",", order));

            var first = source.RandomSplit(new[] { 32, 32 })[0];
            splits.Add(string.Join(",", Enumerable.Range(0, first.Count).Select(k => first[k])));
        }
        Assert.Equal(16, shuffles.Count);
        Assert.Equal(16, splits.Count);
    }

    [Fact]
    public void SeededShufflePipesRandomSplitsAndRandomPruningReproduce()
    {
        var source = new Range(40);
        string Shuffled()
        {
            var e = source.Shuffle(bufferSize: 16, seed: 9).GetEnumerator();
            var order = new List<int>();
            while (e.MoveNext()) order.Add(e.Current);
            return string.Join(",", order);
        }
        Assert.Equal(Shuffled(), Shuffled());

        string Split()
        {
            var part = source.RandomSplit(new[] { 25, 15 }, seed: 9)[1];
            return string.Join(",", Enumerable.Range(0, part.Count).Select(k => part[k]));
        }
        Assert.Equal(Split(), Split());

        var weights = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 8, 3 });
        var a = AiDotNet.Tensors.NN.Pruning.Pruning.RandomStructured(weights, 0.5, dim: 0, seed: 3);
        var b = AiDotNet.Tensors.NN.Pruning.Pruning.RandomStructured(weights, 0.5, dim: 0, seed: 3);
        int kept = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(a.IsKept(i), b.IsKept(i));
            if (a.IsKept(i)) kept++;
        }
        // Half of the 8 rows (3 elements each) are pruned.
        Assert.Equal(12, kept);
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
