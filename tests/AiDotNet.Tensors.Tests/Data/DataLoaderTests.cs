// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Data;
using AiDotNet.Tensors.Data.Samplers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Data;

/// <summary>
/// Acceptance tests for issue #216 — Dataset / DataLoader / Samplers /
/// DataPipes / CheckpointableLoader.
/// </summary>
public class DataLoaderTests
{
    // ── Datasets ────────────────────────────────────────────────────────

    [Fact]
    public void TensorDataset_PerSampleSlice_ReturnsRankMinusOneTensor()
    {
        var x = new Tensor<float>(new[] { 4, 3 });
        for (int i = 0; i < x.Length; i++) x.AsWritableSpan()[i] = i;
        var ds = new TensorDataset<float>(x);
        Assert.Equal(4, ds.Count);

        var sample = ds[1];
        Assert.Single(sample);
        Assert.Equal(new[] { 3 }, sample[0]._shape);
        Assert.Equal(3f, sample[0].AsSpan()[0]);
        Assert.Equal(5f, sample[0].AsSpan()[2]);
    }

    [Fact]
    public void TensorDataset_MismatchedLeadingAxis_Throws()
    {
        var a = new Tensor<float>(new[] { 4 });
        var b = new Tensor<float>(new[] { 3 });
        Assert.Throws<ArgumentException>(() => new TensorDataset<float>(a, b));
    }

    [Fact]
    public void Subset_OnlySeesProvidedIndices()
    {
        var parent = new SquareDataset(10);
        var subset = new Subset<int>(parent, new[] { 9, 3, 7 });
        Assert.Equal(3, subset.Count);
        Assert.Equal(81, subset[0]);
        Assert.Equal(9, subset[1]);
        Assert.Equal(49, subset[2]);
    }

    [Fact]
    public void ConcatDataset_WalksDatasetsInOrder()
    {
        var a = new SquareDataset(3); // [0,1,4]
        var b = new SquareDataset(2); // [0,1]
        var concat = new ConcatDataset<int>(a, b);
        Assert.Equal(5, concat.Count);
        Assert.Equal(0, concat[0]);
        Assert.Equal(1, concat[1]);
        Assert.Equal(4, concat[2]);
        Assert.Equal(0, concat[3]);
        Assert.Equal(1, concat[4]);
    }

    [Fact]
    public void RandomSplit_PartitionsParent()
    {
        var ds = new SquareDataset(10);
        var splits = ds.RandomSplit(new[] { 6, 3, 1 }, seed: 42);
        Assert.Equal(3, splits.Length);
        Assert.Equal(6, splits[0].Count);
        Assert.Equal(3, splits[1].Count);
        Assert.Equal(1, splits[2].Count);

        // Partition: every parent index appears exactly once across the splits.
        var seen = new HashSet<int>();
        foreach (var s in splits)
            for (int i = 0; i < s.Count; i++)
            {
                int v = s[i];
                int rootIdx = (int)System.Math.Round(System.Math.Sqrt(v));
                Assert.True(seen.Add(rootIdx), $"Index {rootIdx} appears in two splits.");
            }
        Assert.Equal(10, seen.Count);
    }

    [Fact]
    public void RandomSplit_LengthsMustSumToCount()
    {
        var ds = new SquareDataset(5);
        Assert.Throws<ArgumentException>(() => ds.RandomSplit(new[] { 2, 2 }));
    }

    [Fact]
    public void CachedDataset_RetrievesViaCache()
    {
        var counted = new CountingDataset(10);
        var cached = new CachedDataset<int>(counted, capacity: 4);

        // First access — miss + parent fetch.
        Assert.Equal(0, cached[0]);
        Assert.Equal(1, counted.AccessCount);

        // Second access — hit, no extra parent fetch.
        Assert.Equal(0, cached[0]);
        Assert.Equal(1, counted.AccessCount);

        // Fill cache + evict the oldest. Indexer reads can't be plain
        // statements in C#; discard explicitly.
        _ = cached[1]; _ = cached[2]; _ = cached[3]; _ = cached[4]; // capacity=4, so 0 is evicted by 4
        Assert.Equal(5, counted.AccessCount);
        Assert.Equal(0, cached[0]); // re-fetched
        Assert.Equal(6, counted.AccessCount);
    }

    // ── Samplers ────────────────────────────────────────────────────────

    [Fact]
    public void SequentialSampler_YieldsZeroToCount()
    {
        var s = new SequentialSampler(5);
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, s.GetIndices().ToArray());
    }

    [Fact]
    public void RandomSampler_SeedReproducesShuffle()
    {
        var s1 = new RandomSampler(20, seed: 17);
        var s2 = new RandomSampler(20, seed: 17);
        Assert.Equal(s1.GetIndices().ToArray(), s2.GetIndices().ToArray());
    }

    [Fact]
    public void RandomSampler_DifferentEpochs_DifferentOrders()
    {
        var s = new RandomSampler(50, seed: 1);
        s.SetEpoch(0);
        var ep0 = s.GetIndices().ToArray();
        s.SetEpoch(1);
        var ep1 = s.GetIndices().ToArray();
        Assert.NotEqual(ep0, ep1);
    }

    [Fact]
    public void RandomSampler_FullPermutation_NoRepeatsNoMisses()
    {
        var s = new RandomSampler(100, seed: 7);
        var seen = s.GetIndices().ToArray();
        Assert.Equal(100, seen.Length);
        Assert.Equal(100, seen.Distinct().Count());
    }

    [Fact]
    public void WeightedRandomSampler_SkewedDistribution()
    {
        // Weight index 0 100x heavier than the rest of 10. Over a large
        // sample, ~90% of draws should be index 0.
        var weights = new double[10];
        weights[0] = 100;
        for (int i = 1; i < 10; i++) weights[i] = 1;
        var s = new WeightedRandomSampler(weights, numSamples: 5000, seed: 3);
        int zeroCount = s.GetIndices().Count(i => i == 0);
        Assert.True(zeroCount > 4000, $"Expected >4000 zero draws, got {zeroCount}.");
    }

    [Fact]
    public void BatchSampler_DropLast()
    {
        var inner = new SequentialSampler(7);
        var bs = new BatchSampler(inner, batchSize: 3, dropLast: true);
        Assert.Equal(2, bs.Count);
        var batches = bs.GetBatches().ToArray();
        Assert.Equal(2, batches.Length);
        Assert.Equal(new[] { 0, 1, 2 }, batches[0].ToArray());
        Assert.Equal(new[] { 3, 4, 5 }, batches[1].ToArray());
    }

    [Fact]
    public void BatchSampler_KeepLast()
    {
        var inner = new SequentialSampler(7);
        var bs = new BatchSampler(inner, batchSize: 3, dropLast: false);
        Assert.Equal(3, bs.Count);
        var batches = bs.GetBatches().ToArray();
        Assert.Equal(new[] { 6 }, batches[2].ToArray());
    }

    [Fact]
    public void DistributedSampler_RanksSeeDisjointShards_UnionEqualsDataset()
    {
        const int N = 100, World = 4;
        var seen = new HashSet<int>();
        for (int rank = 0; rank < World; rank++)
        {
            var s = new DistributedSampler(N, World, rank, shuffle: true, seed: 42);
            foreach (var idx in s.GetIndices()) seen.Add(idx);
        }
        // With dropLast=false (default), padding may revisit a few low
        // indices, so we test "every original index appears at least once."
        for (int i = 0; i < N; i++) Assert.Contains(i, seen);
    }

    [Fact]
    public void DistributedSampler_PerRankCountStable()
    {
        const int N = 100, World = 4;
        var s = new DistributedSampler(N, World, rank: 0, shuffle: false);
        Assert.Equal(25, s.Count);
        Assert.Equal(25, s.GetIndices().Count());
    }

    // ── DataLoader ──────────────────────────────────────────────────────

    [Fact]
    public void DataLoader_Sequential_BatchesInOrder()
    {
        var x = new Tensor<float>(new float[] { 0, 1, 2, 3, 4, 5, 6, 7 }, new[] { 8 });
        var ds = new TensorDataset<float>(x);
        var loader = new DataLoader<Tensor<float>[], Tensor<float>>(
            ds,
            samples => Collators.Stack(samples.Select(s => s[0]).ToArray()),
            new DataLoaderOptions { BatchSize = 4 });

        var batches = loader.ToArray();
        Assert.Equal(2, batches.Length);
        Assert.Equal(new[] { 0f, 1, 2, 3 }, batches[0].AsSpan().ToArray());
        Assert.Equal(new[] { 4f, 5, 6, 7 }, batches[1].AsSpan().ToArray());
    }

    [Fact]
    public void DataLoader_Shuffle_SeedReproducesIterationOrder()
    {
        var x = new Tensor<float>(new float[] { 0, 1, 2, 3, 4, 5, 6, 7 }, new[] { 8 });
        var ds = new TensorDataset<float>(x);
        Tensor<float> Stack(IReadOnlyList<Tensor<float>[]> ss)
            => Collators.Stack(ss.Select(s => s[0]).ToArray());

        var run1 = new DataLoader<Tensor<float>[], Tensor<float>>(
            ds, Stack, new DataLoaderOptions { BatchSize = 2, Shuffle = true, Seed = 5 }).ToArray();
        var run2 = new DataLoader<Tensor<float>[], Tensor<float>>(
            ds, Stack, new DataLoaderOptions { BatchSize = 2, Shuffle = true, Seed = 5 }).ToArray();

        for (int b = 0; b < run1.Length; b++)
            Assert.Equal(run1[b].AsSpan().ToArray(), run2[b].AsSpan().ToArray());
    }

    [Fact]
    public void DataLoader_DropLast_DropsPartial()
    {
        var x = new Tensor<float>(new float[] { 0, 1, 2, 3, 4 }, new[] { 5 });
        var ds = new TensorDataset<float>(x);
        var loader = new DataLoader<Tensor<float>[], Tensor<float>>(
            ds,
            ss => Collators.Stack(ss.Select(s => s[0]).ToArray()),
            new DataLoaderOptions { BatchSize = 2, DropLast = true });
        Assert.Equal(2, loader.ToArray().Length);
    }

    [Fact]
    public void DataLoader_AsyncWorkers_SameOutputAsSync()
    {
        var x = new Tensor<float>(new float[16], new[] { 16 });
        for (int i = 0; i < x.Length; i++) x.AsWritableSpan()[i] = i;
        var ds = new TensorDataset<float>(x);

        Tensor<float> Stack(IReadOnlyList<Tensor<float>[]> ss)
            => Collators.Stack(ss.Select(s => s[0]).ToArray());

        var sync = new DataLoader<Tensor<float>[], Tensor<float>>(
            ds, Stack, new DataLoaderOptions { BatchSize = 4 }).ToArray();
        var async4 = new DataLoader<Tensor<float>[], Tensor<float>>(
            ds, Stack, new DataLoaderOptions { BatchSize = 4, NumWorkers = 4 }).ToArray();

        Assert.Equal(sync.Length, async4.Length);
        // Async workers preserve the sampler's index order — order must match.
        for (int b = 0; b < sync.Length; b++)
            Assert.Equal(sync[b].AsSpan().ToArray(), async4[b].AsSpan().ToArray());
    }

    // ── DataPipes ───────────────────────────────────────────────────────

    [Fact]
    public void DataPipes_MapFilterBatch()
    {
        var src = new RangeIterableDataset(10);
        var pipeline = src.Map(x => x * 2).Filter(x => x > 5).Batch(2);
        var batches = pipeline.AsEnumerable().ToList();
        // Source: 0..9. Map: 0,2,4,6,8,10,12,14,16,18. Filter>5: 6,8,10,12,14,16,18.
        // Batch 2: [6,8],[10,12],[14,16],[18].
        Assert.Equal(4, batches.Count);
        Assert.Equal(new[] { 6, 8 }, batches[0].ToArray());
        Assert.Equal(new[] { 18 }, batches[3].ToArray());
    }

    [Fact]
    public void DataPipes_ShufflePreservesElements()
    {
        var src = new RangeIterableDataset(20);
        var shuffled = src.Shuffle(bufferSize: 8, seed: 1).AsEnumerable().ToList();
        Assert.Equal(20, shuffled.Count);
        Assert.Equal(20, shuffled.Distinct().Count());
    }

    [Fact]
    public void DataPipes_Zip_StopsAtShorterStream()
    {
        var a = new RangeIterableDataset(5);
        var b = new RangeIterableDataset(3);
        var zipped = a.Zip(b).AsEnumerable().ToList();
        Assert.Equal(3, zipped.Count);
        Assert.Equal((0, 0), zipped[0]);
        Assert.Equal((2, 2), zipped[2]);
    }

    [Fact]
    public void DataPipes_ShardingFilter_DistributesAcrossWorkers()
    {
        var src = new RangeIterableDataset(20);
        var shardR0 = src.ShardingFilter(rank: 0, worldSize: 4).AsEnumerable().ToList();
        var shardR1 = src.ShardingFilter(rank: 1, worldSize: 4).AsEnumerable().ToList();
        Assert.Equal(5, shardR0.Count);
        Assert.Equal(5, shardR1.Count);
        Assert.Equal(0, shardR0[0]);
        Assert.Equal(1, shardR1[0]);
        Assert.Empty(shardR0.Intersect(shardR1));
    }

    // ── CheckpointableLoader ────────────────────────────────────────────

    [Fact]
    public void CheckpointableLoader_ResumeMidEpoch_ContinuesIdentically()
    {
        var x = new Tensor<float>(new float[16], new[] { 16 });
        for (int i = 0; i < x.Length; i++) x.AsWritableSpan()[i] = i;
        var ds = new TensorDataset<float>(x);
        Tensor<float> Stack(IReadOnlyList<Tensor<float>[]> ss)
            => Collators.Stack(ss.Select(s => s[0]).ToArray());

        // Run 1 — full uninterrupted iteration.
        var full = new CheckpointableLoader<Tensor<float>[], Tensor<float>>(
            ds, Stack, new DataLoaderOptions { BatchSize = 2, Shuffle = true, Seed = 11 }).ToArray();

        // Run 2 — stop after 3 batches, snapshot, resume.
        var loader = new CheckpointableLoader<Tensor<float>[], Tensor<float>>(
            ds, Stack, new DataLoaderOptions { BatchSize = 2, Shuffle = true, Seed = 11 });
        var firstHalf = loader.Take(3).ToArray();
        var checkpoint = loader.StateDict();

        // Restore in a fresh loader.
        var resumed = new CheckpointableLoader<Tensor<float>[], Tensor<float>>(
            ds, Stack, new DataLoaderOptions { BatchSize = 2, Shuffle = true, Seed = 11 });
        resumed.LoadStateDict(checkpoint);
        var secondHalf = resumed.ToArray();

        Assert.Equal(full.Length, firstHalf.Length + secondHalf.Length);
        for (int i = 0; i < firstHalf.Length; i++)
            Assert.Equal(full[i].AsSpan().ToArray(), firstHalf[i].AsSpan().ToArray());
        for (int i = 0; i < secondHalf.Length; i++)
            Assert.Equal(full[firstHalf.Length + i].AsSpan().ToArray(), secondHalf[i].AsSpan().ToArray());
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    private sealed class SquareDataset : IDataset<int>
    {
        public SquareDataset(int n) { Count = n; }
        public int Count { get; }
        public int this[int i] => i * i;
    }

    private sealed class CountingDataset : IDataset<int>
    {
        public CountingDataset(int n) { Count = n; }
        public int Count { get; }
        public int AccessCount { get; private set; }
        public int this[int i] { get { AccessCount++; return i; } }
    }

    private sealed class RangeIterableDataset : IIterableDataset<int>
    {
        private readonly int _n;
        public RangeIterableDataset(int n) { _n = n; }
        public IEnumerator<int> GetEnumerator()
        {
            for (int i = 0; i < _n; i++) yield return i;
        }
    }
}

internal static class IterableDatasetExtensions
{
    /// <summary>Test-helper: iterate an iterable dataset as IEnumerable for LINQ.</summary>
    public static IEnumerable<T> AsEnumerable<T>(this IIterableDataset<T> ds)
    {
        using var e = ds.GetEnumerator();
        while (e.MoveNext()) yield return e.Current;
    }
}
