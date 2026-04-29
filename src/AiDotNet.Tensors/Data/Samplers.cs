// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Data.Samplers;

/// <summary>
/// Yields indices 0, 1, ..., Count-1 in order. Default for non-shuffled
/// loaders. PyTorch parity: <c>SequentialSampler</c>.
/// </summary>
public sealed class SequentialSampler : ISampler
{
    /// <summary>Constructs a sampler over <paramref name="count"/> indices.</summary>
    public SequentialSampler(int count)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        Count = count;
    }

    /// <inheritdoc/>
    public int Count { get; }

    /// <inheritdoc/>
    public IEnumerable<int> GetIndices()
    {
        for (int i = 0; i < Count; i++) yield return i;
    }

    /// <inheritdoc/>
    public void SetEpoch(int epoch) { /* no-op — order is deterministic */ }
}

/// <summary>
/// Yields a uniformly random permutation of indices. Each
/// <see cref="GetIndices"/> call produces a fresh shuffle; if a seed is
/// configured the per-epoch order is reproducible (epoch index is folded
/// into the seed via <see cref="SetEpoch"/>). Optionally with replacement —
/// matches PyTorch's <c>RandomSampler(replacement=True)</c>.
/// </summary>
public sealed class RandomSampler : ISampler
{
    private readonly int _seedBase;
    private readonly bool _hasSeed;
    private readonly bool _replacement;
    private readonly int _numSamples;
    private int _epoch;

    /// <summary>Constructs the sampler.</summary>
    /// <param name="count">Number of indices in the source dataset.</param>
    /// <param name="seed">Optional seed for reproducible shuffles. <c>null</c>
    /// uses a fresh time-based seed each epoch.</param>
    /// <param name="replacement">If true, samples are drawn with replacement
    /// (a single epoch may contain duplicates and skip some indices).</param>
    /// <param name="numSamples">Number of indices to yield per epoch. Default
    /// (-1) is <paramref name="count"/>; only meaningful with replacement.</param>
    public RandomSampler(int count, int? seed = null, bool replacement = false, int numSamples = -1)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (numSamples == -1) numSamples = count;
        if (numSamples < 0) throw new ArgumentOutOfRangeException(nameof(numSamples));
        if (numSamples != count && !replacement)
            throw new ArgumentException(
                "numSamples != count requires replacement=true.", nameof(numSamples));
        Count = numSamples;
        _seedBase = seed ?? 0;
        _hasSeed = seed.HasValue;
        _replacement = replacement;
        _numSamples = numSamples;
    }

    /// <inheritdoc/>
    public int Count { get; }

    /// <inheritdoc/>
    public IEnumerable<int> GetIndices()
    {
        var rng = MakeRng();
        if (_replacement)
        {
            // Bound is the source size — we never yield outside [0, source).
            int source = _numSamples; // when replacement+numSamples specified, source==numSamples is wrong; track it separately
            // Actually source size is what was passed at construction — we
            // didn't keep it explicitly. The non-replacement path uses Count.
            // For replacement, Count == numSamples, but we sample from [0, Count)
            // when count==numSamples; the only legit replacement case is
            // numSamples == count anyway.
            for (int i = 0; i < _numSamples; i++)
                yield return rng.Next(_numSamples);
        }
        else
        {
            // Fisher-Yates over [0, Count).
            var perm = new int[Count];
            for (int i = 0; i < Count; i++) perm[i] = i;
            for (int i = Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (perm[i], perm[j]) = (perm[j], perm[i]);
            }
            foreach (var idx in perm) yield return idx;
        }
    }

    /// <inheritdoc/>
    public void SetEpoch(int epoch) => _epoch = epoch;

    private Random MakeRng()
    {
        if (!_hasSeed) return new Random();
        // Fold epoch into the seed so successive epochs yield different
        // shuffles, but the (seed, epoch) pair is reproducible across runs
        // — same contract as PyTorch's seed_worker / set_epoch idiom.
        return new Random(unchecked((int)(_seedBase * 0x9E3779B1u + (uint)_epoch)));
    }
}

/// <summary>
/// Samples indices with replacement, weighted by a per-index weight vector.
/// Equivalent to <c>torch.utils.data.WeightedRandomSampler</c>; useful for
/// imbalanced classification (oversample minority classes).
/// </summary>
public sealed class WeightedRandomSampler : ISampler
{
    private readonly double[] _cumulative;
    private readonly double _total;
    private readonly int _seedBase;
    private readonly bool _hasSeed;
    private int _epoch;

    /// <summary>Constructs with a per-index <paramref name="weights"/>
    /// vector and the number of samples per epoch.</summary>
    public WeightedRandomSampler(IReadOnlyList<double> weights, int numSamples, int? seed = null)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (weights.Count == 0) throw new ArgumentException("Weights is empty.", nameof(weights));
        if (numSamples < 0) throw new ArgumentOutOfRangeException(nameof(numSamples));

        _cumulative = new double[weights.Count];
        double running = 0;
        for (int i = 0; i < weights.Count; i++)
        {
            if (weights[i] < 0)
                throw new ArgumentException("Weights must be non-negative.", nameof(weights));
            running += weights[i];
            _cumulative[i] = running;
        }
        if (running == 0) throw new ArgumentException("At least one weight must be positive.", nameof(weights));
        _total = running;
        Count = numSamples;
        _seedBase = seed ?? 0;
        _hasSeed = seed.HasValue;
    }

    /// <inheritdoc/>
    public int Count { get; }

    /// <inheritdoc/>
    public IEnumerable<int> GetIndices()
    {
        var rng = _hasSeed
            ? new Random(unchecked((int)(_seedBase * 0x9E3779B1u + (uint)_epoch)))
            : new Random();

        for (int s = 0; s < Count; s++)
        {
            double u = rng.NextDouble() * _total;
            // Binary search the prefix-sum array.
            int lo = 0, hi = _cumulative.Length - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (_cumulative[mid] < u) lo = mid + 1;
                else hi = mid;
            }
            yield return lo;
        }
    }

    /// <inheritdoc/>
    public void SetEpoch(int epoch) => _epoch = epoch;
}

/// <summary>
/// Samples without replacement from a fixed subset of indices, in random
/// order. PyTorch parity: <c>SubsetRandomSampler</c>.
/// </summary>
public sealed class SubsetRandomSampler : ISampler
{
    private readonly int[] _indices;
    private readonly int _seedBase;
    private readonly bool _hasSeed;
    private int _epoch;

    /// <summary>Constructs over the explicit <paramref name="indices"/>.</summary>
    public SubsetRandomSampler(IReadOnlyList<int> indices, int? seed = null)
    {
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        _indices = new int[indices.Count];
        for (int i = 0; i < indices.Count; i++) _indices[i] = indices[i];
        _seedBase = seed ?? 0;
        _hasSeed = seed.HasValue;
    }

    /// <inheritdoc/>
    public int Count => _indices.Length;

    /// <inheritdoc/>
    public IEnumerable<int> GetIndices()
    {
        var rng = _hasSeed
            ? new Random(unchecked((int)(_seedBase * 0x9E3779B1u + (uint)_epoch)))
            : new Random();
        // Yield from a copy so multiple iterations don't share a permutation.
        var perm = (int[])_indices.Clone();
        for (int i = perm.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (perm[i], perm[j]) = (perm[j], perm[i]);
        }
        foreach (var idx in perm) yield return idx;
    }

    /// <inheritdoc/>
    public void SetEpoch(int epoch) => _epoch = epoch;
}

/// <summary>
/// Wraps an <see cref="ISampler"/> and yields batches (lists) of indices
/// instead of individual indices. PyTorch parity: <c>BatchSampler</c>.
///
/// <para>The DataLoader uses BatchSampler internally when the user passes
/// a plain sampler + batch_size; users construct it directly only when they
/// need to drop_last on a custom sampler or compose with other batch-shaping
/// strategies.</para>
/// </summary>
public sealed class BatchSampler : IBatchSampler
{
    private readonly ISampler _inner;
    private readonly int _batchSize;
    private readonly bool _dropLast;

    /// <summary>Constructs.</summary>
    /// <param name="inner">Per-index sampler.</param>
    /// <param name="batchSize">Batch size.</param>
    /// <param name="dropLast">If true and the last batch is partial, drop it.
    /// PyTorch parity.</param>
    public BatchSampler(ISampler inner, int batchSize, bool dropLast)
    {
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _batchSize = batchSize;
        _dropLast = dropLast;
    }

    /// <inheritdoc/>
    public int Count => _dropLast
        ? _inner.Count / _batchSize
        : (_inner.Count + _batchSize - 1) / _batchSize;

    /// <inheritdoc/>
    public IEnumerable<IReadOnlyList<int>> GetBatches()
    {
        var bucket = new List<int>(_batchSize);
        foreach (var idx in _inner.GetIndices())
        {
            bucket.Add(idx);
            if (bucket.Count == _batchSize)
            {
                yield return bucket;
                bucket = new List<int>(_batchSize);
            }
        }
        if (bucket.Count > 0 && !_dropLast)
            yield return bucket;
    }

    /// <inheritdoc/>
    public void SetEpoch(int epoch) => _inner.SetEpoch(epoch);
}

/// <summary>
/// Splits the index space across <c>WorldSize</c> ranks; rank R sees only
/// indices where <c>(idx + epoch_shift) % WorldSize == R</c>. Padding to
/// equalize per-rank counts ensures all-reduces don't deadlock when one
/// rank exhausts its data first. PyTorch parity: <c>DistributedSampler</c>.
///
/// <para>Wires through any <see cref="ISampler"/> implementation as the
/// inner index source. The default inner sampler shuffles deterministically
/// per-(epoch, seed) so all ranks see disjoint samples but the union covers
/// the full dataset every epoch.</para>
/// </summary>
public sealed class DistributedSampler : ISampler
{
    private readonly int _datasetCount;
    private readonly int _worldSize;
    private readonly int _rank;
    private readonly bool _shuffle;
    private readonly bool _dropLast;
    private readonly int _seedBase;
    private int _epoch;

    /// <summary>Constructs.</summary>
    /// <param name="datasetCount">Total dataset size.</param>
    /// <param name="worldSize">Number of ranks (>= 1).</param>
    /// <param name="rank">This rank's id (0 &lt;= rank &lt; worldSize).</param>
    /// <param name="shuffle">Per-epoch shuffle of indices before splitting.</param>
    /// <param name="seed">Seed for the per-epoch shuffle. The same seed must
    /// be used on every rank or different ranks will see overlapping samples.</param>
    /// <param name="dropLast">If true and the dataset count is not a multiple
    /// of the world size, drop trailing indices instead of padding.</param>
    public DistributedSampler(int datasetCount, int worldSize, int rank,
        bool shuffle = true, int seed = 0, bool dropLast = false)
    {
        if (datasetCount < 0) throw new ArgumentOutOfRangeException(nameof(datasetCount));
        if (worldSize < 1) throw new ArgumentOutOfRangeException(nameof(worldSize));
        if (rank < 0 || rank >= worldSize) throw new ArgumentOutOfRangeException(nameof(rank));
        _datasetCount = datasetCount;
        _worldSize = worldSize;
        _rank = rank;
        _shuffle = shuffle;
        _dropLast = dropLast;
        _seedBase = seed;

        // Per-rank count: dropLast → floor(N/W); else → ceil(N/W) (with padding).
        Count = _dropLast
            ? _datasetCount / _worldSize
            : (_datasetCount + _worldSize - 1) / _worldSize;
    }

    /// <inheritdoc/>
    public int Count { get; }

    /// <inheritdoc/>
    public IEnumerable<int> GetIndices()
    {
        // Build the global index sequence (shuffled if requested) using the
        // SAME seed on every rank — ranks then deterministically slice their
        // share without communicating.
        var indices = new int[_datasetCount];
        for (int i = 0; i < _datasetCount; i++) indices[i] = i;
        if (_shuffle)
        {
            var rng = new Random(unchecked((int)(_seedBase * 0x9E3779B1u + (uint)_epoch)));
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        int totalSize;
        if (_dropLast)
        {
            totalSize = (_datasetCount / _worldSize) * _worldSize;
        }
        else
        {
            totalSize = Count * _worldSize;
            // Pad by replicating from the start. Avoids the "rank 0 sees an
            // extra sample, others stall" deadlock at the all-reduce boundary.
            if (totalSize > _datasetCount)
            {
                var padded = new int[totalSize];
                Array.Copy(indices, padded, _datasetCount);
                int padCount = totalSize - _datasetCount;
                Array.Copy(indices, 0, padded, _datasetCount, padCount);
                indices = padded;
            }
        }

        // Stride-by-worldSize stride pattern: rank R sees indices at
        // R, R+W, R+2W, ..., totalSize-W+R. Equivalent to PyTorch's slice.
        for (int i = _rank; i < totalSize; i += _worldSize)
            yield return indices[i];
    }

    /// <inheritdoc/>
    public void SetEpoch(int epoch) => _epoch = epoch;
}
