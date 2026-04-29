// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Data;

/// <summary>
/// Wraps one or more <see cref="Tensor{T}"/>s as a map-style dataset. Item N
/// is the tuple of element N from each tensor's leading axis. Direct PyTorch
/// equivalent: <c>torch.utils.data.TensorDataset(*tensors)</c>.
///
/// <para>Used when the entire dataset already fits in memory as tensors —
/// canonical example: a small classification problem where features and
/// labels are loaded once at the start of training. The DataLoader drives
/// random access via the per-index <see cref="this[int]"/> accessor.</para>
/// </summary>
public sealed class TensorDataset<T> : IDataset<Tensor<T>[]>
{
    private readonly Tensor<T>[] _tensors;
    private readonly int _length;

    /// <summary>
    /// Constructs a dataset from <paramref name="tensors"/>. All tensors must
    /// share the same leading-axis length (the sample count). Empty array is
    /// rejected — a zero-tensor dataset would have no shape to compute the
    /// per-sample type from.
    /// </summary>
    public TensorDataset(params Tensor<T>[] tensors)
    {
        if (tensors is null || tensors.Length == 0)
            throw new ArgumentException("TensorDataset requires at least one tensor.", nameof(tensors));
        // Reject scalars (rank 0) explicitly — without this, indexing
        // `_shape[0]` would throw IndexOutOfRangeException with no
        // diagnostic context. TensorDataset requires a leading sample
        // axis on every member tensor by definition.
        if (tensors[0]._shape.Length == 0)
            throw new ArgumentException(
                "TensorDataset requires tensors with at least one dimension; tensor 0 is rank 0.",
                nameof(tensors));
        int n = tensors[0]._shape[0];
        for (int i = 1; i < tensors.Length; i++)
        {
            if (tensors[i]._shape.Length == 0)
                throw new ArgumentException(
                    $"TensorDataset requires tensors with at least one dimension; tensor {i} is rank 0.",
                    nameof(tensors));
            if (tensors[i]._shape[0] != n)
                throw new ArgumentException(
                    $"TensorDataset tensors must share their leading dimension. " +
                    $"Tensor 0 has {n}, tensor {i} has {tensors[i]._shape[0]}.");
        }
        _tensors = tensors;
        _length = n;
    }

    /// <inheritdoc/>
    public int Count => _length;

    /// <summary>
    /// Returns the index-th slice of every wrapped tensor. The returned
    /// <see cref="Tensor{T}"/> array has length equal to the constructor's
    /// tensor count; each element is a rank-(R-1) slice along axis 0.
    /// </summary>
    public Tensor<T>[] this[int index]
    {
        get
        {
            if ((uint)index >= (uint)_length)
                throw new ArgumentOutOfRangeException(nameof(index));
            var sample = new Tensor<T>[_tensors.Length];
            for (int i = 0; i < _tensors.Length; i++)
                sample[i] = _tensors[i].Slice(0, index, index + 1).Squeeze(0);
            return sample;
        }
    }
}

/// <summary>
/// View over a parent dataset that exposes only a fixed subset of indices.
/// Used by <see cref="DatasetExtensions.RandomSplit{T}"/> to produce
/// non-overlapping train/val/test partitions; can also be constructed
/// directly to drop bad samples without copying the underlying storage.
/// PyTorch parity: <c>torch.utils.data.Subset</c>.
/// </summary>
public sealed class Subset<TSample> : IDataset<TSample>
{
    private readonly IDataset<TSample> _parent;
    private readonly IReadOnlyList<int> _indices;

    /// <summary>Constructs a view over <paramref name="parent"/> at the
    /// positions in <paramref name="indices"/>. Indices are not validated
    /// eagerly — out-of-range entries surface as <see cref="ArgumentOutOfRangeException"/>
    /// from the parent on access.</summary>
    public Subset(IDataset<TSample> parent, IReadOnlyList<int> indices)
    {
        _parent = parent ?? throw new ArgumentNullException(nameof(parent));
        _indices = indices ?? throw new ArgumentNullException(nameof(indices));
    }

    /// <inheritdoc/>
    public int Count => _indices.Count;

    /// <inheritdoc/>
    public TSample this[int index] => _parent[_indices[index]];
}

/// <summary>
/// Concatenates multiple map-style datasets end-to-end. Index 0 maps to
/// dataset[0][0]; the boundary between datasets is computed via prefix sums
/// over per-dataset counts. PyTorch parity: <c>torch.utils.data.ConcatDataset</c>.
/// </summary>
public sealed class ConcatDataset<TSample> : IDataset<TSample>
{
    private readonly IDataset<TSample>[] _datasets;
    private readonly int[] _cumulativeCounts; // length = datasets.Length, cumulativeCounts[i] = total count up to and including dataset i

    /// <summary>Constructs a concatenated view over <paramref name="datasets"/>.</summary>
    public ConcatDataset(params IDataset<TSample>[] datasets)
    {
        if (datasets is null || datasets.Length == 0)
            throw new ArgumentException("ConcatDataset requires at least one dataset.", nameof(datasets));
        _datasets = datasets;
        _cumulativeCounts = new int[datasets.Length];
        int running = 0;
        for (int i = 0; i < datasets.Length; i++)
        {
            running += datasets[i].Count;
            _cumulativeCounts[i] = running;
        }
    }

    /// <inheritdoc/>
    public int Count => _cumulativeCounts[_cumulativeCounts.Length - 1];

    /// <inheritdoc/>
    public TSample this[int index]
    {
        get
        {
            if ((uint)index >= (uint)Count)
                throw new ArgumentOutOfRangeException(nameof(index));
            // Linear scan beats binary search until ~16 partitions; the
            // common case is 2-3 (train/val/test) so the simple loop wins.
            int dsIdx = 0;
            while (index >= _cumulativeCounts[dsIdx]) dsIdx++;
            int local = dsIdx == 0 ? index : index - _cumulativeCounts[dsIdx - 1];
            return _datasets[dsIdx][local];
        }
    }
}

/// <summary>
/// Concatenates iterable datasets — emits all samples from the first, then
/// all from the second, etc. Used for streaming pipelines where multiple
/// upstream feeds share a downstream training loop. PyTorch parity:
/// <c>torch.utils.data.ChainDataset</c>.
/// </summary>
public sealed class ChainDataset<TSample> : IIterableDataset<TSample>
{
    private readonly IEnumerable<IIterableDataset<TSample>> _datasets;

    /// <summary>Constructs a chain over <paramref name="datasets"/>.</summary>
    public ChainDataset(IEnumerable<IIterableDataset<TSample>> datasets)
    {
        _datasets = datasets ?? throw new ArgumentNullException(nameof(datasets));
    }

    /// <inheritdoc/>
    public IEnumerator<TSample> GetEnumerator()
    {
        foreach (var ds in _datasets)
        {
            using var e = ds.GetEnumerator();
            while (e.MoveNext()) yield return e.Current;
        }
    }
}

/// <summary>
/// LRU-cached view over a (typically expensive-to-decode) dataset. Used for
/// pipelines where each sample costs significant CPU / I/O to materialize
/// (image decode + augment, audio resample, JSON parse) and the working set
/// fits in memory. Cache hits skip the parent's
/// <see cref="IDataset{TSample}.this[int]"/> entirely.
///
/// <para>The implementation is a doubly-linked-list LRU keyed by sample
/// index. Get is O(1) on hit (dict lookup + list head splice) and O(1) on
/// miss (parent fetch + list head insert + tail eviction). All methods are
/// thread-safe — multi-worker loaders share the cache across worker threads.</para>
/// </summary>
public sealed class CachedDataset<TSample> : IDataset<TSample>
{
    private readonly IDataset<TSample> _parent;
    private readonly int _capacity;
    private readonly LinkedList<(int Key, TSample Value)> _lru = new();
    private readonly Dictionary<int, LinkedListNode<(int Key, TSample Value)>> _index = new();
    private readonly object _gate = new();

    /// <summary>Constructs a cache wrapping <paramref name="parent"/> with
    /// at most <paramref name="capacity"/> samples retained.</summary>
    public CachedDataset(IDataset<TSample> parent, int capacity)
    {
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be at least 1.");
        _parent = parent ?? throw new ArgumentNullException(nameof(parent));
        _capacity = capacity;
    }

    /// <inheritdoc/>
    public int Count => _parent.Count;

    /// <inheritdoc/>
    public TSample this[int index]
    {
        get
        {
            lock (_gate)
            {
                if (_index.TryGetValue(index, out var node))
                {
                    // LRU touch — move to head.
                    _lru.Remove(node);
                    _lru.AddFirst(node);
                    return node.Value.Value;
                }
            }

            // Miss — parent fetch outside the gate so cold loads don't
            // block other readers' hits. Re-check after the fetch in case
            // another thread filled the slot meanwhile.
            var fresh = _parent[index];
            lock (_gate)
            {
                if (_index.TryGetValue(index, out var existing))
                    return existing.Value.Value;

                var node = new LinkedListNode<(int, TSample)>((index, fresh));
                _lru.AddFirst(node);
                _index[index] = node;
                if (_index.Count > _capacity)
                {
                    var tail = _lru.Last!;
                    _lru.RemoveLast();
                    _index.Remove(tail.Value.Key);
                }
                return fresh;
            }
        }
    }
}

/// <summary>
/// Convenience helpers over <see cref="IDataset{TSample}"/>. RandomSplit is
/// the headline — produces non-overlapping subsets that together partition
/// the parent's index space. PyTorch parity:
/// <c>torch.utils.data.random_split</c>.
/// </summary>
public static class DatasetExtensions
{
    /// <summary>
    /// Splits <paramref name="dataset"/> into <paramref name="lengths"/>-sized
    /// disjoint subsets. The lengths must sum to <paramref name="dataset"/>.
    /// <see cref="IDataset{TSample}.Count"/>; a deterministic shuffle (seeded
    /// from <paramref name="seed"/>) decides the partition.
    /// </summary>
    public static Subset<TSample>[] RandomSplit<TSample>(this IDataset<TSample> dataset, int[] lengths, int? seed = null)
    {
        if (dataset is null) throw new ArgumentNullException(nameof(dataset));
        if (lengths is null || lengths.Length == 0) throw new ArgumentException("Lengths is required.", nameof(lengths));
        long sum = 0;
        foreach (var l in lengths)
        {
            if (l < 0) throw new ArgumentException("Lengths must be non-negative.", nameof(lengths));
            sum += l;
        }
        if (sum != dataset.Count)
            throw new ArgumentException(
                $"Sum of lengths ({sum}) must equal dataset count ({dataset.Count}).", nameof(lengths));

        var perm = new int[dataset.Count];
        for (int i = 0; i < perm.Length; i++) perm[i] = i;
        var rng = seed is null ? new Random() : new Random(seed.Value);
        // Fisher-Yates.
        for (int i = perm.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (perm[i], perm[j]) = (perm[j], perm[i]);
        }

        var splits = new Subset<TSample>[lengths.Length];
        int offset = 0;
        for (int i = 0; i < lengths.Length; i++)
        {
            var slice = new int[lengths[i]];
            Array.Copy(perm, offset, slice, 0, lengths[i]);
            splits[i] = new Subset<TSample>(dataset, slice);
            offset += lengths[i];
        }
        return splits;
    }
}
