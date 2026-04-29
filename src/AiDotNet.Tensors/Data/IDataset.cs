// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;

namespace AiDotNet.Tensors.Data;

/// <summary>
/// Map-style dataset — supports indexed lookup. Mirrors PyTorch's
/// <c>torch.utils.data.Dataset</c>: implementations expose a known
/// <see cref="Count"/> and a per-index <see cref="this[int]"/> accessor that
/// returns one sample. Samplers iterate over the index space and the
/// <see cref="DataLoader{TSample, TBatch}"/> assembles batches.
///
/// <para>Use this when:</para>
/// <list type="bullet">
///   <item>The full dataset size is known up-front (file count, row count).</item>
///   <item>Random access is needed (most ML training pipelines — shuffling,
///     subsetting, k-fold splits all depend on indexed access).</item>
/// </list>
///
/// <para>For unbounded streams (e.g. log tail, network socket), use
/// <see cref="IIterableDataset{TSample}"/> instead.</para>
/// </summary>
/// <typeparam name="TSample">The shape of one sample. Often a tuple of
/// <c>(Tensor&lt;float&gt; input, int label)</c> or a custom record type.</typeparam>
public interface IDataset<out TSample>
{
    /// <summary>Total number of samples. Must remain stable for the dataset's
    /// lifetime — the sampler caches it and a mid-iteration change would
    /// produce out-of-range indices.</summary>
    int Count { get; }

    /// <summary>Returns the sample at <paramref name="index"/>. Implementations
    /// should be thread-safe — multi-worker loaders call this from
    /// <see cref="System.Threading.Tasks.Task"/>-pool threads in parallel.</summary>
    /// <exception cref="System.ArgumentOutOfRangeException">If
    /// <paramref name="index"/> is outside <c>[0, Count)</c>.</exception>
    TSample this[int index] { get; }
}

/// <summary>
/// Stream-style dataset — produces samples from a source whose total length
/// may be unknown or unbounded (network feed, log file being written,
/// streaming Kafka topic). Mirrors PyTorch's
/// <c>torch.utils.data.IterableDataset</c>.
///
/// <para>The <see cref="DataLoader{TSample, TBatch}"/> drives an iterable
/// dataset by enumerating it and grouping into batches; samplers do not
/// participate (PyTorch's contract — there is no index space to sample).</para>
/// </summary>
public interface IIterableDataset<out TSample>
{
    /// <summary>Returns an enumerator over the stream's samples. Each call
    /// produces a fresh enumerator — implementations should support
    /// re-iteration (the loader calls this once per epoch).</summary>
    IEnumerator<TSample> GetEnumerator();
}

/// <summary>
/// Sampler — yields a sequence of indices into an
/// <see cref="IDataset{TSample}"/>. The DataLoader composes a sampler with a
/// batch size to drive the per-iteration index sequence.
///
/// <para>Implementations:</para>
/// <list type="bullet">
///   <item><see cref="Samplers.SequentialSampler"/> — 0..N-1 in order.</item>
///   <item><see cref="Samplers.RandomSampler"/> — uniform random permutation.</item>
///   <item><see cref="Samplers.WeightedRandomSampler"/> — weighted with replacement.</item>
///   <item><see cref="Samplers.SubsetRandomSampler"/> — random over a fixed subset.</item>
///   <item><see cref="Samplers.BatchSampler"/> — wraps another sampler and
///     yields batches of indices instead of individual indices.</item>
///   <item><see cref="Samplers.DistributedSampler"/> — splits indices across
///     ranks for distributed training.</item>
/// </list>
/// </summary>
public interface ISampler
{
    /// <summary>Number of indices the sampler will yield in one full pass.
    /// Used by the DataLoader to compute batch counts and progress bars.</summary>
    int Count { get; }

    /// <summary>Returns the index sequence for one epoch. Each call should
    /// produce a fresh sequence — the DataLoader calls this once per epoch
    /// and a stateful sampler (<see cref="Samplers.RandomSampler"/>) reseeds
    /// here to give the next epoch a different shuffle.</summary>
    IEnumerable<int> GetIndices();

    /// <summary>
    /// Notifies the sampler that we are starting <paramref name="epoch"/>.
    /// Distributed and shuffled samplers fold the epoch into their seed so
    /// the per-epoch shuffle is deterministic across ranks.
    /// </summary>
    void SetEpoch(int epoch);
}

/// <summary>
/// Sampler that yields batches (lists of indices) directly. Used for
/// bucket-batching and other patterns where the batch composition itself is
/// the sampling decision (variable-length sequences grouped by length, etc).
/// </summary>
public interface IBatchSampler
{
    /// <summary>Number of batches in one full pass.</summary>
    int Count { get; }

    /// <summary>Returns the per-batch index sequence for one epoch. Each
    /// inner <see cref="IReadOnlyList{T}"/> is a batch; the loader wraps each
    /// batch's samples through the collate function.</summary>
    IEnumerable<IReadOnlyList<int>> GetBatches();

    /// <summary>Sets the epoch — see <see cref="ISampler.SetEpoch"/>.</summary>
    void SetEpoch(int epoch);
}
