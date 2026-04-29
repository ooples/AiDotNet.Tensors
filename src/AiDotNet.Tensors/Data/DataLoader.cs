// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using AiDotNet.Tensors.Data.Samplers;

namespace AiDotNet.Tensors.Data;

/// <summary>
/// Configuration knobs for <see cref="DataLoader{TSample, TBatch}"/>.
/// Mirrors the keyword arguments of PyTorch's
/// <c>torch.utils.data.DataLoader</c>.
/// </summary>
public sealed class DataLoaderOptions
{
    /// <summary>Samples per batch. Required for the auto-sampler path.</summary>
    public int BatchSize { get; set; } = 1;

    /// <summary>If true, randomly shuffles indices each epoch (uses
    /// <see cref="RandomSampler"/>). Ignored when <see cref="Sampler"/>
    /// or <see cref="BatchSampler"/> is provided.</summary>
    public bool Shuffle { get; set; }

    /// <summary>Drops the trailing partial batch when the dataset count
    /// isn't a multiple of <see cref="BatchSize"/>. PyTorch parity.</summary>
    public bool DropLast { get; set; }

    /// <summary>Number of background workers that prefetch batches via the
    /// async pipeline. <c>0</c> uses the calling thread (sync); &gt; 0
    /// spawns that many <see cref="Task"/>-pool workers writing into a
    /// bounded <see cref="Channel{T}"/>. PyTorch parity:
    /// <c>num_workers</c>.</summary>
    public int NumWorkers { get; set; }

    /// <summary>How many batches to keep buffered ahead of the consumer.
    /// Also serves as the worker channel's bounded capacity. Default 2 —
    /// matches PyTorch's <c>prefetch_factor=2</c>.</summary>
    public int PrefetchFactor { get; set; } = 2;

    /// <summary>Optional explicit per-index sampler. Overrides
    /// <see cref="Shuffle"/> if non-null. <see cref="BatchSize"/> still
    /// determines batch grouping.</summary>
    public ISampler? Sampler { get; set; }

    /// <summary>Optional explicit batch sampler. Overrides both
    /// <see cref="Sampler"/> and <see cref="BatchSize"/>.</summary>
    public IBatchSampler? BatchSampler { get; set; }

    /// <summary>Seed for the default <see cref="RandomSampler"/> when
    /// <see cref="Shuffle"/> is true and no explicit sampler is given.
    /// <c>null</c> = unseeded (different shuffle every run).</summary>
    public int? Seed { get; set; }

    /// <summary>Throws if validation finds an inconsistent setting.</summary>
    internal void Validate()
    {
        if (BatchSize < 1) throw new ArgumentOutOfRangeException(nameof(BatchSize));
        if (NumWorkers < 0) throw new ArgumentOutOfRangeException(nameof(NumWorkers));
        if (PrefetchFactor < 1) throw new ArgumentOutOfRangeException(nameof(PrefetchFactor));
        if (BatchSampler is not null && (Sampler is not null || Shuffle))
        {
            throw new ArgumentException(
                "BatchSampler is mutually exclusive with Sampler/Shuffle. " +
                "When BatchSampler is set, BatchSize is ignored too.");
        }
    }
}

/// <summary>
/// Drives a per-epoch iteration over an <see cref="IDataset{TSample}"/> —
/// applies a sampler to choose indices, fetches samples, applies a collate
/// function to assemble batches, and (optionally) prefetches batches in
/// parallel from a worker pool. Mirrors
/// <c>torch.utils.data.DataLoader</c>.
/// </summary>
/// <typeparam name="TSample">Per-sample type as exposed by the dataset.</typeparam>
/// <typeparam name="TBatch">Per-batch type as produced by the collator.</typeparam>
public class DataLoader<TSample, TBatch> : IEnumerable<TBatch>
{
    private readonly IDataset<TSample> _dataset;
    private readonly CollateFn<TSample, TBatch> _collate;
    private readonly DataLoaderOptions _options;
    private readonly IBatchSampler _batchSampler;
    private int _epoch;

    /// <summary>Constructs.</summary>
    /// <param name="dataset">Map-style dataset to iterate.</param>
    /// <param name="collate">Function that turns a list of samples into a batch.</param>
    /// <param name="options">Loader options. <c>null</c> uses defaults.</param>
    public DataLoader(IDataset<TSample> dataset, CollateFn<TSample, TBatch> collate, DataLoaderOptions? options = null)
    {
        _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));
        _collate = collate ?? throw new ArgumentNullException(nameof(collate));
        _options = options ?? new DataLoaderOptions();
        _options.Validate();
        _batchSampler = ResolveBatchSampler();
    }

    /// <summary>The underlying dataset.</summary>
    public IDataset<TSample> Dataset => _dataset;

    /// <summary>Number of batches per epoch.</summary>
    public int BatchCount => _batchSampler.Count;

    /// <summary>Current epoch index. Bumped automatically on each fresh
    /// enumeration, or set explicitly via <see cref="SetEpoch"/>.</summary>
    public int Epoch => _epoch;

    /// <summary>Sets the epoch — typically not needed (auto-increments per
    /// enumeration), but exposed so distributed-training code can sync the
    /// per-epoch shuffle seed across ranks before iteration.</summary>
    public void SetEpoch(int epoch)
    {
        _epoch = epoch;
        _batchSampler.SetEpoch(epoch);
    }

    /// <summary>Enumerates one full epoch.</summary>
    public IEnumerator<TBatch> GetEnumerator()
    {
        _batchSampler.SetEpoch(_epoch);
        try
        {
            return _options.NumWorkers <= 0
                ? IterateSync().GetEnumerator()
                : IterateAsync().GetEnumerator();
        }
        finally
        {
            // Don't auto-increment here — finally runs on enumerator dispose
            // and the user may dispose mid-epoch (early-stop). Bump epoch at
            // the START of the next enumeration instead.
        }
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    private IEnumerable<TBatch> IterateSync()
    {
        foreach (var batchIdx in _batchSampler.GetBatches())
        {
            var samples = new TSample[batchIdx.Count];
            for (int i = 0; i < batchIdx.Count; i++) samples[i] = _dataset[batchIdx[i]];
            yield return _collate(samples);
        }
        _epoch++;
    }

    private IEnumerable<TBatch> IterateAsync()
    {
        // Order-preserving multi-worker prefetch. Each batch carries its
        // sampler-order sequence number through the worker pool; the
        // consumer thread holds a small reorder buffer (capped by the
        // prefetch factor) and yields strictly in sampler order. Matches
        // PyTorch's DataLoader contract: batches with `num_workers>0` are
        // returned in the same order as `num_workers=0`.
        int capacity = Math.Max(1, _options.PrefetchFactor * Math.Max(1, _options.NumWorkers));
        var resultChannel = Channel.CreateBounded<(int Seq, TBatch Batch)>(
            new BoundedChannelOptions(capacity)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = false,
            });

        var jobChannel = Channel.CreateBounded<(int Seq, IReadOnlyList<int> Idx)>(
            new BoundedChannelOptions(capacity * 2)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = false,
                SingleWriter = true,
            });

        var pumpCts = new CancellationTokenSource();
        var pumpTask = Task.Run(async () =>
        {
            try
            {
                int seq = 0;
                foreach (var batch in _batchSampler.GetBatches())
                {
                    if (pumpCts.IsCancellationRequested) break;
                    await jobChannel.Writer.WriteAsync((seq++, batch)).ConfigureAwait(false);
                }
            }
            finally { jobChannel.Writer.TryComplete(); }
        });

        var workers = new Task[_options.NumWorkers];
        int activeWorkers = _options.NumWorkers;
        for (int w = 0; w < _options.NumWorkers; w++)
        {
            workers[w] = Task.Run(async () =>
            {
                try
                {
                    // Manual WaitToRead/TryRead loop instead of ReadAllAsync —
                    // ReadAllAsync is netstandard2.1+ and we still target net471.
                    while (await jobChannel.Reader.WaitToReadAsync().ConfigureAwait(false))
                    {
                        while (jobChannel.Reader.TryRead(out var job))
                        {
                            if (pumpCts.IsCancellationRequested) break;
                            var samples = new TSample[job.Idx.Count];
                            for (int i = 0; i < job.Idx.Count; i++) samples[i] = _dataset[job.Idx[i]];
                            var batch = _collate(samples);
                            await resultChannel.Writer.WriteAsync((job.Seq, batch)).ConfigureAwait(false);
                        }
                        if (pumpCts.IsCancellationRequested) break;
                    }
                }
                finally
                {
                    if (Interlocked.Decrement(ref activeWorkers) == 0)
                        resultChannel.Writer.TryComplete();
                }
            });
        }

        // Reorder buffer: holds out-of-order batches until the next
        // expected sequence number is available. Bounded by the prefetch
        // factor — workers can't run more batches ahead than the buffer
        // allows because the result channel back-pressures when full.
        var reorderBuffer = new Dictionary<int, TBatch>();
        int nextExpected = 0;
        try
        {
            while (true)
            {
                if (!TryReadNext(resultChannel.Reader, out var item)) break;

                if (item.Seq == nextExpected)
                {
                    nextExpected++;
                    yield return item.Batch;
                    while (reorderBuffer.TryGetValue(nextExpected, out var pending))
                    {
                        reorderBuffer.Remove(nextExpected);
                        nextExpected++;
                        yield return pending;
                    }
                }
                else
                {
                    reorderBuffer[item.Seq] = item.Batch;
                }
            }
            // Drain leftover buffered batches in order — guards against the
            // edge case where the channel closes between reads.
            while (reorderBuffer.TryGetValue(nextExpected, out var pending))
            {
                reorderBuffer.Remove(nextExpected);
                nextExpected++;
                yield return pending;
            }
        }
        finally
        {
            pumpCts.Cancel();
            try { Task.WaitAll(workers, TimeSpan.FromSeconds(5)); } catch { /* best-effort */ }
            try { pumpTask.Wait(TimeSpan.FromSeconds(5)); } catch { /* best-effort */ }
            pumpCts.Dispose();
        }
        _epoch++;
    }

    /// <summary>
    /// Tries to dequeue one (seq, batch) from <paramref name="reader"/>,
    /// blocking the caller until either an item is available or the channel
    /// is fully drained + completed. Wraps the WaitToReadAsync /
    /// ChannelClosedException dance so the calling iterator doesn't need a
    /// catch block (illegal alongside yield).
    /// </summary>
    private static bool TryReadNext(ChannelReader<(int Seq, TBatch Batch)> reader, out (int Seq, TBatch Batch) item)
    {
        item = default;
        while (true)
        {
            bool got;
            try { got = reader.WaitToReadAsync().AsTask().GetAwaiter().GetResult(); }
            catch (ChannelClosedException) { return false; }
            if (!got) return false;
            if (reader.TryRead(out item)) return true;
        }
    }

    private IBatchSampler ResolveBatchSampler()
    {
        if (_options.BatchSampler is not null) return _options.BatchSampler;

        ISampler inner;
        if (_options.Sampler is not null)
        {
            inner = _options.Sampler;
        }
        else if (_options.Shuffle)
        {
            inner = new RandomSampler(_dataset.Count, seed: _options.Seed);
        }
        else
        {
            inner = new SequentialSampler(_dataset.Count);
        }
        return new BatchSampler(inner, _options.BatchSize, _options.DropLast);
    }
}
