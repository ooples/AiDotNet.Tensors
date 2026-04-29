// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Data.Samplers;

namespace AiDotNet.Tensors.Data;

/// <summary>
/// Snapshot of a <see cref="CheckpointableLoader{TSample, TBatch}"/>'s
/// position. Save/restore so a crashed training run resumes from the same
/// batch in the same epoch — no re-shuffling, no skipped or repeated
/// samples. PyTorch parity: <c>torchdata.StatefulDataLoader.state_dict()</c>.
/// </summary>
public readonly record struct LoaderCheckpoint(int Epoch, int BatchIndex);

/// <summary>
/// Single-process DataLoader that supports mid-epoch resume. Maintains the
/// same per-epoch ordering as the regular DataLoader (same sampler, same
/// seed) but tracks how many batches have been emitted so far in
/// <see cref="LoaderCheckpoint.BatchIndex"/>. After a crash, restoring the
/// checkpoint replays the sampler and skips the already-consumed prefix —
/// the next yielded batch is exactly the one that would have been yielded
/// next, had the crash not happened.
///
/// <para>The async/multi-worker path is not used here — mid-epoch resume
/// + parallel prefetch is genuinely incompatible (workers may have decoded
/// past the checkpoint by the time a crash hits). For multi-worker
/// resumption, snapshot at end-of-epoch only.</para>
/// </summary>
public sealed class CheckpointableLoader<TSample, TBatch> : IEnumerable<TBatch>
{
    private readonly IDataset<TSample> _dataset;
    private readonly CollateFn<TSample, TBatch> _collate;
    private readonly DataLoaderOptions _options;
    private readonly IBatchSampler _batchSampler;
    private int _epoch;
    private int _batchIndex;

    /// <summary>Constructs.</summary>
    public CheckpointableLoader(IDataset<TSample> dataset,
        CollateFn<TSample, TBatch> collate, DataLoaderOptions? options = null)
    {
        _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));
        _collate = collate ?? throw new ArgumentNullException(nameof(collate));
        _options = options ?? new DataLoaderOptions();
        _options.Validate();
        if (_options.NumWorkers > 0)
            throw new ArgumentException(
                "CheckpointableLoader does not support NumWorkers > 0. " +
                "Use the regular DataLoader for prefetch, or snapshot at end-of-epoch.");
        _batchSampler = ResolveBatchSampler();
    }

    /// <summary>Captures the current loader position. Pass to a future
    /// <see cref="LoadStateDict"/> call to resume.</summary>
    public LoaderCheckpoint StateDict() => new(_epoch, _batchIndex);

    /// <summary>Restores from a checkpoint produced by an earlier
    /// <see cref="StateDict"/> call.</summary>
    public void LoadStateDict(LoaderCheckpoint checkpoint)
    {
        if (checkpoint.Epoch < 0) throw new ArgumentOutOfRangeException(nameof(checkpoint));
        if (checkpoint.BatchIndex < 0) throw new ArgumentOutOfRangeException(nameof(checkpoint));
        _epoch = checkpoint.Epoch;
        _batchIndex = checkpoint.BatchIndex;
    }

    /// <summary>Number of batches per epoch.</summary>
    public int BatchCount => _batchSampler.Count;

    /// <summary>Enumerates an epoch from the current checkpoint position.</summary>
    public IEnumerator<TBatch> GetEnumerator()
    {
        _batchSampler.SetEpoch(_epoch);
        int skip = _batchIndex;
        int seen = 0;
        foreach (var batchIdx in _batchSampler.GetBatches())
        {
            if (seen < skip) { seen++; continue; }
            var samples = new TSample[batchIdx.Count];
            for (int i = 0; i < batchIdx.Count; i++) samples[i] = _dataset[batchIdx[i]];
            var batch = _collate(samples);
            _batchIndex++;
            yield return batch;
            seen++;
        }
        // Epoch finished — bump epoch and reset batch index.
        _epoch++;
        _batchIndex = 0;
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    private IBatchSampler ResolveBatchSampler()
    {
        if (_options.BatchSampler is not null) return _options.BatchSampler;
        ISampler inner = _options.Sampler
            ?? (_options.Shuffle
                ? new RandomSampler(_dataset.Count, seed: _options.Seed)
                : new SequentialSampler(_dataset.Count));
        return new BatchSampler(inner, _options.BatchSize, _options.DropLast);
    }
}
