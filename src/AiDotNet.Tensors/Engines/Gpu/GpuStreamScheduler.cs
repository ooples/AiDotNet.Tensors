// Copyright (c) AiDotNet. All rights reserved.
// PR #333 / issue #335 — schedules independent kernel launches across the
// streams in a GpuStreamPool. Solves the "every kernel serializes on
// _defaultStream" perf gap by round-robining independent work across
// pool streams. Used by BatchMatMul (per-batch-slice GEMMs), multi-head
// attention (per-head launches), and any other consumer that has a list
// of data-independent launches.

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Dispatches independent GPU kernel launches across the streams in a
/// <see cref="GpuStreamPool"/> for compute-compute overlap. Each launch
/// runs on its own stream, so the GPU's stream-multiprocessor scheduler
/// can interleave them based on resource availability.
///
/// <para>This is the high-level consumer API for <see cref="GpuStreamPool"/>.
/// Callers describe their independent work as a list of <c>Action&lt;IGpuStream&gt;</c>
/// delegates; the scheduler picks a stream from the pool for each, runs
/// the delegate (which is expected to call kernel-launch methods bound to
/// the supplied stream), and returns completion events the caller can
/// wait on or chain into downstream work.</para>
///
/// <para><b>Concurrency model:</b> Each launch in a single <see cref="Dispatch"/>
/// call is independent of every other launch in that call — the caller is
/// responsible for guaranteeing no data dependencies between them. The
/// scheduler does NOT analyze dependencies; it just round-robins streams.
/// For DAG-ordered dispatch (batch N depends on batch N-1), use
/// <see cref="DispatchSequential"/>.</para>
///
/// <para><b>Stream count:</b> Bounded by <see cref="GpuExecutionOptions.MaxComputeStreams"/>.
/// When the number of launches exceeds the stream count, multiple launches
/// share each stream and serialize on it; the scheduler still spreads work
/// across all available streams.</para>
///
/// <para><b>Hot-path use cases:</b>
/// <list type="bullet">
/// <item>BatchMatMul over independent batch slices — each slice GEMM is
/// dispatched on a distinct stream, so cuBLAS / cuDNN can keep multiple
/// SMs busy concurrently.</item>
/// <item>Multi-head attention forward: scores = Q · K^T per head, then
/// softmax, then output = scores · V per head — each head's three GEMMs
/// run on the same stream (preserving per-head ordering) but heads run
/// across different streams (compute-compute overlap).</item>
/// <item>ResNet bottleneck blocks with parallel 1×1 conv branches — each
/// branch on its own stream.</item>
/// </list></para>
/// </summary>
public sealed class GpuStreamScheduler
{
    private readonly GpuStreamPool _pool;
    private readonly GpuStreamType _streamType;
    private int _nextStream;

    /// <summary>
    /// Creates a scheduler bound to a stream pool. Launches dispatched
    /// through this scheduler will be drawn from the pool's
    /// <paramref name="streamType"/> tier; the default is <see cref="GpuStreamType.Compute"/>
    /// which matches the most common training/inference use case.
    /// </summary>
    public GpuStreamScheduler(GpuStreamPool pool, GpuStreamType streamType = GpuStreamType.Compute)
    {
        _pool = pool ?? throw new ArgumentNullException(nameof(pool));
        _streamType = streamType;
    }

    /// <summary>
    /// Dispatches a batch of independent launches across pool streams.
    /// Returns one completion event per launch, in the same order as the
    /// input list. Callers waiting for all to complete should call
    /// <see cref="SynchronizeEvents"/> with the returned events or call
    /// <see cref="GpuStreamPool.SynchronizeAll"/> on the underlying pool.
    /// </summary>
    /// <remarks>
    /// The launches must be data-independent — neither reads nor writes
    /// shared GPU memory between launches in this batch. If two launches
    /// touch the same buffer (even if logically disjoint regions), the
    /// underlying CUDA / HIP / Metal / OpenCL runtime may emit memory-
    /// access warnings under racy access patterns. Use
    /// <see cref="DispatchSequential"/> when ordering matters.
    /// </remarks>
    public IReadOnlyList<IGpuEvent> Dispatch(IReadOnlyList<Action<IGpuStream>> launches)
    {
        if (launches is null) throw new ArgumentNullException(nameof(launches));
        if (launches.Count == 0) return Array.Empty<IGpuEvent>();

        var events = new IGpuEvent[launches.Count];
        // Acquire streams round-robin from the pool. The pool caps stream
        // count at GpuExecutionOptions.MaxComputeStreams; when launches.Count
        // exceeds that, we wrap and reuse streams — which serializes the
        // wrapped launches on each stream, but still spreads work across
        // the cap-limited stream count.
        for (int i = 0; i < launches.Count; i++)
        {
            var stream = AcquireNextStream();
            try
            {
                launches[i](stream);
                events[i] = stream.RecordEvent();
            }
            finally
            {
                _pool.ReleaseStream(stream);
            }
        }
        return events;
    }

    /// <summary>
    /// Dispatches a sequence of DAG-ordered batches. Each batch in
    /// <paramref name="batches"/> runs in parallel across streams (same
    /// semantics as <see cref="Dispatch"/>), but batches run in order —
    /// batch N's launches wait on completion events from batch N-1 before
    /// they start. Use when each layer of the model is parallel internally
    /// but downstream layers depend on upstream completion (forward pass
    /// over per-attention-head Q · K^T → softmax → output).
    /// </summary>
    public void DispatchSequential(IReadOnlyList<IReadOnlyList<Action<IGpuStream>>> batches)
    {
        if (batches is null) throw new ArgumentNullException(nameof(batches));

        IReadOnlyList<IGpuEvent>? prevEvents = null;
        for (int b = 0; b < batches.Count; b++)
        {
            var batch = batches[b];
            if (batch is null || batch.Count == 0) continue;

            var thisBatchEvents = new IGpuEvent[batch.Count];
            for (int i = 0; i < batch.Count; i++)
            {
                var stream = AcquireNextStream();
                try
                {
                    // Block this stream on every event from the previous
                    // batch — batch N can't start until batch N-1 is fully
                    // done. The GPU still runs batch N's launches in
                    // parallel across streams; it just won't start them
                    // until the cross-stream wait is satisfied.
                    if (prevEvents is not null)
                        foreach (var ev in prevEvents)
                            stream.WaitEvent(ev);

                    batch[i](stream);
                    thisBatchEvents[i] = stream.RecordEvent();
                }
                finally
                {
                    _pool.ReleaseStream(stream);
                }
            }
            prevEvents = thisBatchEvents;
        }
    }

    /// <summary>
    /// Waits for every event in <paramref name="events"/> to complete on
    /// the host. Cheaper than synchronizing the whole pool when you only
    /// need a subset of dispatched launches done — e.g. waiting on the
    /// last layer's events before reading the loss to CPU.
    /// </summary>
    public void SynchronizeEvents(IReadOnlyList<IGpuEvent> events)
    {
        if (events is null) return;
        foreach (var ev in events)
            ev?.Synchronize();
    }

    private IGpuStream AcquireNextStream()
    {
        // Round-robin across pool streams. We use AcquireStream rather than
        // tracking our own stream pool because the GpuStreamPool already
        // caps total streams at MaxComputeStreams and reuses them.
        int idx = Interlocked.Increment(ref _nextStream);
        _ = idx; // suppress unused; kept for future logging / debugging
        return _pool.AcquireStream(_streamType);
    }
}
