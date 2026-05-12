// Copyright (c) AiDotNet. All rights reserved.
// PR #333 / issue #335 — schedules independent kernel launches across the
// streams in a GpuStreamPool. Solves the "every kernel serializes on
// _defaultStream" perf gap by round-robining independent work across
// pool streams. Used by BatchMatMul (per-batch-slice GEMMs), multi-head
// attention (per-head launches), and any other consumer that has a list
// of data-independent launches.

using System.Collections;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Dispatches independent GPU kernel launches across the streams in a
/// <see cref="GpuStreamPool"/> for compute-compute overlap. Each launch
/// runs on its own stream, so the GPU's stream-multiprocessor scheduler
/// can interleave them based on resource availability.
///
/// <para>This is the high-level consumer API for <see cref="GpuStreamPool"/>.
/// Callers describe their independent work as a list of <c>Action&lt;IGpuStream&gt;</c>
/// delegates; the scheduler leases up to <see cref="GpuExecutionOptions.MaxComputeStreams"/>
/// streams from the pool, round-robins the launches across the leased set,
/// and returns a <see cref="GpuEventBatch"/> the caller can wait on or
/// chain into downstream work. The batch owns the recorded events and
/// disposes them when the caller disposes the batch.</para>
///
/// <para><b>Concurrency model:</b> Each launch in a single <see cref="Dispatch"/>
/// call is independent of every other launch in that call — the caller is
/// responsible for guaranteeing no data dependencies between them. The
/// scheduler does NOT analyze dependencies; it just round-robins streams.
/// For DAG-ordered dispatch (batch N depends on batch N-1), use
/// <see cref="DispatchSequential"/>.</para>
///
/// <para><b>Stream leasing model:</b> <see cref="Dispatch"/> leases
/// <c>min(launches.Count, MaxComputeStreams)</c> streams for the entire
/// call, indexes into the leased set with a round-robin counter, and
/// releases all leased streams when the dispatch returns. This avoids the
/// trap where per-launch acquire+release through <c>ConcurrentBag</c>
/// returns the same stream to the calling thread (LIFO cache) and
/// serializes every launch onto a single stream. If the pool is exhausted
/// by concurrent callers and we can't lease anything, the scheduler falls
/// back to the pool's default compute stream so the work still completes
/// — without parallelism, but correctly.</para>
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
    private readonly int _maxStreams;
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
        _maxStreams = pool.GetMaxStreams(streamType);
        if (_maxStreams < 1) _maxStreams = 1;
    }

    /// <summary>
    /// Dispatches a batch of independent launches across pool streams.
    /// Returns a <see cref="GpuEventBatch"/> with one completion event per
    /// launch in the same order as the input list. The returned batch owns
    /// the events — callers SHOULD <c>using var batch = scheduler.Dispatch(...)</c>
    /// or call <see cref="GpuEventBatch.Dispose"/> explicitly to release
    /// the native event handles. Failing to dispose leaks native GPU event
    /// objects.
    /// </summary>
    /// <remarks>
    /// The launches must be data-independent — neither reads nor writes
    /// shared GPU memory between launches in this batch. If two launches
    /// touch the same buffer (even if logically disjoint regions), the
    /// underlying CUDA / HIP / Metal / OpenCL runtime may emit memory-
    /// access warnings under racy access patterns. Use
    /// <see cref="DispatchSequential"/> when ordering matters.
    /// </remarks>
    public GpuEventBatch Dispatch(IReadOnlyList<Action<IGpuStream>> launches)
    {
        if (launches is null) throw new ArgumentNullException(nameof(launches));
        if (launches.Count == 0) return GpuEventBatch.Empty;

        // Lease the streams up front so each launch lands on a distinct
        // stream via round-robin. With per-launch acquire+release, the
        // pool's ConcurrentBag returns the same stream to the calling
        // thread (LIFO thread-local cache) and every launch serializes
        // onto a single stream — the very perf gap this scheduler exists
        // to close.
        int target = Math.Min(launches.Count, _maxStreams);
        var leased = LeaseStreams(target);
        try
        {
            int streamCount = leased.Count;
            if (streamCount == 0)
            {
                // Pool exhausted by concurrent callers. Fall back to the
                // default compute stream — work still executes, just
                // without parallelism on this call.
                return DispatchOnSingleStream(_pool.DefaultComputeStream, launches);
            }

            var events = new IGpuEvent[launches.Count];
            int start = (int)((uint)Interlocked.Increment(ref _nextStream)) % streamCount;
            for (int i = 0; i < launches.Count; i++)
            {
                var stream = leased[(start + i) % streamCount];
                launches[i](stream);
                events[i] = stream.RecordEvent();
            }
            return new GpuEventBatch(events);
        }
        finally
        {
            for (int i = 0; i < leased.Count; i++)
                _pool.ReleaseStream(leased[i]);
        }
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
    /// <returns>
    /// A <see cref="GpuEventBatch"/> covering the FINAL batch's recorded
    /// events. Earlier batches' events are owned by the scheduler and
    /// disposed internally once the next batch has installed waits on
    /// them. The returned batch is the caller's responsibility — dispose
    /// it (or wrap in <c>using</c>) to release the final batch's native
    /// event handles. Returns <see cref="GpuEventBatch.Empty"/> when every
    /// supplied batch is null or empty.
    /// </returns>
    public GpuEventBatch DispatchSequential(IReadOnlyList<IReadOnlyList<Action<IGpuStream>>> batches)
    {
        if (batches is null) throw new ArgumentNullException(nameof(batches));
        if (batches.Count == 0) return GpuEventBatch.Empty;

        int maxBatchSize = 0;
        for (int b = 0; b < batches.Count; b++)
        {
            var batch = batches[b];
            if (batch is not null && batch.Count > maxBatchSize) maxBatchSize = batch.Count;
        }
        if (maxBatchSize == 0) return GpuEventBatch.Empty;

        int target = Math.Min(maxBatchSize, _maxStreams);
        var leased = LeaseStreams(target);
        try
        {
            // Build the effective stream set. If the pool was exhausted by
            // concurrent callers and we couldn't lease anything, fall back
            // to the default compute stream — single-stream execution still
            // respects the DAG ordering because everything runs in submit
            // order on one stream.
            IGpuStream[] streams = leased.Count > 0
                ? leased.ToArray()
                : new[] { _pool.DefaultComputeStream };

            IGpuEvent[]? prevEvents = null;
            IGpuEvent[]? lastEvents = null;
            int rotation = (int)((uint)Interlocked.Increment(ref _nextStream)) % streams.Length;

            for (int b = 0; b < batches.Count; b++)
            {
                var batch = batches[b];
                if (batch is null || batch.Count == 0) continue;

                var thisBatchEvents = new IGpuEvent[batch.Count];
                for (int i = 0; i < batch.Count; i++)
                {
                    var stream = streams[(rotation + i) % streams.Length];

                    if (prevEvents is not null)
                    {
                        for (int p = 0; p < prevEvents.Length; p++)
                            stream.WaitEvent(prevEvents[p]);
                    }

                    batch[i](stream);
                    thisBatchEvents[i] = stream.RecordEvent();
                }

                // The previous batch's events have been consumed by every
                // stream in the current batch that needed them. CUDA /
                // HIP / Metal / OpenCL all capture the event state at
                // WaitEvent submission time, so disposing the wrapper now
                // does not break the wait that's already queued on the
                // GPU. This is the only path that keeps DispatchSequential
                // from leaking event handles proportional to batch count.
                if (prevEvents is not null)
                {
                    for (int p = 0; p < prevEvents.Length; p++)
                        prevEvents[p]?.Dispose();
                }

                prevEvents = thisBatchEvents;
                lastEvents = thisBatchEvents;
            }

            return lastEvents is null ? GpuEventBatch.Empty : new GpuEventBatch(lastEvents);
        }
        finally
        {
            for (int i = 0; i < leased.Count; i++)
                _pool.ReleaseStream(leased[i]);
        }
    }

    /// <summary>
    /// Waits for every event in <paramref name="events"/> to complete on
    /// the host. Cheaper than synchronizing the whole pool when you only
    /// need a subset of dispatched launches done — e.g. waiting on the
    /// last layer's events before reading the loss to CPU. Equivalent to
    /// <see cref="GpuEventBatch.SynchronizeAll"/> for the batch case.
    /// </summary>
    public void SynchronizeEvents(IReadOnlyList<IGpuEvent> events)
    {
        if (events is null) return;
        for (int i = 0; i < events.Count; i++)
            events[i]?.Synchronize();
    }

    private GpuEventBatch DispatchOnSingleStream(IGpuStream stream, IReadOnlyList<Action<IGpuStream>> launches)
    {
        var events = new IGpuEvent[launches.Count];
        for (int i = 0; i < launches.Count; i++)
        {
            launches[i](stream);
            events[i] = stream.RecordEvent();
        }
        return new GpuEventBatch(events);
    }

    private List<IGpuStream> LeaseStreams(int count)
    {
        var leased = new List<IGpuStream>(count);
        for (int i = 0; i < count; i++)
        {
            try
            {
                leased.Add(_pool.AcquireStream(_streamType));
            }
            catch (InvalidOperationException)
            {
                // Pool exhausted by concurrent callers — break and use
                // whatever subset we managed to lease. Caller path also
                // tolerates Count == 0 by routing to the default stream.
                break;
            }
        }
        return leased;
    }
}

/// <summary>
/// Disposable list of recorded GPU events returned by
/// <see cref="GpuStreamScheduler.Dispatch"/> and
/// <see cref="GpuStreamScheduler.DispatchSequential"/>. Owns the underlying
/// <see cref="IGpuEvent"/> handles — dispose the batch (e.g. via
/// <c>using var batch = scheduler.Dispatch(...)</c>) to release the native
/// event objects. Failing to dispose leaks GPU event handles.
/// </summary>
public sealed class GpuEventBatch : IReadOnlyList<IGpuEvent>, IDisposable
{
    /// <summary>
    /// Shared empty batch — returned when there is nothing to dispatch.
    /// Safe to dispose any number of times.
    /// </summary>
    public static readonly GpuEventBatch Empty = new GpuEventBatch(Array.Empty<IGpuEvent>());

    private readonly IGpuEvent[] _events;
    private bool _disposed;

    internal GpuEventBatch(IGpuEvent[] events)
    {
        _events = events ?? throw new ArgumentNullException(nameof(events));
    }

    /// <inheritdoc />
    public IGpuEvent this[int index] => _events[index];

    /// <inheritdoc />
    public int Count => _events.Length;

    /// <inheritdoc />
    public IEnumerator<IGpuEvent> GetEnumerator() => ((IEnumerable<IGpuEvent>)_events).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => _events.GetEnumerator();

    /// <summary>
    /// Blocks the calling thread until every event in this batch
    /// completes on the device. Equivalent to calling
    /// <see cref="IGpuEvent.Synchronize"/> on each event in order.
    /// </summary>
    public void SynchronizeAll()
    {
        for (int i = 0; i < _events.Length; i++)
            _events[i]?.Synchronize();
    }

    /// <summary>
    /// Disposes every event in the batch, releasing the underlying native
    /// GPU event handles. Idempotent. The batch is unusable after disposal.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _events.Length; i++)
            _events[i]?.Dispose();
    }
}
