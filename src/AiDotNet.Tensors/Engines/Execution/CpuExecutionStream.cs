using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Compilation;

namespace AiDotNet.Tensors.Engines.Execution;

/// <summary>
/// CPU-backed <see cref="IExecutionStream{T}"/>: a single long-lived
/// worker thread reads work items from an unbounded
/// <see cref="Channel{T}"/> and runs each step's kernel sequentially.
/// FIFO ordering is enforced by the single-reader channel; submission
/// has sub-microsecond cost (one struct write to a lock-free queue);
/// no per-submit <see cref="Task"/> allocation.
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
/// <remarks>
/// <para>
/// <b>Why a single worker, not a pool:</b> a stream is a sequential
/// queue by contract — step N's writes must be visible to step N+1's
/// reads. Running steps in parallel within one stream would break that
/// invariant. To extract pipelining, the caller allocates multiple
/// streams (e.g. one per inflight batch) and lets the kernel-level
/// thread-pool inside each step (BLAS, AVX) saturate cores naturally.
/// This matches CUDA's stream model: streams overlap, ordering within
/// a stream is sequential.
/// </para>
/// <para>
/// <b>Why not <see cref="Task.Run"/>:</b> issue #296's measured
/// counterfactual showed <c>Task.Run</c>-per-submit regressing wall
/// time by 7%–65% on stages of sub-millisecond size — Task scheduler
/// overhead (continuation queuing, worker-thread context switches)
/// outweighs any pipelining win. A long-lived worker draining a
/// channel pays the scheduler cost once at stream construction,
/// then submits are zero-allocation.
/// </para>
/// <para>
/// <b>Drain signalling:</b> <see cref="SyncAsync"/> snapshots the
/// current submitted count and waits until completed catches up.
/// A single shared <see cref="TaskCompletionSource{TResult}"/> wakes
/// every waiter when the worker drains; it's reset after firing so
/// subsequent waits get a fresh signal. No per-submit allocation.
/// </para>
/// </remarks>
internal sealed class CpuExecutionStream<T> : IExecutionStream<T>
{
    // Work item is a struct so Channel writes don't allocate per submit.
    // Carries the step + engine pair so the worker can execute without
    // any captured-closure GC overhead.
    private readonly struct WorkItem
    {
        public readonly CompiledStep<T> Step;
        public readonly IEngine Engine;
        public WorkItem(CompiledStep<T> step, IEngine engine)
        {
            Step = step;
            Engine = engine;
        }
    }

    private readonly Channel<WorkItem> _channel;
    private readonly Task _workerTask;
    private long _submitted;
    private long _completed;
    // Lock-protected drain TCS — null when no waiter is currently parked.
    // Replaced (not reset) on each drain firing so a stale reference can't
    // race with the next SyncAsync's TCS allocation.
    private readonly object _syncLock = new();
    private TaskCompletionSource<bool>? _drainTcs;
    private int _disposed; // 0 = alive, 1 = disposed

    public CpuExecutionStream()
    {
        _channel = Channel.CreateUnbounded<WorkItem>(new UnboundedChannelOptions
        {
            // SingleReader = the worker; SingleWriter = false so the same
            // stream can be fed from multiple producer threads (e.g. a
            // batched-async caller queueing batches off a thread pool).
            SingleReader = true,
            SingleWriter = false,
            // Kick continuations to the thread pool rather than running them
            // synchronously on the writer's thread. Submit must stay sub-µs
            // and not run kernel work on the caller.
            AllowSynchronousContinuations = false,
        });
        _workerTask = Task.Factory.StartNew(
            WorkerLoopAsync,
            CancellationToken.None,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default).Unwrap();
    }

    /// <inheritdoc/>
    public void Submit(CompiledStep<T> step, IEngine engine)
    {
        if (step is null) throw new ArgumentNullException(nameof(step));
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(CpuExecutionStream<T>));

        // Increment BEFORE write so a SyncAsync on a fully-drained stream
        // never observes (_completed > _submitted). Atomic so multi-producer
        // submission stays consistent.
        Interlocked.Increment(ref _submitted);
        if (!_channel.Writer.TryWrite(new WorkItem(step, engine)))
        {
            // TryWrite on an unbounded channel only fails when the writer
            // has been completed — i.e. concurrent dispose. Roll back the
            // counter so SyncAsync doesn't deadlock waiting for work that
            // will never run.
            Interlocked.Decrement(ref _submitted);
            throw new InvalidOperationException(
                $"{nameof(CpuExecutionStream<T>)} has been completed and cannot accept new work.");
        }
    }

    /// <inheritdoc/>
    public ValueTask SyncAsync(CancellationToken cancellationToken = default)
    {
        // Fast path: if the worker has already caught up to the current
        // submission count, return synchronously with no allocation. This
        // is the steady-state SyncAsync after a graph-capture replay or
        // when the stream is intentionally idle.
        long target = Volatile.Read(ref _submitted);
        if (Volatile.Read(ref _completed) >= target)
            return default;

        return SyncAsyncCold(target, cancellationToken);
    }

    private async ValueTask SyncAsyncCold(long target, CancellationToken cancellationToken)
    {
        while (Volatile.Read(ref _completed) < target)
        {
            cancellationToken.ThrowIfCancellationRequested();
            TaskCompletionSource<bool> tcs;
            lock (_syncLock)
            {
                // Re-check inside the lock — the worker may have caught up
                // between the outer loop's read and our lock acquisition.
                if (Volatile.Read(ref _completed) >= target) return;
                tcs = _drainTcs ??= new TaskCompletionSource<bool>(
                    TaskCreationOptions.RunContinuationsAsynchronously);
            }
            // .WaitAsync(ct) is the cheapest cancellable Task wait on
            // net6+; on net471 we fall back to Task.Run-free cancellation
            // via the linked CTS pattern in the polyfill below.
            await WaitWithCancellationAsync(tcs.Task, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <inheritdoc/>
    public void Sync()
    {
        // Block-wait variant for the synchronous Execute() path. We cannot
        // simply .GetAwaiter().GetResult() on SyncAsync because that allocates
        // an awaiter and risks deadlock when called on a sync-context-bound
        // thread. Instead use the same target-snapshot loop with a
        // ManualResetEventSlim wakeup — same correctness, sync semantics,
        // zero per-step allocation.
        long target = Volatile.Read(ref _submitted);
        while (Volatile.Read(ref _completed) < target)
        {
            TaskCompletionSource<bool> tcs;
            lock (_syncLock)
            {
                if (Volatile.Read(ref _completed) >= target) return;
                tcs = _drainTcs ??= new TaskCompletionSource<bool>(
                    TaskCreationOptions.RunContinuationsAsynchronously);
            }
            tcs.Task.Wait();
        }
    }

    private async Task WorkerLoopAsync()
    {
        // Manual drain loop instead of ReadAllAsync — the IAsyncEnumerable
        // extension is net6+ only, and this assembly targets net471 too.
        // WaitToReadAsync + TryRead is functionally identical and works on
        // every framework.
        try
        {
            var reader = _channel.Reader;
            while (await reader.WaitToReadAsync().ConfigureAwait(false))
            {
                while (reader.TryRead(out var item))
                {
                    try
                    {
                        item.Step.Execute(item.Engine, item.Step.OutputBuffer);
                    }
                    finally
                    {
                        long completed = Interlocked.Increment(ref _completed);
                        long submitted = Volatile.Read(ref _submitted);
                        if (completed >= submitted)
                        {
                            // Fire any parked SyncAsync. Replace the TCS
                            // with null so the next SyncAsync allocates a
                            // fresh one — re-arming a fired TCS isn't
                            // supported.
                            TaskCompletionSource<bool>? tcs;
                            lock (_syncLock)
                            {
                                tcs = _drainTcs;
                                _drainTcs = null;
                            }
                            tcs?.TrySetResult(true);
                        }
                    }
                }
            }
        }
        finally
        {
            // After Complete, any parked SyncAsync should be released so
            // callers awaiting drain don't hang on a dead worker.
            TaskCompletionSource<bool>? tcs;
            lock (_syncLock)
            {
                tcs = _drainTcs;
                _drainTcs = null;
            }
            tcs?.TrySetResult(true);
        }
    }

    private static Task WaitWithCancellationAsync(Task task, CancellationToken cancellationToken)
    {
        // .WaitAsync(CancellationToken) is net6+ only; this polyfill keeps
        // net471 callers cancellable too. The wait races the task against
        // a TaskCompletionSource that fires from the cancellation
        // registration; whichever finishes first wins. The losing path
        // simply returns once the original task observes its end-state.
        if (!cancellationToken.CanBeCanceled || task.IsCompleted)
            return task;

        var tcs = new TaskCompletionSource<bool>(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var registration = cancellationToken.Register(
            static state => ((TaskCompletionSource<bool>)state!).TrySetCanceled(),
            tcs);

        return WaitImplAsync(task, tcs, registration);

        static async Task WaitImplAsync(
            Task workTask,
            TaskCompletionSource<bool> cancelTcs,
            CancellationTokenRegistration reg)
        {
            try
            {
                var winner = await Task.WhenAny(workTask, cancelTcs.Task).ConfigureAwait(false);
                await winner.ConfigureAwait(false); // surfaces cancellation if it won
            }
            finally
            {
                reg.Dispose();
            }
        }
    }

    public void Dispose()
    {
        // Sync dispose forwards to the async path and blocks. Test/cleanup
        // code that doesn't need the async pattern can still use using {}.
        DisposeAsync().AsTask().GetAwaiter().GetResult();
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _channel.Writer.TryComplete();
        try
        {
            await _workerTask.ConfigureAwait(false);
        }
        catch
        {
            // Swallow worker-loop exceptions on dispose — surfacing them
            // here would mask the user's own dispose-time problems and
            // they've already been observed by anyone awaiting SyncAsync.
        }
    }
}
