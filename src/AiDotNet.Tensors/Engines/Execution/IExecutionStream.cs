using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Compilation;

namespace AiDotNet.Tensors.Engines.Execution;

/// <summary>
/// An ordered, backend-specific work queue for asynchronous compiled-plan
/// step submission. CPU streams are backed by a long-lived worker thread
/// reading from a <see cref="System.Threading.Channels.Channel{T}"/>;
/// GPU streams wrap a native <c>cudaStream_t</c> (or HIP stream / OpenCL
/// command queue) so kernels submitted via <see cref="Submit"/>
/// execute concurrently with the host. Both backends preserve FIFO
/// ordering within a single stream — step N's writes are guaranteed to
/// be visible to step N+1's reads.
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
/// <remarks>
/// <para>
/// <b>Why a per-engine abstraction:</b> the compiled-plan async path
/// (<c>ICompiledPlan&lt;T&gt;.ExecuteAsync</c> / <c>ChainAsync</c>) needs
/// a uniform "submit step, return immediately, sync at pipeline boundary"
/// contract. CUDA's native stream API gives us this for free; CPU has
/// to fake it with a worker pool. Using one interface for both lets the
/// plan code stay backend-agnostic — the same <c>ExecuteAsync</c> body
/// runs on either side.
/// </para>
/// <para>
/// <b>Sub-microsecond dispatch:</b> <see cref="Submit"/> deliberately
/// avoids per-call <c>Task</c> allocation. CPU streams write a struct
/// work-item to an unbounded channel (≈100 ns); GPU streams call the
/// engine method with the active stream context selected. Acceptance
/// criterion #9 of issue #296 forbids <c>Task.Run</c> on the
/// <see cref="Submit"/> hot path — instead a single long-lived worker
/// drains the queue.
/// </para>
/// <para>
/// <b>FIFO ordering, NOT parallelism within a stream:</b> a single
/// stream is sequential by contract. To overlap batch N+1's stage 1
/// with batch N's stage 2 (the multi-batch throughput win documented
/// in issue #296's <c>StreamThroughputBenchmark</c>), allocate
/// <i>multiple</i> streams and round-robin batches across them. This
/// matches PyTorch's CUDA-stream programming model.
/// </para>
/// <para>
/// <b>CUDA Graph capture compatibility:</b> when <see cref="Submit"/>
/// is called inside a <see cref="AiDotNet.Tensors.Engines.Gpu.CudaGraphScope"/>, the kernel
/// dispatch is captured into the graph rather than executed eagerly.
/// <see cref="SyncAsync"/> outside the capture replays the captured
/// graph deterministically — the same path PyTorch's
/// <c>cudaGraphLaunch</c> uses.
/// </para>
/// </remarks>
internal interface IExecutionStream<T> : IAsyncDisposable, IDisposable
{
    /// <summary>
    /// Queues <paramref name="step"/>'s kernel for execution on this
    /// stream and returns immediately. The kernel runs on a worker
    /// thread (CPU) or native stream (GPU) — control returns to the
    /// caller before the kernel completes. Submission is FIFO-ordered
    /// against earlier <see cref="Submit"/> calls on the same stream:
    /// <paramref name="step"/> is guaranteed to start only after every
    /// previously-submitted step has finished.
    /// </summary>
    /// <param name="step">The compiled step to execute.</param>
    /// <param name="engine">The engine whose kernels the step's
    /// closures dispatch through.</param>
    /// <exception cref="ArgumentNullException"><paramref name="step"/>
    /// or <paramref name="engine"/> is null.</exception>
    /// <exception cref="ObjectDisposedException">This stream has been
    /// disposed.</exception>
    /// <exception cref="InvalidOperationException">The stream's
    /// underlying queue has been completed (post-DisposeAsync).</exception>
    void Submit(CompiledStep<T> step, IEngine engine);

    /// <summary>
    /// Asynchronously waits until every step submitted before this call
    /// has completed. Used at pipeline boundaries — when the caller
    /// needs to read the final output tensor, dispose the stream, or
    /// rebind storage. Subsequent <see cref="Submit"/> calls after
    /// <see cref="SyncAsync"/> resumes ARE permitted; they queue
    /// behind the sync barrier.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token. If cancelled
    /// before the queue drains, the returned task transitions to
    /// cancelled and any in-flight work runs to completion in the
    /// background (cancellation is observation, not termination).</param>
    /// <returns>A <see cref="ValueTask"/> that completes when every
    /// previously-submitted step has finished.</returns>
    ValueTask SyncAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Synchronous counterpart to <see cref="SyncAsync"/>. Blocks the
    /// calling thread until every previously-submitted step has
    /// completed. Used by the synchronous adapter on
    /// <c>ICompiledPlan&lt;T&gt;.Execute()</c> — the public sync API
    /// that runs the pipeline through the stream and blocks for the
    /// final result.
    /// </summary>
    void Sync();
}
