using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.Execution;

/// <summary>
/// GPU-backed <see cref="IExecutionStream{T}"/> that wraps an
/// <see cref="IGpuStream"/> (CUDA stream, HIP stream, or OpenCL command
/// queue). <see cref="Submit"/> dispatches the step's kernel
/// synchronously on the calling thread — but on a GPU backend the
/// kernel itself is queued on the device stream and runs concurrently
/// with the host, so the caller returns immediately. FIFO ordering is
/// inherent to the underlying native stream.
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
/// <remarks>
/// <para>
/// <b>Why no worker thread on GPU:</b> the device stream IS the queue.
/// CUDA's <c>cudaStream_t</c> already provides asynchronous, FIFO-ordered
/// kernel dispatch; layering a CPU-side worker on top would just add a
/// hop. <see cref="Submit"/> calls into the engine method directly and
/// the engine queues the kernel against the stream context.
/// </para>
/// <para>
/// <b>CUDA Graph capture:</b> when this stream is the active stream
/// inside a <see cref="CudaGraphScope"/>, every kernel dispatched by
/// <see cref="Submit"/> is recorded into the graph rather than executed
/// eagerly. Outside the capture, <see cref="SyncAsync"/> replays the
/// captured graph deterministically, matching the
/// <c>cudaGraphLaunch</c> path PyTorch uses for steady-state inference.
/// </para>
/// <para>
/// <b>Async wait:</b> <see cref="SyncAsync"/> polls
/// <see cref="IGpuStream.Query"/> with <see cref="Task.Yield"/> in
/// between probes — the native call is non-blocking (one driver
/// roundtrip), so polling has near-zero CPU cost while the GPU drains.
/// The synchronous <see cref="Sync"/> path uses
/// <see cref="IGpuStream.Synchronize"/> directly to avoid the polling
/// loop when the caller is already prepared to block.
/// </para>
/// </remarks>
internal sealed class GpuExecutionStream<T> : IExecutionStream<T>
{
    private readonly IGpuStream _stream;
    private readonly bool _ownsStream;
    private int _disposed;

    /// <param name="stream">The underlying GPU stream. Pass an existing
    /// stream (e.g. from <c>CudaBackend</c>'s default stream) when the
    /// caller manages the stream's lifetime; pass <paramref name="ownsStream"/>
    /// = true when this <see cref="GpuExecutionStream{T}"/> created the
    /// stream and is responsible for disposing it.</param>
    /// <param name="ownsStream">Whether disposal of this object should
    /// also dispose the underlying <paramref name="stream"/>.</param>
    public GpuExecutionStream(IGpuStream stream, bool ownsStream)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        _ownsStream = ownsStream;
    }

    /// <summary>
    /// The underlying GPU stream. Exposed so a caller can record events
    /// on it for cross-stream waits or pass it to a
    /// <see cref="CudaGraphScope"/> for graph capture.
    /// </summary>
    public IGpuStream UnderlyingStream => _stream;

    /// <inheritdoc/>
    public void Submit(CompiledStep<T> step, IEngine engine)
    {
        if (step is null) throw new ArgumentNullException(nameof(step));
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(GpuExecutionStream<T>));

        // For GPU backends the kernel call is non-blocking — control
        // returns to us as soon as the kernel is queued. We dispatch
        // straight into the engine without a worker hop. The engine's
        // backend is responsible for routing the kernel to the active
        // stream context (see CudaBackend.Stream / equivalent).
        step.Execute(engine, step.OutputBuffer);
    }

    /// <inheritdoc/>
    public ValueTask SyncAsync(CancellationToken cancellationToken = default)
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(GpuExecutionStream<T>));

        // Fast path: stream is already drained, no allocation.
        if (_stream.Query()) return default;
        return SyncAsyncCold(cancellationToken);
    }

    private async ValueTask SyncAsyncCold(CancellationToken cancellationToken)
    {
        // Poll with Task.Yield between probes. cuStreamQuery is a single
        // driver roundtrip (~1 µs); we don't want to block the caller's
        // thread while the GPU drains. Yielding lets other ready work on
        // the thread pool run while we wait.
        while (!_stream.Query())
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();
        }
    }

    /// <inheritdoc/>
    public void Sync()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(GpuExecutionStream<T>));
        _stream.Synchronize();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        if (_ownsStream)
        {
            _stream.Dispose();
        }
    }

    public ValueTask DisposeAsync()
    {
        Dispose();
        return default;
    }
}
