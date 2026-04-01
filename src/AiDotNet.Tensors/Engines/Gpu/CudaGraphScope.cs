using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Records a sequence of GPU operations as a CUDA graph, then replays it
/// with zero kernel launch overhead. Equivalent to PyTorch's
/// <c>torch.cuda.CUDAGraph</c> with <c>torch.cuda.graph()</c>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Every time you run a GPU operation, there's a small
/// overhead for the CPU to tell the GPU what to do (kernel launch). For training loops
/// that repeat the same operations millions of times, this overhead adds up.
/// CUDA graphs record the entire sequence of GPU operations once, then replay
/// it as a single pre-compiled command — eliminating launch overhead entirely.</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// // Warmup run (required by CUDA graphs)
/// var output = model.Forward(input);
///
/// // Record the graph
/// using var graph = new CudaGraphScope(backend);
/// graph.BeginCapture();
/// var recorded = model.Forward(input);  // Operations are recorded, not executed
/// graph.EndCapture();
///
/// // Replay with zero launch overhead
/// for (int epoch = 0; epoch &lt; 1000; epoch++)
/// {
///     graph.Replay();  // Instant replay of all recorded operations
/// }
/// </code>
/// </remarks>
public sealed class CudaGraphScope : IDisposable
{
    private readonly IGpuBatchExecution _backend;
    private readonly IntPtr _stream;
    private IntPtr _graph;
    private IntPtr _graphExec;
    private bool _isCapturing;
    private bool _hasGraph;
    private bool _disposed;

    /// <summary>
    /// Gets whether the backend supports graph capture (requires CUDA backend).
    /// </summary>
    public bool IsSupported { get; }

    /// <summary>
    /// Gets whether a graph is currently being captured.
    /// </summary>
    public bool IsCapturing => _isCapturing;

    /// <summary>
    /// Gets whether a graph has been captured and is ready for replay.
    /// </summary>
    public bool HasGraph => _hasGraph;

    /// <summary>
    /// Creates a new CUDA graph scope.
    /// </summary>
    /// <param name="backend">The GPU backend to use for graph operations.</param>
    /// <param name="stream">The CUDA stream to capture on. Use IntPtr.Zero for the default stream.</param>
    public CudaGraphScope(IGpuBatchExecution backend, IntPtr stream = default)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _stream = stream;

        // Check if this is actually a CUDA backend that supports graph capture
        IsSupported = CudaNativeBindings.IsAvailable;
    }

    /// <summary>
    /// Begins capturing GPU operations into a graph.
    /// All subsequent GPU operations on this stream will be recorded instead of executed.
    /// </summary>
    public void BeginCapture()
    {
        if (_isCapturing)
            throw new InvalidOperationException("Graph capture is already in progress.");

        if (!IsSupported)
            throw new NotSupportedException("CUDA graph capture requires a CUDA-capable GPU.");

        // Clean up any previous graph
        CleanupGraph();

        var result = CudaNativeBindings.cuStreamBeginCapture(
            _stream, CudaNativeBindings.CU_STREAM_CAPTURE_MODE_GLOBAL);

        if (result != CudaResult.Success)
            throw new InvalidOperationException($"cuStreamBeginCapture failed: {result}");

        _isCapturing = true;
    }

    /// <summary>
    /// Ends the graph capture. The recorded operations can be replayed via <see cref="Replay"/>.
    /// </summary>
    public void EndCapture()
    {
        if (!_isCapturing)
            throw new InvalidOperationException("No graph capture in progress.");

        var result = CudaNativeBindings.cuStreamEndCapture(_stream, out _graph);
        _isCapturing = false;

        if (result != CudaResult.Success)
            throw new InvalidOperationException($"cuStreamEndCapture failed: {result}");

        // Instantiate the graph for execution
        result = CudaNativeBindings.cuGraphInstantiate(
            out _graphExec, _graph, IntPtr.Zero, IntPtr.Zero, 0);

        if (result != CudaResult.Success)
        {
            CudaNativeBindings.cuGraphDestroy(_graph);
            _graph = IntPtr.Zero;
            throw new InvalidOperationException($"cuGraphInstantiate failed: {result}");
        }

        _hasGraph = true;
    }

    /// <summary>
    /// Replays the captured graph with zero kernel launch overhead.
    /// </summary>
    public void Replay()
    {
        if (!_hasGraph)
            throw new InvalidOperationException("No graph has been captured. Call BeginCapture/EndCapture first.");

        var result = CudaNativeBindings.cuGraphLaunch(_graphExec, _stream);
        if (result != CudaResult.Success)
            throw new InvalidOperationException($"cuGraphLaunch failed: {result}");

        // Synchronize the stream to ensure the graph execution completes
        CudaNativeBindings.cuStreamSynchronize(_stream);
    }

    private void CleanupGraph()
    {
        if (_graphExec != IntPtr.Zero)
        {
            CudaNativeBindings.cuGraphExecDestroy(_graphExec);
            _graphExec = IntPtr.Zero;
        }
        if (_graph != IntPtr.Zero)
        {
            CudaNativeBindings.cuGraphDestroy(_graph);
            _graph = IntPtr.Zero;
        }
        _hasGraph = false;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_isCapturing)
        {
            // Cancel the capture
            try { CudaNativeBindings.cuStreamEndCapture(_stream, out var discardGraph); }
            catch { }
            _isCapturing = false;
        }

        CleanupGraph();
    }
}

/// <summary>
/// Configuration for pinned (page-locked) memory allocation.
/// Pinned memory enables faster CPU-GPU transfers by avoiding OS page faults.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Normal memory can be moved around by the operating system
/// (paged to disk, relocated, etc.). Pinned memory is locked in place, which means the GPU
/// can transfer data directly without waiting for the OS. This can make CPU-GPU transfers
/// 2-3x faster, but uses more memory.</para>
/// </remarks>
public static class PinnedMemoryConfig
{
    /// <summary>
    /// Gets or sets whether to use pinned memory for GPU uploads by default.
    /// When enabled, AllocateBuffer uses page-locked memory for faster DMA transfers.
    /// Default is false — enable for training workloads with frequent CPU-GPU transfers.
    /// </summary>
    public static bool UsePinnedMemoryByDefault { get; set; }

    /// <summary>
    /// Gets or sets the minimum tensor size (in elements) to use pinned memory.
    /// Tensors smaller than this threshold use regular memory (pinning overhead not worth it).
    /// Default is 4096 elements.
    /// </summary>
    public static int PinnedMemoryThreshold { get; set; } = 4096;
}
