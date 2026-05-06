using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.Execution;

/// <summary>
/// Public factory for <see cref="IExecutionStream{T}"/> instances. The
/// concrete <c>Cpu</c> / <c>Gpu</c> stream classes are internal so the
/// stream contract stays uniform; consumers obtain a stream by passing
/// an engine and the framework picks the right backend (CPU worker pool
/// vs native GPU stream). This keeps the
/// <see cref="ICompiledPlan{T}.ExecuteAsync"/> caller code backend-
/// agnostic — same call site, same await semantics on either side.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a factory instead of an <see cref="IEngine"/> member:</b>
/// adding a member to <see cref="IEngine"/> is binary-breaking for
/// external implementers (the same constraint repeatedly cited in this
/// codebase under issues #166 / #170 / #199 — net471 has no default
/// interface members so we can't polyfill). Routing through a static
/// factory keeps the interface frozen and lets the framework evolve
/// the backend pickers without breaking downstream.
/// </para>
/// <para>
/// <b>Backend selection:</b> if the engine exposes a non-null
/// <see cref="IEngine.DirectGpu"/> with an active CUDA stream, a
/// <c>GpuExecutionStream</c> wrapping that stream is returned. CPU
/// engines (and any GPU engine without an active stream) get a
/// <c>CpuExecutionStream</c> backed by a <see cref="System.Threading.Channels.Channel{T}"/>
/// + long-lived worker. Disposing the returned stream is the caller's
/// responsibility.
/// </para>
/// </remarks>
internal static class ExecutionStreamFactory
{
    /// <summary>
    /// Creates a fresh execution stream appropriate for the given
    /// engine. CPU engines get a worker-pool-backed stream;
    /// CUDA-capable engines get a wrapper around the engine's existing
    /// <see cref="CudaBackend.DefaultStream"/> (NOT a new
    /// <c>cudaStream_t</c> — kernels dispatched through the engine
    /// route via the backend's internally-managed stream context, so
    /// using a different stream would let the kernels enqueue against
    /// the wrong target). Closes review-comment #298.8R5W.
    /// </summary>
    /// <typeparam name="T">The tensor element type.</typeparam>
    /// <param name="engine">The engine the stream will dispatch through.</param>
    /// <returns>A new <see cref="IExecutionStream{T}"/>. The caller owns
    /// the disposable; call <see cref="IExecutionStream{T}.DisposeAsync"/>
    /// (or <see cref="IDisposable.Dispose"/>) when the stream is no
    /// longer needed.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/>
    /// is null.</exception>
    internal static IExecutionStream<T> CreateForEngine<T>(IEngine engine)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));

        // Probe the engine's DirectGpu surface for a CUDA backend with
        // an active stream. If present, wrap that stream — kernels
        // dispatched through the engine route via the stream context
        // already set up by CudaBackend. We DO NOT create a new
        // cudaStream_t here because the engine wires kernels against
        // its own _stream field; using a different stream would let
        // the kernels enqueue against the wrong target.
        var directGpu = engine.DirectGpu;
        if (directGpu is not null)
        {
            var cudaStream = TryGetCudaStream(directGpu);
            if (cudaStream is not null)
            {
                // ownsStream=false because the stream's lifetime is
                // bound to the CudaBackend, not to this wrapper.
                return new GpuExecutionStream<T>(cudaStream, ownsStream: false);
            }
        }

        // CPU fallback. The Channel<>-backed worker is appropriate for
        // any engine that doesn't expose a native device stream — most
        // notably CpuEngine, but also DirectGpuEngine instances built
        // against backends without an active CUDA context (early init,
        // unit tests on CI without a GPU, etc.).
        return new CpuExecutionStream<T>();
    }

    /// <summary>
    /// Direct lookup of the active GPU stream on a
    /// <see cref="DirectGpuEngine"/>. Routes through the public
    /// <see cref="DirectGpuEngine.Backend"/> property and the public
    /// <c>DefaultStream</c> on whichever <c>IDirectGpuBackend</c>
    /// implementation is wired in (CUDA / OpenCL / Vulkan / Metal —
    /// all expose <c>IGpuStream DefaultStream</c> via their public
    /// API). No reflection: a future backend rename surfaces as a
    /// compile-time error rather than a silent CPU fallback. Returns
    /// null when there's no active backend, in which case the factory
    /// falls through to the CPU stream.
    /// </summary>
    private static IGpuStream? TryGetCudaStream(DirectGpuEngine directGpu)
    {
        var backend = directGpu.Backend;
        if (backend is null) return null;

        // CudaBackend exposes DefaultStream as a public IGpuStream
        // property. The IDirectGpuBackend interface itself doesn't
        // declare DefaultStream (other backends — OpenCL, Vulkan, Metal
        // — manage streams differently), so the lookup is concrete-type
        // routed: typed cast on CudaBackend, fall through for everything
        // else. Switching to a direct cast (no reflection) means a
        // future CudaBackend rename surfaces as a compile-time error
        // rather than a silent CPU fallback.
        if (backend is CudaBackend cudaBackend)
        {
            try
            {
                return cudaBackend.DefaultStream;
            }
            catch (InvalidOperationException)
            {
                // DefaultStream throws if the backend isn't fully
                // initialized yet (lazy-load races). Fall through to
                // CPU so the plan still runs.
                return null;
            }
        }

        // Non-CUDA direct-GPU backends: no execution-stream wrapper
        // surface yet. Future work for HIP / Vulkan / Metal goes here.
        return null;
    }
}
