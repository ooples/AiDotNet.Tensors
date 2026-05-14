namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Extension methods that surface the GPU stream-scheduler API
/// (<see cref="DirectGpuTensorEngine.GetStreamScheduler"/> /
/// <see cref="DirectGpuTensorEngine.CreateStreamPool"/>) on the
/// <see cref="IEngine"/> abstraction so interface-typed consumers can
/// discover the capability without a concrete-type cast.
///
/// PR #344 review: interface-typed consumers shouldn't need to know
/// the concrete engine type to participate in multi-stream execution.
/// These extensions probe the engine for the multi-stream capability
/// at runtime — CPU engines and GPU engines without multi-stream
/// backends return null cleanly, matching the same null-return contract
/// that <c>DirectGpuTensorEngine</c> itself uses.
///
/// This pattern (capability extension on the generic engine surface)
/// avoids polluting <see cref="IEngine"/> with GPU-specific types
/// while still giving callers a single interface-level API. Same
/// approach <see cref="GpuRoutingHelpers"/> (#342) uses for the
/// deferred-execution capability.
/// </summary>
public static class EngineStreamSchedulerExtensions
{
    /// <summary>
    /// Builds a <see cref="GpuStreamScheduler"/> on
    /// <paramref name="engine"/> using <paramref name="streamPool"/>, when
    /// the engine is a multi-stream-capable GPU engine. Returns null for
    /// CPU engines, GPU engines lacking an async backend, or GPU engines
    /// whose backend doesn't advertise <c>SupportsMultiStream</c>.
    /// </summary>
    /// <exception cref="System.ArgumentNullException">If
    /// <paramref name="streamPool"/> is null.</exception>
    /// <exception cref="System.ArgumentException">If
    /// <paramref name="streamPool"/> was created with a different backend
    /// than this engine's — cross-backend stream misuse would otherwise
    /// surface as opaque CUDA errors deep inside cuMemcpy.</exception>
    public static GpuStreamScheduler? GetStreamScheduler(
        this IEngine engine,
        GpuStreamPool streamPool,
        GpuStreamType streamType = GpuStreamType.Compute)
        => engine is DirectGpuTensorEngine gpu
            ? gpu.GetStreamScheduler(streamPool, streamType)
            : null;

    /// <summary>
    /// Builds a <see cref="GpuStreamPool"/> sized to
    /// <paramref name="engine"/>'s async backend's
    /// <see cref="IAsyncGpuBackend.MaxConcurrentStreams"/>. Returns null
    /// for engines without a multi-stream-capable backend (CPU, single-
    /// stream GPU). Caller owns the returned pool's lifetime.
    /// </summary>
    public static GpuStreamPool? CreateStreamPool(
        this IEngine engine,
        GpuExecutionOptions? options = null)
        => engine is DirectGpuTensorEngine gpu
            ? gpu.CreateStreamPool(options)
            : null;
}
