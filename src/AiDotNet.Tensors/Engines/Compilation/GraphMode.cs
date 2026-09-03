using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal enum GraphTraceKind : byte
{
    Compatibility,
    Inference,
    Training
}

/// <summary>
/// Type-safe reasons an IEngine contract cannot currently be represented by a homogeneous,
/// fixed-shape inference plan.
/// </summary>
internal enum GraphCaptureLimitation : byte
{
    DataDependentOutputShape,
    HeterogeneousInput,
    HeterogeneousOutput,
    MixedElementTypes,
    HostBoundary,
    Stateful
}

/// <summary>
/// Ambient context toggle for lazy tensor evaluation. When active, tensor operations
/// record into a computation graph instead of executing immediately. The graph is then
/// optimized (fused) and compiled into a flat execution plan.
///
/// This is completely internal — the facade (PredictionModelBuilder) enables it
/// automatically during training/inference for maximum performance.
///
/// Pattern: [ThreadStatic] ambient context, same as GradientTape and DeferredScope.
/// Overhead when inactive: single null check (~2ns per operation).
/// </summary>
internal static class GraphMode
{
    [ThreadStatic]
    private static LazyTensorScope? _current;

    /// <summary>Gets the active lazy tensor scope for this thread, or null.</summary>
    internal static LazyTensorScope? Current
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _current;
    }

    /// <summary>Whether graph mode is active on this thread.</summary>
    internal static bool IsActive
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _current is not null;
    }

    /// <summary>
    /// Whether the active scope was explicitly opened for inference. Composite kernels may retain
    /// opaque fused nodes only in this mode. The compatibility scope deliberately returns false:
    /// callers that choose whether to compile inference or training only after tracing must receive
    /// a differentiable primitive graph rather than a silently non-differentiable training plan.
    /// </summary>
    internal static bool IsInferenceTrace
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _current?.TraceKind == GraphTraceKind.Inference;
    }

    /// <summary>
    /// Fails before an inference trace can partially execute an operation whose public contract is
    /// not representable by the current plan format. Compatibility and training traces are left
    /// untouched so their existing eager/decomposed behavior remains available.
    /// </summary>
    internal static void ThrowIfInferenceUnsupported(GraphCaptureLimitation limitation)
    {
        if (!IsInferenceTrace) return;

        string reason = limitation switch
        {
            GraphCaptureLimitation.DataDependentOutputShape =>
                "its output shape depends on runtime tensor values",
            GraphCaptureLimitation.HeterogeneousInput =>
                "its tensor inputs use different element types",
            GraphCaptureLimitation.HeterogeneousOutput =>
                "it returns tensor outputs with different element types",
            GraphCaptureLimitation.MixedElementTypes =>
                "its graph crosses tensor element types",
            GraphCaptureLimitation.HostBoundary =>
                "it crosses a host-only data boundary",
            GraphCaptureLimitation.Stateful =>
                "it mutates state during execution",
            _ => throw new ArgumentOutOfRangeException(nameof(limitation))
        };

        throw new NotSupportedException(
            $"This operation cannot be captured in an inference graph because {reason}.");
    }

    /// <summary>
    /// Enables graph mode and returns a scope. All tensor operations on this thread
    /// will record into the scope's computation graph until it is disposed.
    /// </summary>
    internal static LazyTensorScope Enable()
    {
        var scope = new LazyTensorScope(_current, GraphTraceKind.Compatibility);
        _current = scope;
        return scope;
    }

    /// <summary>
    /// Enables a graph scope whose result will be compiled for inference. This explicit intent lets
    /// inference-only composite kernels remain fused while <see cref="Enable"/> retains the safe,
    /// differentiable compatibility behavior required by older training callers.
    /// </summary>
    internal static LazyTensorScope EnableInference()
    {
        var scope = new LazyTensorScope(_current, GraphTraceKind.Inference);
        _current = scope;
        return scope;
    }

    /// <summary>
    /// Enables graph mode for a training trace. Copy-on-write parameters are
    /// privatized before any graph node or parameter-derived view can capture
    /// their storage. Every production training compiler must enter through
    /// this method rather than <see cref="Enable"/>.
    /// </summary>
    internal static LazyTensorScope EnableTraining<T>(Tensor<T>[] parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        for (int i = 0; i < parameters.Length; i++)
        {
            if (parameters[i] is null)
                throw new ArgumentException("Training parameters cannot contain null tensors.", nameof(parameters));
            parameters[i].PrepareForInPlaceWrite();
        }

        var scope = new LazyTensorScope(_current, GraphTraceKind.Training);
        _current = scope;
        return scope;
    }

    /// <summary>
    /// Temporarily disables graph recording while a compiled plan replays.
    /// Generic plan delegates call ordinary engine operations; without this
    /// boundary they can accidentally append to an ambient trace instead of
    /// executing, producing silent zero or stale gradients.
    /// </summary>
    internal static RecordingSuspension SuspendRecording()
    {
        var previous = _current;
        _current = null;
        return new RecordingSuspension(previous);
    }

    internal readonly struct RecordingSuspension : IDisposable
    {
        private readonly LazyTensorScope? _previous;

        internal RecordingSuspension(LazyTensorScope? previous)
        {
            _previous = previous;
        }

        public void Dispose()
        {
            _current = _previous;
        }
    }

    /// <summary>Sets the current scope (used by LazyTensorScope.Dispose).</summary>
    internal static void SetCurrent(LazyTensorScope? scope) => _current = scope;
}
