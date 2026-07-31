using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

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
    /// Enables graph mode and returns a scope. All tensor operations on this thread
    /// will record into the scope's computation graph until it is disposed.
    /// </summary>
    internal static LazyTensorScope Enable()
    {
        var scope = new LazyTensorScope(_current);
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

        var scope = new LazyTensorScope(_current, trainingParametersPreparedBeforeTrace: true);
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
