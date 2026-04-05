using System.Runtime.CompilerServices;

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

    /// <summary>Sets the current scope (used by LazyTensorScope.Dispose).</summary>
    internal static void SetCurrent(LazyTensorScope? scope) => _current = scope;
}
