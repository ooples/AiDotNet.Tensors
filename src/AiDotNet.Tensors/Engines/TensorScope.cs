// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Frees the GPU memory of the results created inside it when it ends, except the ones passed to
/// <see cref="Keep{T}"/>, as TorchSharp's <c>DisposeScope</c> does.
/// </summary>
/// <remarks>
/// <para>
/// A GPU result owns its device buffer, and .NET frees it only when the garbage collector finalizes the tensor. An
/// inference call allocates far more device memory than managed memory, so the dead intermediates of a forward pass
/// were reclaimed by a forced collection: about 1.3 ms of a 10 ms ResNet-50 predict on OpenCL (#1432). Inside a scope,
/// those intermediates go back to the pool when it ends, with no collection:
/// </para>
/// <code>
/// using var scope = TensorScope.Begin();
/// var logits = model.Forward(input);
/// return scope.Keep(logits);
/// </code>
/// <para>
/// Every GPU result created on this thread while the scope is open belongs to it, including a layer's cached input or
/// a result handed to a gradient tape. Reading one after the scope ends throws an
/// <see cref="InvalidOperationException"/>; it never returns stale data. Keep what outlives the scope, and do not run
/// a backward pass across a scope's end. Scopes nest: <see cref="Keep{T}"/> inside a nested scope hands the result to
/// the enclosing one, which frees it when it ends unless it is kept there too. Results on the CPU are unaffected.
/// </para>
/// </remarks>
public sealed class TensorScope : IDisposable
{
    [ThreadStatic]
    private static TensorScope? s_current;

    private readonly TensorScope? _outer;
    private readonly int _threadId;
    // Keyed by storage (the vector), so a kept view keeps the result it views.
    private readonly Dictionary<object, Action> _results = new(Autodiff.ReferenceEqualityComparer<object>.Instance);
    private bool _disposed;

    private TensorScope(TensorScope? outer)
    {
        _outer = outer;
        _threadId = Environment.CurrentManagedThreadId;
    }

    /// <summary>Opens a scope on the calling thread. Dispose it on the same thread, innermost first.</summary>
    public static TensorScope Begin()
    {
        var scope = new TensorScope(s_current);
        s_current = scope;
        return scope;
    }

    /// <summary>The innermost scope open on the calling thread, or null.</summary>
    internal static TensorScope? Current => s_current;

    /// <summary>
    /// Keeps <paramref name="tensor"/> (and every view of its storage) past this scope: to the enclosing scope when one
    /// is open, otherwise to the tensor's own lifetime. Returns <paramref name="tensor"/>.
    /// </summary>
    public Tensor<T> Keep<T>(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        ThrowIfDisposed();
        object storage = tensor.DataVector;
        if (_results.TryGetValue(storage, out var release))
        {
            _results.Remove(storage);
            _outer?.Adopt(storage, release);
        }
        return tensor;
    }

    /// <summary>The number of results this scope will free when it ends.</summary>
    public int TrackedCount => _results.Count;

    /// <summary>
    /// Stops tracking on this thread until the returned handle is disposed: for engine caches built during a call (a
    /// folded kernel, say), which outlive the scope that happened to be open.
    /// </summary>
    internal static SuspendHandle Suspend()
    {
        var suspended = s_current;
        s_current = null;
        return new SuspendHandle(suspended);
    }

    internal readonly struct SuspendHandle : IDisposable
    {
        private readonly TensorScope? _suspended;
        internal SuspendHandle(TensorScope? suspended) => _suspended = suspended;
        public void Dispose() => s_current = _suspended;
    }
    /// <summary>Records a GPU result created on this thread with the innermost open scope, if any.</summary>
    internal static void Track(object storage, Action release)
    {
        var scope = s_current;
        if (scope is null || scope._threadId != Environment.CurrentManagedThreadId) return;
        scope._results[storage] = release;
    }

    private void Adopt(object storage, Action release)
    {
        if (!_disposed) _results[storage] = release;
    }

    /// <summary>Frees the device memory of every result created in this scope and not kept.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        if (!ReferenceEquals(s_current, this))
            throw new InvalidOperationException(
                "TensorScope disposed out of order or on another thread: dispose nested scopes innermost first, on the thread that began them.");
        _disposed = true;
        s_current = _outer;
        foreach (var release in _results.Values) release();
        _results.Clear();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TensorScope));
    }
}
