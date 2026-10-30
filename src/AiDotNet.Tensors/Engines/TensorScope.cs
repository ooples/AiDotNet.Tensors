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
/// a result handed to a gradient tape. A tensor that existed before the scope stays its owner's even when an op inside
/// the scope writes it (a destination handed to an *Into op). Reading a freed result after the scope ends throws an
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
    // Storage keys (vectors and host arrays) of the kept results: the activation-cache release below spares them.
    private readonly HashSet<object> _kept = new(Autodiff.ReferenceEqualityComparer<object>.Instance);
    // Engines whose activation cache gained entries in this scope, with the cache timestamp from before the first one.
    private Dictionary<DirectGpuTensorEngine, long>? _activationOwners;
    // Storage keys (vectors and host arrays) of the tensors created on this thread while the scope, or a scope nested in it,
    // was open. The activation-cache release frees only entries keyed by these: a tensor that existed before the scope (a
    // destination handed to an *Into op, a parameter written in place) is the caller's, and so is an engine cache built while
    // the scope was suspended (a folded convolution kernel). Held weakly: a whole inference call's intermediates, host
    // arrays included, stay collectable while the scope is open.
    private readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, object> _created = new();
    // Declared before s_mark: static fields initialize in order, and the callback returns it.
    private static readonly object s_marker = new();
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, object>.CreateValueCallback s_mark = _ => s_marker;
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
        _kept.Add(storage);
        if (tensor.GetBackingArrayForCacheLookupUnsafe() is { } array) _kept.Add(array);
        if (_results.TryGetValue(storage, out var release))
        {
            _results.Remove(storage);
            _outer?.Adopt(storage, release);
        }
        return tensor;
    }

    /// <summary>
    /// Hands <paramref name="tensor"/> (and every view of its storage) back to its own lifetime: no scope open on the
    /// calling thread frees it, the enclosing ones included, as TorchSharp's <c>DetachFromDisposeScope</c> does. For a
    /// tensor something longer-lived than every open scope holds, such as a cache a layer builds on its first forward.
    /// Returns <paramref name="tensor"/>.
    /// </summary>
    public static Tensor<T> Detach<T>(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        object storage = tensor.DataVector;
        var array = tensor.GetBackingArrayForCacheLookupUnsafe();
        // A result kept in a nested scope was adopted by the enclosing one, and each scope's activation release spares
        // only the storages kept in it or around it: every open scope forgets the result and spares it.
        for (var scope = s_current; scope is not null; scope = scope._outer)
        {
            scope._results.Remove(storage);
            scope._kept.Add(storage);
            if (array is not null) scope._kept.Add(array);
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
    /// <summary>
    /// Records a tensor storage created on this thread (its vector, and its host array when it has one) with the innermost
    /// open scope and each scope enclosing it. Callers check <see cref="Current"/> first, so a storage made with no scope
    /// open costs one thread-static read.
    /// </summary>
    internal static void OnStorageCreated(object vector, object? array)
    {
        for (var scope = s_current; scope is not null; scope = scope._outer)
        {
            scope._created.GetValue(vector, s_mark);
            if (array is not null) scope._created.GetValue(array, s_mark);
        }
    }

    private bool Created(object key) => _created.TryGetValue(key, out _);

    /// <summary>Records a GPU result created on this thread with the innermost open scope, if any.</summary>
    internal static void Track(object storage, Action release)
    {
        var scope = s_current;
        // A buffer created while a graph is captured belongs to the compiled plan, which outlives any scope.
        if (scope is null || scope._threadId != Environment.CurrentManagedThreadId || Compilation.GraphMode.IsActive) return;
        scope._results[storage] = release;
    }

    /// <summary>
    /// Records that <paramref name="engine"/> is about to cache an activation on this thread, with its cache timestamp
    /// from before that entry: when the scope (and each enclosing one) ends, the engine frees the entries created after
    /// it that nothing kept, as a finished gradient tape does. Results that live in the activation cache instead of
    /// owning their buffers (most element-wise and reduction results) are freed this way.
    /// </summary>
    internal static void EnlistActivationOwner(DirectGpuTensorEngine engine, long timestampBeforeEntry)
    {
        var scope = s_current;
        if (scope is null || scope._threadId != Environment.CurrentManagedThreadId || Compilation.GraphMode.IsActive) return;
        for (; scope is not null; scope = scope._outer)
        {
            scope._activationOwners ??= new Dictionary<DirectGpuTensorEngine, long>();
            if (scope._activationOwners.ContainsKey(engine)) break;   // enclosing scopes enlisted it earlier
            scope._activationOwners[engine] = timestampBeforeEntry;
        }
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
        // Inside a deferred-execution scope the recorded graph still references these buffers until it executes, and not every
        // backend's buffers consult the deferral gate on disposal (HIP's do not): queue the whole release behind the graph.
        if (!Gpu.GpuBufferReleaseDeferral.TryDefer(Release)) Release();
    }

    private void Release()
    {
        foreach (var release in _results.Values) release();
        _results.Clear();
        if (_activationOwners is not null)
        {
            // The kept results of enclosing scopes are this scope's results too until they end: spare them all.
            var spared = new HashSet<object>(_kept, Autodiff.ReferenceEqualityComparer<object>.Instance);
            for (var outer = _outer; outer is not null; outer = outer._outer) spared.UnionWith(outer._kept);
            foreach (var owner in _activationOwners)
                owner.Key.EvictActivationsCreatedAfter(owner.Value, spared, ActivationReleaseMode.Release,
                    releasableOnly: Created, afterQueuedWork: true);
        }
        _activationOwners = null;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TensorScope));
    }
}
