// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Implemented by <see cref="VectorBase{T}"/> so <see cref="TensorScope"/> can record scope creation
/// via a monotonic thread-local tick without allocating <see cref="ConditionalWeakTable{TKey, TValue}"/>
/// dependent handles on every GPU-resident intermediate tensor.
/// </summary>
internal interface IScopeTrackableStorage
{
    int CreatedThreadId { get; set; }
    long CreatedScopeTick { get; set; }
}

/// <summary>
/// Implemented by <see cref="Tensor{T}"/> so <see cref="TensorScope"/> can release device-resident
/// intermediate tensors directly without per-op closure and delegate allocations.
/// </summary>
internal interface IScopedDeviceTensor
{
    void ReleaseFromScope(DirectGpuTensorEngine engine);
}

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

    [ThreadStatic]
    private static int s_suspendedDepth;

    [ThreadStatic]
    private static long s_clock;

    private TensorScope? _outer;
    private readonly int _threadId;
    private readonly long _openTick;
    // Keyed by storage (the vector), so a kept view keeps the result it views.
    // Value is either (DirectGpuTensorEngine, IScopedDeviceTensor) for zero-closure GPU tensor tracking
    // or (null, Action) for custom callbacks.
    private readonly Dictionary<object, (DirectGpuTensorEngine? Engine, object Target)> _results =
        new(Autodiff.ReferenceEqualityComparer<object>.Instance);
    // Storage keys (vectors and host arrays) of the kept results: the activation-cache release below spares them.
    private readonly HashSet<object> _kept = new(Autodiff.ReferenceEqualityComparer<object>.Instance);
    // Engines whose activation cache gained entries in this scope, with the cache timestamp from before the first one.
    private Dictionary<DirectGpuTensorEngine, long>? _activationOwners;
    // Fallback weak table used only for raw host arrays or storages created while an outer scope is suspended.
    private ConditionalWeakTable<object, object>? _created;
    // Declared before s_mark: static fields initialize in order, and the callback returns it.
    private static readonly object s_marker = new();
    private static readonly ConditionalWeakTable<object, object>.CreateValueCallback s_mark = _ => s_marker;
    private Func<object, bool>? _isReleasablePredicate;
    private bool _releasing;
    private bool _disposed;

    private TensorScope(TensorScope? outer)
    {
        _outer = outer;
        _threadId = Environment.CurrentManagedThreadId;
        _openTick = ++s_clock;
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

    /// <summary>Whether the calling thread currently has an open or temporarily suspended <see cref="TensorScope"/>.</summary>
    internal static bool HasActiveOrSuspendedScope => s_current is not null || s_suspendedDepth > 0;

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
        if (_results.TryGetValue(storage, out var entry))
        {
            _results.Remove(storage);
            _outer?.AdoptEntry(storage, entry);
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
        for (var scope = s_current; scope is not null; scope = scope._outer)
        {
            bool removed = scope._results.Remove(storage);
            // The array is checked too: a scope-created alias of a pre-existing tensor wraps the same array in a new
            // vector, registering the array (not this tensor's vector) with the scope.
            if (removed || scope.Created(storage) || (array is not null && scope.Created(array)))
            {
                scope._kept.Add(storage);
                if (array is not null) scope._kept.Add(array);
            }
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
        if (suspended is not null)
        {
            s_current = null;
            s_suspendedDepth++;
        }
        return new SuspendHandle(suspended, Environment.CurrentManagedThreadId);
    }

    internal struct SuspendHandle : IDisposable
    {
        private TensorScope? _suspended;
        private readonly int _threadId;
        internal SuspendHandle(TensorScope? suspended, int threadId)
        {
            _suspended = suspended;
            _threadId = threadId;
        }

        public void Dispose()
        {
            if (Environment.CurrentManagedThreadId != _threadId) return;
            var suspended = System.Threading.Interlocked.Exchange(ref _suspended, null);
            if (suspended is null) return;
            s_suspendedDepth--;
            if (s_current is null)
            {
                s_current = suspended;
            }
            else
            {
                var root = s_current;
                while (root._outer is not null)
                    root = root._outer;
                root._outer = suspended;
            }
        }
    }

    /// <summary>
    /// Records a tensor storage created on this thread (its vector, and its host array when it has one) with the innermost
    /// open scope and each scope enclosing it. Callers check <see cref="Current"/> first, so a storage made with no scope
    /// open costs one thread-static read.
    /// </summary>
    internal static void OnStorageCreated(object vector, object? array)
    {
        var current = s_current;
        if (current is null) return;

        if (s_suspendedDepth == 0 && vector is IScopeTrackableStorage trackable)
        {
            trackable.CreatedThreadId = current._threadId;
            trackable.CreatedScopeTick = ++s_clock;
            if (array is not null)
            {
                for (var scope = current; scope is not null; scope = scope._outer)
                {
                    (scope._created ??= new ConditionalWeakTable<object, object>()).GetValue(array, s_mark);
                }
            }
            return;
        }

        for (var scope = current; scope is not null; scope = scope._outer)
        {
            var table = scope._created ??= new ConditionalWeakTable<object, object>();
            table.GetValue(vector, s_mark);
            if (array is not null) table.GetValue(array, s_mark);
        }
    }

    private bool Created(object key)
    {
        if (key is IScopeTrackableStorage trackable
            && trackable.CreatedScopeTick > _openTick
            && trackable.CreatedThreadId == _threadId)
        {
            return true;
        }
        return _created is not null && _created.TryGetValue(key, out _);
    }

    /// <summary>Records a GPU result created on this thread with the innermost open scope, if any.</summary>
    internal static void Track(object storage, Action release)
    {
        var scope = s_current;
        // A buffer created while a graph is captured belongs to the compiled plan, which outlives any scope.
        if (scope is null || scope._threadId != Environment.CurrentManagedThreadId || Compilation.GraphMode.IsActive) return;
        scope._results[storage] = (null, release);
    }

    /// <summary>
    /// Records a device-resident tensor result on the innermost open scope without allocating a closure or delegate.
    /// </summary>
    internal static void TrackTensor(object storage, DirectGpuTensorEngine engine, IScopedDeviceTensor tensor)
    {
        var scope = s_current;
        if (scope is null || scope._threadId != Environment.CurrentManagedThreadId || Compilation.GraphMode.IsActive) return;
        scope._results[storage] = (engine, tensor);
    }

    /// <summary>
    /// Drops a storage from active scopes once its device buffer has already been released mid-scope
    /// (for example by <c>BackwardStorageReleasePlan</c> or <c>ReleaseDroppedGradients</c>), allowing the
    /// managed <see cref="Tensor{T}"/> wrapper to be collected in Gen-0 instead of surviving to Gen-2.
    /// </summary>
    internal static void Untrack(object storage)
    {
        int tid = Environment.CurrentManagedThreadId;
        for (var scope = s_current; scope is not null; scope = scope._outer)
        {
            if (scope._threadId == tid && !scope._releasing)
            {
                scope._results.Remove(storage);
            }
        }
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

    private void AdoptEntry(object storage, (DirectGpuTensorEngine? Engine, object Target) entry)
    {
        if (!_disposed) _results[storage] = entry;
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
        if (!(Gpu.GpuBufferReleaseDeferral.IsActive && Gpu.GpuBufferReleaseDeferral.TryDefer(Release))) Release();
    }

    private bool IsReleasable(object key)
    {
        if (!Created(key)) return false;
        for (var scope = this; scope is not null; scope = scope._outer)
        {
            if (scope._kept.Contains(key)) return false;
        }
        return true;
    }

    private void Release()
    {
        _releasing = true;
        try
        {
            foreach (var entry in _results.Values)
            {
                if (entry.Engine is { } engine && entry.Target is IScopedDeviceTensor scopedTensor)
                {
                    scopedTensor.ReleaseFromScope(engine);
                }
                else if (entry.Target is Action release)
                {
                    release();
                }
            }
        }
        finally
        {
            _releasing = false;
            _results.Clear();
        }
        if (_activationOwners is not null)
        {
            var releasable = _isReleasablePredicate ??= IsReleasable;
            foreach (var owner in _activationOwners)
                owner.Key.EvictActivationsCreatedAfter(owner.Value, protect: null, ActivationReleaseMode.Release,
                    releasableOnly: releasable, afterQueuedWork: true);
        }
        _activationOwners = null;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TensorScope));
    }
}
