using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Small deterministic LRU for loaded modules. Callers serialize lookup,
/// launch, and eviction so an executing module can never be unloaded.
/// </summary>
internal sealed class DirectPtxKernelCache<TKey, TKernel> : IDisposable
    where TKey : notnull
    where TKernel : class, IDisposable
{
    private readonly int _capacity;
    private readonly Dictionary<TKey, LinkedListNode<Entry>> _entries = new();
    private readonly LinkedList<Entry> _lru = new();

    private sealed class Entry
    {
        internal Entry(TKey key, TKernel kernel)
        {
            Key = key;
            Kernel = kernel;
        }

        internal TKey Key { get; }
        internal TKernel Kernel { get; }
        internal bool IsPermanentlyPinned { get; set; }
        internal int CapturePinCount { get; set; }
        internal bool IsPinned => IsPermanentlyPinned || CapturePinCount != 0;
    }

    internal DirectPtxKernelCache(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        _capacity = capacity;
    }

    internal int Count => _entries.Count;
    internal int Capacity => _capacity;
    internal int PinnedCount
    {
        get
        {
            int count = 0;
            foreach (Entry entry in _lru)
                if (entry.IsPinned) count++;
            return count;
        }
    }

    internal bool TryGetValue(TKey key, out TKernel kernel)
    {
        if (_entries.TryGetValue(key, out LinkedListNode<Entry>? node))
        {
            _lru.Remove(node);
            _lru.AddFirst(node);
            kernel = node.Value.Kernel;
            return true;
        }
        kernel = null!;
        return false;
    }

    /// <summary>
    /// Permanently prevents eviction of a loaded module whose lifetime cannot
    /// be observed by the cache. Captures owned by <c>CudaBackend.CaptureGraph</c>
    /// use the reference-counted capture-pin methods below instead.
    /// </summary>
    internal bool Pin(TKey key)
    {
        if (!_entries.TryGetValue(key, out LinkedListNode<Entry>? node))
            return false;
        node.Value.IsPermanentlyPinned = true;
        return true;
    }

    /// <summary>
    /// Acquires a graph-lifetime pin. Unlike <see cref="Pin"/>, this pin can be
    /// released when the graph-exec that retains the module is destroyed.
    /// </summary>
    internal bool AcquireCapturePin(TKey key)
    {
        if (!_entries.TryGetValue(key, out LinkedListNode<Entry>? node))
            return false;
        node.Value.CapturePinCount = checked(node.Value.CapturePinCount + 1);
        return true;
    }

    internal void ReleaseCapturePin(TKey key)
    {
        if (!_entries.TryGetValue(key, out LinkedListNode<Entry>? node) ||
            node.Value.CapturePinCount == 0)
            throw new InvalidOperationException(
                "The direct-PTX capture pin is not owned by this cache entry.");
        node.Value.CapturePinCount--;
    }

    internal TKernel GetOrAdd(TKey key, Func<TKernel> factory)
    {
        PtxCompat.ThrowIfNull(factory, nameof(factory));
        if (TryGetValue(key, out TKernel existing)) return existing;
        if (!HasRoomOrEvictableEntry())
            throw new InvalidOperationException(
                "The direct-PTX module cache is full of CUDA-graph-pinned kernels.");

        return AddOrGetExisting(key, factory());
    }

    /// <summary>
    /// Adds an already-created kernel. This overload lets allocation-sensitive
    /// dispatchers keep factory delegates and their closure objects entirely on
    /// the cache-miss path.
    /// </summary>
    internal TKernel AddOrGetExisting(TKey key, TKernel created)
    {
        PtxCompat.ThrowIfNull(created, nameof(created));
        if (TryGetValue(key, out TKernel existing))
        {
            created.Dispose();
            return existing;
        }
        if (_entries.Count >= _capacity)
        {
            LinkedListNode<Entry>? victim = _lru.Last;
            while (victim is not null && victim.Value.IsPinned)
                victim = victim.Previous;
            if (victim is null)
            {
                created.Dispose();
                throw new InvalidOperationException(
                    "The direct-PTX module cache is full of CUDA-graph-pinned kernels.");
            }
            _lru.Remove(victim);
            _entries.Remove(victim.Value.Key);
            victim.Value.Kernel.Dispose();
        }
        var node = new LinkedListNode<Entry>(new Entry(key, created));
        _entries.Add(key, node);
        _lru.AddFirst(node);
        return created;
    }

    private bool HasRoomOrEvictableEntry()
    {
        if (_entries.Count < _capacity) return true;
        for (LinkedListNode<Entry>? node = _lru.Last; node is not null; node = node.Previous)
            if (!node.Value.IsPinned) return true;
        return false;
    }

    public void Dispose()
    {
        foreach (Entry entry in _lru) entry.Kernel.Dispose();
        _entries.Clear();
        _lru.Clear();
    }
}

/// <summary>Bounded LRU for small immutable dispatch-plan values.</summary>
internal sealed class DirectPtxPlanCache<TKey, TValue> where TKey : notnull
{
    private readonly int _capacity;
    private readonly Dictionary<TKey, LinkedListNode<(TKey Key, TValue Value)>> _entries = new();
    private readonly LinkedList<(TKey Key, TValue Value)> _lru = new();

    internal DirectPtxPlanCache(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        _capacity = capacity;
    }

    internal bool TryGetValue(TKey key, out TValue value)
    {
        if (_entries.TryGetValue(key, out var node))
        {
            _lru.Remove(node);
            _lru.AddFirst(node);
            value = node.Value.Value;
            return true;
        }
        value = default!;
        return false;
    }

    internal void Set(TKey key, TValue value)
    {
        if (_entries.TryGetValue(key, out var existing))
        {
            existing.Value = (key, value);
            _lru.Remove(existing);
            _lru.AddFirst(existing);
            return;
        }
        var node = new LinkedListNode<(TKey Key, TValue Value)>((key, value));
        _entries.Add(key, node);
        _lru.AddFirst(node);
        if (_entries.Count <= _capacity) return;
        LinkedListNode<(TKey Key, TValue Value)> victim = _lru.Last!;
        _lru.RemoveLast();
        _entries.Remove(victim.Value.Key);
    }

    internal void Clear()
    {
        _entries.Clear();
        _lru.Clear();
    }
}
