using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Arena-style allocator for tensor training loops. Pre-allocates arrays during the first
/// iteration, then reuses them for subsequent iterations — zero allocation, zero GC after warmup.
///
/// <para><b>Why this beats PyTorch:</b> PyTorch CPU tensors use malloc/free per tensor, causing
/// fragmentation and GC pauses. TensorArena eliminates per-tensor allocation after the first
/// iteration by reusing the same arrays via a ring buffer of pooled arrays.</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// using var arena = TensorArena.Create();
/// for (int epoch = 0; epoch &lt; epochs; epoch++)
/// {
///     arena.Reset(); // Reuse arrays from previous iteration
///     var output = model.Forward(input);  // Allocations come from arena
///     var grads = model.Backward(loss);   // Zero new GC allocations after warmup
/// }
/// </code>
///
/// <para><b>How it works:</b> On first use (warmup), arrays are allocated normally and tracked.
/// On Reset(), the cursor rewinds to 0 — subsequent Rent calls return the same arrays.
/// Arrays are only allocated if the arena doesn't have one of the right size.</para>
///
/// <para><b>Thread safety:</b> Each thread gets its own arena via [ThreadStatic]. No locks.</para>
/// </summary>
public sealed class TensorArena : IDisposable
{
    [ThreadStatic]
    private static TensorArena? _current;

    /// <summary>
    /// SCRATCH pool: short-lived per-step intermediates. Pool of arrays allocated
    /// during warmup, keyed by (element type, element count). Cursors rewind on
    /// <see cref="Reset"/> so the next iteration reuses these arrays. This is
    /// what every <see cref="TensorAllocator.Rent{T}(int[])"/> call lands in.
    /// </summary>
    private readonly Dictionary<(Type, int), List<Array>> _pool = new();

    /// <summary>
    /// Scratch-pool reuse cursor — how many arrays of each (type, size) have
    /// been handed out since the last <see cref="Reset"/>.
    /// </summary>
    private readonly Dictionary<(Type, int), int> _cursor = new();

    /// <summary>
    /// Reusable scratch buffer for <see cref="Reset"/>'s key snapshot, so the
    /// steady-state reset is allocation-free (the arena's key set is stable
    /// across training iterations). Grown only when the key count rises.
    /// </summary>
    private (Type, int)[]? _resetKeyScratch;

    /// <summary>
    /// PINNED list: long-lived allocations that survive <see cref="Reset"/>.
    /// Model weights (from layer EnsureInitialized), optimizer moment state
    /// (Adam m / v), running BatchNorm statistics, positional embeddings —
    /// anything carried across training steps lives here. Tracked so
    /// <see cref="Dispose"/> can drop references for GC, but the cursor never
    /// rewinds, so Reset can't accidentally hand a pinned array out as scratch
    /// on the next iteration. This eliminates the corruption hazard that
    /// would otherwise happen if model weights and per-step intermediates
    /// shared one cursor: rewinding the cursor would let Rent re-issue the
    /// weight tensor's underlying array as scratch storage.
    /// </summary>
    private readonly List<Array> _pinnedArrays = new();

    // Backing arrays handed out by the tensor RING (TryRentTensor) this
    // lifetime, tracked with their (element type, element count) so Dispose
    // can return them to the cross-arena persistent pool. The ring caches
    // Tensor<T> WRAPPERS in _tensorRing; those are cheap and dropped on
    // Dispose, but their large backing arrays must survive to the next arena.
    private readonly List<(Type Type, int Size, Array Arr)> _ringBackingArrays = new();

    // ---------------------------------------------------------------------
    // Cross-arena PERSISTENT pool (issue #478)
    //
    // The per-instance _pool / ring are dropped on Dispose. Production callers
    // that create + dispose a fresh arena PER training step (e.g.
    // NeuralNetworkBase.TrainWithTape) therefore re-allocate and gen2-collect
    // the entire transient working set (~param-count × sizeof(T), ~1 GB for a
    // paper-scale model) every step — the bottleneck profiled in #478. To make
    // that pattern allocation-free after warmup WITHOUT requiring callers to
    // hold a long-lived arena, Dispose RETURNS large backing arrays to this
    // thread-static pool and allocation RENTS from it first. The next arena on
    // the same thread reuses the buffers instead of asking the GC for new ones.
    //
    // Thread-static: each thread keeps its own pool (no locks, matches the
    // [ThreadStatic] _current contract). Bounded: only arrays at/above
    // PersistThresholdElems are retained (small buffers GC cheaply and would
    // bloat the dictionary), and at most MaxPersistPerSize per (type,size) so
    // a model with many distinct shapes can't grow the pool without bound.
    // ---------------------------------------------------------------------
    [ThreadStatic]
    private static Dictionary<(Type, int), Stack<Array>>? _persistent;

    // Test/diagnostic: counts how many times a large buffer was served from the
    // cross-arena persistent pool (a reuse hit) rather than freshly allocated.
    [ThreadStatic]
    private static long _persistentReuseHits;

    /// <summary>Number of cross-arena persistent-pool reuse hits on this thread
    /// since the last <see cref="ClearPersistentPool"/>. Test-only signal that
    /// large buffers are actually being recycled across arena lifetimes.</summary>
    internal static long PersistentReuseHits => _persistentReuseHits;

    /// <summary>Minimum element count for an array to be worth persisting across
    /// arena lifetimes. 65536 elems = 256 KB (float) / 512 KB (double) — matches
    /// <see cref="TensorAllocator.ArrayPoolThresholdValue"/>'s intent: below this,
    /// allocation/GC is cheap and pooling churns the dictionary for no gain.</summary>
    private const int PersistThresholdElems = 64 * 1024;

    /// <summary>Max retained buffers per (type, size). A single-threaded training
    /// step rents each distinct size O(1) times, so a small cap fully covers
    /// steady-state reuse while bounding worst-case retained memory to a few ×
    /// the model's transient working set.</summary>
    private const int MaxPersistPerSize = 4;

    private static Array? RentPersistent(Type type, int elementCount)
    {
        if (elementCount < PersistThresholdElems) return null;
        var pool = _persistent;
        if (pool is null) return null;
        if (pool.TryGetValue((type, elementCount), out var stack) && stack.Count > 0)
        {
            var arr = stack.Pop();
            _persistentReuseHits++;
            // Zero the recycled buffer. The arena's previous behaviour allocated
            // every first-use-per-arena buffer via `new T[]`, which the CLR
            // always zeroes — so consumers (e.g. GroupNorm reductions, any op
            // that accumulates into a "fresh" scratch tensor) latently rely on
            // zero-init even on the "uninitialized" ring path. A recycled buffer
            // carries the previous lifetime's bytes, so it MUST be cleared to
            // preserve that contract. This costs the same memset `new T[]` paid
            // anyway — we just skip the allocation + GC. Without this, the
            // cross-arena reuse corrupts those consumers (caught by GroupNorm
            // correctness tests).
            Array.Clear(arr, 0, arr.Length);
            return arr;
        }
        return null;
    }

    private static void ReturnPersistent(Type type, int elementCount, Array arr)
    {
        if (elementCount < PersistThresholdElems) return; // let small buffers GC
        var pool = _persistent ??= new Dictionary<(Type, int), Stack<Array>>();
        var key = (type, elementCount);
        if (!pool.TryGetValue(key, out var stack))
        {
            stack = new Stack<Array>(MaxPersistPerSize);
            pool[key] = stack;
        }
        if (stack.Count < MaxPersistPerSize)
            stack.Push(arr);
        // else: at cap — drop the reference, let GC reclaim (don't hoard).
    }

    /// <summary>
    /// Drops every buffer held in the calling thread's cross-arena persistent
    /// pool. For tests / explicit memory-pressure handling; production code
    /// never needs this (the pool is bounded by <see cref="MaxPersistPerSize"/>).
    /// </summary>
    public static void ClearPersistentPool()
    {
        _persistent?.Clear();
        _persistentReuseHits = 0;
    }

    private bool _disposed;
    private readonly TensorArena? _previous;

    /// <summary>
    /// Gets the currently active arena for this thread, or null if none.
    /// </summary>
    internal static TensorArena? Current => _current;

    private TensorArena(TensorArena? previous)
    {
        _previous = previous;
    }

    /// <summary>
    /// Creates and activates a new arena for this thread.
    /// All <see cref="TensorAllocator.Rent{T}"/> calls on this thread will allocate from the arena
    /// until it is disposed.
    /// </summary>
    /// <returns>An arena that must be disposed to deactivate.</returns>
    public static TensorArena Create()
    {
        var arena = new TensorArena(_current);
        _current = arena;
        return arena;
    }

    /// <summary>
    /// Temporarily detaches the calling thread's active arena for the lifetime of the returned
    /// scope, so allocations made inside it do NOT land in the (transient, per-iteration) arena
    /// scratch pool. Restores the previous arena on dispose.
    ///
    /// <para><b>Why (compiled-plan buffer liveness):</b> a <see cref="Compilation.CompiledTrainingPlan{T}"/>
    /// is traced+compiled ONCE and then replayed for the life of the model, so its forward-activation
    /// and gradient buffers are LONG-LIVED plan state — not per-iteration scratch. But the trace runs
    /// inside whatever transient arena the caller opened for the current training step. Without this
    /// suspension those persistent buffers were allocated from that step's arena and RETURNED to the
    /// shared pool when the step's arena disposed; a later step's backward temporary (e.g. the
    /// <c>ReduceSum</c> gradient broadcast, which tiles a fresh <c>[B,V]</c> buffer) could then re-rent
    /// a live forward activation's array and overwrite it BEFORE its backward consumer read it —
    /// silently zeroing gradients on REPLAY for large activations (≥ the ArrayPool bucket boundary),
    /// freezing training at large vocab. Compiling with the arena suspended makes the plan own its
    /// buffers independently of any transient arena, so they can never be recycled underneath a replay.</para>
    /// </summary>
    internal static IDisposable Suspend()
    {
        var saved = _current;
        _current = null;
        return new SuspendScope(saved);
    }

    private sealed class SuspendScope : IDisposable
    {
        private readonly TensorArena? _saved;
        private bool _disposed;
        internal SuspendScope(TensorArena? saved) { _saved = saved; }
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _current = _saved;
        }
    }

    /// <summary>
    /// Tries to rent an array of the given size from the arena.
    /// During warmup (first iteration): allocates a new array and tracks it.
    /// After Reset() (subsequent iterations): returns a previously-allocated array.
    /// Returns null only if disposed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal T[]? TryAllocate<T>(int elementCount)
    {
        return TryAllocateCore<T>(elementCount, clear: true);
    }

    /// <summary>
    /// Like TryAllocate but skips Array.Clear. Use ONLY when the caller guarantees it will
    /// write every element before reading (e.g., backward kernels that overwrite the entire buffer).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal T[]? TryAllocateUninitialized<T>(int elementCount)
    {
        return TryAllocateCore<T>(elementCount, clear: false);
    }

    /// <summary>
    /// Allocates a long-lived array tracked in the PINNED tier. Unlike
    /// <see cref="TryAllocate{T}"/> (scratch), pinned allocations are NOT
    /// rewound by <see cref="Reset"/> — the caller can hold a reference
    /// across many training iterations without risk of the arena re-issuing
    /// the same backing array as scratch on the next iteration. Use for
    /// model weights (layer EnsureInitialized), optimizer state (Adam m / v),
    /// running BatchNorm statistics, anything that's part of the network's
    /// learnable state. Returns null only when the arena is disposed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal T[]? TryAllocatePinned<T>(int elementCount)
    {
        if (_disposed) return null;
        var arr = new T[elementCount];
        _pinnedArrays.Add(arr);
        return arr;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private T[]? TryAllocateCore<T>(int elementCount, bool clear)
    {
        if (_disposed) return null;

        var key = (typeof(T), elementCount);
        if (!_pool.TryGetValue(key, out var bucket))
        {
            bucket = new List<Array>(4);
            _pool[key] = bucket;
            _cursor[key] = 0;
        }

        int cursor = _cursor[key];

        if (cursor < bucket.Count)
        {
            // Reuse path: return existing array, advance cursor
            var existing = (T[])bucket[cursor];
            _cursor[key] = cursor + 1;
            if (clear) Array.Clear(existing, 0, elementCount);
            return existing;
        }

        // Warmup path: reuse a buffer from the cross-arena persistent pool if
        // one is available for this (type,size), else allocate fresh. Both
        // sources yield ZEROED memory — RentPersistent clears recycled buffers,
        // and `new T[]` is CLR-zeroed — so the `clear` contract is already
        // satisfied without an extra Array.Clear here. (The within-arena reuse
        // path above still clears, because those buffers were handed out earlier
        // THIS lifetime and may hold stale data the caller wrote.)
        var arr = RentPersistent(typeof(T), elementCount) as T[] ?? new T[elementCount];
        bucket.Add(arr);
        _cursor[key] = cursor + 1;
        return arr;
    }

    // Flat tensor ring buffer — sequential scan is faster than dictionary hash for <10 sizes
    private object[]? _tensorRing;
    private int[]? _tensorRingSizes;
    private int[]? _tensorRingCursors;
    private int _tensorRingCount;
    private const int MaxTensorRingSlots = 32;

    /// <summary>
    /// Rents a Tensor object from the arena. Returns a cached Tensor with its
    /// backing array already pooled. Uses a flat array with linear scan instead of
    /// Dictionary — eliminates hash computation overhead for the 3-5 unique sizes
    /// in a typical training step.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal LinearAlgebra.Tensor<T>? TryRentTensor<T>(int totalSize, int[] shape)
    {
        if (_disposed) return null;

        // Lazy init
        if (_tensorRing == null)
        {
            _tensorRing = new object[MaxTensorRingSlots];
            _tensorRingSizes = new int[MaxTensorRingSlots];
            _tensorRingCursors = new int[MaxTensorRingSlots];
        }

        // Linear scan for matching size (fast for <10 entries)
        for (int i = 0; i < _tensorRingCount; i++)
        {
            if (_tensorRingSizes![i] == totalSize && _tensorRing[i] is List<object> bucket)
            {
                int cursor = _tensorRingCursors![i];
                if (cursor < bucket.Count)
                {
                    _tensorRingCursors[i] = cursor + 1;
                    var cached = (LinearAlgebra.Tensor<T>)bucket[cursor];
                    // The ring buckets by element COUNT, not shape. The cached wrapper
                    // carries the shape it was FIRST created with, so a same-count /
                    // different-shape request (e.g. an [B,L] vs [L,B] permute in N-BEATS)
                    // must be re-issued under the REQUESTED shape — otherwise a stale-shaped
                    // tensor flows downstream and crashes shape checks on this post-Reset
                    // reuse path (AiDotNet #1804). The buffer is contiguous with matching
                    // element count, so this is a zero-copy metadata swap that does NOT
                    // record a Reshape node on the active tape.
                    if (!cached._shape.AsSpan().SequenceEqual(shape))
                        cached.ArenaReshapeInPlace(shape);
                    return cached;
                }

                // Need one more tensor of this size — reuse a persistent-pool
                // backing array if available (ring tensors are uninitialized:
                // the caller overwrites every element, so no clear needed).
                var newArr = RentPersistent(typeof(T), totalSize) as T[] ?? new T[totalSize];
                _ringBackingArrays.Add((typeof(T), totalSize, newArr));
                var newTensor = LinearAlgebra.Tensor<T>.FromMemory(new Memory<T>(newArr, 0, totalSize), shape);
                bucket.Add(newTensor);
                _tensorRingCursors[i] = cursor + 1;
                return newTensor;
            }
        }

        // New size — add slot
        if (_tensorRingCount < MaxTensorRingSlots)
        {
            var arr = RentPersistent(typeof(T), totalSize) as T[] ?? new T[totalSize];
            _ringBackingArrays.Add((typeof(T), totalSize, arr));
            var tensor = LinearAlgebra.Tensor<T>.FromMemory(new Memory<T>(arr, 0, totalSize), shape);
            var newBucket = new List<object>(4) { tensor };
            int idx = _tensorRingCount++;
            _tensorRing[idx] = newBucket;
            _tensorRingSizes![idx] = totalSize;
            _tensorRingCursors![idx] = 1;
            return tensor;
        }

        // Ring is full: this model produces more distinct tensor sizes than
        // MaxTensorRingSlots (deep models exhaust the ring with forward
        // activation shapes before the large backward gradient shapes arrive).
        // Returning null sends the caller to ArrayPool.Shared, whose largest
        // retained bucket is 2^20 elements — every paper-scale weight gradient
        // exceeds that, so ArrayPool.Rent returns a FRESH array and Return drops
        // it (the ~1 GB/step GC churn in #478). Instead fall back to the
        // UNCAPPED dictionary scratch tier (which itself rents from / returns to
        // the cross-arena persistent pool), so large gradients pool+reuse no
        // matter how many distinct sizes the model has. Uninitialized — matches
        // the ring's contract that the caller writes every element first.
        var overflowArr = TryAllocateUninitialized<T>(totalSize);
        if (overflowArr is null) return null; // disposed
        return LinearAlgebra.Tensor<T>.FromMemory(new Memory<T>(overflowArr, 0, totalSize), shape);
    }

    /// <summary>
    /// Resets the arena for the next iteration. After this, subsequent Rent calls
    /// will reuse arrays from the previous iteration — zero allocation.
    /// </summary>
    public void Reset()
    {
        // Rewind all cursors to 0 — arrays and tensors stay pooled.
        // NOTE: must snapshot the keys before mutating. On .NET Framework
        // (net471), Dictionary<,>.this[key] = value increments the collection
        // version even when updating an EXISTING key, so iterating
        // _cursor.Keys while assigning _cursor[key] throws "Collection was
        // modified". (.NET Core / net5+ doesn't bump version on update, which
        // is why this only reproduced on the net471 TFM.)
        int n = _cursor.Count;
        if (n > 0)
        {
            // Reuse a scratch buffer so steady-state Reset() allocates nothing
            // (the key set is stable across iterations); grow only when it rises.
            if (_resetKeyScratch is null || _resetKeyScratch.Length < n)
                _resetKeyScratch = new (Type, int)[n];
            _cursor.Keys.CopyTo(_resetKeyScratch, 0);
            for (int i = 0; i < n; i++)
                _cursor[_resetKeyScratch[i]] = 0;
        }
        // Reset flat tensor ring cursors
        if (_tensorRingCursors != null)
        {
            for (int i = 0; i < _tensorRingCount; i++)
                _tensorRingCursors[i] = 0;
        }
    }

    /// <summary>
    /// Gets the total number of arrays tracked by this arena.
    /// </summary>
    public int TrackedArrayCount
    {
        get
        {
            int count = 0;
            foreach (var bucket in _pool.Values)
                count += bucket.Count;
            return count;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_current == this)
            _current = _previous; // restore outer arena if nested

        // Return large backing arrays to the cross-arena persistent pool so the
        // NEXT arena on this thread reuses them instead of GC-allocating fresh
        // (issue #478). Small buffers (< PersistThresholdElems) are skipped by
        // ReturnPersistent and simply dropped for GC. The pinned tier is NOT
        // returned — those are long-lived (weights / optimizer state) and the
        // owner still holds references to them.
        //
        // ONLY a TOP-LEVEL arena (no outer arena active) may pool its scratch/ring
        // buffers. A NESTED arena (_previous != null) disposes back into a still-live
        // outer scope, and its op-output tensors routinely ESCAPE there — a layer caches
        // the forward activation for backward, the caller retains the model output, etc.
        // Pooling an escaped buffer lets a later RentPersistent zero+reuse it and silently
        // corrupt the live tensor (issue #1221: MobileNetV3 trained under an outer arena,
        // then a per-step training arena's escaped activation buffer was recycled into the
        // following eval forward, making Predict non-idempotent — first forward ≠ rest).
        // Top-level per-step training arenas (the #478 pattern: TrainWithTape's own
        // `using var arena`, no outer wrapper) keep pooling, so the GC win is preserved.
        if (_previous is null)
        {
            foreach (var kvp in _pool)
            {
                var (type, size) = kvp.Key;
                var bucket = kvp.Value;
                for (int i = 0; i < bucket.Count; i++)
                    ReturnPersistent(type, size, bucket[i]);
            }
            for (int i = 0; i < _ringBackingArrays.Count; i++)
            {
                var (type, size, arr) = _ringBackingArrays[i];
                ReturnPersistent(type, size, arr);
            }
        }

        _pool.Clear();
        _cursor.Clear();
        _pinnedArrays.Clear();
        _ringBackingArrays.Clear();

        // Clear tensor ring pools (the Tensor<T> wrappers; their backing arrays
        // were just handed to the persistent pool above).
        if (_tensorRing != null)
        {
            Array.Clear(_tensorRing, 0, _tensorRingCount);
            _tensorRingCount = 0;
        }
    }

    /// <summary>
    /// Number of arrays currently held in the PINNED tier. Diagnostic — lets
    /// tests / benchmarks verify that long-lived allocations (weights,
    /// optimizer state) are landing in the right tier and not bloating the
    /// scratch pool. Pinned count grows monotonically across <see cref="Reset"/>
    /// calls (only <see cref="Dispose"/> drops these references).
    /// </summary>
    public int PinnedArrayCount => _pinnedArrays.Count;
}
