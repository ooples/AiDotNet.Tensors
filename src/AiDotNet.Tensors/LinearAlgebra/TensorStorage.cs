using System;
using System.Runtime.CompilerServices;
using System.Threading;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Shared storage for tensor data. Multiple tensors (views) can reference the same storage
/// with different shapes, strides, and offsets — enabling O(1) Transpose, Reshape, and Slice.
/// </summary>
/// <remarks>
/// <para>This follows the PyTorch Storage model where the actual data lives in a Storage object
/// and Tensor is a view into that storage with shape/stride metadata.</para>
/// <para>Thread-safe reference counting tracks how many tensors/views share this storage.
/// When a tensor is disposed, it calls Release() to decrement the count. When the count
/// reaches zero the storage is eligible for GC (future: pool return).</para>
/// </remarks>
/// <typeparam name="T">The numeric type of the stored elements.</typeparam>
internal sealed class TensorStorage<T>
{
    private readonly Vector<T> _data;
    private int _refCount;

    // Streaming-pool zero-copy mmap alias (PR #604, CodeRabbit-Major): when
    // WeightRegistry.TryAliasZeroCopy installs an MmapTensorMemoryManager as
    // this storage's backing Vector, the mapping HAS to outlive every view /
    // RebindStorageFrom that shares the storage — those views can outlive the
    // tensor that installed the alias. So the owner lives at storage scope,
    // disposed exactly when the last ref releases (or an explicit
    // TryClaimExclusive succeeds). Also exposes IsReadOnlyMapped so the
    // write paths (AsWritableSpan / AsWritableMemory / GetDataArray) can
    // throw a clear error instead of faulting in the mapped pages.
    private IDisposable? _mmapOwner;
    // Writable mmap aliases (#1715 param-IO round-trip) back a tensor whose mutations must persist
    // to the mapped file (MAP_SHARED). Only READ-ONLY aliases gate the write paths — a writable
    // alias's whole purpose is to be written. Set before _mmapOwner is published (the CAS below is
    // the release barrier), so a reader that sees a non-null owner also sees the correct writability.
    private bool _mmapWritable;
    internal bool IsReadOnlyMapped => Volatile.Read(ref _mmapOwner) != null && !_mmapWritable;

    /// <summary>
    /// Attaches an IDisposable that will be disposed when this storage's
    /// refcount finally reaches 0. Used by the streaming-pool zero-copy
    /// alias to tie the lifetime of the underlying memory-mapped file to
    /// the lifetime of the shared storage (not the lifetime of any single
    /// tensor that holds a ref). Must be called BEFORE the alias is exposed
    /// to other tensors. <paramref name="writable"/> (#1715) marks a
    /// read-WRITE mapping whose mutations persist — such aliases do NOT gate
    /// the write paths (read-only aliases do, so writes fail loud rather than
    /// faulting the mapped pages).
    /// </summary>
    internal void AttachMmapOwner(IDisposable owner, bool writable = false)
    {
        if (owner is null) throw new ArgumentNullException(nameof(owner));
        _mmapWritable = writable;
        if (Interlocked.CompareExchange(ref _mmapOwner, owner, null) != null)
            throw new InvalidOperationException(
                "TensorStorage already has an attached mmap owner; replacing it would leak the prior mapping.");
    }

    /// <summary>
    /// Creates a new storage wrapping an existing Vector (zero-copy).
    /// </summary>
    /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
    internal TensorStorage(Vector<T> data)
    {
        _data = data ?? throw new ArgumentNullException(nameof(data));
        _refCount = 1;
    }

    /// <summary>
    /// Creates a new storage with the specified size.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when size is not positive.</exception>
    internal TensorStorage(int size)
    {
        if (size < 0) throw new ArgumentOutOfRangeException(nameof(size), "Storage size must be non-negative.");
        _data = new Vector<T>(size);
        _refCount = 1;
    }

    /// <summary>
    /// Gets the total number of elements in this storage.
    /// </summary>
    internal int Length => _data.Length;

    /// <summary>
    /// Gets the current reference count (number of tensor views sharing this storage).
    /// </summary>
    internal int RefCount => Volatile.Read(ref _refCount);

    /// <summary>
    /// Increments the reference count atomically. Called when a new view is created from this storage.
    /// Uses compare-and-swap loop to prevent TOCTOU race between disposal check and increment.
    /// </summary>
    /// <exception cref="ObjectDisposedException">Thrown when storage has been fully released.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void AddRef()
    {
        while (true)
        {
            int current = Volatile.Read(ref _refCount);
            if (current <= 0)
                throw new ObjectDisposedException(nameof(TensorStorage<T>),
                    "Cannot acquire reference to released storage.");
            if (Interlocked.CompareExchange(ref _refCount, current + 1, current) == current)
                return;
            // CAS failed — another thread modified refCount, retry
        }
    }

    /// <summary>
    /// Decrements the reference count. When it reaches zero, the storage is marked as disposed
    /// and the pooled array (if any) can be reclaimed by TensorAllocator.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown on double-release (refCount already 0).</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void Release()
    {
        int newCount = Interlocked.Decrement(ref _refCount);
        if (newCount < 0)
        {
            // Restore and throw — this is a bug in the caller.
            Interlocked.Increment(ref _refCount);
            throw new InvalidOperationException("TensorStorage released more times than it was acquired.");
        }
        if (newCount == 0)
        {
            // Last ref out — dispose the streaming-pool mmap owner (if any) so
            // the underlying mapping doesn't outlive its last viewer.
            // Interlocked.Exchange so a concurrent (broken) double-release
            // can't double-dispose.
            var owner = Interlocked.Exchange(ref _mmapOwner, null);
            owner?.Dispose();
        }
    }

    /// <summary>
    /// Atomically claims sole ownership of this storage. Returns <c>true</c>
    /// when the caller was the only ref-holder at the moment of the claim
    /// (refcount was exactly 1, now 0); <c>false</c> when one or more
    /// additional refs exist (refcount &gt; 1 stays unchanged).
    ///
    /// <para>After a successful claim the storage's refcount is 0 and any
    /// concurrent <see cref="AddRef"/> will throw <see cref="ObjectDisposedException"/>;
    /// the caller must NOT call <see cref="Release"/> on this storage
    /// (that would underflow). The expected pattern is
    /// <c>DropStorageForStreaming</c>: caller swaps in fresh storage and
    /// abandons this one for GC.</para>
    ///
    /// <para>This is the atomic version of "check refcount == 1 then
    /// swap". A naive read-then-swap is racy with <see cref="AddRef"/>
    /// from a sibling <c>RebindStorageFrom</c>: between the read and the
    /// swap, another thread could AddRef this storage, leaving the
    /// caller swapping out shared bytes — its rebound peer would observe
    /// the original storage and the two views would diverge. CAS-based
    /// claim closes that window.</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal bool TryClaimExclusive()
    {
        // CAS 1 → 0. Succeeds only when no other ref exists.
        if (Interlocked.CompareExchange(ref _refCount, 0, 1) != 1) return false;
        // Sole-claim succeeded → no other ref exists → dispose the mmap
        // owner too. Caller is about to abandon this storage.
        var owner = Interlocked.Exchange(ref _mmapOwner, null);
        owner?.Dispose();
        return true;
    }

    /// <summary>
    /// Gets the underlying data as a read-only span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal ReadOnlySpan<T> AsSpan() => _data.AsSpan();

    /// <summary>
    /// Gets the underlying data as a writable span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Span<T> AsWritableSpan()
    {
        ThrowIfReadOnlyMapped();
        return _data.AsWritableSpan();
    }

    /// <summary>
    /// Gets the underlying data as Memory&lt;T&gt; for pinning/GPU transfer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Memory<T> AsMemory()
    {
        ThrowIfReadOnlyMapped();
        return _data.AsWritableMemory();
    }

    /// <summary>
    /// Gets the raw backing array. Use with care — bypasses safety checks.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal T[] GetDataArray()
    {
        ThrowIfReadOnlyMapped();
        return _data.GetDataArray();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ThrowIfReadOnlyMapped()
    {
        if (IsReadOnlyMapped)
            throw new InvalidOperationException(
                "Cannot obtain a writable view of TensorStorage: this storage aliases a read-only " +
                "memory-mapped weight (streaming-pool zero-copy materialization). Writing into the " +
                "mapped pages would fault. Call DropStorageForStreaming / Rebind into fresh storage " +
                "before mutating, or stage the write in a separate tensor.");
    }

    /// <summary>
    /// Gets the underlying Vector for compatibility with existing code.
    /// </summary>
    internal Vector<T> GetVector() => _data;
}
