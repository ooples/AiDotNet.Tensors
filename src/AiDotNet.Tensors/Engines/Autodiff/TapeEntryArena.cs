using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Pre-allocated ring buffer for TapeEntry structs. Eliminates per-record GC allocation
/// by writing entries into a fixed-size array. When the array is full, it doubles in capacity.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para><b>Performance:</b> Recording an op writes a TapeEntry struct into the next slot
/// in the backing array — no heap allocation, no GC pressure. The backing array grows
/// geometrically (like List&lt;T&gt;) but since TapeEntry is a struct, the array stores
/// entries inline (no boxing, no pointer chasing).</para>
/// <para><b>Why not List&lt;TapeEntry&gt;?</b> List&lt;T&gt; for value types copies the struct
/// on every Add (into the backing array). Our arena avoids the copy by returning a ref
/// to the next slot, which DifferentiableOps writes into directly. However, since C#
/// doesn't support ref returns from indexers in all TFMs, we use List&lt;T&gt; as the backing
/// store but pre-size it to avoid resizing allocations.</para>
/// </remarks>
internal sealed class TapeEntryArena<T>
{
    private TapeEntry<T>[] _entries;
    private int _count;

    /// <summary>Initial capacity — enough for a typical forward pass.</summary>
    private const int InitialCapacity = 256;

    internal TapeEntryArena()
    {
        _entries = new TapeEntry<T>[InitialCapacity];
        _count = 0;
    }

    /// <summary>Number of entries recorded.</summary>
    internal int Count => _count;

    /// <summary>
    /// Records an entry into the next arena slot. Grows the backing array if needed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void Add(TapeEntry<T> entry)
    {
        if (_count >= _entries.Length)
        {
            Grow();
        }
        _entries[_count++] = entry;
    }

    /// <summary>Gets an entry by index. Used during backward traversal.</summary>
    internal ref TapeEntry<T> this[int index] => ref _entries[index];

    /// <summary>Removes the last entry. Used by fused ops that replace multiple entries.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void RemoveLast()
    {
        if (_count > 0)
        {
            _count--;
            _entries[_count] = default; // Clear reference fields for GC
        }
    }

    /// <summary>Removes the last N entries.</summary>
    internal void RemoveLast(int n)
    {
        int newCount = Math.Max(0, _count - n);
        for (int i = newCount; i < _count; i++)
            _entries[i] = default;
        _count = newCount;
    }

    /// <summary>Resets the arena for reuse. Keeps the backing array allocated.</summary>
    internal void Reset()
    {
        // Clear entries to release tensor references for GC
        Array.Clear(_entries, 0, _count);
        _count = 0;
    }

    private void Grow()
    {
        int newCapacity = _entries.Length * 2;
        var newArray = new TapeEntry<T>[newCapacity];
        Array.Copy(_entries, newArray, _count);
        _entries = newArray;
    }
}
