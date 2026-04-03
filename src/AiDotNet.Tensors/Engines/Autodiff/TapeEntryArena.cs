using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Growable array for TapeEntry structs with ref-return slot allocation.
/// Eliminates per-record GC allocation by writing fields directly into array slots.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para><b>Performance:</b> <see cref="AllocateSlot"/> returns a ref to the next array
/// element, allowing DifferentiableOps to write 9 fields directly into the backing array
/// with zero struct copy and zero heap allocation. The backing array grows geometrically
/// (doubles when full). Thread-local caching in GradientTape ensures the array is reused
/// across training steps — zero allocation after warmup.</para>
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

    /// <summary>
    /// Returns a ref to the next arena slot for direct field writes.
    /// This avoids the 80-byte struct copy in Add() — the caller writes
    /// fields directly into the backing array. ~50ns faster per recording.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal ref TapeEntry<T> AllocateSlot()
    {
        if (_count >= _entries.Length)
        {
            Grow();
        }
        return ref _entries[_count++];
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
