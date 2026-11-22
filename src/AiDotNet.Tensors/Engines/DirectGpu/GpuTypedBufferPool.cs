// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Idle device buffers kept for reuse, keyed by what they hold: element type and element count. A buffer is handed out
/// only for a request of its own type and count, so a caller never receives memory of another width or a longer buffer
/// it would under-fill.
/// </summary>
/// <remarks>
/// The pool belongs to one backend at a time (the first to return a buffer); buffers of another backend are not
/// accepted, and nothing is handed to another backend. Not thread-safe: the owner serializes access.
/// </remarks>
internal sealed class GpuTypedBufferPool
{
    private readonly Dictionary<(GpuElementType Type, int Length), Stack<IGpuBuffer>> _idle = new();
    private readonly Func<IGpuBuffer, bool> _isLive;
    private IDirectGpuBackend? _backend;

    /// <param name="isLive">Whether a buffer still names device memory (a released one has only its wrapper left).</param>
    public GpuTypedBufferPool(Func<IGpuBuffer, bool> isLive)
        => _isLive = isLive ?? throw new ArgumentNullException(nameof(isLive));

    /// <summary>Bytes of device memory the idle buffers hold.</summary>
    public long Bytes { get; private set; }

    /// <summary>The number of distinct (type, count) sizes the pool holds buffers of.</summary>
    public int Sizes => _idle.Count;

    /// <summary>The number of idle buffers.</summary>
    public int Count => _idle.Values.Sum(stack => stack.Count);

    private static long BytesOf(GpuElementType type, int length) => (long)length * GpuElementTypes.SizeInBytes(type);

    /// <summary>
    /// Takes an idle buffer of exactly <paramref name="length"/> <paramref name="type"/> elements on
    /// <paramref name="backend"/>. Idle buffers of that key that are no longer usable (another backend's, or released)
    /// are disposed on the way.
    /// </summary>
    public bool TryTake(IDirectGpuBackend backend, GpuElementType type, int length, [NotNullWhen(true)] out IGpuBuffer? buffer)
    {
        buffer = null;
        if (!_idle.TryGetValue((type, length), out var stack)) return false;
        bool sameBackend = ReferenceEquals(_backend, backend);
        while (stack.Count > 0)
        {
            // Judged while the pool still holds it: a liveness check that throws leaves the buffer pooled, not adrift
            // between the pop and a dispose. Only then does it leave the pool, to the caller or to its disposal.
            bool usable = sameBackend && _isLive(stack.Peek());
            Bytes -= BytesOf(type, length);
            if (usable)
            {
                buffer = stack.Pop();
                // A drained size leaves no entry behind: the pool holds a key only for sizes it has buffers of.
                if (stack.Count == 0) _idle.Remove((type, length));
                return true;
            }

            using (stack.Pop())
            {
                // Disposed as it leaves the pool.
            }
        }

        _idle.Remove((type, length));
        return false;
    }

    /// <summary>
    /// Offers <paramref name="buffer"/> for reuse. It is kept when it is live, of <paramref name="backend"/> (the pool's
    /// backend, or the first), and holds the type and count it reports; otherwise it is left with the caller, who
    /// disposes it.
    /// </summary>
    /// <returns>Whether the pool now owns the buffer.</returns>
    public bool TryReturn(IDirectGpuBackend backend, IGpuBuffer buffer)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (buffer.ElementType == GpuElementType.Unspecified || !_isLive(buffer)) return false;
        if (_backend is not null && !ReferenceEquals(_backend, backend)) return false;
        _backend = backend;
        var key = (buffer.ElementType, buffer.Size);
        if (!_idle.TryGetValue(key, out var stack)) _idle[key] = stack = new Stack<IGpuBuffer>();
        stack.Push(buffer);
        Bytes += BytesOf(key.ElementType, key.Size);
        return true;
    }

    /// <summary>Frees every idle buffer, except those of <paramref name="keepType"/> and <paramref name="keepLength"/>
    /// when both are given (the size a caller is about to ask for again).</summary>
    public void Trim(GpuElementType keepType = GpuElementType.Unspecified, int keepLength = -1)
    {
        // A snapshot: entries are removed as they drain.
        foreach (var entry in _idle.Where(idle => idle.Key != (keepType, keepLength)).ToArray())
        {
            // Removed first: a buffer whose free throws leaves no half-drained entry behind.
            _idle.Remove(entry.Key);
            while (entry.Value.Count > 0)
            {
                // Accounted before the dispose: a buffer whose free throws is gone from the pool either way.
                Bytes -= BytesOf(entry.Key.Type, entry.Key.Length);
                entry.Value.Pop().Dispose();
            }
        }
    }

    /// <summary>Frees every idle buffer and forgets the backend, so the pool can serve another.</summary>
    public void Clear()
    {
        Trim();
        _backend = null;
    }
}
