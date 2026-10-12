using System;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// A read-only view of a tensor's, matrix's or vector's host data that keeps the object alive while it is in scope.
/// </summary>
/// <remarks>
/// <para>A span alone keeps the ARRAY alive, not the tensor, matrix or vector that owns it. Large results are recycled
/// once their owner is collected (<see cref="Helpers.ResultBufferTracker"/>), and in optimized code the JIT treats a
/// local as dead after its last use: in <c>var s = Op().AsSpan(); Op2(); Use(s);</c> the first result can be collected
/// during <c>Op2</c> and its array handed to <c>Op2</c>'s result while <c>s</c> still reads it. A lease reads its owner
/// in <see cref="Dispose"/>, so <c>using var lease = Op().Lease();</c> keeps the result alive to the end of the
/// scope.</para>
/// <para>Use it for any span over a result that something else could outlive: a temporary, or a local whose last use
/// is the span. Spans over parameters the caller holds, or over a local used again afterwards, are already safe.</para>
/// </remarks>
internal readonly ref struct ReadLease<T>
{
    private readonly object _owner;

    internal ReadLease(object owner, ReadOnlySpan<T> span)
    {
        _owner = owner;
        Span = span;
    }

    /// <summary>The leased data.</summary>
    public ReadOnlySpan<T> Span { get; }

    /// <summary>Ends the lease; the owner may be collected (and its array recycled) after this.</summary>
    public void Dispose() => GC.KeepAlive(_owner);
}

/// <summary>
/// A writable view of a tensor's, matrix's or vector's host data that keeps the object alive while it is in scope.
/// See <see cref="ReadLease{T}"/> for why a bare span is not enough.
/// </summary>
internal readonly ref struct WriteLease<T>
{
    private readonly object _owner;

    internal WriteLease(object owner, Span<T> span)
    {
        _owner = owner;
        Span = span;
    }

    /// <summary>The leased data.</summary>
    public Span<T> Span { get; }

    /// <summary>Ends the lease; the owner may be collected (and its array recycled) after this.</summary>
    public void Dispose() => GC.KeepAlive(_owner);
}

/// <summary>
/// Keeps an object alive until the scope ends. For members that read another instance's backing storage directly
/// (<c>other._memory</c>): without it, <c>other</c> is dead after that read in optimized code, and a large result's array
/// could be recycled while the member still uses it. <c>using var keep = new KeepAliveScope(other);</c> as the first
/// statement covers every path out, including return expressions (disposal runs after they are evaluated).
/// </summary>
internal readonly ref struct KeepAliveScope
{
    private readonly object? _target;

    internal KeepAliveScope(object? target) => _target = target;

    public void Dispose() => GC.KeepAlive(_target);
}

/// <summary>
/// The backing array of a tensor, matrix or vector, with the object kept alive until the lease is disposed: the
/// array form of <see cref="ReadLease{T}"/>, for code that needs the T[] itself (pinning, interop, legacy kernels).
/// </summary>
internal readonly ref struct ArrayLease<T>
{
    private readonly object _owner;

    internal ArrayLease(object owner, T[] array)
    {
        _owner = owner;
        Array = array;
    }

    /// <summary>The leased array (the live backing array, or a copy for layouts that have none).</summary>
    public T[] Array { get; }

    /// <summary>Ends the lease; the owner may be collected (and its array recycled) after this.</summary>
    public void Dispose() => GC.KeepAlive(_owner);
}
