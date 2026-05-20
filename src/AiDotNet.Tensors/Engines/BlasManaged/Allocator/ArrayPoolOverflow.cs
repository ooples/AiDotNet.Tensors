using System;
using System.Buffers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Layer 2 of the BlasManaged allocator stack: overflow into
/// <see cref="ArrayPool{T}.Shared"/> when the per-thread pool can't satisfy a
/// request (typically: shape exceeds <see cref="PerThreadPool"/>'s held
/// buffer size by a wide margin, e.g., LLM-scale GEMMs).
///
/// <para>
/// Rent returns a disposable token wrapping the pooled buffer. Callers
/// should use <c>using var rent = ArrayPoolOverflow.Rent(bytes);</c> to
/// ensure the buffer is returned to the pool.
/// </para>
///
/// <para>
/// Unlike <see cref="PerThreadPool"/>, this layer does NOT hold buffers
/// across calls — every rent allocates (or reuses from the shared pool) and
/// the dispose returns the buffer. The shared pool's internal caching keeps
/// the alloc cost down across hot-path calls.
/// </para>
/// </summary>
internal static class ArrayPoolOverflow
{
    /// <summary>
    /// Rent a buffer of at least <paramref name="bytes"/> bytes from the
    /// shared pool. Returns a disposable token; the buffer is returned to
    /// the pool on <see cref="ArrayPoolByteRent.Dispose"/>.
    /// </summary>
    /// <param name="bytes">Byte count requested. Must be &gt;= 0; 0 or negative returns an empty rent.</param>
    public static ArrayPoolByteRent Rent(int bytes)
    {
        if (bytes <= 0) return ArrayPoolByteRent.Empty;
        var buffer = ArrayPool<byte>.Shared.Rent(bytes);
        return new ArrayPoolByteRent(buffer, bytes);
    }
}

/// <summary>
/// Disposable rent token from <see cref="ArrayPoolOverflow.Rent"/>. Returns
/// the underlying buffer to <see cref="ArrayPool{T}.Shared"/> on dispose.
/// Implements <see cref="IDisposable"/> so it can be used with <c>using</c>
/// statements and <c>using var</c> declarations (C# 8+).
/// The <see cref="Dispose"/> method is idempotent — safe to call more than once.
///
/// <para>PR #402 CodeRabbit fix: changed from mutable struct to sealed class.
/// As a struct the type could be silently copied (parameter pass-by-value,
/// foreach iteration variables, local assignment) — each copy carried the same
/// reference to <c>_buffer</c>, and each <see cref="Dispose"/> call attempted
/// to return the array to the pool, producing a double-return that corrupts
/// the shared <see cref="ArrayPool{T}"/>. Reference semantics enforce
/// single-ownership: copies share a reference to the one instance, so the
/// idempotent dispose check at the bottom of <see cref="Dispose"/> serializes
/// correctly across all aliases.</para>
/// </summary>
internal sealed class ArrayPoolByteRent : IDisposable
{
    /// <summary>Shared empty rent returned by <see cref="ArrayPoolOverflow.Rent"/> for
    /// zero/negative byte requests. Safe to dispose multiple times because the buffer
    /// is already null. Replaces the old <c>default(ArrayPoolByteRent)</c> sentinel
    /// (struct era) that became a null-reference under the class refactor.</summary>
    internal static readonly ArrayPoolByteRent Empty = new ArrayPoolByteRent(null, 0);

    private byte[]? _buffer;
    private readonly int _length;

    internal ArrayPoolByteRent(byte[]? buffer, int length)
    {
        _buffer = buffer;
        _length = length;
    }

    /// <summary>
    /// The rented buffer span. Exactly <c>length</c> bytes; the backing
    /// array may be larger. Returns <see cref="Span{T}.Empty"/> after
    /// <see cref="Dispose"/> has been called.
    /// </summary>
    public Span<byte> Span => _buffer is null ? Span<byte>.Empty : _buffer.AsSpan(0, _length);

    /// <summary>
    /// Return the buffer to the shared pool. Idempotent — subsequent calls
    /// after the first are no-ops.
    /// </summary>
    public void Dispose()
    {
        if (_buffer != null)
        {
            ArrayPool<byte>.Shared.Return(_buffer);
            _buffer = null;
        }
    }
}
