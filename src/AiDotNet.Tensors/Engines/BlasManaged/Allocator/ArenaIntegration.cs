using System;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Layer 4 of the BlasManaged allocator stack: integration with
/// <see cref="TensorArena"/>. When an arena is active on the current thread
/// (typical inside a compiled training plan), prefers arena-allocated pack
/// buffers over the per-thread pool.
///
/// <para>
/// Arena allocation has two advantages over the per-thread pool:
/// </para>
/// <list type="bullet">
///   <item>Contiguous memory within a training step → better TLB / page-cache behavior across mixed ops.</item>
///   <item>Releases all memory at iteration end → useful for memory-constrained inference where each thread shouldn't permanently hold ~700 KB of pack buffers.</item>
/// </list>
///
/// <para>
/// When no arena is active (<see cref="TensorArena.Current"/> returns null),
/// callers fall through to <see cref="PerThreadPool"/>. The arena uses
/// <see cref="TensorArena.TryAllocateUninitialized{T}(int)"/> because pack
/// routines write every byte before any read.
/// </para>
/// </summary>
internal static class ArenaIntegration
{
    /// <summary>
    /// Try to rent a byte buffer of the requested size from the active arena.
    /// Returns empty when no arena is active, in which case the caller falls back
    /// to <see cref="PerThreadPool"/>.
    /// </summary>
    /// <param name="bytes">Byte count requested. Must be &gt; 0.</param>
    /// <returns>An arena-allocated <see cref="Span{Byte}"/>, or empty when no arena is active.</returns>
    public static Span<byte> TryRentBytes(int bytes)
    {
        if (bytes <= 0) return Span<byte>.Empty;

        var arena = TensorArena.Current;
        if (arena is null) return Span<byte>.Empty;

        var buffer = arena.TryAllocateUninitialized<byte>(bytes);
        if (buffer is null) return Span<byte>.Empty;  // Arena disposed.

        return buffer.AsSpan(0, bytes);
    }

    /// <summary>
    /// True when <see cref="TensorArena.Current"/> is non-null.
    /// </summary>
    public static bool IsArenaActive => TensorArena.Current != null;
}
