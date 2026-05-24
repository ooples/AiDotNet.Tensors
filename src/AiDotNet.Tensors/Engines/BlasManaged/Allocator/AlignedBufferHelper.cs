using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-P (#406): aligned-buffer rental helper. Over-rents a managed byte[] by
/// the alignment amount and exposes a sub-range that starts at a properly
/// aligned offset. Used by <see cref="PackBothStrategy"/> to obtain a 32-byte
/// aligned pack-B buffer so the <see cref="Microkernels.Avx2.Avx2Pack"/>
/// non-temporal-store fast path actually fires.
///
/// <para>
/// .NET array allocations are pointer-aligned (8 bytes on x64) but not
/// AVX-aligned (32 bytes). Over-renting by <c>alignment-1</c> bytes guarantees
/// we can find an aligned start within the buffer. The non-aligned head and
/// tail are unused.
/// </para>
/// </summary>
internal static class AlignedBufferHelper
{
    /// <summary>
    /// Rent a byte[] from <see cref="ArrayPool{T}.Shared"/> with at least
    /// <paramref name="usableBytes"/> contiguous bytes at <paramref name="alignment"/>-byte
    /// alignment. The caller MUST return the underlying array via
    /// <see cref="ReturnAlignedBuffer"/> when done — the wrapper tracks the
    /// raw rental and the alignment offset internally.
    /// </summary>
    /// <param name="usableBytes">Number of bytes the caller will actually use, starting at the aligned offset.</param>
    /// <param name="alignment">Alignment boundary in bytes (must be a power of 2). Typical: 32 for AVX, 64 for AVX-512 / cache line.</param>
    /// <returns>An <see cref="AlignedRental"/> exposing <c>Span</c>, the underlying array, and the offset.</returns>
    public static AlignedRental RentAligned(int usableBytes, int alignment = 32)
    {
        if (usableBytes <= 0) throw new ArgumentOutOfRangeException(nameof(usableBytes));
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
            throw new ArgumentException("Alignment must be a power of 2.", nameof(alignment));

        int rentSize = usableBytes + alignment - 1;
        byte[] raw = ArrayPool<byte>.Shared.Rent(rentSize);
        int offset = ComputeAlignedOffset(raw, alignment);
        return new AlignedRental(raw, offset, usableBytes);
    }

    /// <summary>
    /// Return a buffer rented via <see cref="RentAligned"/>. Hands the raw
    /// (non-aligned) backing array back to the pool.
    /// </summary>
    public static void ReturnAlignedBuffer(AlignedRental rental)
    {
        if (rental.Raw != null) ArrayPool<byte>.Shared.Return(rental.Raw);
    }

    /// <summary>
    /// Compute the byte offset within <paramref name="buffer"/> such that
    /// <c>&amp;buffer[offset]</c> is <paramref name="alignment"/>-aligned.
    /// </summary>
    private static unsafe int ComputeAlignedOffset(byte[] buffer, int alignment)
    {
        fixed (byte* p = buffer)
        {
            nuint addr = (nuint)p;
            nuint mask = (nuint)(alignment - 1);
            nuint alignmentN = (nuint)alignment;
            nuint padding = (alignmentN - (addr & mask)) & mask;
            return (int)padding;
        }
    }

    /// <summary>
    /// Tracks a rental from <see cref="RentAligned"/>. The <see cref="Raw"/>
    /// array is what the pool gave us; <see cref="AlignedOffset"/> is the
    /// alignment skip; <see cref="UsableBytes"/> is the caller-requested size.
    /// </summary>
    internal readonly struct AlignedRental
    {
        public readonly byte[] Raw;
        public readonly int AlignedOffset;
        public readonly int UsableBytes;

        public AlignedRental(byte[] raw, int alignedOffset, int usableBytes)
        {
            Raw = raw;
            AlignedOffset = alignedOffset;
            UsableBytes = usableBytes;
        }

        /// <summary>Aligned span view: <c>UsableBytes</c> bytes starting at the aligned offset.</summary>
        public Span<byte> Span => Raw.AsSpan(AlignedOffset, UsableBytes);

        /// <summary>True when this rental is non-empty.</summary>
        public bool IsValid => Raw != null && UsableBytes > 0;
    }
}
