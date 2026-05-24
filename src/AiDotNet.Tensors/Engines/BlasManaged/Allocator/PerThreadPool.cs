using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Layer 1 of the BlasManaged allocator stack: a per-thread persistent pool
/// for pack-A, pack-B, and K-split partial-C buffers. Uses
/// <see cref="ThreadStaticAttribute"/> so each .NET thread has its own
/// instance — zero cross-thread contention.
///
/// <para>
/// Buffers grow monotonically (never shrink). After the first GEMM call on a
/// thread, subsequent calls reuse the same backing arrays — zero allocation
/// in the steady state, which is critical for the training-loop hot path.
/// </para>
///
/// <para>
/// Memory ceiling per thread is ~700 KB at Skylake-X defaults (Mc=240, Kc=240,
/// Nc=3072 for FP64). 16 threads = ~11 MB total — acceptable for the
/// throughput gain.
/// </para>
///
/// <para>
/// The pool itself is not GC-finalized; .NET reclaims the buffers when the
/// thread exits. For long-lived process worker threads (typical for training
/// loops), the pool persists for the process lifetime.
/// </para>
/// </summary>
internal sealed class PerThreadPool
{
    [ThreadStatic] private static PerThreadPool? _instance;

    /// <summary>
    /// The current thread's pool. Lazily allocated on first access.
    /// </summary>
    public static PerThreadPool Current => _instance ??= new PerThreadPool();

    private byte[]? _packA;
    private byte[]? _packB;
    private byte[]? _kSplitC;

    /// <summary>
    /// Rent a pack-A buffer of at least <paramref name="bytes"/> bytes. The
    /// returned span has exactly <paramref name="bytes"/> length; the backing
    /// array may be larger. Growth is monotonic — repeated calls with smaller
    /// sizes reuse the existing buffer.
    /// </summary>
    public Span<byte> RentPackA(int bytes)
    {
        if (bytes <= 0) return Span<byte>.Empty;
        if (_packA is null || _packA.Length < bytes)
            _packA = new byte[bytes];
        return _packA.AsSpan(0, bytes);
    }

    /// <summary>
    /// Rent a pack-B buffer of at least <paramref name="bytes"/> bytes. Same
    /// semantics as <see cref="RentPackA"/>.
    /// </summary>
    public Span<byte> RentPackB(int bytes)
    {
        if (bytes <= 0) return Span<byte>.Empty;
        if (_packB is null || _packB.Length < bytes)
            _packB = new byte[bytes];
        return _packB.AsSpan(0, bytes);
    }

    /// <summary>
    /// Rent a K-split partial-C buffer of at least <paramref name="bytes"/>
    /// bytes. Lazily allocated — only consumed when the parallelism axis is K
    /// (Phase G).
    /// </summary>
    public Span<byte> RentKSplitC(int bytes)
    {
        if (bytes <= 0) return Span<byte>.Empty;
        if (_kSplitC is null || _kSplitC.Length < bytes)
            _kSplitC = new byte[bytes];
        return _kSplitC.AsSpan(0, bytes);
    }

    /// <summary>
    /// Diagnostic: total bytes held by this thread's pool. Useful for tests
    /// and for the <see cref="BlasManagedStats"/> diagnostic surface.
    /// </summary>
    public long TotalBytesHeld =>
        (_packA?.Length ?? 0L) + (_packB?.Length ?? 0L) + (_kSplitC?.Length ?? 0L);

    /// <summary>
    /// Test-only: reset the per-thread pool. Used by tests that need to
    /// verify allocation behavior from a known starting state.
    /// </summary>
    internal void ResetForTest()
    {
        _packA = null;
        _packB = null;
        _kSplitC = null;
    }
}
