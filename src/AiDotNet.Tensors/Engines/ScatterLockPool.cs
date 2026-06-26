using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Shared, fixed-size striped lock pool for thread-safe gradient scatter/accumulation in backward
/// kernels. Replaces the recurring anti-pattern of allocating a per-element <c>new object[N]</c> lock
/// array (plus one <c>new object()</c> per element) on EVERY backward call — at scatter scale that is
/// hundreds of KB to MB of pure GC churn per call. A fixed power-of-two pool, allocated once at type
/// init, maps each scatter index to a stable lock via <c>index &amp; (Count - 1)</c>, so the same element
/// always serializes on the same lock (correctness preserved); distinct elements only rarely collide
/// (brief, harmless extra contention). Zero per-call allocation.
/// </summary>
/// <remarks>
/// ONLY safe for single-lock critical sections (acquire one lock, mutate, release). Do NOT use where a
/// thread holds two of these locks at once with an index-ordering deadlock-avoidance scheme: striping
/// can map two distinct ordered indices to the same stripe across threads and reintroduce a deadlock.
/// Such call sites (e.g. symmetric matrix updates that lock min(i,j) then max(i,j)) must keep their own
/// per-element locks.
/// </remarks>
internal static class ScatterLockPool
{
    /// <summary>Number of stripes. Power of two so <c>index &amp; (Count - 1)</c> is a valid mask.</summary>
    internal const int Count = 16384;

    /// <summary>The shared lock objects (length <see cref="Count"/>, a power of two).</summary>
    internal static readonly object[] Locks = Create();

    private static object[] Create()
    {
        var locks = new object[Count];
        for (int i = 0; i < locks.Length; i++) locks[i] = new object();
        return locks;
    }

    /// <summary>Returns the stable stripe lock for the given scatter index.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static object For(long index) => Locks[(int)(index & (Count - 1))];
}
