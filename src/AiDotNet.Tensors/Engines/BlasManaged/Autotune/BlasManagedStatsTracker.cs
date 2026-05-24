using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Process-wide diagnostic counters for BlasManaged. Thread-safe increments
/// via <see cref="Interlocked.Increment(ref long)"/>. Returned to callers as
/// an immutable <see cref="BlasManagedStats"/> snapshot.
/// </summary>
internal static class BlasManagedStatsTracker
{
    private static long _autotuneHits;
    private static long _autotuneMisses;
    private static long _jitEmissions;
    private static long _jitCacheHits;
    private static long _packCacheHits;
    private static long _packCacheMisses;
    private static long _packCacheBytes;

    public static void IncrementAutotuneHit() => Interlocked.Increment(ref _autotuneHits);
    public static void IncrementAutotuneMiss() => Interlocked.Increment(ref _autotuneMisses);
    public static void IncrementJitEmission() => Interlocked.Increment(ref _jitEmissions);
    public static void IncrementJitCacheHit() => Interlocked.Increment(ref _jitCacheHits);
    public static void IncrementPackCacheHit() => Interlocked.Increment(ref _packCacheHits);
    public static void IncrementPackCacheMiss() => Interlocked.Increment(ref _packCacheMisses);
    public static void AddPackCacheBytes(long bytes) => Interlocked.Add(ref _packCacheBytes, bytes);

    public static BlasManagedStats Snapshot() => new BlasManagedStats
    {
        AutotuneHits = Interlocked.Read(ref _autotuneHits),
        AutotuneMisses = Interlocked.Read(ref _autotuneMisses),
        JitEmissions = Interlocked.Read(ref _jitEmissions),
        JitCacheHits = Interlocked.Read(ref _jitCacheHits),
        PackCacheHits = Interlocked.Read(ref _packCacheHits),
        PackCacheMisses = Interlocked.Read(ref _packCacheMisses),
        PackCacheBytes = Interlocked.Read(ref _packCacheBytes),
    };

    public static void Reset()
    {
        Interlocked.Exchange(ref _autotuneHits, 0);
        Interlocked.Exchange(ref _autotuneMisses, 0);
        Interlocked.Exchange(ref _jitEmissions, 0);
        Interlocked.Exchange(ref _jitCacheHits, 0);
        Interlocked.Exchange(ref _packCacheHits, 0);
        Interlocked.Exchange(ref _packCacheMisses, 0);
        Interlocked.Exchange(ref _packCacheBytes, 0);
    }
}
