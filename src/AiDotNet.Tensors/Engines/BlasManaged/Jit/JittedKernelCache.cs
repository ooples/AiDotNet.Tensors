using System;
using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Cache of shape-specialized JIT-emitted microkernel delegates, keyed by
/// <see cref="KernelKey"/>. Each entry is a <see cref="System.Reflection.Emit.DynamicMethod"/>-emitted
/// closure tuned to a specific (M, N, K, trans, precision, arch, packing,
/// epilogue) tuple.
///
/// <para>
/// Phase J1 ships the cache infrastructure WITHOUT actual IL emission. Every
/// <see cref="TryGetJittedKernel"/> call returns null for keys that have not
/// been explicitly stored; callers fall back to the hand-written microkernels.
/// The cache structure is in place for a future task to add real IL emission
/// (J2-J5 in the original plan, deferred because the hand-written 8×16
/// AVX-512 kernel is already near peak).
/// </para>
///
/// <para>
/// When IL emission lands, the cache will:
/// </para>
/// <list type="number">
///   <item>Look up the key in the ConcurrentDictionary (hit = use cached delegate)</item>
///   <item>On miss after 3+ calls to the same shape, fire emit on a background <c>Task.Run</c></item>
///   <item>Subsequent calls hit the cache</item>
///   <item>LRU eviction when total emitted IL exceeds <see cref="DefaultMaxJitCacheBytes"/></item>
/// </list>
///
/// <para>
/// Honours <see cref="NativeAotDetector.IsDynamicCodeSupported"/>: under
/// NativeAOT the cache is permanently empty and emission is never attempted.
/// </para>
/// </summary>
internal static class JittedKernelCache
{
    private static readonly ConcurrentDictionary<KernelKey, Delegate> _cache = new();

    /// <summary>
    /// Default cache capacity in emitted-IL bytes. Cache entries beyond this
    /// total trigger LRU eviction (Phase J8 — currently deferred since no
    /// entries are added by the JIT emitter).
    /// </summary>
    public const long DefaultMaxJitCacheBytes = 64L * 1024 * 1024;  // 64 MB

    /// <summary>
    /// Try to look up a previously-stored kernel delegate. Returns null on
    /// cache miss. Always returns null when dynamic code is not supported
    /// (e.g., NativeAOT).
    /// </summary>
    public static Delegate? TryGetJittedKernel(KernelKey key)
    {
        if (!NativeAotDetector.IsDynamicCodeSupported) return null;
        if (_cache.TryGetValue(key, out var del))
        {
            BlasManagedStatsTracker.IncrementJitCacheHit();
            return del;
        }
        return null;
    }

    /// <summary>
    /// Store an emitted delegate under a given key. Idempotent — last write
    /// wins. Increments <see cref="BlasManagedStats.JitEmissions"/>.
    /// </summary>
    public static void Store(KernelKey key, Delegate emittedDelegate)
    {
        if (emittedDelegate is null) throw new ArgumentNullException(nameof(emittedDelegate));
        if (!NativeAotDetector.IsDynamicCodeSupported) return;
        _cache[key] = emittedDelegate;
        BlasManagedStatsTracker.IncrementJitEmission();
    }

    /// <summary>
    /// Total entries currently in the cache. Diagnostic only.
    /// </summary>
    public static int Count => _cache.Count;

    /// <summary>
    /// Clear the entire cache. Used by <see cref="BlasManaged.ClearCaches"/>
    /// and by tests.
    /// </summary>
    public static void Clear() => _cache.Clear();
}
