using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-F3 (#374 follow-up): per-shape autotune cache that picks managed vs native
/// dispatch based on measurement.
///
/// <para>
/// On cache miss: allocate matching temp buffers, run both <see cref="BlasManagedLib.Gemm{T}"/>
/// and <see cref="BlasProvider.TryGemmEx"/> (with autotune bypass), time each, cache the winner.
/// On cache hit: return the cached preference in O(1).
/// </para>
///
/// <para>
/// Bypass mechanism: <see cref="BypassAutotune"/> is a <see cref="ThreadStaticAttribute"/>
/// bool that the measurement helpers flip while calling <see cref="BlasProvider.TryGemmEx"/>.
/// This routes the measurement call past the autotune check (so native dispatch fires)
/// without affecting concurrent threads in production code.
/// </para>
///
/// <para>
/// Memory: in-memory <see cref="ConcurrentDictionary{TKey, TValue}"/>. On-disk
/// persistence is deferred to a follow-up — F.3 ships the measurement logic; F.4
/// would add the file-backed cache so warmup amortizes across processes.
/// </para>
/// </summary>
public static class PrefersManagedCache
{
    private readonly record struct ShapeKey(
        int M, int N, int K, bool TransA, bool TransB, byte Dtype);

    private static readonly ConcurrentDictionary<ShapeKey, bool> _cache = new();

    /// <summary>
    /// Thread-local bypass flag. Set to true inside the measurement helpers so
    /// the nested <see cref="BlasProvider.TryGemmEx"/> call routes to native
    /// without re-entering this cache.
    /// </summary>
    [ThreadStatic] internal static bool BypassAutotune;

    /// <summary>
    /// Number of probe iterations per backend during measurement. Small (3) keeps
    /// first-call overhead low; the noise floor is acceptable because the goal
    /// is to pick the better path, not to produce a precise speedup number.
    /// </summary>
    private const int ProbeIters = 3;

    /// <summary>
    /// Clears the cache. Used by tests to ensure measurement fires fresh.
    /// </summary>
    public static void Clear() => _cache.Clear();

    /// <summary>Returns the current number of cached decisions.</summary>
    public static int Count => _cache.Count;

    /// <summary>
    /// Returns true if BlasManaged is the preferred dispatch for this shape.
    /// Measures on cache miss; returns cached result on hit.
    /// </summary>
    /// <param name="m">Rows of C.</param>
    /// <param name="n">Cols of C.</param>
    /// <param name="k">Inner dim.</param>
    /// <param name="transA">Whether A is transposed.</param>
    /// <param name="transB">Whether B is transposed.</param>
    /// <param name="dtype">Element type (typeof(float) or typeof(double)).</param>
    public static bool PrefersManaged(
        int m, int n, int k, bool transA, bool transB, Type dtype)
    {
        byte dtypeKey = dtype == typeof(float) ? (byte)1
                       : dtype == typeof(double) ? (byte)2 : (byte)0;
        if (dtypeKey == 0) return true;  // unknown dtype — managed path is the only supported

        var key = new ShapeKey(m, n, k, transA, transB, dtypeKey);
        if (_cache.TryGetValue(key, out bool cached)) return cached;

        // Cache miss — measure. If native isn't available, prefer managed.
        if (!BlasProvider.IsAvailable)
        {
            _cache[key] = true;
            return true;
        }

        bool prefersManaged = dtypeKey == 1
            ? MeasureFp32(m, n, k, transA, transB)
            : MeasureFp64(m, n, k, transA, transB);
        _cache[key] = prefersManaged;
        return prefersManaged;
    }

    private static bool MeasureFp32(int m, int n, int k, bool transA, bool transB)
    {
        int aRows = transA ? k : m;
        int aCols = transA ? m : k;
        int bRows = transB ? n : k;
        int bCols = transB ? k : n;
        int aLen = aRows * aCols;
        int bLen = bRows * bCols;
        int cLen = m * n;

        var a = ArrayPool<float>.Shared.Rent(aLen);
        var b = ArrayPool<float>.Shared.Rent(bLen);
        var c = ArrayPool<float>.Shared.Rent(cLen);
        try
        {
            var rng = new Random(42);
            for (int i = 0; i < aLen; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < bLen; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            int lda = aCols, ldb = bCols, ldc = n;

            // Warmup each path once.
            BlasManagedLib.Gemm<float>(
                new ReadOnlySpan<float>(a, 0, aLen), lda, transA,
                new ReadOnlySpan<float>(b, 0, bLen), ldb, transB,
                new Span<float>(c, 0, cLen), ldc,
                m, n, k);

            BypassAutotune = true;
            try
            {
                BlasProvider.TryGemmEx(
                    m, n, k,
                    a, 0, lda, transA,
                    b, 0, ldb, transB,
                    c, 0, ldc);
            }
            finally { BypassAutotune = false; }

            // Time managed.
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < ProbeIters; i++)
            {
                BlasManagedLib.Gemm<float>(
                    new ReadOnlySpan<float>(a, 0, aLen), lda, transA,
                    new ReadOnlySpan<float>(b, 0, bLen), ldb, transB,
                    new Span<float>(c, 0, cLen), ldc,
                    m, n, k);
            }
            sw.Stop();
            double managedMs = sw.Elapsed.TotalMilliseconds / ProbeIters;

            // Time native (with autotune bypass).
            BypassAutotune = true;
            double nativeMs;
            try
            {
                sw.Restart();
                for (int i = 0; i < ProbeIters; i++)
                {
                    BlasProvider.TryGemmEx(
                        m, n, k,
                        a, 0, lda, transA,
                        b, 0, ldb, transB,
                        c, 0, ldc);
                }
                sw.Stop();
                nativeMs = sw.Elapsed.TotalMilliseconds / ProbeIters;
            }
            finally { BypassAutotune = false; }

            return managedMs <= nativeMs;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(a);
            ArrayPool<float>.Shared.Return(b);
            ArrayPool<float>.Shared.Return(c);
        }
    }

    private static bool MeasureFp64(int m, int n, int k, bool transA, bool transB)
    {
        int aRows = transA ? k : m;
        int aCols = transA ? m : k;
        int bRows = transB ? n : k;
        int bCols = transB ? k : n;
        int aLen = aRows * aCols;
        int bLen = bRows * bCols;
        int cLen = m * n;

        var a = ArrayPool<double>.Shared.Rent(aLen);
        var b = ArrayPool<double>.Shared.Rent(bLen);
        var c = ArrayPool<double>.Shared.Rent(cLen);
        try
        {
            var rng = new Random(42);
            for (int i = 0; i < aLen; i++) a[i] = rng.NextDouble() * 2 - 1;
            for (int i = 0; i < bLen; i++) b[i] = rng.NextDouble() * 2 - 1;

            int lda = aCols, ldb = bCols, ldc = n;

            BlasManagedLib.Gemm<double>(
                new ReadOnlySpan<double>(a, 0, aLen), lda, transA,
                new ReadOnlySpan<double>(b, 0, bLen), ldb, transB,
                new Span<double>(c, 0, cLen), ldc,
                m, n, k);

            BypassAutotune = true;
            try
            {
                BlasProvider.TryGemmEx(
                    m, n, k,
                    a, 0, lda, transA,
                    b, 0, ldb, transB,
                    c, 0, ldc);
            }
            finally { BypassAutotune = false; }

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < ProbeIters; i++)
            {
                BlasManagedLib.Gemm<double>(
                    new ReadOnlySpan<double>(a, 0, aLen), lda, transA,
                    new ReadOnlySpan<double>(b, 0, bLen), ldb, transB,
                    new Span<double>(c, 0, cLen), ldc,
                    m, n, k);
            }
            sw.Stop();
            double managedMs = sw.Elapsed.TotalMilliseconds / ProbeIters;

            BypassAutotune = true;
            double nativeMs;
            try
            {
                sw.Restart();
                for (int i = 0; i < ProbeIters; i++)
                {
                    BlasProvider.TryGemmEx(
                        m, n, k,
                        a, 0, lda, transA,
                        b, 0, ldb, transB,
                        c, 0, ldc);
                }
                sw.Stop();
                nativeMs = sw.Elapsed.TotalMilliseconds / ProbeIters;
            }
            finally { BypassAutotune = false; }

            return managedMs <= nativeMs;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(a);
            ArrayPool<double>.Shared.Return(b);
            ArrayPool<double>.Shared.Return(c);
        }
    }
}
