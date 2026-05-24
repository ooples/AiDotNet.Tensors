using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Helpers.Autotune;
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
    private static readonly object _diskLock = new object();
    private static bool _diskLoaded;
    private static string? _diskPath;

    /// <summary>
    /// Thread-local bypass flag. Set to true inside the measurement helpers so
    /// the nested <see cref="BlasProvider.TryGemmEx"/> call routes to native
    /// without re-entering this cache.
    /// </summary>
    [ThreadStatic] internal static bool BypassAutotune;

    /// <summary>
    /// Sub-F4 (#374 follow-up): on-disk persistence. The cache reads + writes
    /// JSON at <see cref="DiskPath"/>. Hardware fingerprint is validated on
    /// load — entries from a different host are ignored. Set to <c>null</c>
    /// to disable persistence entirely.
    /// </summary>
    public static string? DiskPath
    {
        get
        {
            if (_diskPath is not null) return _diskPath;
            // Default: ~/.aidotnet/autotune/blas-managed-routing.json
            try
            {
                string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                if (string.IsNullOrEmpty(home)) return null;
                return Path.Combine(home, ".aidotnet", "autotune", "blas-managed-routing.json");
            }
            catch { return null; }
        }
        set { _diskPath = value; _diskLoaded = false; }
    }

    /// <summary>
    /// File schema written to <see cref="DiskPath"/>. The fingerprint is checked
    /// on load — mismatches discard the cache (different hardware, different optima).
    /// </summary>
    private sealed class DiskFormat
    {
        public string SchemaVersion { get; set; } = "1";
        public string HardwareFingerprint { get; set; } = string.Empty;
        public System.Collections.Generic.List<DiskEntry> Entries { get; set; } = new();
    }

    private sealed class DiskEntry
    {
        public int M { get; set; }
        public int N { get; set; }
        public int K { get; set; }
        public bool TransA { get; set; }
        public bool TransB { get; set; }
        public byte Dtype { get; set; }
        public bool PrefersManaged { get; set; }
    }

    /// <summary>
    /// Loads cached entries from <see cref="DiskPath"/> if the file exists AND
    /// its hardware fingerprint matches the current host. No-op on subsequent calls.
    /// Idempotent and thread-safe.
    /// </summary>
    public static void LoadFromDisk()
    {
        lock (_diskLock)
        {
            if (_diskLoaded) return;
            _diskLoaded = true;

            string? path = DiskPath;
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;

            try
            {
                string json = File.ReadAllText(path);
                var data = JsonSerializer.Deserialize<DiskFormat>(json);
                if (data is null) return;

                string current = HardwareFingerprint.Current.ToString();
                if (data.HardwareFingerprint != current)
                {
                    // Different hardware — discard the file's entries.
                    return;
                }

                foreach (var entry in data.Entries)
                {
                    var key = new ShapeKey(entry.M, entry.N, entry.K, entry.TransA, entry.TransB, entry.Dtype);
                    _cache.TryAdd(key, entry.PrefersManaged);
                }
            }
            catch
            {
                // Corrupted file or IO failure — proceed with empty cache.
            }
        }
    }

    /// <summary>
    /// Persists the current cache to <see cref="DiskPath"/>. Atomic via
    /// temp-file + rename. Best-effort: failures are swallowed (the cache
    /// remains valid in-memory; persistence is a perf optimization).
    /// </summary>
    public static void SaveToDisk()
    {
        string? path = DiskPath;
        if (string.IsNullOrEmpty(path)) return;

        try
        {
            var data = new DiskFormat
            {
                SchemaVersion = "1",
                HardwareFingerprint = HardwareFingerprint.Current.ToString(),
            };
            foreach (var kv in _cache)
            {
                data.Entries.Add(new DiskEntry
                {
                    M = kv.Key.M, N = kv.Key.N, K = kv.Key.K,
                    TransA = kv.Key.TransA, TransB = kv.Key.TransB,
                    Dtype = kv.Key.Dtype,
                    PrefersManaged = kv.Value,
                });
            }

            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

            // Atomic write via temp + rename.
            string tmp = path + ".tmp";
            File.WriteAllText(tmp, JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true }));
            if (File.Exists(path)) File.Delete(path);
            File.Move(tmp, path);
        }
        catch
        {
            // Best-effort; cache stays valid in-memory.
        }
    }

    /// <summary>
    /// Number of probe iterations per backend during measurement.
    ///
    /// <para>
    /// Sub-F5: bumped from 3 → 10. The F.3/F.4 diagnostic showed sub-ms shapes
    /// (M=1 inference) get noisy results at 3 iters — MobileNetV2_fc oscillated
    /// between managed-routed and native-routed across runs. 10 iters reduces
    /// variance enough that borderline shapes converge to a stable decision.
    /// First-call cost rises to ~10× normal GEMM, but only once per (shape, host)
    /// over the cache's lifetime.
    /// </para>
    /// </summary>
    private const int ProbeIters = 10;

    /// <summary>
    /// Clears the in-memory cache and resets the disk-loaded flag. Used by tests
    /// to ensure measurement fires fresh. Does NOT delete <see cref="DiskPath"/>.
    /// </summary>
    public static void Clear()
    {
        _cache.Clear();
        lock (_diskLock) { _diskLoaded = false; }
    }

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

        // Sub-F4: ensure on-disk cache is loaded (idempotent, thread-safe).
        if (!_diskLoaded) LoadFromDisk();

        var key = new ShapeKey(m, n, k, transA, transB, dtypeKey);
        if (_cache.TryGetValue(key, out bool cached)) return cached;

        // Cache miss — measure. If native isn't available, prefer managed.
        if (!BlasProvider.IsAvailable)
        {
            _cache[key] = true;
            SaveToDisk();
            return true;
        }

        bool prefersManaged = dtypeKey == 1
            ? MeasureFp32(m, n, k, transA, transB)
            : MeasureFp64(m, n, k, transA, transB);
        _cache[key] = prefersManaged;
        // Sub-F4: persist after every cache update so a crashed process still
        // contributes its decisions to the next run.
        SaveToDisk();
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
            // PR #402 CodeRabbit fix: bail to "prefer managed" when the native
            // probe fails. Pre-fix the return values from TryGemmEx were
            // ignored, so when native execution failed (missing libopenblas,
            // unsupported shape, etc.) the timed loop completed in ~0 ms and
            // the cache was poisoned with "native is faster" routing — but no
            // native call ever succeeded, so subsequent calls would also fail
            // silently or worse, produce garbage output.
            bool nativeOk;
            try
            {
                nativeOk = BlasProvider.TryGemmEx(
                    m, n, k,
                    a, 0, lda, transA,
                    b, 0, ldb, transB,
                    c, 0, ldc);
            }
            finally { BypassAutotune = false; }
            if (!nativeOk) return true;

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
                    if (!BlasProvider.TryGemmEx(
                        m, n, k,
                        a, 0, lda, transA,
                        b, 0, ldb, transB,
                        c, 0, ldc))
                    {
                        // Native started failing mid-loop — abandon the probe
                        // and prefer managed. The partial timing isn't trustworthy.
                        return true;
                    }
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

            // PR #402 CodeRabbit fix: same native-probe-failure guard as the
            // FP32 path above. Bail to "prefer managed" if the warmup or any
            // timed iteration of the native probe reports failure.
            BypassAutotune = true;
            bool nativeOk;
            try
            {
                nativeOk = BlasProvider.TryGemmEx(
                    m, n, k,
                    a, 0, lda, transA,
                    b, 0, ldb, transB,
                    c, 0, ldc);
            }
            finally { BypassAutotune = false; }
            if (!nativeOk) return true;

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
                    if (!BlasProvider.TryGemmEx(
                        m, n, k,
                        a, 0, lda, transA,
                        b, 0, ldb, transB,
                        c, 0, ldc))
                    {
                        return true;
                    }
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
