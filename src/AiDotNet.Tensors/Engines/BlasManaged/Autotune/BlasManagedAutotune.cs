using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Autotune cache facade for BlasManaged. Wraps the codebase's existing
/// <see cref="AutotuneCache"/> with BlasManaged-specific encode/decode of
/// GEMM parameters and chosen strategy.
///
/// <para>
/// The cache learns the winning (axis, blocking) per (M, N, K, trans, precision)
/// shape on the current hardware across calls. First call to a shape pays the
/// benchmark cost; subsequent calls are a cache hit and use the cached choice
/// directly. Cache persists across process restarts via JSON-on-disk under
/// <c>~/.aidotnet/autotune/{fingerprint}/</c>.
/// </para>
///
/// <para>
/// This facade only does encode/decode and cache lookup/store. The first-call
/// benchmark logic (running candidate strategies, measuring, picking winner)
/// lives in the AutotuneDispatcher (Task H3).
/// </para>
/// </summary>
internal static class BlasManagedAutotune
{
    private static readonly KernelId GemmKernelId = new("blas-managed", "gemm");

    // Precision tag encoding for ShapeProfile (we can't put Type into an int[]).
    private const int PrecisionTagFp32 = 1;
    private const int PrecisionTagFp64 = 2;

    /// <summary>
    /// Encode a BlasManaged GEMM signature into the existing cache's
    /// <see cref="ShapeProfile"/>. The int-array layout is:
    /// <c>[M, N, K, transA, transB, precisionTag, mr, nr, hasEpilogue, isDeterministic]</c>.
    /// PR #402 CodeRabbit fix: <c>isDeterministic</c> is part of the key because
    /// <c>AxisSelector.Select</c> branches on it — without this discriminator,
    /// deterministic and non-deterministic calls reuse the same cache slot and
    /// the wrong axis can be replayed under determinism mode.
    /// </summary>
    /// <summary>
    /// Source-compat overload that defaults <c>isDeterministic</c> to false. Existing
    /// callers (and tests) compiled before the determinism-keyed cache landed continue
    /// to work without changes. New code SHOULD pass <c>isDeterministic</c> explicitly
    /// to preserve the deterministic/non-deterministic cache separation.
    /// </summary>
    public static ShapeProfile EncodeShape<T>(
        int m, int n, int k,
        bool transA, bool transB,
        int mr, int nr,
        bool hasEpilogue) where T : unmanaged
        => EncodeShape<T>(m, n, k, transA, transB, mr, nr, hasEpilogue, isDeterministic: false);

    public static ShapeProfile EncodeShape<T>(
        int m, int n, int k,
        bool transA, bool transB,
        int mr, int nr,
        bool hasEpilogue,
        bool isDeterministic) where T : unmanaged
    {
        int precisionTag;
        if (typeof(T) == typeof(double)) precisionTag = PrecisionTagFp64;
        else if (typeof(T) == typeof(float)) precisionTag = PrecisionTagFp32;
        else throw new NotSupportedException($"Autotune does not support T={typeof(T).Name}.");

        return new ShapeProfile(
            m, n, k,
            transA ? 1 : 0,
            transB ? 1 : 0,
            precisionTag,
            mr, nr,
            hasEpilogue ? 1 : 0,
            isDeterministic ? 1 : 0);
    }

    /// <summary>
    /// Encode a chosen autotune result (axis + blocking + thread count) as
    /// a <see cref="KernelChoice"/> ready for cache storage.
    /// </summary>
    public static KernelChoice EncodeChoice(
        ParallelismAxis axis,
        int mc, int nc, int kc,
        int threadCount,
        double measuredTimeMs)
    {
        var choice = new KernelChoice
        {
            Variant = $"axis={axis},mc={mc},nc={nc},kc={kc}",
            Parameters = new Dictionary<string, string>
            {
                { "axis", axis.ToString() },
                { "mc", mc.ToString() },
                { "nc", nc.ToString() },
                { "kc", kc.ToString() },
                { "threadCount", threadCount.ToString() },
            },
            MeasuredTimeMs = measuredTimeMs,
            MeasuredGflops = 0,  // Optional; can be computed by caller if desired.
        };
        return choice;
    }

    /// <summary>
    /// Decode a <see cref="KernelChoice"/> back into the BlasManaged autotune fields.
    /// </summary>
    public static (ParallelismAxis Axis, int Mc, int Nc, int Kc, int ThreadCount) DecodeChoice(KernelChoice choice)
    {
        if (choice is null) throw new ArgumentNullException(nameof(choice));
        // PR #402 CodeRabbit fix: harden against malformed / older on-disk
        // entries where Parameters may be null or contain non-positive
        // blocking values that would destabilize the strategy pipeline if
        // honored. On any defect, fall back to the safe (M-axis, 64×64×64,
        // single-thread) default that matches FallbackToHeuristic.
        if (choice.Parameters is null)
        {
            return (ParallelismAxis.M, 64, 64, 64, 1);
        }

        ParallelismAxis axis = ParallelismAxis.M;
        if (choice.Parameters.TryGetValue("axis", out string? axisStr)
            && Enum.TryParse<ParallelismAxis>(axisStr, ignoreCase: false, out var parsed))
        {
            axis = parsed;
        }

        int mc = ParsePositiveIntOrDefault(choice.Parameters, "mc", 64);
        int nc = ParsePositiveIntOrDefault(choice.Parameters, "nc", 64);
        int kc = ParsePositiveIntOrDefault(choice.Parameters, "kc", 64);
        // threadCount: 0 is a sentinel for "auto" (procs at run time); allow it.
        int threadCount = ParseIntOrDefault(choice.Parameters, "threadCount", 0);
        if (threadCount < 0) threadCount = 0;

        return (axis, mc, nc, kc, threadCount);
    }

    private static int ParseIntOrDefault(IDictionary<string, string> dict, string key, int defaultValue)
    {
        if (dict.TryGetValue(key, out string? value)
            && int.TryParse(value, System.Globalization.NumberStyles.Integer,
                             System.Globalization.CultureInfo.InvariantCulture, out int result))
        {
            return result;
        }
        return defaultValue;
    }

    // PR #402 CodeRabbit fix: reject zero / negative blocking-parameter
    // values from on-disk cache entries so the strategy pipeline isn't
    // handed an invalid mc/nc/kc that would divide-by-zero in the inner
    // tile loops or allocate a zero-byte packed buffer.
    private static int ParsePositiveIntOrDefault(IDictionary<string, string> dict, string key, int defaultValue)
    {
        int parsed = ParseIntOrDefault(dict, key, defaultValue);
        return parsed > 0 ? parsed : defaultValue;
    }

    /// <summary>
    /// Look up a cached autotune choice for the given shape. Returns null on miss.
    /// </summary>
    public static (ParallelismAxis Axis, int Mc, int Nc, int Kc, int ThreadCount)? TryLookup(ShapeProfile shape)
    {
        KernelChoice? choice = AutotuneCache.Lookup(GemmKernelId, shape);
        if (choice is null) return null;
        return DecodeChoice(choice);
    }

    /// <summary>
    /// Store a chosen tuning result. Idempotent (last-write-wins).
    /// </summary>
    public static void Store(ShapeProfile shape, ParallelismAxis axis, int mc, int nc, int kc, int threadCount, double measuredTimeMs)
    {
        var choice = EncodeChoice(axis, mc, nc, kc, threadCount, measuredTimeMs);
        AutotuneCache.Store(GemmKernelId, shape, choice);
    }

    /// <summary>
    /// #375: Store a tuned (strategy + blocking) unit for a shape, tagged with the kernel
    /// version (G2/G11). Strategy and blocking are persisted together so a learned strategy
    /// is never paired with blocking tuned for a different one.
    /// </summary>
    public static void StoreStrategy(ShapeProfile shape, PackingMode mode, ParallelismAxis axis,
        int mc, int nc, int kc, int threadCount, string kernelVersion)
    {
        var choice = EncodeChoice(axis, mc, nc, kc, threadCount, measuredTimeMs: 0);
        choice.Parameters["packingMode"] = mode.ToString();
        choice.Parameters["kernelVersion"] = kernelVersion;
        AutotuneCache.Store(GemmKernelId, shape, choice);
    }

    /// <summary>
    /// #375: Look up a tuned (strategy + blocking) unit. Returns null on miss OR when the
    /// stored entry's kernel version doesn't match the current build (stale → ignore, G2).
    /// </summary>
    public static (PackingMode Mode, ParallelismAxis Axis, int Mc, int Nc, int Kc, int ThreadCount)?
        TryLookupStrategy(ShapeProfile shape)
    {
        EnsurePrewarmLoaded();
        KernelChoice? choice = AutotuneCache.Lookup(GemmKernelId, shape);
        if (choice?.Parameters is null) return null;
        if (!choice.Parameters.TryGetValue("kernelVersion", out var ver)
            || ver != BlasKernelVersion.Current)
            return null; // stale or pre-strategy entry → treat as miss (G2)
        if (!choice.Parameters.TryGetValue("packingMode", out var modeStr)
            || !Enum.TryParse<PackingMode>(modeStr, out var mode))
            return null;
        var (axis, mc, nc, kc, tc) = DecodeChoice(choice);
        return (mode, axis, mc, nc, kc, tc);
    }

    /// <summary>
    /// #375 Phase 4: seed a strategy entry from the shipped pre-warm ONLY if no
    /// version-matching local entry exists. Local learned entries always win — the
    /// shipped pre-warm is a cold-start convenience, not an override.
    /// </summary>
    public static void SeedFromShippedIfAbsent(ShapeProfile shape, PackingMode mode,
        ParallelismAxis axis, int mc, int nc, int kc, int threadCount)
    {
        if (TryLookupStrategy(shape) is not null) return;  // local/learned wins
        StoreStrategy(shape, mode, axis, mc, nc, kc, threadCount, BlasKernelVersion.Current);
    }

    private static int _prewarmLoaded;

    /// <summary>
    /// #375 Phase 4: load shipped pre-warm entries for the current fingerprint once.
    /// Best-effort — missing/garbled resource → no-op. Each line:
    /// "M N K fp64 transA transB strategy mc nc kc threadCount". Seeds only where no
    /// local learned entry exists (local always wins); version-tagged via StoreStrategy.
    /// </summary>
    internal static void EnsurePrewarmLoaded()
    {
        if (System.Threading.Interlocked.CompareExchange(ref _prewarmLoaded, 1, 0) != 0) return;
        try
        {
            string fp = Helpers.Autotune.HardwareFingerprint.Current;
            string resourceName =
                $"AiDotNet.Tensors.Engines.BlasManaged.Autotune.prewarm.{fp}.prewarm.json";
            using var stream = typeof(BlasManagedAutotune).Assembly.GetManifestResourceStream(resourceName);
            if (stream is null) return; // no pre-warm shipped for this fingerprint
            using var reader = new System.IO.StreamReader(stream);
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                var p = line.Split(' ');
                if (p.Length < 11) continue;
                if (!int.TryParse(p[0], out int m) || !int.TryParse(p[1], out int n)
                    || !int.TryParse(p[2], out int k)) continue;
                bool fp64 = p[3] == "1";
                bool transA = p[4] == "1", transB = p[5] == "1";
                if (!Enum.TryParse<PackingMode>(p[6], out var mode)) continue;
                if (!int.TryParse(p[7], out int mc) || !int.TryParse(p[8], out int nc)
                    || !int.TryParse(p[9], out int kc) || !int.TryParse(p[10], out int tc)) continue;
                var shape = fp64
                    ? EncodeShape<double>(m, n, k, transA, transB, 0, 0, false, false)
                    : EncodeShape<float>(m, n, k, transA, transB, 0, 0, false, false);
                SeedFromShippedIfAbsent(shape, mode, ParallelismAxis.M, mc, nc, kc, tc);
            }
        }
        catch { /* best-effort */ }
    }
}
