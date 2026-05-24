using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;

/// <summary>
/// Test-only shape instrumenter. Records every (m, n, k, transA, transB, dtype)
/// tuple that flows through <c>BlasProvider</c>'s public Try* GEMM entry points
/// when <see cref="Enabled"/> is true. Deduplicates by shape key and counts
/// frequency so the catalog can be sorted by hottest shapes.
///
/// <para>
/// Wire-up: a test fixture or test body sets <c>BlasProvider.ShapeLogHook</c> to
/// call <see cref="Record"/>. The hook is null in production builds; recording
/// has zero cost when disabled.
/// </para>
///
/// <para>
/// Thread-safety: <see cref="_counts"/> is a <see cref="ConcurrentDictionary{TKey, TValue}"/>;
/// concurrent test threads can <see cref="Record"/> safely. The <see cref="Enabled"/>
/// flag is a primitive bool — readers may observe a stale value briefly across thread
/// boundaries but every reader settles on the right value before <see cref="Snapshot"/>
/// is read.
/// </para>
/// </summary>
public static class ShapeInstrumenter
{
    /// <summary>
    /// Master switch. Defaults to off (and off when not running under the test-suite
    /// instrumentation pass). Can be flipped by env var <c>AIDOTNET_INSTRUMENT_SHAPES=1</c>
    /// at process start, or set directly by a test body.
    /// </summary>
    public static bool Enabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_INSTRUMENT_SHAPES") == "1";

    // ValueTuple as dictionary key: net471-safe (avoids C# 10 record struct syntax
    // concerns) and equality-comparable without writing GetHashCode.
    private static readonly ConcurrentDictionary<(int M, int N, int K, bool TA, bool TB, DType D), int> _counts
        = new();

    /// <summary>
    /// Records one observation of a GEMM shape. No-op when <see cref="Enabled"/> is false.
    /// </summary>
    public static void Record(int m, int n, int k, bool transA, bool transB, DType dtype)
    {
        if (!Enabled) return;
        var key = (m, n, k, transA, transB, dtype);
        _counts.AddOrUpdate(key, 1, (_, count) => count + 1);
    }

    /// <summary>
    /// Returns the current observations sorted by descending frequency.
    /// </summary>
    public static IReadOnlyList<Shape> Snapshot()
    {
        return _counts
            .Select(kv => new Shape(
                Name: $"Instrumented_{kv.Key.M}x{kv.Key.N}x{kv.Key.K}_{(kv.Key.TA ? "TA" : "NA")}_{(kv.Key.TB ? "TB" : "NB")}_{kv.Key.D}",
                M: kv.Key.M, N: kv.Key.N, K: kv.Key.K,
                TransA: kv.Key.TA, TransB: kv.Key.TB,
                Dtype: kv.Key.D,
                Frequency: kv.Value,
                Source: "instrumented:test-suite"))
            .OrderByDescending(s => s.Frequency)
            .ToList();
    }

    /// <summary>Clears all recorded observations.</summary>
    public static void Reset() => _counts.Clear();

    /// <summary>
    /// Writes the current snapshot as pretty-printed JSON to <paramref name="path"/>.
    /// Creates the parent directory if missing. Used by the assembly fixture that
    /// fires at end-of-test-run to commit <c>artifacts/perf/instrumented-shapes.json</c>.
    /// </summary>
    public static void DumpToJson(string path)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        var json = JsonSerializer.Serialize(Snapshot(), new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }
}
