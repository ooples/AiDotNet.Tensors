using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;

/// <summary>
/// Numeric element type for a benchmark shape.
/// </summary>
public enum DType
{
    Single,
    Double,
}

/// <summary>
/// A single GEMM shape in the BlasManaged benchmark catalog.
/// </summary>
/// <param name="Name">Stable identifier for output rows and dedup.</param>
/// <param name="M">Rows of C / rows of op(A).</param>
/// <param name="N">Cols of C / cols of op(B).</param>
/// <param name="K">Inner dimension (cols of op(A) / rows of op(B)).</param>
/// <param name="TransA">Whether A is transposed.</param>
/// <param name="TransB">Whether B is transposed.</param>
/// <param name="Dtype">FP32 or FP64.</param>
/// <param name="Frequency">
/// How often this shape was observed during instrumentation. 0 for curated workload shapes
/// that are kept regardless of frequency to force architecture coverage.
/// </param>
/// <param name="Source">
/// Origin tag: <c>workload:BERT-base FFN</c>, <c>instrumented:test-suite</c>, etc.
/// </param>
public record Shape(
    string Name,
    int M, int N, int K,
    bool TransA, bool TransB,
    DType Dtype,
    int Frequency,
    string Source);

/// <summary>
/// Sub-issue A (#369) deliverable: the unified shape catalog for the BlasManaged
/// perf bench. Merges curated workload shapes (always kept, force coverage) with
/// the top-N most-frequent instrumented shapes (when <c>artifacts/perf/instrumented-shapes.json</c>
/// is present). Final catalog size is bounded to 50-80 by the <see cref="ShapeCatalogTest"/>
/// gate.
/// </summary>
public static class ShapeCatalog
{
    /// <summary>
    /// Maximum number of shapes in the catalog. <see cref="ShapeCatalogTest"/>'s
    /// in-range assertion is (50, 80) — 80 is the hard upper.
    /// </summary>
    internal const int MaxCatalogSize = 80;

    /// <summary>
    /// All shapes used by the perf harness. Lazy-initialized from <see cref="WorkloadShapes"/>
    /// (always) plus the top-N instrumented shapes by frequency (when the JSON harvest
    /// exists). Deduplicated by (M, N, K, TransA, TransB, Dtype); when a shape appears
    /// in both sources the workload version wins (richer <see cref="Shape.Source"/>).
    /// </summary>
    public static IReadOnlyList<Shape> All { get; } = LoadCatalog();

    private static IReadOnlyList<Shape> LoadCatalog()
    {
        var result = new List<Shape>(WorkloadShapes.All);

        // Load instrumented shapes if the harvest file exists. Resolved by walking up
        // from the test binary directory to find the repo root.
        var instrumentedPath = FindInstrumentedJson();
        if (instrumentedPath is not null)
        {
            try
            {
                var json = File.ReadAllText(instrumentedPath);
                var instrumented = JsonSerializer.Deserialize<List<Shape>>(json) ?? new List<Shape>();

                int slots = MaxCatalogSize - result.Count;
                if (slots > 0)
                    result.AddRange(instrumented.Take(slots));
            }
            catch
            {
                // Malformed JSON or IO failure — fall back to workload-only catalog.
                // The shape-count test will fail, which is the correct gate.
            }
        }

        // Dedupe by (M, N, K, TransA, TransB, Dtype). Workload entries win when the
        // same shape is also instrumented — their Source field is more informative.
        var deduped = result
            .GroupBy(s => (s.M, s.N, s.K, s.TransA, s.TransB, s.Dtype))
            .Select(g => g.OrderBy(s => s.Source.StartsWith("workload:") ? 0 : 1).First())
            .ToList();

        return deduped;
    }

    private static string? FindInstrumentedJson()
    {
        // First honour explicit env override (matches the bootstrap's output path resolution).
        var envOverride = Environment.GetEnvironmentVariable("AIDOTNET_INSTRUMENT_OUT");
        if (!string.IsNullOrWhiteSpace(envOverride) && File.Exists(envOverride))
            return envOverride;

        // Walk up from the test binary directory looking for artifacts/perf/instrumented-shapes.json.
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir != null; i++)
        {
            var candidate = Path.Combine(dir, "artifacts", "perf", "instrumented-shapes.json");
            if (File.Exists(candidate)) return candidate;
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }
}
