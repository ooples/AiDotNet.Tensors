using System.Collections.Generic;

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
/// perf bench. Starts empty until A.2 (instrumented shapes) and A.3 (workload shapes)
/// land, then A.4 merges them. Once populated, <see cref="ShapeCatalogTest"/>'s count
/// assertion passes.
/// </summary>
public static class ShapeCatalog
{
    /// <summary>
    /// All shapes used by the perf harness. Empty until A.2/A.3/A.4 land — the count
    /// assertion in <see cref="ShapeCatalogTest.Catalog_Has_Between_50_And_80_Shapes"/>
    /// is the intentional gate that forces those tasks to land before downstream
    /// sub-issues (B/C/D/E/F/G) can claim they're measured against this catalog.
    /// </summary>
    public static IReadOnlyList<Shape> All { get; } = new List<Shape>();
}
