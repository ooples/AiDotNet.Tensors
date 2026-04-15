namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// A previously-benchmarked kernel configuration chosen as the winner for a
/// specific <see cref="KernelId"/> + <see cref="ShapeProfile"/> combination on
/// the current hardware. Round-trippable through JSON for on-disk persistence.
/// </summary>
/// <remarks>
/// <para><b>Variant</b> is a kernel-specific string identifying the winning
/// implementation (e.g. <c>"blocked-4x4"</c>, <c>"warp-shuffle-v3"</c>,
/// <c>"mkl-sgemm"</c>). <b>Parameters</b> carry the tuned hyperparameters as
/// a free-form string dictionary so each kernel category can serialise its
/// own parameter schema without the cache needing category-specific types.</para>
///
/// <para><b>MeasuredGflops</b> and <b>MeasuredTimeMs</b> are advisory telemetry —
/// useful for diagnostics and for future "re-tune if hardware moved" heuristics,
/// but the cache lookup doesn't depend on them.</para>
/// </remarks>
public sealed class KernelChoice
{
    /// <summary>Kernel-specific identifier for the winning variant.</summary>
    public string Variant { get; set; } = "";

    /// <summary>
    /// Free-form tuned hyperparameters. Keys and values are strings; each kernel
    /// category defines the expected schema (e.g. GEMM might store
    /// <c>{"TileM":"128","TileN":"128","TileK":"16"}</c>).
    /// </summary>
    public Dictionary<string, string> Parameters { get; set; } = new();

    /// <summary>Measured throughput when the winner was chosen (GFLOPS).</summary>
    public double MeasuredGflops { get; set; }

    /// <summary>Measured execution time when the winner was chosen (milliseconds).</summary>
    public double MeasuredTimeMs { get; set; }

    /// <summary>
    /// UTC timestamp when this entry was recorded. Used for diagnostics and to
    /// support "tune again after N days" policies without breaking current
    /// lookups.
    /// </summary>
    public DateTime RecordedAtUtc { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Version of the <see cref="KernelChoice"/> schema. Bumped when the JSON
    /// layout changes in a non-backward-compatible way so the cache reader
    /// can reject stale files as corruption/mismatch.
    /// </summary>
    public int SchemaVersion { get; set; } = 1;
}
