namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Options controlling ONNX model import. Zero-config by default: pass
/// <c>null</c> to <see cref="OnnxImporter.Import{T}"/> and the importer uses
/// the most permissive settings (collect unsupported ops, fail only on truly
/// malformed input).
/// </summary>
public sealed class OnnxImportOptions
{
    /// <summary>
    /// When <c>true</c>, the importer throws <see cref="NotSupportedException"/>
    /// as soon as it encounters an operator that has no translator registered.
    /// Default <c>false</c>: unsupported operators are collected into
    /// <see cref="OnnxImportResult{T}.UnsupportedOperators"/> and the returned
    /// <c>Plan</c> is <c>null</c> so the caller can diagnose what's missing.
    /// </summary>
    public bool StrictMode { get; set; }

    /// <summary>
    /// Concrete sizes for dynamic axes. ONNX models saved with named dim params
    /// (e.g. <c>batch_size</c>, <c>sequence_length</c>) need those resolved to
    /// ints before the plan can execute. Map the dim name → resolved value.
    /// <para>
    /// Example: a BERT model with inputs of shape <c>[batch, sequence]</c>
    /// where both are dim params. Pass
    /// <c>new Dictionary&lt;string,int&gt; { ["batch"] = 1, ["sequence"] = 128 }</c>.
    /// </para>
    /// </summary>
    public IReadOnlyDictionary<string, int>? DimensionOverrides { get; set; }

    /// <summary>
    /// Full-shape overrides per graph input. Takes precedence over
    /// <see cref="DimensionOverrides"/> if both name a given axis. Used when
    /// a caller knows the complete input shape up front and wants to skip
    /// per-dim resolution.
    /// </summary>
    public IReadOnlyDictionary<string, int[]>? OverrideInputShapes { get; set; }
}
