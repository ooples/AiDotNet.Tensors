namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Options for <see cref="OnnxExporter"/>. Every property is optional; null takes the documented default.
/// </summary>
public sealed class OnnxExportOptions
{
    /// <summary>The minimum opset the exporter can write: LayerNormalization first exists in opset 17.</summary>
    public const long MinimumOpsetVersion = 17;

    /// <summary>
    /// Default-domain opset version of the exported model. Null = 20, the first opset with <c>Gelu</c>; ONNX Runtime
    /// 1.16+ runs it. Lower values (down to <see cref="MinimumOpsetVersion"/>) write GELU as its tanh-form
    /// decomposition instead.
    /// </summary>
    public long? OpsetVersion { get; set; }

    /// <summary>The model's <c>producer_name</c>. Null = "AiDotNet.Tensors.Onnx".</summary>
    public string? ProducerName { get; set; }

    /// <summary>The graph's name. Null = "main".</summary>
    public string? GraphName { get; set; }

    internal long EffectiveOpsetVersion
    {
        get
        {
            long opset = OpsetVersion ?? 20;
            if (opset < MinimumOpsetVersion)
                throw new ArgumentOutOfRangeException(nameof(OpsetVersion),
                    $"Opset {opset} is below the minimum {MinimumOpsetVersion} the exporter writes.");
            return opset;
        }
    }
}
