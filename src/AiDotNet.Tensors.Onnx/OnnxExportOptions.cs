namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Options for <see cref="OnnxExporter"/>. Every property is optional; null takes the documented default.
/// </summary>
public sealed class OnnxExportOptions
{
    /// <summary>The minimum opset the exporter can write: LayerNormalization first exists in opset 17.</summary>
    public const long MinimumOpsetVersion = 17;

    /// <summary>The opset written when <see cref="OpsetVersion"/> is null: 20, the first with <c>Gelu</c>.</summary>
    public const long DefaultOpsetVersion = 20;

    /// <summary>
    /// The highest opset the exporter writes: the translator's opset-dependent forms (<c>Gelu</c> from 20,
    /// <c>ReduceMean</c> axes as an input from 18) are verified against ONNX Runtime from 17 to 20, and a later opset
    /// may change an operator's signature.
    /// </summary>
    public const long MaximumOpsetVersion = 20;

    /// <summary>
    /// Default-domain opset version of the exported model, from <see cref="MinimumOpsetVersion"/> to
    /// <see cref="MaximumOpsetVersion"/>. Null = <see cref="DefaultOpsetVersion"/>; ONNX Runtime 1.16+ runs it. Opsets
    /// below 20 write GELU as its tanh-form decomposition instead.
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
            long opset = OpsetVersion ?? DefaultOpsetVersion;
            if (opset < MinimumOpsetVersion || opset > MaximumOpsetVersion)
                throw new ArgumentOutOfRangeException(nameof(OpsetVersion),
                    $"Opset {opset} is outside the range [{MinimumOpsetVersion}, {MaximumOpsetVersion}] the exporter writes.");
            return opset;
        }
    }
}
