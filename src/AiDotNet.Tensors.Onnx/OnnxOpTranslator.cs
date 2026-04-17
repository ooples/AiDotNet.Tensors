using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Translates one ONNX operator into the equivalent <c>IEngine</c> call(s).
/// The importer calls <see cref="Translate"/> under a GraphMode scope, so
/// the translator just invokes the engine method and ties the result back
/// to the node's output name via <see cref="OnnxTranslationContext{T}.PutTensor"/>.
/// </summary>
/// <typeparam name="T">Element type of the model (typically <c>float</c>).</typeparam>
public interface IOnnxOpTranslator<T> where T : unmanaged
{
    /// <summary>The ONNX <c>op_type</c> this translator handles.</summary>
    string OpType { get; }

    /// <summary>
    /// The ONNX operator domain. <c>null</c> or empty means the default
    /// onnx.ai domain. Custom domains are used for extensions like
    /// <c>"com.microsoft"</c>.
    /// </summary>
    string? Domain { get; }

    /// <summary>
    /// Emits engine ops for the given node. The translator reads inputs via
    /// <see cref="OnnxTranslationContext{T}.GetTensor"/>, calls engine methods
    /// on <see cref="OnnxTranslationContext{T}.Engine"/>, and stores every
    /// output via <see cref="OnnxTranslationContext{T}.PutTensor"/>.
    /// </summary>
    void Translate(OnnxTranslationContext<T> ctx, NodeProto node);
}

/// <summary>
/// Registry of <see cref="IOnnxOpTranslator{T}"/> instances. Built-in
/// translators are registered once per process via <see cref="BuildDefault"/>;
/// callers can add custom translators for extension operators or their own
/// polyfills.
/// </summary>
public sealed class OnnxOpTranslatorRegistry<T> where T : unmanaged
{
    private readonly Dictionary<(string domain, string opType), IOnnxOpTranslator<T>> _translators = new();

    /// <summary>
    /// Registers a translator. Keyed by <c>(domain, op_type)</c>; duplicate
    /// registrations replace the earlier translator, letting callers override
    /// a built-in with a specialized implementation.
    /// </summary>
    public void Register(IOnnxOpTranslator<T> translator)
    {
        if (translator is null) throw new ArgumentNullException(nameof(translator));
        _translators[(translator.Domain ?? string.Empty, translator.OpType)] = translator;
    }

    /// <summary>
    /// Looks up a translator by ONNX node's <c>op_type</c> and <c>domain</c>.
    /// Returns <c>null</c> if no translator is registered.
    /// </summary>
    public IOnnxOpTranslator<T>? Find(string opType, string? domain)
    {
        _translators.TryGetValue((domain ?? string.Empty, opType), out var t);
        return t;
    }

    /// <summary>
    /// Builds a registry populated with every built-in translator shipped
    /// in this assembly. The full Phase 1 operator set — arithmetic,
    /// activations, normalizations, convolutions, tensor manipulation,
    /// and the com.microsoft.Attention extension.
    /// </summary>
    public static OnnxOpTranslatorRegistry<T> BuildDefault()
    {
        var r = new OnnxOpTranslatorRegistry<T>();
        Operators.ArithOperators.Register(r);
        Operators.ActivationOperators.Register(r);
        Operators.NormOperators.Register(r);
        Operators.ConvOperators.Register(r);
        Operators.TensorManipOperators.Register(r);
        Operators.AttentionOperator.Register(r);
        Operators.RecurrentOperators.Register(r);
        Operators.QuantizedOperators.Register(r);
        Operators.MathOperators.Register(r);
        return r;
    }
}
