using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Result of <see cref="OnnxImporter.Import{T}"/>. When every operator in the
/// model has a translator and the graph topologically resolves, <see cref="Plan"/>
/// is non-null and ready to <c>Execute()</c>. When one or more operators are
/// unsupported, <see cref="UnsupportedOperators"/> names them and
/// <see cref="Plan"/> is <c>null</c> (unless
/// <see cref="OnnxImportOptions.StrictMode"/> is set, in which case the import
/// throws instead of returning).
/// </summary>
/// <typeparam name="T">Element type of the compiled plan (typically <c>float</c>).</typeparam>
public sealed class OnnxImportResult<T> where T : unmanaged
{
    /// <summary>
    /// The compiled inference plan, or <c>null</c> if any operator was
    /// unsupported. Execute with <c>Plan.Execute()</c> after writing the
    /// input data into the buffer returned by <see cref="NamedInputs"/>.
    /// </summary>
    public ICompiledPlan<T>? Plan { get; }

    /// <summary>
    /// ONNX operator names (with domain prefix if non-default) that had no
    /// translator registered. Empty when the import succeeded. Users can
    /// register custom translators via
    /// <see cref="OnnxOpTranslatorRegistry{T}.Register"/> to cover these.
    /// </summary>
    public IReadOnlyList<string> UnsupportedOperators { get; }

    /// <summary>Shape profiles for the model's declared graph inputs.</summary>
    public IReadOnlyDictionary<string, ShapeProfile> NamedInputs { get; }

    /// <summary>Shape profiles for the model's declared graph outputs.</summary>
    public IReadOnlyDictionary<string, ShapeProfile> NamedOutputs { get; }

    /// <summary>Model producer name (e.g. "pytorch", "tensorflow").</summary>
    public string ProducerName { get; }

    /// <summary>ONNX IR version the model declares.</summary>
    public long IrVersion { get; }

    /// <summary>
    /// The placeholder input tensors keyed by ONNX input name. Callers write
    /// their per-inference input data into the returned buffers (using
    /// <c>Tensor.Data.AsSpan()</c> or similar) before calling
    /// <c>Plan.Execute()</c>. The compiled plan references these exact
    /// tensor instances, so in-place updates are visible on the next Execute.
    /// </summary>
    public IReadOnlyDictionary<string, Tensor<T>> Inputs { get; }

    internal OnnxImportResult(
        ICompiledPlan<T>? plan,
        IReadOnlyList<string> unsupportedOperators,
        IReadOnlyDictionary<string, ShapeProfile> namedInputs,
        IReadOnlyDictionary<string, ShapeProfile> namedOutputs,
        IReadOnlyDictionary<string, Tensor<T>> inputs,
        string producerName,
        long irVersion)
    {
        Plan = plan;
        UnsupportedOperators = unsupportedOperators ?? throw new ArgumentNullException(nameof(unsupportedOperators));
        NamedInputs = namedInputs ?? throw new ArgumentNullException(nameof(namedInputs));
        NamedOutputs = namedOutputs ?? throw new ArgumentNullException(nameof(namedOutputs));
        Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        ProducerName = producerName ?? string.Empty;
        IrVersion = irVersion;
    }
}
