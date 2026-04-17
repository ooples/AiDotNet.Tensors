namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Describes the expected shape of an ONNX graph input or output. Dimensions
/// that are unresolved (dynamic axes / named parameters without a concrete
/// value after <see cref="OnnxImportOptions.OverrideInputShapes"/>) are
/// represented as <c>-1</c>, matching the convention used elsewhere in
/// AiDotNet.Tensors (e.g. <c>SymbolicShape</c>).
/// </summary>
public sealed class ShapeProfile
{
    /// <summary>The ONNX value name (as written in the model file).</summary>
    public string Name { get; }

    /// <summary>
    /// Concrete dimension sizes. <c>-1</c> means the dim is dynamic and was
    /// not resolved via <see cref="OnnxImportOptions.OverrideInputShapes"/>.
    /// Callers that need to Execute the plan must pass an input tensor whose
    /// shape matches this profile, with dynamic dims bound to concrete values.
    /// </summary>
    public int[] Shape { get; }

    /// <summary>ONNX element type name, e.g. <c>"float"</c>, <c>"int64"</c>.</summary>
    public string ElementType { get; }

    public ShapeProfile(string name, int[] shape, string elementType)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        ElementType = elementType ?? throw new ArgumentNullException(nameof(elementType));
    }
}
