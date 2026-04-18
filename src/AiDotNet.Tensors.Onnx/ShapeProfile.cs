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
    private readonly int[] _shape;

    /// <summary>The ONNX value name (as written in the model file).</summary>
    public string Name { get; }

    /// <summary>
    /// Concrete dimension sizes. <c>-1</c> means the dim is dynamic and was
    /// not resolved via <see cref="OnnxImportOptions.OverrideInputShapes"/>.
    /// Callers that need to Execute the plan must pass an input tensor whose
    /// shape matches this profile, with dynamic dims bound to concrete values.
    /// Exposed as a read-only view so callers can't mutate the profile's
    /// internal array through the reference they get here.
    /// </summary>
    public IReadOnlyList<int> Shape => _shape;

    /// <summary>
    /// Copies the dimension sizes into a fresh <see cref="int"/> array.
    /// Use this when you need to pass the shape to an API that takes
    /// <c>int[]</c> (e.g. <c>new Tensor&lt;T&gt;(shape)</c>).
    /// </summary>
    public int[] ToShapeArray() => (int[])_shape.Clone();

    /// <summary>ONNX element type name, e.g. <c>"float"</c>, <c>"int64"</c>.</summary>
    public string ElementType { get; }

    public ShapeProfile(string name, int[] shape, string elementType)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        // Defensive copy — the caller is free to mutate the array they
        // passed in without silently corrupting this profile.
        _shape = (int[])shape.Clone();
        ElementType = elementType ?? throw new ArgumentNullException(nameof(elementType));
    }
}
