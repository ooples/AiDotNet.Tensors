using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Mutable state threaded through ONNX operator translation. Holds the
/// engine, the live tensor registry (node outputs keyed by ONNX name), and
/// helpers for reading ONNX attributes.
/// </summary>
public sealed class OnnxTranslationContext<T> where T : unmanaged
{
    private readonly Dictionary<string, Tensor<T>> _tensorsByName;
    private readonly HashSet<string> _placeholderInputs;

    internal OnnxTranslationContext(IEngine engine, Dictionary<string, Tensor<T>> tensorsByName, OnnxImportOptions options, long defaultOpset = 13, HashSet<string>? placeholderInputs = null)
    {
        Engine = engine;
        _tensorsByName = tensorsByName;
        Options = options;
        DefaultOpset = defaultOpset;
        _placeholderInputs = placeholderInputs ?? new HashSet<string>(StringComparer.Ordinal);
    }

    /// <summary>
    /// Returns <c>true</c> if <paramref name="name"/> refers to a graph input
    /// placeholder (user-supplied runtime tensor) rather than a constant
    /// initializer or a node-output tensor. Translators use this to route
    /// ops whose semantics depend on runtime input content (e.g. OneHot,
    /// ConstantOfShape) through the lazy path instead of eagerly baking
    /// trace-time placeholder contents into the plan.
    /// </summary>
    public bool IsPlaceholderInput(string name) => _placeholderInputs.Contains(name);

    /// <summary>The engine against which engine ops are traced.</summary>
    public IEngine Engine { get; }

    /// <summary>The options passed to the importer.</summary>
    public OnnxImportOptions Options { get; }

    /// <summary>
    /// Opset version for the default ONNX domain (""), as declared by the
    /// model's opset_import entries. Translators whose default-attribute
    /// semantics changed with opset version (e.g. <c>Softmax</c>'s axis
    /// default flipped from 1 to -1 at opset 13) read this to pick the
    /// right default. Falls back to 13 (the modern default) when the
    /// model declares no opset.
    /// </summary>
    public long DefaultOpset { get; }

    /// <summary>
    /// Resolves a tensor by its ONNX name. Tensors come from three sources,
    /// all unified under this lookup: graph inputs (placeholders), initializers
    /// (frozen weights), and outputs of previously-translated nodes.
    /// </summary>
    public Tensor<T> GetTensor(string name)
    {
        if (string.IsNullOrEmpty(name))
            throw new ArgumentException("Empty tensor name. Malformed ONNX graph.", nameof(name));
        if (!_tensorsByName.TryGetValue(name, out var t))
            throw new InvalidDataException(
                $"ONNX tensor '{name}' referenced before being produced. The graph is malformed " +
                $"or the topological sort missed a dependency.");
        return t;
    }

    /// <summary>
    /// Stores a translated node's output tensor under its ONNX name so
    /// downstream nodes can look it up via <see cref="GetTensor"/>.
    /// </summary>
    public void PutTensor(string name, Tensor<T> tensor)
    {
        if (string.IsNullOrEmpty(name))
            throw new ArgumentException("Empty output name.", nameof(name));
        _tensorsByName[name] = tensor ?? throw new ArgumentNullException(nameof(tensor));
    }

    /// <summary>
    /// Returns <c>true</c> if a tensor with the given name exists in the
    /// registry. Used by translators for optional inputs (e.g. Conv's bias
    /// input name may be empty when the op has no bias).
    /// </summary>
    public bool HasTensor(string? name) =>
        !string.IsNullOrEmpty(name) && _tensorsByName.ContainsKey(name!);

    // ─── Attribute accessors ──────────────────────────────────────────

    /// <summary>
    /// Returns the named attribute on the node, or <c>null</c> if absent.
    /// Attributes with a typed getter below should prefer those accessors —
    /// this is the untyped fallback.
    /// </summary>
    public AttributeProto? GetAttribute(NodeProto node, string name)
    {
        for (int i = 0; i < node.Attribute.Count; i++)
            if (node.Attribute[i].Name == name)
                return node.Attribute[i];
        return null;
    }

    public long GetIntAttr(NodeProto node, string name, long defaultValue)
    {
        var a = GetAttribute(node, name);
        return a is null ? defaultValue : a.I;
    }

    public int GetIntAttrAsInt(NodeProto node, string name, int defaultValue)
    {
        var a = GetAttribute(node, name);
        return a is null ? defaultValue : checked((int)a.I);
    }

    public float GetFloatAttr(NodeProto node, string name, float defaultValue)
    {
        var a = GetAttribute(node, name);
        return a is null ? defaultValue : a.F;
    }

    public string? GetStringAttr(NodeProto node, string name, string? defaultValue)
    {
        var a = GetAttribute(node, name);
        if (a is null) return defaultValue;
        return System.Text.Encoding.UTF8.GetString(a.S.ToByteArray());
    }

    public int[]? GetIntArrayAttr(NodeProto node, string name)
    {
        var a = GetAttribute(node, name);
        if (a is null) return null;
        var result = new int[a.Ints.Count];
        for (int i = 0; i < a.Ints.Count; i++)
            result[i] = checked((int)a.Ints[i]);
        return result;
    }
}
